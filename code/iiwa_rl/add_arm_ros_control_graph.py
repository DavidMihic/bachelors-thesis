"""
add_arm_ros_control_graph.py - Dodaje ROS2 Action Graph (JointState publish/
subscribe + ArticulationController) na kmr_iiwa_full_base.usd, POSLIJE
URDF->USD konverzije. Isti obrazac kao add_camera_ros_graph.py - graf se bakea
u sam USD i izvrsava se automatski svaki tick unutar bilo kojeg skripta koji
ucita ovaj USD i pokrene world.step() (npr. cmd_vel_bridge.py), nema potrebe
za posebnim procesom.

Ovo je Isaac Sim strana infrastrukture za MoveIt + ros2_control: MoveIt
racuna trajektoriju i salje je preko FollowJointTrajectory akcije
joint_trajectory_controlleru (ros2_control), koji svaki fizicki korak salje
poziciju/brzinu na topic_based_ros2_control (TopicBasedSystem hardware
interface), konfiguriran s:
    joint_states_topic:   /isaac_joint_states
    joint_commands_topic: /isaac_joint_commands
Ovaj graf je Isaac Sim strana tog para topica - subscribe na komande,
primijeni ih preko ArticulationControllera, publish trenutno stanje natrag.

NAMJERNO ne dira gripper ni bazu: gripper prsti se i dalje kontroliraju
odvojeno preko gripper_driver.py (/gripper_cmd sucelje), baza odvojeno preko
cmd_vel_bridge.py (direktna set_linear_velocities/set_angular_velocities na
artikulacijski root). SubscribeJointState prosljedjuje ArticulationControlleru
SAMO imena zglobova koja stvarno stignu u dolaznoj JointState poruci - koji
su to zglobovi odredjuje ros2_control strana (topic_based_ros2_control.xacro
konfiguracija bi trebala navesti samo iiwa_joint_1..7), ne ovaj graf.

PublishJointState objavljuje stanje CIJELE artikulacije (svi DOF, ukljucujuci
gripper prste) - ros2_control na drugoj strani cita po imenu i ignorira
zglobove koje ne ocekuje, pa ovo nije problem, samo vrijedi znati da poruka
sadrzi vise od samih 7 zglobova ruke.

Pokretanje (headless), nakon svake convert_urdf_usd.py konverzije robota:
    ./isaaclab.sh -p add_arm_ros_control_graph.py assets/configuration/kmr_iiwa_full_base.usd

Idempotentno - siguran za visestruko pokretanje na istom fajlu (brise i
ponovno gradi Graph prim ako vec postoji, umjesto da duplicira).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Dodaje ROS2 JointState publish/subscribe + ArticulationController graf za ruku, nakon URDF->USD konverzije."
)
parser.add_argument("usd_path", type=str, help="Putanja do _base.usd (mijenja se in-place).")
parser.add_argument(
    "--articulation-prim-name",
    type=str,
    default="base_link",
    help="Ime (ne puna putanja) prima artikulacijskog roota - trazi se po imenu, isti pristup kao add_camera_ros_graph.py.",
)
parser.add_argument("--commands-topic", type=str, default="/isaac_joint_commands")
parser.add_argument("--states-topic", type=str, default="/isaac_joint_states")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.graph.core as og  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from pxr import Usd  # noqa: E402

# Headless AppLauncher ne ucitava ROS2 bridge ekstenziju automatski (isto
# ogranicenje kao u add_camera_ros_graph.py).
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

GRAPH_NAME = "ROS_ArmControl"


def find_prim_by_name(stage: Usd.Stage, name: str):
    """Link moze zavrsiti pod razlicitim root prim imenima ovisno o
    defaultPrimu - trazimo po imenu, ne po punoj putanji."""
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def main():
    usd_context = omni.usd.get_context()
    success = usd_context.open_stage(args_cli.usd_path)
    if not success:
        raise ValueError(f"Ne mogu otvoriti USD kroz omni.usd context: {args_cli.usd_path}")
    for _ in range(5):
        simulation_app.update()
    stage = usd_context.get_stage()
    if stage is None:
        raise ValueError("omni.usd context nema aktivan stage nakon open_stage - neocekivano.")

    articulation_prim = find_prim_by_name(stage, args_cli.articulation_prim_name)
    if articulation_prim is None:
        raise ValueError(
            f"Artikulacijski prim '{args_cli.articulation_prim_name}' ne postoji na stageu - provjeri ime."
        )
    articulation_path = str(articulation_prim.GetPath())

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise ValueError("Stage nema postavljen defaultPrim - ne mogu sigurno smjestiti Graph.")
    graph_path = default_prim.GetPath().AppendPath(f"Graph/{GRAPH_NAME}")

    if stage.GetPrimAtPath(str(graph_path)).IsValid():
        print(f"Graph vec postoji na {graph_path}, brisem i gradim iznova")
        stage.RemovePrim(str(graph_path))
        stage.GetRootLayer().Save()

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": str(graph_path), "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                # Imena zglobova dolaze iz dolazne poruke (SubscribeJointState
                # ih parsira iz JointState.name polja) - ArticulationController
                # ih mapira po imenu na DOF indekse u ciljanoj artikulaciji.
                # Namjerno NEMA staticki postavljenog jointNames popisa ovdje -
                # koji se zglobovi stvarno salju odredjuje ros2_control
                # konfiguracija na ROS strani.
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
            ],
            keys.SET_VALUES: [
                ("SubscribeJointState.inputs:topicName", args_cli.commands_topic),
                ("PublishJointState.inputs:topicName", args_cli.states_topic),
                ("PublishJointState.inputs:targetPrim", [articulation_path]),
                ("ArticulationController.inputs:targetPrim", [articulation_path]),
            ],
        },
    )
    print(f"ROS2 Action Graph za ruku kreiran na {graph_path}")
    print(f"  Artikulacija: {articulation_path}")
    print(f"  Subscribe (komande): {args_cli.commands_topic}")
    print(f"  Publish (stanje):    {args_cli.states_topic}")

    stage.GetRootLayer().Save()
    print(f"\nSpremljeno u {args_cli.usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
