"""
add_lidar_ros_graph.py - Dodaje RTX Lidar prim + ROS2 Action Graph (LaserScan
publish) na kmr_iiwa_full_base.usd, POSLIJE URDF->USD konverzije.

Isti razlog postojanja kao add_camera_ros_graph.py: URDF ne nosi senzorske
tagove za nas format, RTX Lidar prim (OmniLidar) se ne moze dobiti
konverzijom vec ga treba stvoriti programski preko IsaacSensorCreateRtxLidar
komande, a ROS2 OmniGraph nikad ne nastaje automatski.

lidar_link (goli frame, vidi lidar_mount.xacro) definira SAMO mjesto i
orijentaciju montaze - sam senzorski prim (OmniLidar) se vjesa kao njegovo
dijete s lokalnim identity transformom, pa cijela pozicija dolazi iz URDF-a.

TF za lidar_link NE treba dodavati ovdje - add_camera_ros_graph.py vec
publisha PublishTF s tf_root=base_link, sto pokriva cijelo kinematicko
stablo (pa i lidar_link) bez obzira koja se skripta pokrene prva.

Pokretanje (headless), nakon svake convert_urdf_usd.py konverzije robota:
    ./isaaclab.sh -p add_lidar_ros_graph.py assets/configuration/kmr_iiwa_full_base.usd

Idempotentno - siguran za visestruko pokretanje na istom fajlu.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Dodaje RTX Lidar prim + ROS2 Action Graph na robota nakon URDF->USD konverzije."
)
parser.add_argument(
    "usd_path", type=str, help="Putanja do _base.usd (mijenja se in-place)."
)
parser.add_argument(
    "--lidar-parent",
    type=str,
    default="lidar_link",
    help="Ime linka na koji se Lidar prim vjesa (mora vec postojati iz URDF-a).",
)
parser.add_argument(
    "--config",
    type=str,
    default="Example_Rotary_2D",
    help="NVIDIA RTX Lidar profil. Example_Rotary_2D = generican 360-stupanjeva "
    "rotirajuci 2D lidar, dovoljan dok se ne bira stvarni model senzora.",
)
parser.add_argument("--topic", type=str, default="scan")
parser.add_argument("--frame-id", type=str, default="lidar_link")
parser.add_argument(
    "--publish-point-cloud",
    action="store_true",
    default=False,
    help="Uz LaserScan, dodaj i drugi ROS2RtxLidarHelper koji publisha "
    "PointCloud2 (korisno za RViz debug, nepotrebno za samu navigaciju).",
)
parser.add_argument(
    "--near-range-m",
    type=float,
    default=0.05,
    help="Minimalni domet senzora (omni:sensor:Core:nearRangeM). Default profila "
    "Example_Rotary_2D je 1.0 m - preveliko za ovog robota: vrata su nakon "
    "hvata kvake samo ~0.6 m od senzora, pa postaju nevidljiva ispod te "
    "granice bez ove izmjene.",
)
parser.add_argument("--point-cloud-topic", type=str, default="lidar_points")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.graph.core as og  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from pxr import Gf, Usd  # noqa: E402

# Isto kao u add_camera_ros_graph.py - headless AppLauncher ne ucitava ove
# ekstenzije automatski. Bez isaacsim.sensors.rtx, IsaacSensorCreateRtxLidar
# puca s "unknown command"; bez isaacsim.ros2.bridge, ROS2Context/
# ROS2RtxLidarHelper pucaju s "unrecognized type" (isto kao kod kamere).
enable_extension("isaacsim.ros2.bridge")
enable_extension("isaacsim.sensors.rtx")
simulation_app.update()

LIDAR_NAME = "Lidar"


def find_prim_by_name(stage: Usd.Stage, name: str):
    """Link moze zavrsiti pod razlicitim root prim imenima ovisno o defaultPrimu -
    trazimo po imenu, ne po punoj putanji."""
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def main():
    usd_context = omni.usd.get_context()
    success = usd_context.open_stage(args_cli.usd_path)
    if not success:
        raise ValueError(
            f"Ne mogu otvoriti USD kroz omni.usd context: {args_cli.usd_path}"
        )
    for _ in range(5):
        simulation_app.update()
    stage = usd_context.get_stage()
    if stage is None:
        raise ValueError(
            "omni.usd context nema aktivan stage nakon open_stage - neocekivano."
        )

    parent = find_prim_by_name(stage, args_cli.lidar_parent)
    if parent is None:
        raise ValueError(
            f"Link '{args_cli.lidar_parent}' ne postoji na stageu - provjeri ime/URDF "
            "(lidar_mount.xacro / kmr_iiwa.urdf.xacro)."
        )

    # Graph MORA biti unutar defaultPrim podstabla - isto obrazlozenje kao u
    # add_camera_ros_graph.py (reference/payload arc povlaci samo to podstablo).
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise ValueError(
            "Stage nema postavljen defaultPrim - ne mogu sigurno smjestiti Graph."
        )
    graph_path = default_prim.GetPath().AppendPath("Graph/ROS_Lidar")

    lidar_path = parent.GetPath().AppendChild(LIDAR_NAME)

    # ---- RTX Lidar prim (OmniLidar) ----
    # Identity transform - lidar_link vec nosi pravu poziciju/orijentaciju
    # iz URDF-a (vidi lidar_mount.xacro), pa senzor sjedi TOCNO na frameu.
    if stage.GetPrimAtPath(lidar_path).IsValid():
        print(f"Lidar prim vec postoji na {lidar_path}, brisem i gradim iznova")
        stage.RemovePrim(lidar_path)

    sensor_attributes = {"omni:sensor:Core:nearRangeM": args_cli.near_range_m}
    _, lidar_prim = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path=f"/{LIDAR_NAME}",
        parent=str(parent.GetPath()),
        config=args_cli.config,
        translation=Gf.Vec3d(0.0, 0.0, 0.0),
        orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
        **sensor_attributes,
    )
    print(f"RTX Lidar prim ({args_cli.config}) kreiran na {lidar_prim.GetPath()}")

    # ---- ROS2 Action Graph ----
    # OnPlaybackTick -> RunOnce -> RenderProduct(cameraPrim=Lidar) -> LaserScanPublish.
    # Isti obrazac kao add_camera_ros_graph.py; RenderProduct ovdje samo gleda
    # Lidar prim umjesto Camera prim, width/height se ne postavljaju (lidar ne
    # proizvodi sliku, sluzbeni GUI tutorial ih isto ne trazi).
    if stage.GetPrimAtPath(str(graph_path)).IsValid():
        print(f"Graph vec postoji na {graph_path}, brisem i gradim iznova")
        stage.RemovePrim(str(graph_path))
        stage.GetRootLayer().Save()

    keys = og.Controller.Keys
    create_nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
        ("LaserScanPublish", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
    ]
    connect = [
        ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
        ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),
        ("RenderProduct.outputs:execOut", "LaserScanPublish.inputs:execIn"),
        (
            "RenderProduct.outputs:renderProductPath",
            "LaserScanPublish.inputs:renderProductPath",
        ),
        ("Context.outputs:context", "LaserScanPublish.inputs:context"),
    ]
    set_values = [
        ("RenderProduct.inputs:cameraPrim", [str(lidar_path)]),
        ("RenderProduct.inputs:enabled", True),
        ("LaserScanPublish.inputs:frameId", args_cli.frame_id),
        ("LaserScanPublish.inputs:nodeNamespace", ""),
        ("LaserScanPublish.inputs:topicName", args_cli.topic),
        ("LaserScanPublish.inputs:type", "laser_scan"),
    ]

    if args_cli.publish_point_cloud:
        create_nodes.append(
            ("PointCloudPublish", "isaacsim.ros2.bridge.ROS2RtxLidarHelper")
        )
        connect += [
            ("RenderProduct.outputs:execOut", "PointCloudPublish.inputs:execIn"),
            (
                "RenderProduct.outputs:renderProductPath",
                "PointCloudPublish.inputs:renderProductPath",
            ),
            ("Context.outputs:context", "PointCloudPublish.inputs:context"),
        ]
        set_values += [
            ("PointCloudPublish.inputs:frameId", args_cli.frame_id),
            ("PointCloudPublish.inputs:nodeNamespace", ""),
            ("PointCloudPublish.inputs:topicName", args_cli.point_cloud_topic),
            ("PointCloudPublish.inputs:type", "point_cloud"),
        ]

    og.Controller.edit(
        {"graph_path": str(graph_path), "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: create_nodes,
            keys.CONNECT: connect,
            keys.SET_VALUES: set_values,
        },
    )
    print(
        f"ROS2 Action Graph kreiran na {graph_path} (lidar topic '/{args_cli.topic}')"
    )

    stage.GetRootLayer().Save()
    print(f"\nSpremljeno u {args_cli.usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
