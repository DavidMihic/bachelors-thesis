"""measure_door_friction.py - u kojim je jedinicama friction_coefficient?

PITANJE: stari physxJoint:jointFriction bio je APSOLUTNI moment/sila (Nm, N).
Novi PhysX model zove isti parametar "coefficient", sto sugerira bezdimenzijski
mnozitelj sile u zglobu. O tome ovisi znace li rasponi u door_cfg.py ono sto
mislimo da znace - uz krivu interpretaciju vrata su ili bez otpora ili
zabetonirana, a RL rezultati u oba slucaja nemaju veze sa stvarnim vratima.

METODA: nekoliko env-ova, svaki s drugom vrijednoscu trenja, bez robota.
Krutost i prigusenje na nuli, pa je trenje jedini otpor. Rampa momenta na
door_dof_joint dok se zglob ne pokrene; biljezi se moment pri pokretanju.

ODGOVOR: ako je breakaway ~= zadana vrijednost, jedinice su apsolutne i
rasponi u door_cfg.py stoje. Ako nije, omjer daje faktor skaliranja.

Pokretanje:
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/bachelors-thesis/code/iiwa_rl/scripts/measure_door_friction.py \
        --door sliding
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Izmjeri breakaway moment vrata.")
parser.add_argument("--door", choices=["sliding", "revolute"], default="sliding")
parser.add_argument(
    "--max-effort",
    type=float,
    default=40.0,
    help="Gornja granica rampe (Nm za zakretna, N za klizna).",
)
parser.add_argument("--ramp-seconds", type=float, default=4.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from iiwa_rl.tasks.door.door_cfg import (  # noqa: E402
    DOOR_DOF_JOINT,
    REVOLUTE_DOOR_CFG,
    SLIDING_DOOR_CFG,
)

# Pokriva cijeli raspon iz door_cfg.py: zakretna 0.5-5, klizna 5-60.
FRICTION_VALUES = [0.5, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0]

# Zglob se smatra pokrenutim iznad ove brzine. Dovoljno iznad numerickog suma
# da ga puzanje solvera ne okine, dovoljno nisko da uhvati pravi trenutak.
#
# MJERNA PRISTRANOST: krilo od 15 kg treba vremena da nakon pokretanja
# nakupi ovu brzinu, a rampa u tom vremenu jos raste - pa izmjereni breakaway
# bude nesto visi od stvarnog. Apsolutno je to konstantnih ~0.2 N, dakle
# zanemarivo pri 10 N i vise, ali pri 0.5 N daje omjer preko 3. Nije nelinearno
# trenje nego mjerna metoda; za tocnije male vrijednosti usporite rampu.
MOTION_THRESHOLD = 0.01  # rad/s ili m/s


@configclass
class FrictionTestSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0)
    )
    door = None  # postavlja se u main()


def main():
    base_cfg = SLIDING_DOOR_CFG if args_cli.door == "sliding" else REVOLUTE_DOOR_CFG

    scene_cfg = FrictionTestSceneCfg(num_envs=len(FRICTION_VALUES), env_spacing=4.0)
    scene_cfg.door = base_cfg.replace(prim_path="{ENV_REGEX_NS}/Door")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0))
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    door = scene["door"]
    joint_ids, _ = door.find_joints(DOOR_DOF_JOINT)
    device = sim.device
    num_envs = len(FRICTION_VALUES)

    zeros = torch.zeros((num_envs, 1), device=device)
    friction = torch.tensor(FRICTION_VALUES, device=device).unsqueeze(-1)

    # Trenje je jedini otpor: krutost i prigusenje na nulu.
    door.write_joint_stiffness_to_sim(zeros, joint_ids=joint_ids)
    door.write_joint_damping_to_sim(zeros, joint_ids=joint_ids)
    door.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids)
    # Effort limit mora biti iznad rampe, inace mjerimo limit a ne trenje.
    door.write_joint_effort_limit_to_sim(
        torch.full((num_envs, 1), args_cli.max_effort * 2.0, device=device),
        joint_ids=joint_ids,
    )

    breakaway = torch.full((num_envs,), float("nan"), device=device)
    steps = int(args_cli.ramp_seconds / sim.get_physics_dt())

    for step in range(steps):
        effort = args_cli.max_effort * step / steps
        door.set_joint_effort_target(
            torch.full((num_envs, 1), effort, device=device), joint_ids=joint_ids
        )
        door.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

        velocity = door.data.joint_vel[:, joint_ids[0]].abs()
        moving = (velocity > MOTION_THRESHOLD) & breakaway.isnan()
        breakaway = torch.where(moving, torch.full_like(breakaway, effort), breakaway)

        if not breakaway.isnan().any():
            break

    unit = "Nm" if args_cli.door == "revolute" else "N"
    print(f"\n=== {args_cli.door} vrata ===")
    print(f"{'zadano trenje':>16} {'breakaway':>12} {'omjer':>8}")
    for value, measured in zip(FRICTION_VALUES, breakaway.tolist()):
        ratio = "-" if measured != measured else f"{measured / value:.2f}"
        shown = "nije se pomaklo" if measured != measured else f"{measured:.2f} {unit}"
        print(f"{value:>16.2f} {shown:>12} {ratio:>8}")
    print(
        "\nOmjer ~1.0 kroz sve retke -> jedinice su apsolutne, rasponi u "
        "door_cfg.py stoje.\nKonstantan omjer razlicit od 1 -> faktor skaliranja.\n"
        "Omjer koji varira -> trenje ovisi o sili u zglobu, dakle pravi "
        "koeficijent,\ni rasponi se moraju preracunati preko ocekivane sile "
        "vucenja."
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
