"""measure_grasp_strength.py - koliko sile hvat uopce moze prenijeti?

PITANJE: kod zakretnih vrata poluga je zavarena za krilo, pa sav moment
sarke ide kroz trenje cetiri prsta na sipci promjera 28 mm. Nikad nismo
izmjerili gdje je granica. Ako je blizu sile potrebne za otvaranje, politika
radi na rubu proklizavanja i nikakvo podesavanje nagrade to ne rjesava -
grasp_lost od 0.4-0.9 tada nije neuspjeh ucenja nego fizikalna granica.

METODA: robot se postavi u nominalnu pozu hvata i prsti se zatvore, isto kao
u reset_grasp_and_door. Vrata se ZAKLJUCAJU (limiti DOF-a na nulu) da se ne
mogu otvoriti - time sve sto ruka razvije ide u hvat umjesto u gibanje
vrata. Zatim se rampom povecava sila na vrh alata dok TCP ne pobjegne od
kvake, i biljezi se sila pri kojoj se to dogodi.

Sila se zadaje kroz OSC istim putem kojim je zadaje i politika (pose_rel +
krutost), pa je izmjerena granica ono sto politika STVARNO ima na
raspolaganju, a ne teorijski maksimum trenja.

ODGOVOR: usporedi izmjerenu granicu s momentom koji zakretna vrata traze.
Pri trenju sarke 0.5-5 Nm i kraku 0.72 m to je 0.7-7 N; ako granica hvata
ispadne reda desetak njutna, zadatak je izvediv i problem je drugdje. Ako
ispadne blizu, hvat je usko grlo.

Pokretanje:
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/bachelors-thesis/code/iiwa_rl/scripts/measure_grasp_strength.py \
        --task Isaac-Door-Revolute-KMR-iiwa-v0 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Izmjeri granicu nosivosti hvata.")
parser.add_argument("--task", type=str, default="Isaac-Door-Revolute-KMR-iiwa-v0")
parser.add_argument(
    "--ramp-seconds",
    type=float,
    default=8.0,
    help="Trajanje rampe. Sporija rampa = tocnije, ali dulje.",
)
parser.add_argument(
    "--slip-threshold",
    type=float,
    default=0.03,
    help="Pomak TCP-a od kvake iznad kojeg se hvat smatra proklizalim (m).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import iiwa_rl.tasks.door  # noqa: E402, F401
from iiwa_rl.tasks.door import door_mdp as mdp  # noqa: E402
from iiwa_rl.tasks.door.door_cfg import (  # noqa: E402
    DOOR_DOF_JOINT,
    HANDLE_LOCAL_REVOLUTE,
    HANDLE_LOCAL_SLIDING,
)

# Jedan env po smjeru vucenja. Granica ovisi o smjeru jer prsti nose silu
# razlicito ovisno o tome vuce li se duz osi sipke ili okomito na nju.
DIRECTIONS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


def main() -> None:
    handle_local = (
        HANDLE_LOCAL_REVOLUTE if "Revolute" in args_cli.task else HANDLE_LOCAL_SLIDING
    )

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=len(DIRECTIONS)
    )
    # Baza mora mirovati: admitancijski zakon bi se pomicao i time mijenjao
    # smjer vucenja usred mjerenja.
    env_cfg.actions.base.max_linear_speed = 0.0
    env_cfg.actions.base.max_angular_speed = 0.0
    # Bez terminacija - epizoda mora trajati cijelu rampu.
    env_cfg.terminations.grasp_lost = None
    env_cfg.terminations.overforce = None
    env_cfg.episode_length_s = args_cli.ramp_seconds + 2.0

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    n = len(DIRECTIONS)

    env.reset()

    robot = unwrapped.scene["robot"]
    door = unwrapped.scene["door"]

    # Zakljucaj vrata: sve sto ruka razvije ide u hvat, ne u gibanje vrata.
    door_joint = door.find_joints(DOOR_DOF_JOINT)[0]
    door.write_joint_limits_to_sim(
        torch.zeros(n, 1, 2, device=device), joint_ids=door_joint
    )

    directions = torch.tensor(list(DIRECTIONS.values()), device=device)
    names = list(DIRECTIONS.keys())

    # Krutost na maksimum, pomak reference raste rampom -> sila raste rampom.
    stiffness_action = torch.full((n, 6), 10.0, device=device)
    slip_force = torch.full((n,), float("nan"), device=device)

    steps = int(args_cli.ramp_seconds / unwrapped.step_dt)
    for step in range(steps):
        scale = step / steps
        actions = torch.zeros((n, 12), device=device)
        actions[:, :3] = directions * scale * 3.0
        actions[:, 6:] = stiffness_action

        with torch.inference_mode():
            env.step(actions)

            tcp = robot.data.body_pos_w[:, robot.find_bodies("gripper_tcp")[0][0]]
            handle = mdp.handle_pos_w(unwrapped, handle_local, SceneEntityCfg("door"))
            distance = (tcp - handle).norm(dim=-1)

            force = mdp.tcp_wrench(unwrapped, SceneEntityCfg("robot"))[:, :3]
            magnitude = force.norm(dim=-1)

            slipping = (distance > args_cli.slip_threshold) & slip_force.isnan()
            slip_force = torch.where(slipping, magnitude, slip_force)

        if not slip_force.isnan().any():
            break

    print("\n=== granica nosivosti hvata ===")
    print(f"{'smjer':>8} {'sila pri proklizavanju':>24}")
    for name, value in zip(names, slip_force.tolist()):
        shown = "nije proklizalo" if value != value else f"{value:.1f} N"
        print(f"{name:>8} {shown:>24}")
    print(
        "\nZakretna vrata pri trenju sarke 0.5-5 Nm i kraku 0.72 m traze 0.7-7 N.\n"
        "Ako je najslabiji smjer usporediv s tim, hvat je usko grlo i nikakvo\n"
        "podesavanje nagrade nece pomoci - treba jaci stisak, vece trenje\n"
        "prstiju ili drugacija geometrija kvake."
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
