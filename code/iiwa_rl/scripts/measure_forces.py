"""Sile i uspjesnost naucene politike pri determinstickom izvodjenju.

Mjeri se po epizodi, pa se preko epizoda daju raspodjele:

  sila        prosjecna, efektivna i najveca vrijednost iznosa sile na
              zapescu, UMANJENE za tezinu prihvatnice.

              Staticki doprinos se oduzima VEKTORSKI, po koraku: referenca je
              vektor sile rano u epizodi, dok se jos nista ne giba, i oduzima
              se od svakog kasnijeg ocitanja prije nego se uzme iznos.
              Skalarno oduzimanje od prosjeka ne bi dalo ispravan efektivni
              iznos ni maksimum.

  otvorenost  najveci kut odnosno hod postignut u epizodi. Uzima se najveci, a
              ne konacni, jer se epizoda moze prekinuti nakon sto su vrata vec
              bila otvorena.

  hod platforme  najveci pomak platforme od poze na pocetku epizode. Sluzi za
              podjelu epizoda: ishod je dvojak, jer se zatvoreni kinematicki
              lanac platforma-ruka-vrata ili odlijepi ili zakljuca. U
              zakljucanim epizodama platforma se ne pomakne uopce, pa bi
              mijesanje dviju skupina dalo brojku koja ne opisuje ni jednu.

Pokretanje:

    ./isaaclab.sh -p code/iiwa_rl/scripts/measure_forces.py \\
        --task Isaac-Door-Revolute-LearnedBase-KMR-iiwa-v0 \\
        --load_run revolute --num_envs 64 --episodes 1024 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Sile i uspjesnost naucene politike.")
parser.add_argument("--task", type=str, default=None, help="Naziv zadatka.")
parser.add_argument("--num_envs", type=int, default=64, help="Broj okruzenja.")
parser.add_argument(
    "--episodes", type=int, default=1024, help="Koliko epizoda skupiti."
)
parser.add_argument(
    "--stall-threshold",
    type=float,
    default=0.05,
    help="Ispod ovog hoda platforme epizoda se smatra zakljucanom (m).",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Entry point konfiguracije agenta.",
)
parser.add_argument("--seed", type=int, default=None, help="Sjeme okruzenja.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import importlib.metadata as metadata  # noqa: E402

installed_version = metadata.version("rsl-rl-lib")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import iiwa_rl.tasks.door  # noqa: E402, F401
from iiwa_rl.tasks.door import door_mdp as mdp  # noqa: E402
from iiwa_rl.tasks.door.door_cfg import DOOR_DOF_JOINT  # noqa: E402
from iiwa_rl.tasks.door.robot_cfg import BASE_BODY  # noqa: E402

# Pragovi uspjesnosti po tipu vrata: (oznaka, vrijednost).
REVOLUTE_THRESHOLDS = (("50 st.", 0.8727), ("90 st.", 1.5708))
SLIDING_THRESHOLDS = (("0.60 m", 0.60), ("0.90 m", 0.90))

# Koliko se koraka preskoci prije uzimanja staticke reference. U prvim
# koracima nakon reseta ocitanje sile jos nije osvjezeno i vraca nule.
STATIC_REFERENCE_STEP = 4


def describe(name: str, values: torch.Tensor) -> None:
    """Raspodjela jedne velicine preko epizoda."""
    if values.numel() == 0:
        print(f"    {name:<12} nema epizoda")
        return
    quartiles = torch.quantile(
        values, torch.tensor([0.25, 0.5, 0.75], device=values.device)
    )
    print(
        f"    {name:<12} sred. {values.mean():8.3f}   "
        f"Q1 {quartiles[0]:8.3f}   med. {quartiles[1]:8.3f}   "
        f"Q3 {quartiles[2]:8.3f}   maks. {values.max():8.3f}"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # Otpor se postavlja na PUN raspon, onakav kakav je na kraju treninga.
    #
    # SLIDING_RESISTANCE u door_cfg.py drzi samo POCETNE vrijednosti rampe;
    # pun raspon postize kurikulum, koji racuna iteraciju iz
    # common_step_counter. Pri evaluaciji taj brojac krece od nule i ne
    # dosegne prag, pa bi se mjerilo na bitno lakim vratima. Rasponi se
    # postavljaju izravno, jer kurikulum mijenja vrijednosti tek pri resetu i
    # ovisi o redoslijedu naspram dogadjaja koji ih upisuje u simulator.
    curriculum = getattr(env_cfg, "curriculum", None)
    if curriculum is not None and hasattr(curriculum, "door_resistance"):
        params = curriculum.door_resistance.params
        ranges = env_cfg.events.door_resistance.params["ranges"]
        ranges.friction = (ranges.friction[0], params["max_friction"])
        ranges.damping = (ranges.damping[0], params["max_damping"])
        print(
            f"[INFO] Otpor na punom rasponu: trenje {ranges.friction}, "
            f"prigusenje {ranges.damping}"
        )

    log_root_path = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    print(f"[INFO] Checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    device = unwrapped.device
    n_env = unwrapped.num_envs

    door = unwrapped.scene["door"]
    robot = unwrapped.scene["robot"]
    dof_idx = door.find_joints(DOOR_DOF_JOINT)[0][0]
    base_idx = robot.find_bodies(BASE_BODY)[0][0]
    robot_cfg = SceneEntityCfg("robot")

    is_sliding = "sliding" in str(unwrapped.cfg.scene.door.spawn.usd_path).lower()
    thresholds = SLIDING_THRESHOLDS if is_sliding else REVOLUTE_THRESHOLDS
    dof_unit = "m" if is_sliding else "rad"

    # Zbroj i zbroj kvadrata umjesto liste svih ocitanja, jer epizoda ima 600
    # koraka po okruzenju.
    force_sum = torch.zeros(n_env, device=device)
    force_sq_sum = torch.zeros(n_env, device=device)
    force_max = torch.zeros(n_env, device=device)
    static_vec = torch.zeros(n_env, 3, device=device)
    steps = torch.zeros(n_env, device=device)
    counted = torch.zeros(n_env, device=device)
    peak_dof = torch.zeros(n_env, device=device)
    travel_max = torch.zeros(n_env, device=device)
    base_ref = robot.data.body_pos_w[:, base_idx, :2].clone()

    rec_mean, rec_rms, rec_max = [], [], []
    rec_peak, rec_travel, rec_friction = [], [], []
    collected = 0

    obs = env.get_observations()
    print(f"[INFO] Skupljam {args_cli.episodes} epizoda na {n_env} okruzenja...")

    while collected < args_cli.episodes and simulation_app.is_running():
        with torch.inference_mode():
            wrench = mdp.tcp_wrench(unwrapped, robot_cfg)[:, :3]

            # Referenca statike: rano u epizodi, dok se jos nista ne giba.
            static_vec = torch.where(
                (steps == STATIC_REFERENCE_STEP).unsqueeze(-1), wrench, static_vec
            )

            # Cista sila: staticki vektor se oduzima PRIJE uzimanja iznosa.
            net = (wrench - static_vec).norm(dim=-1)
            valid = steps > STATIC_REFERENCE_STEP
            zero = torch.zeros_like(net)
            force_sum += torch.where(valid, net, zero)
            force_sq_sum += torch.where(valid, net.square(), zero)
            force_max = torch.maximum(force_max, torch.where(valid, net, zero))
            counted += valid.float()
            steps += 1.0

            peak_dof = torch.maximum(peak_dof, door.data.joint_pos[:, dof_idx].abs())
            travel = (robot.data.body_pos_w[:, base_idx, :2] - base_ref).norm(dim=-1)
            travel_max = torch.maximum(travel_max, travel)

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)

            done_ids = dones.bool().nonzero(as_tuple=False).flatten()
            if done_ids.numel() == 0:
                continue

            count = counted[done_ids].clamp(min=1.0)
            rec_mean.append((force_sum[done_ids] / count).cpu())
            rec_rms.append((force_sq_sum[done_ids] / count).sqrt().cpu())
            rec_max.append(force_max[done_ids].cpu())
            rec_peak.append(peak_dof[done_ids].cpu())
            rec_travel.append(travel_max[done_ids].cpu())
            rec_friction.append(door.data.joint_friction_coeff[done_ids, dof_idx].cpu())

            collected += done_ids.numel()
            print(f"  ...{collected}/{args_cli.episodes} epizoda", flush=True)

            force_sum[done_ids] = 0.0
            force_sq_sum[done_ids] = 0.0
            force_max[done_ids] = 0.0
            steps[done_ids] = 0.0
            counted[done_ids] = 0.0
            peak_dof[done_ids] = 0.0
            travel_max[done_ids] = 0.0
            base_ref[done_ids] = robot.data.body_pos_w[done_ids, base_idx, :2]

    mean_force = torch.cat(rec_mean)
    rms_force = torch.cat(rec_rms)
    max_force = torch.cat(rec_max)
    peak = torch.cat(rec_peak)
    travel = torch.cat(rec_travel)
    friction = torch.cat(rec_friction)

    total = peak.numel()
    moved = travel >= args_cli.stall_threshold
    limit = unwrapped.cfg.terminations.overforce.params["limit"]
    reference = unwrapped.cfg.rewards.excess_force.params["reference_force"]

    print(f"\n{'=' * 72}")
    print(f"{total} epizoda, {'klizna' if is_sliding else 'zakretna'} vrata")
    # Kontrola da je otpor doista bio na punom rasponu.
    print(
        f"otpor vrata: {friction.min():.2f} - {friction.max():.2f}, "
        f"sred. {friction.mean():.2f}"
    )
    print(f"prag kazne {reference:.0f} N, prag prekida {limit:.0f} N")
    print("=" * 72)

    for label, mask in (
        ("SVE EPIZODE", torch.ones_like(moved)),
        ("platforma se pomakla", moved),
        ("platforma zapela", ~moved),
    ):
        n = int(mask.sum())
        print(f"\n{label}: {n} epizoda ({n / total:.1%})")
        if n == 0:
            continue

        print("  sila na zapescu, bez tezine prihvatnice [N]")
        describe("prosjecna", mean_force[mask])
        describe("efektivna", rms_force[mask])
        describe("najveca", max_force[mask])

        print(f"  najveca otvorenost [{dof_unit}]")
        describe("otvorenost", peak[mask])

        print("  udio epizoda iznad praga")
        for name, value in thresholds:
            print(f"    {name:<12} {(peak[mask] >= value).float().mean():.3f}")

    print(f"\n{'=' * 72}")
    print("prekoracenje praga sile, sve epizode")
    print(f"  preko {reference:.0f} N: {(max_force >= reference).float().mean():.3f}")
    print(f"  preko {limit:.0f} N: {(max_force >= limit).float().mean():.3f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
