"""diagnose_failures.py - zasto neke epizode ne uspiju.

Pusta naucenu politiku kroz mnogo epizoda i za svaku biljezi:

  otpor       trenje i prigusenje sarke koje je taj env dobio pri resetu
              (door_events.randomize_door_resistance, raspon iz door_cfg.py)
  napredak    najveci dof_progress postignut u epizodi (1.0 = prag uspjeha)
  hod baze    najveci pomak base_linka od poze na pocetku epizode

Pitanje na koje odgovara: jesu li neuspjesi vezani uz TEZE uzorke domain
randomizacije, ili se javljaju ravnomjerno? Ako koreliraju s trenjem, raspon
randomizacije je preširok za trenutnu politiku. Ako ne koreliraju ni s cim,
uzrok je istrazivanje - politika jednostavno nije naucila za dio raspodjele
pocetnih stanja, i poluga je entropy_coef / init_noise_std u
rsl_rl_ppo_cfg.py, a ne raspon randomizacije.

"Hod baze" je tu zbog opazanja da baza ponekad samo stoji i epizoda istekne.
Ako neuspjesi imaju hod baze blizu nule, to je zaseban nacin otkaza od onog
gdje se baza giba ali zadatak svejedno ne uspije - i te dvije stvari se ne
lijece istom polugom.

Pokretanje (headless, bez prozora, puno brze):

    ./isaaclab.sh -p code/iiwa_rl/scripts/diagnose_failures.py \
        --task Isaac-Door-Revolute-LearnedBase-KMR-iiwa-v0 \
        --num_envs 64 --episodes 200 --headless

Uz --force-base politika se zaobilazi i svi env-ovi dobiju ISTU konstantnu
naredbu baze. Time se provjerava je li razlika u ishodu uopce moguca bez
politike; ako se dio env-ova pomakne a dio ne uz identicnu naredbu, uzrok je
u postavu scene ili fizici, ne u naucenoj mrezi:

    ./isaaclab.sh -p code/iiwa_rl/scripts/diagnose_failures.py \
        --task Isaac-Door-Revolute-LearnedBase-KMR-iiwa-v0 \
        --num_envs 4 --episodes 4 --headless --force-base
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import math

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Dijagnostika neuspjelih epizoda.")
parser.add_argument("--task", type=str, default=None, help="Naziv zadatka.")
parser.add_argument(
    "--num_envs", type=int, default=64, help="Broj paralelnih okruzenja."
)
parser.add_argument(
    "--episodes",
    type=int,
    default=200,
    help="Koliko epizoda skupiti prije ispisa. Vise = stabilnija statistika.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Entry point konfiguracije agenta.",
)
parser.add_argument("--seed", type=int, default=None, help="Sjeme okruzenja.")
parser.add_argument(
    "--force-base",
    action="store_true",
    default=False,
    help=(
        "Zanemari politiku i posalji svim env-ovima istu konstantnu naredbu "
        "baze (-x). Ako se jedni pomaknu a drugi ne, uzrok nije u mrezi."
    ),
)
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

import iiwa_rl.tasks.door  # noqa: E402, F401  - registrira Isaac-Door-* zadatke
from iiwa_rl.tasks.door import door_mdp as mdp  # noqa: E402
from iiwa_rl.tasks.door.door_cfg import DOOR_DOF_JOINT  # noqa: E402
from iiwa_rl.tasks.door.robot_cfg import (  # noqa: E402
    BASE_BODY,
    BASE_JOINTS,
    GRIPPER_OPEN,
)


def quantile_table(name: str, values: torch.Tensor, progress: torch.Tensor) -> None:
    """Podijeli epizode u cetiri skupine po `values` i pokazi uspjeh po skupini.

    Ako otpor doista odreduje ishod, uspjeh mora monotono padati niz tablicu.
    Ako su svi redovi slicni, ta velicina NIJE uzrok.
    """
    order = torch.argsort(values)
    chunks = torch.chunk(order, 4)
    print(f"\n  po velicini '{name}' (rastuce):")
    print(f"    {'raspon':>18} {'n':>5} {'uspjeh':>8} {'prosj. napredak':>16}")
    for i, idx in enumerate(chunks):
        if idx.numel() == 0:
            continue
        lo, hi = values[idx].min().item(), values[idx].max().item()
        success = (progress[idx] >= 1.0).float().mean().item()
        mean_progress = progress[idx].mean().item()
        print(
            f"    {f'{lo:.2f} - {hi:.2f}':>18} {idx.numel():>5}"
            f" {success:>8.2f} {mean_progress:>16.2f}"
        )


def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearsonov koeficijent. Blizu nule = ta velicina ne objasnjava ishod."""
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom < 1e-9:
        return 0.0
    return (a @ b / denom).item()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

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

    # Kurikulum racuna iteraciju iz common_step_counter, koji pri evaluaciji
    # krece od nule, pa bi otpor ostao na pocetnoj vrijednosti. Ovim se rampa
    # odmah dovrsi i mjeri se na punom rasponu, kakav je na kraju treninga.
    if "door_resistance" in unwrapped.curriculum_manager.active_terms:
        term = unwrapped.curriculum_manager.get_term_cfg("door_resistance")
        term.params["start_iterations"] = 0
        term.params["full_iterations"] = 1

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
    base_joint_ids = [robot.find_joints(name)[0][0] for name in BASE_JOINTS]
    finger_ids = robot.find_joints("gripper_finger_[1-4]_joint")[0]
    full_travel = unwrapped.cfg.rewards.progress.params["full_travel"]

    def approach_angle() -> torch.Tensor:
        """Kut izmedju smjera baze i normale vrata, (num_envs,).

        Pri resetu je to tocno delta koju uzorkuje reset_grasp_and_door iz
        delta_range (+-0.35 rad). Referenca je door_frame (nepomican korijen),
        a ne krilo.
        """
        _, base_quat = mdp._base_frame(robot)
        axis_x = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n_env, 3)
        fwd = mdp.quat_apply(base_quat, axis_x)
        nrm = mdp.quat_apply(door.data.root_quat_w, axis_x)
        approach = torch.atan2(
            fwd[:, 0] * nrm[:, 1] - fwd[:, 1] * nrm[:, 0],
            fwd[:, 0] * nrm[:, 0] + fwd[:, 1] * nrm[:, 1],
        )
        # Vrata gledaju PREMA robotu, pa je nominalni kut pi, a ne nula.
        # Odstupanje treba omotati u (-pi, pi] jer atan2 puca bas na pi:
        # -3.14 i +3.14 su ista orijentacija, a razlika u predznaku bi inace
        # razdvojila fizicki susjedne poze na suprotne krajeve raspona.
        offset = approach - math.pi
        return torch.atan2(torch.sin(offset), torch.cos(offset))

    # Tekuce stanje epizode po env-u.
    max_progress = torch.zeros(n_env, device=device)
    max_travel = torch.zeros(n_env, device=device)
    base_ref = robot.data.body_pos_w[:, base_idx, :2].clone()
    yaw_start = approach_angle()

    # Skupljeni uzorci (jedan redak po zavrsenoj epizodi).
    rec_friction, rec_damping = [], []
    rec_progress, rec_travel = [], []
    rec_yaw = []
    rec_env = []
    collected = 0
    step_count = 0

    obs = env.get_observations()

    # Usporedba stanja nakon reseta. Politika ne vidi indeks env-a, pa ako se
    # ishod savrseno dijeli po env-u, razlika MORA biti u ovim brojkama.
    if n_env <= 8:
        print("\n=== stanje nakon reseta ===")
        for i in range(n_env):
            print(f"\nenv {i}:")
            print(f"  env origin  : {unwrapped.scene.env_origins[i].tolist()}")
            print(f"  fiktivni zgl: {robot.data.joint_pos[i, base_joint_ids].tolist()}")
            print(f"  brzine zgl  : {robot.data.joint_vel[i, base_joint_ids].tolist()}")
            print(f"  obs         : {obs['policy'][i].tolist()}")
        ref = obs["policy"][0]
        for i in range(1, n_env):
            diff = (obs["policy"][i] - ref).abs()
            big = (diff > 1e-4).nonzero().flatten().tolist()
            print(f"\nobs env{i} vs env0 - indeksi koji se razlikuju: {big}")

        # Parametri fiktivnih zglobova PO ENV-U. Naredba do sim-a stize
        # identicna u svim env-ovima (vidi 'trag naredbe'), pa ako se stvarna
        # brzina svejedno razlikuje, uzrok mora biti ovdje ili u samom
        # kloniranju artikulacije. Sve vrijednosti MORAJU biti jednake kroz
        # env-ove; svaka razlika je izravan uzrok.
        print("\n=== parametri zglobova baze (x, y, theta) po env-u ===")
        for label, arr in (
            ("stiffness", robot.data.joint_stiffness),
            ("damping", robot.data.joint_damping),
            ("armature", robot.data.joint_armature),
            ("friction", robot.data.joint_friction_coeff),
            ("effort_lim", robot.data.joint_effort_limits),
            ("vel_lim", robot.data.joint_vel_limits),
        ):
            rows = [
                [round(v, 4) for v in arr[e, base_joint_ids].tolist()]
                for e in range(n_env)
            ]
            same = all(r == rows[0] for r in rows)
            print(f"  {label:11s}: {rows}   {'' if same else '  <-- RAZLIKA'}")
        pos_lim = [
            robot.data.joint_pos_limits[e, base_joint_ids].tolist()
            for e in range(n_env)
        ]
        same = all(r == pos_lim[0] for r in pos_lim)
        print(f"  pos_limits : {pos_lim}   {'' if same else '  <-- RAZLIKA'}")

    print(f"[INFO] Skupljam {args_cli.episodes} epizoda na {n_env} env-ova...")

    while collected < args_cli.episodes and simulation_app.is_running():
        with torch.inference_mode():
            # Sve se cita PRIJE koraka: ako korak zavrsi epizodu, Isaac Lab
            # odmah resetira taj env, pa bi ocitanje poslije pripadalo vec
            # sljedecoj epizodi i uzorci bi bili pomijesani.
            friction = door.data.joint_friction_coeff[:, dof_idx].clone()
            damping = door.data.joint_damping[:, dof_idx].clone()

            progress = mdp.dof_progress(unwrapped, full_travel)
            max_progress = torch.maximum(max_progress, progress)

            base_now = robot.data.body_pos_w[:, base_idx, :2]
            travel = (base_now - base_ref).norm(dim=-1)
            max_travel = torch.maximum(max_travel, travel)

            actions = policy(obs)

            # DIJAGNOSTIKA (--force-base): politika van igre. Ista konstantna
            # naredba u SVIM env-ovima, pa razlika u ishodu ne moze doci iz
            # mreze - ostaje samo postav scene ili fizika.
            #
            # MORA biti iznad env.step; ispod njega bi se izracunalo i odmah
            # bacilo jer sljedeca iteracija zove policy(obs) ispocetka.
            #
            # Smjer je -x (unatrag), ne +x: vrata su na +x u okviru baze
            # (TCP na x ~ 1.14), pa bi +x gurao bazu u vrata i blokada bi
            # bila legitimna, a ne dokaz bugu. Nulirane dimenzije 6-11 daju
            # krutost 0 koja se klipa na donjih 200 N/m, dakle ruka je
            # popustljiva i ne drzi bazu ukljestenom.
            if args_cli.force_base:
                actions = torch.zeros_like(actions)
                actions[:, 12] = 0.0
                actions[:, 13] = -5.0  # bocno, ne unatrag
                # Prsti OTVORENI: bez hvata ne postoji zatvoreni lanac
                # ruka-vrata, pa zaustavljanje baze vise ne moze doci od
                # zategnute ruke. Bez ovoga je test pristran - OSC uz nultu
                # akciju drzi TCP na mjestu i time sam sidri bazu.
                robot.set_joint_position_target(
                    torch.full((n_env, len(finger_ids)), GRIPPER_OPEN, device=device),
                    joint_ids=finger_ids,
                )

            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)

            if n_env == 2 and step_count < 5:
                pol = obs["policy"]
                diff = (pol[0] - pol[1]).abs()
                print(f"\nkorak {step_count}")
                print(f"  akcija baze env0: {actions[0, 12:15].tolist()}")
                print(f"  akcija baze env1: {actions[1, 12:15].tolist()}")
                print(
                    f"  najveca razlika obs: idx={diff.argmax().item()} "
                    f"vrijednost={diff.max().item():.4f}"
                )
                print(f"  obs env0: {pol[0].tolist()}")
                print(f"  obs env1: {pol[1].tolist()}")
            step_count += 1

            # Trasiranje puta naredbe: raw -> processed -> target u simu ->
            # stvarna brzina. Gdje god se env-ovi PRVI PUT raziđu uz identicnu
            # ulaznu naredbu, ondje je bug:
            #   raw isto / processed razlicito -> JointVelocityAction
            #   processed isto / target razlicit -> upis u sim (env_ids)
            #   target isto / brzina razlicita  -> aktuator ili sam USD robota
            if args_cli.force_base and step_count <= 3:
                term = unwrapped.action_manager.get_term("base")
                print(f"\n--- trag naredbe, korak {step_count - 1} ---")
                print(f"  raw            : {term.raw_actions[:, 0].tolist()}")
                print(f"  processed      : {term.processed_actions[:, 0].tolist()}")
                print(
                    f"  vel target sim : "
                    f"{robot.data.joint_vel_target[:, base_joint_ids[0]].tolist()}"
                )
                print(
                    f"  stvarna brzina : "
                    f"{robot.data.joint_vel[:, base_joint_ids[0]].tolist()}"
                )
                print(
                    f"  polozaj zglobova: "
                    f"{[[round(v, 5) for v in robot.data.joint_pos[e, base_joint_ids].tolist()] for e in range(n_env)]}"
                )

            if args_cli.force_base and step_count % 60 == 0:
                print(
                    f"  korak {step_count:4d}  zglobovi baze po env-u: "
                    f"{[[round(v, 4) for v in robot.data.joint_pos[e, base_joint_ids].tolist()] for e in range(n_env)]}"
                )

            done_ids = dones.nonzero(as_tuple=False).flatten()
            if done_ids.numel() > 0:
                rec_friction.append(friction[done_ids].cpu())
                rec_damping.append(damping[done_ids].cpu())
                rec_progress.append(max_progress[done_ids].cpu())
                rec_travel.append(max_travel[done_ids].cpu())
                rec_yaw.append(yaw_start[done_ids].cpu())
                rec_env.append(done_ids.cpu())

                collected += done_ids.numel()
                print(f"  ...{collected}/{args_cli.episodes} epizoda")

                for j in done_ids.tolist():
                    print(
                        f"  env {j:2d}  delta={yaw_start[j]:+.3f}"
                        f"  napredak={max_progress[j]:5.2f}"
                        f"  hod baze={max_travel[j]:5.2f}"
                    )

                max_progress[done_ids] = 0.0
                max_travel[done_ids] = 0.0
                # Nova referentna poza baze za env-ove koji su upravo resetirani.
                base_ref[done_ids] = robot.data.body_pos_w[done_ids, base_idx, :2]
                yaw_start[done_ids] = approach_angle()[done_ids]

    friction = torch.cat(rec_friction)
    damping = torch.cat(rec_damping)
    progress = torch.cat(rec_progress)
    travel = torch.cat(rec_travel)
    yaw = torch.cat(rec_yaw)
    env_ids = torch.cat(rec_env)
    per_env = torch.zeros(n_env)
    per_env_n = torch.zeros(n_env)
    per_env.index_add_(0, env_ids, (progress >= 1.0).float())
    per_env_n.index_add_(0, env_ids, torch.ones_like(progress))
    rate = per_env / per_env_n.clamp(min=1)
    print(f"\n  uspjeh po env-u: min {rate.min():.2f}  max {rate.max():.2f}")
    print(f"  env-ova s uspjehom 0.0 : {(rate == 0).sum().item()} / {n_env}")
    print(f"  env-ova s uspjehom 1.0 : {(rate == 1).sum().item()} / {n_env}")
    n = progress.numel()

    success = progress >= 1.0
    stalled = travel < 0.05  # baza se prakticki nije ni pomakla

    print(f"\n=== {n} epizoda ===")
    print(f"  uspjeh (napredak >= 1.0) : {success.float().mean():.3f}")
    print(f"  prosjecni napredak       : {progress.mean():.3f}")
    print(f"  prosjecni hod baze       : {travel.mean():.3f} m")
    print(f"  epizoda s mirnom bazom   : {stalled.float().mean():.3f}  (< 5 cm)")

    if stalled.any() and (~stalled).any():
        print(
            f"    napredak kad baza MIRUJE : {progress[stalled].mean():.3f}"
            f"   ({stalled.sum().item()} epizoda)"
        )
        print(
            f"    napredak kad se baza GIBA: {progress[~stalled].mean():.3f}"
            f"   ({(~stalled).sum().item()} epizoda)"
        )

    print("\n  korelacija s napretkom:")
    print(f"    trenje sarke  : {correlation(friction, progress):+.3f}")
    print(f"    prigusenje    : {correlation(damping, progress):+.3f}")
    print(f"    hod baze      : {correlation(travel, progress):+.3f}")
    print(f"    kut prilaza   : {correlation(yaw.abs(), progress):+.3f}")

    quantile_table("trenje sarke", friction, progress)
    quantile_table("prigusenje", damping, progress)
    quantile_table("delta prilaza (rad)", yaw, progress)
    quantile_table("|delta prilaza| (rad)", yaw.abs(), progress)
    print(f"    delta > 0     : {(progress[yaw > 0] >= 1.0).float().mean():.3f}")
    print(f"    delta < 0     : {(progress[yaw < 0] >= 1.0).float().mean():.3f}")

    print(
        "\n  TUMACENJE: jaka negativna korelacija s trenjem znaci da je raspon\n"
        "  randomizacije preširok. Korelacije blizu nule uz i dalje visok udio\n"
        "  neuspjeha znace da uzrok nije tezina uzorka nego istrazivanje.\n"
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
