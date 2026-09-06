# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
parser.add_argument(
    "--debug-grasp",
    action="store_true",
    default=False,
    help="Ispisi dijagnostiku hvata (TCP, racunata poza kvake, poza vrata).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Korijen repozitorija na sys.path, pa uvoz zadatka koji ga registrira.
# MORA biti iznad definicije main() - @hydra_task_config razrjesava
# args_cli.task u gym registru vec pri dekoriranju, dakle pri uvozu modula,
# a ne pri pozivu. Uvoz nize u fajlu bio bi prekasno i zadatak bi ispao
# neregistriran.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import iiwa_rl.tasks.door  # noqa: E402, F401  - registrira Isaac-Door-* zadatke
from iiwa_rl.tasks.door import door_mdp as mdp  # noqa: E402
from iiwa_rl.tasks.door.door_cfg import (  # noqa: E402
    HANDLE_LOCAL_REVOLUTE,
    HANDLE_LOCAL_SLIDING,
)

# PLACEHOLDER: Extension template (do not remove this comment)


def print_diagnostics(env, handle_local, label=""):
    """Gdje je TCP, gdje MISLIMO da je kvaka, i gdje su vrata.

    Kljucna usporedba je |TCP - kvaka|: ako je velika a hvat vizualno drzi,
    onda grasp_lost okida na pogresno izracunatoj pozi kvake, a ne na
    stvarnom klizanju. Poza korijena vrata otkriva je li reset uopce uspio
    pomaknuti vrata - kod fix_root_link=True PhysX drzi korijen na pozi iz
    spawna i write_root_pose_to_sim moze ostati bez ucinka.
    """
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    door = unwrapped.scene["door"]

    tcp_ids, _ = robot.find_bodies("gripper_tcp")
    tcp = robot.data.body_pos_w[0, tcp_ids[0]]
    handle = mdp.handle_pos_w(unwrapped, handle_local, SceneEntityCfg("door"))[0]
    leaf_ids, _ = door.find_bodies("door_leaf")

    print(f"  handle_local   : {handle_local}")
    print(f"  cfg grasp      : {unwrapped.cfg.events.grasp.params['handle_local']}")
    print(
        f"  cfg grasp_lost : {unwrapped.cfg.terminations.grasp_lost.params['handle_local']}"
    )

    print(f"--- {label}")
    print(f"  TCP            : {tcp.tolist()}")
    print(f"  kvaka(racunata): {handle.tolist()}")
    print(f"  |TCP - kvaka|  : {(tcp - handle).norm().item():.4f}")
    print(f"  door root      : {door.data.root_pos_w[0].tolist()}")
    print(f"  door leaf      : {door.data.body_pos_w[0, leaf_ids[0]].tolist()}")
    print(f"  env origin     : {unwrapped.scene.env_origins[0].tolist()}")
    print(f"  door dof       : {door.data.joint_pos[0].tolist()}")

    arm_ids, _ = robot.find_joints("iiwa_joint_[1-7]")
    q = robot.data.joint_pos[0, arm_ids]
    lower = robot.data.soft_joint_pos_limits[0, arm_ids, 0]
    upper = robot.data.soft_joint_pos_limits[0, arm_ids, 1]
    print(f"  zglobovi (udio): {((q - lower) / (upper - lower)).tolist()}")
    # base_link, NE root: korijen artikulacije je link 'world' i fiksan je u
    # ishodistu env-a otkad robot ima fiktivne zglobove za pokretnu bazu.
    base_idx = robot.find_bodies("base_link")[0][0]
    base_pos = robot.data.body_pos_w[0, base_idx]
    print(f"  baza (svijet)  : {base_pos.tolist()}")
    print(f"  |TCP - baza|   : {(tcp - base_pos).norm().item():.4f}")

    force = mdp.tcp_wrench(unwrapped, SceneEntityCfg("robot"))[0, :3]
    print(f"  sila lokalno   : {force.tolist()}")

    obs = unwrapped.observation_manager.compute_group("policy")[0]
    print("  obs :", obs.tolist())


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # lokalni offset kvake ovisi o tipu vrata
    handle_local = (
        HANDLE_LOCAL_REVOLUTE if "Revolute" in task_name else HANDLE_LOCAL_SLIDING
    )

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print(
                "[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task."
            )
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # convert pre-5.0 published checkpoints to the layout expected by rsl-rl >= 5.0 (no-op otherwise)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(
            policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.pt",
        )
        export_policy_as_onnx(
            policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # Epizode se trenutno prekidaju u prvom koraku, pa je stanje ODMAH
    # nakon reseta jedino koje se stigne vidjeti - zato prije prvog stepa.
    print_diagnostics(env, handle_local, "prije prvog koraka")

    # simulate environment
    count = 0
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        # Prvih nekoliko koraka svaki, dalje rjedje: ako epizoda traje
        # jedan korak, ispis na svakih 60 koraka nikad ne uhvati trenutak.
        if count < 5 or count % 60 == 0:
            print_diagnostics(env, handle_local, f"korak {count}")
            print("  krutost:", actions[0, 6:].tolist())

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        count += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
