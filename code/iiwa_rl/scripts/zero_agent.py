# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
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
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os, sys  # noqa: E402
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from isaaclab.managers import SceneEntityCfg
import iiwa_rl.tasks.door  # noqa: E402, F401
from iiwa_rl.tasks.door import door_mdp as mdp

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    count = 0

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # actions[:, 1] = -1.0  # puni pomak reference duz osi klizanja
            # actions[:, 6:] = 1.0  # maksimalna krutost
            # apply actions
            env.step(actions)

            if count % 60 == 0:
                robot = env.unwrapped.scene["robot"]
                arm_ids, arm_names = robot.find_joints("iiwa_joint_[1-7]")
                q = robot.data.joint_pos[0, arm_ids]
                lower = robot.data.soft_joint_pos_limits[0, arm_ids, 0]
                upper = robot.data.soft_joint_pos_limits[0, arm_ids, 1]
                # 0 = na donjem limitu, 1 = na gornjem, ~0.5 = sredina raspona
                print("zglobovi (udio raspona):", ((q - lower) / (upper - lower)))
                tcp_ids, _ = robot.find_bodies("gripper_tcp")
                base = robot.data.root_pos_w[0]
                print(
                    "|TCP - baza|:",
                    (robot.data.body_pos_w[0, tcp_ids[0]] - base).norm(),
                )

            count += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
