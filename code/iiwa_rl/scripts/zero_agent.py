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

import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import iiwa_rl.tasks.door  # noqa: E402, F401  - registrira Isaac-Door-* zadatke

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Zero actions agent with Isaac Lab environment.

    Nulta akcija uz pose_rel znaci "ne pomicaj referencu", pa ruka mirno
    stoji i drzi kvaku. Sluzi za provjeru scene i reseta bez politike u igri:
    ako se ovdje nesto mice ili raspada, uzrok nije u treningu.
    """
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    env.reset()

    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

            if count % 60 == 0:
                robot = env.unwrapped.scene["robot"]
                arm_ids, _ = robot.find_joints("iiwa_joint_[1-7]")
                q = robot.data.joint_pos[0, arm_ids]
                lower = robot.data.soft_joint_pos_limits[0, arm_ids, 0]
                upper = robot.data.soft_joint_pos_limits[0, arm_ids, 1]
                # 0 = na donjem limitu, 1 = na gornjem, ~0.5 = sredina raspona.
                # Zglob na 0 ili 1 znaci da je ruka kinematicki potrosena i da
                # vrata staju zbog toga, a ne zbog upravljanja.
                print("zglobovi (udio raspona):", ((q - lower) / (upper - lower)))
                tcp_ids, _ = robot.find_bodies("gripper_tcp")
                base_idx = robot.find_bodies("base_link")[0][0]
                base_pos = robot.data.body_pos_w[0, base_idx]
                # reach = (
                #     robot.data.body_pos_w[0, tcp_ids[0]] - robot.data.root_pos_w[0]
                # ).norm()
                reach = (robot.data.body_pos_w[0, tcp_ids[0]] - base_pos).norm()
                print("|TCP - baza|:", reach.item())

            count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
