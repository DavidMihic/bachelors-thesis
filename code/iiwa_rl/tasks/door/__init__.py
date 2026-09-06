"""Registracija zadataka otvaranja vrata.

Klizna vrata su prvi cilj (§9.4): jednostavnija su, i za njih postoji
klasicni baseline broj za usporedbu. Zakretna dolaze tek kad se potvrdi da
politika na kliznima nauci nesto smisleno.
"""

import gymnasium as gym

from . import agents
from .door_env_cfg import (
    DoorEnvCfg,
    DoorSlidingLearnedBaseEnvCfg,
    DoorRevoluteEnvCfg,
    DoorRevoluteLearnedBaseEnvCfg,
    DoorRevoluteFixedEnvCfg,
    DoorRevoluteLearnedBaseFixedEnvCfg,
    DoorSlidingFixedEnvCfg,
    DoorSlidingLearnedBaseFixedEnvCfg,
)

gym.register(
    id="Isaac-Door-Sliding-KMR-iiwa-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DoorEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoorPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Door-Sliding-LearnedBase-KMR-iiwa-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DoorSlidingLearnedBaseEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoorPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Door-Revolute-KMR-iiwa-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DoorRevoluteEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoorPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Door-Revolute-LearnedBase-KMR-iiwa-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DoorRevoluteLearnedBaseEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoorPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Door-Revolute-Fixed-KMR-iiwa-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DoorRevoluteFixedEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoorPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Door-Revolute-LearnedBase-Fixed-KMR-iiwa-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DoorRevoluteLearnedBaseFixedEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoorPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Door-Sliding-Fixed-KMR-iiwa-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DoorSlidingFixedEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoorPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Door-Sliding-LearnedBase-Fixed-KMR-iiwa-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DoorSlidingLearnedBaseFixedEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoorPPORunnerCfg",
    },
)
