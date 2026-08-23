"""Registracija zadataka otvaranja vrata.

Klizna vrata su prvi cilj (§9.4): jednostavnija su, i za njih postoji
klasicni baseline broj za usporedbu. Zakretna dolaze tek kad se potvrdi da
politika na kliznima nauci nesto smisleno.
"""

import gymnasium as gym

from . import agents
from .door_env_cfg import DoorEnvCfg, DoorRevoluteEnvCfg

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
    id="Isaac-Door-Revolute-KMR-iiwa-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DoorRevoluteEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoorPPORunnerCfg",
    },
)
