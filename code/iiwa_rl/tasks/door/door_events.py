"""door_events.py - randomizacija otpora vrata po env-u, pri resetu.

Parni dio uz door_cfg.py: tamo su rasponi i USD s pogonom na nuli, ovdje
je pisanje u simulaciju. Sva cetiri poziva idu nad (len(env_ids), 1)
tenzorima - nema petlje po env-u.

Za zakretna vrata se po env-u baca novcic izmedju slobodne sarke i sarke sa
zatvaracem. To NIJE isto sto i sirok raspon krutosti preko svih env-ova:
politika treba vidjeti dva kvalitativno razlicita rezima (otpor koji raste s
kutom naspram otpora koji ne raste), a ne kontinuum izmedju njih, jer o tome
ovisi isplati li se popustiti ili gurati dalje.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from .door_cfg import DOOR_DOF_JOINT, DoorResistanceRanges

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _sample(rng_range: tuple[float, float], shape, device) -> torch.Tensor:
    low, high = rng_range
    if low == high:
        return torch.full(shape, low, device=device)
    return torch.empty(shape, device=device).uniform_(low, high)


def randomize_door_resistance(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    ranges: DoorResistanceRanges,
    alt_ranges: DoorResistanceRanges | None = None,
    alt_probability: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("door"),
) -> None:
    """Postavi otpor DOF-a vrata na nasumicne vrijednosti iz raspona.

    ranges:          osnovni rezim (npr. slobodna sarka)
    alt_ranges:      alternativni rezim (npr. sarka sa zatvaracem)
    alt_probability: vjerojatnost da env dobije alternativni rezim
    """
    asset = env.scene[asset_cfg.name]
    device = env.device

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=device)
    joint_ids, _ = asset.find_joints(DOOR_DOF_JOINT)

    shape = (len(env_ids), len(joint_ids))

    stiffness = _sample(ranges.stiffness, shape, device)
    damping = _sample(ranges.damping, shape, device)
    friction = _sample(ranges.friction, shape, device)
    effort = _sample(ranges.effort_limit, shape, device)

    if alt_ranges is not None and alt_probability > 0.0:
        # Novcic je po env-u, ne po zglobu - inace bi jedna te ista vrata
        # mogla imati zatvarac na jednom DOF-u a ne na drugom, sto nema
        # fizikalnog smisla (i vazno je tek ako asset ikad dobije vise DOF-ova).
        use_alt = torch.rand((len(env_ids), 1), device=device) < alt_probability
        use_alt = use_alt.expand(shape)
        stiffness = torch.where(use_alt, _sample(alt_ranges.stiffness, shape, device), stiffness)
        damping = torch.where(use_alt, _sample(alt_ranges.damping, shape, device), damping)
        friction = torch.where(use_alt, _sample(alt_ranges.friction, shape, device), friction)
        effort = torch.where(use_alt, _sample(alt_ranges.effort_limit, shape, device), effort)

    asset.write_joint_stiffness_to_sim(stiffness, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_joint_damping_to_sim(damping, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_joint_effort_limit_to_sim(effort, joint_ids=joint_ids, env_ids=env_ids)
