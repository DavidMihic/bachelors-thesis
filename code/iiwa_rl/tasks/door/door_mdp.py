"""door_mdp.py - funkcije koje env cfg zice u terme.

Podjela naspram door_events.py: tamo je randomizacija otpora vrata (parna uz
door_cfg.py), ovdje je sve ostalo - nagrada, terminacija, opazanje i reset
hvata.

PRIVILEGIRANE VELICINE: kut/pomak DOF-a vrata i njegov stvarni smjer gibanja
koriste se ISKLJUCIVO u nagradi i terminaciji, nikad u opazanju politike.
Isto pravilo kao u klasicnom pristupu. Nagrada smije znati istinu jer se
racuna offline; politika ne smije jer ju u stvarnosti nema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from .door_cfg import DOOR_DOF_JOINT
from .robot_cfg import GRIPPER_CLOSED

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Link koji je dijete gripper_wrist_jointa - tu se cita transmitirana sila.
# Robot se NE konvertira s merge-joints upravo zato da ovaj link prezivi.
# PROVJERI ime u kmr_iiwa_full.usd prije prvog treninga.
FT_SENSOR_BODY = "gripper_base"

Z_AXIS = torch.tensor([0.0, 0.0, 1.0])


def _door_dof(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Pomak/kut DOF-a vrata, (num_envs,). PRIVILEGIRANO."""
    door: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = door.find_joints(DOOR_DOF_JOINT)
    return door.data.joint_pos[:, joint_ids[0]]


def door_motion_direction_w(
    env: "ManagerBasedRLEnv",
    tcp_pos_w: torch.Tensor,
    door_type: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Jedinicni vektor trenutnog smjera gibanja tocke hvata, u svijetu.

    PRIVILEGIRANO - racuna se iz stvarne poze vrata, ne iz procjene. Koristi
    se samo za kaznu na okomitu silu, gdje je bas i htio bit referenca:
    politika treba biti kaznjena kad se stvarno bori s ogranicenjem, bez
    obzira na to koliko je njena vlastita procjena smjera dobra.
    """
    door: Articulation = env.scene[asset_cfg.name]
    root_quat = door.data.root_quat_w
    root_pos = door.data.root_pos_w

    if door_type == "sliding":
        # Klizanje je duz lokalne +Y osi okvira vrata (konvencija iz URDF-a).
        local = torch.zeros_like(tcp_pos_w)
        local[:, 1] = 1.0
        return quat_apply(root_quat, local)

    # Zakretna: tangenta luka je z x (p - c), gdje je c os sarke. Sarka je u
    # ishodistu okvira vrata (hinge_joint origin = 0 0 0 u URDF-u), pa je
    # radijus-vektor jednostavno p_tcp - p_root, projiciran u vodoravnu ravninu.
    z = Z_AXIS.to(tcp_pos_w.device).expand_as(tcp_pos_w)
    radius = tcp_pos_w - root_pos
    radius[:, 2] = 0.0
    tangent = torch.cross(z, radius, dim=-1)
    return tangent / (tangent.norm(dim=-1, keepdim=True) + 1e-8)


# --------------------------------------------------------------------------
# Opazanje
# --------------------------------------------------------------------------


def tcp_wrench(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sila i moment na zapescu, (num_envs, 6).

    Cita se na istom mjestu kinematickog lanca gdje tcp_wrench_estimator daje
    svoju procjenu pri deploymentu - najmanji moguci train/eval jaz za
    velicinu koja ulazi i u opazanje i u nagradu.

    NAPOMENA: rezultat je u LOKALNOM okviru linka. Tko ga usporedjuje s necim
    u svijetu mora ga prvo rotirati (vidi perpendicular_force_penalty).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    body_ids, _ = robot.find_bodies(FT_SENSOR_BODY)
    forces = robot.root_physx_view.get_link_incoming_joint_force()
    return forces[:, body_ids[0], :]


def tcp_pose_b(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Poza vrha alata u frameu baze, (num_envs, 7): pozicija + kvaternion."""
    robot: Articulation = env.scene[asset_cfg.name]
    body_ids, _ = robot.find_bodies(asset_cfg.body_names)
    idx = body_ids[0]
    pos = robot.data.body_pos_w[:, idx] - robot.data.root_pos_w
    return torch.cat([pos, robot.data.body_quat_w[:, idx]], dim=-1)


def tcp_velocity_b(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Linearna i kutna brzina vrha alata, (num_envs, 6).

    NAPOMENA O OKVIRU: baza je fiksirana s identitetom kao orijentacijom
    (robot_cfg.py, init_state bez rotacije + fix_root_link), pa je okvir baze
    jednak svjetskom do na translaciju - a translacija ne utjece na brzinu.
    Zato nema rotacije iz svijeta u bazu, isto kao u tcp_pose_b gore. Ako
    robot ikad dobije rotiran spawn, OBA terma treba ispraviti.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    body_ids, _ = robot.find_bodies(asset_cfg.body_names)
    idx = body_ids[0]
    return torch.cat(
        [robot.data.body_lin_vel_w[:, idx], robot.data.body_ang_vel_w[:, idx]], dim=-1
    )


# TODO (§5): kad constraint_estimator.py bude vektoriziran, dodaj ObsTerm koji
# vraca procijenjeni smjer ogranicenja i radijus. Do tada politika radi samo s
# pozom, brzinom i wrenchom - dovoljno da se potvrdi da uopce uci nesto
# smisleno (§9.4), ali NE stavljaj ovdje privilegirani smjer kao privremenu
# zamjenu, jer bi to napravilo tocno onaj train/eval jaz koji izbjegavamo.


# --------------------------------------------------------------------------
# Nagrada
# --------------------------------------------------------------------------


def dof_progress(
    env: "ManagerBasedRLEnv",
    full_travel: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("door"),
) -> torch.Tensor:
    """Normirani napredak DOF-a vrata. Glavni clan nagrade."""
    return _door_dof(env, asset_cfg) / full_travel


def perpendicular_force_penalty(
    env: "ManagerBasedRLEnv",
    door_type: str,
    robot_cfg: SceneEntityCfg,
    door_cfg: SceneEntityCfg = SceneEntityCfg("door"),
) -> torch.Tensor:
    """Kazna za komponentu sile okomitu na stvarni smjer gibanja.

    Ovo je clan koji uci politiku da NE bude kruta okomito na ogranicenje -
    tj. izravan odgovor na razlog zasto je kruto pozicijsko pracenje luka
    zakazalo na zakretnim vratima.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    body_ids, _ = robot.find_bodies(FT_SENSOR_BODY)
    idx = body_ids[0]

    # tcp_wrench vraca silu u LOKALNOM okviru linka, a smjer gibanja vrata je
    # u svijetu - bez ove rotacije usporedjuju se dva razlicita okvira.
    force_local = tcp_wrench(env, robot_cfg)[:, :3]
    force_w = quat_apply(robot.data.body_quat_w[:, idx], force_local)

    # Samo vodoravna ravnina: tezina gripera je ~15 N cisto vertikalno i inace
    # bi usla u okomitu komponentu kao konstantna kazna koja nema veze s
    # interakcijom. Vrata se u oba scenarija gibaju vodoravno, pa se Z ne gubi
    # nista bitno. (Alternativa je tariranje pri resetu, kao u klasicnom
    # pipelineu - skuplje i sa stanjem, bez dobitka ovdje.)
    force_w = force_w.clone()
    force_w[:, 2] = 0.0

    tcp_pos_w = robot.data.body_pos_w[:, idx]
    direction = door_motion_direction_w(env, tcp_pos_w, door_type, door_cfg)
    along = (force_w * direction).sum(dim=-1, keepdim=True) * direction
    return (force_w - along).norm(dim=-1)


def total_force_penalty(
    env: "ManagerBasedRLEnv",
    reference_force: float,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Kazna za ukupnu silu iznad reference.

    reference_force je GORNJA REFERENTNA GRANICA (~280 N izmjereno klasicnim
    pristupom), ne cilj. Ispod nje kazne nema - vucenje vrata legitimno trazi
    silu i ne zelimo politiku koja se boji dodira.
    """
    # Vodoravna komponenta, iz istog razloga kao gore. Ovdje je rotacija u
    # svijet nepotrebna jer se uzima norma, ali Z se i dalje mora ponistiti -
    # samo u lokalnom okviru, gdje tezina NIJE nuzno duz Z. Zato rotiramo i tu.
    robot: Articulation = env.scene[robot_cfg.name]
    body_ids, _ = robot.find_bodies(FT_SENSOR_BODY)
    force_local = tcp_wrench(env, robot_cfg)[:, :3]
    force_w = quat_apply(robot.data.body_quat_w[:, body_ids[0]], force_local).clone()
    force_w[:, 2] = 0.0
    magnitude = force_w.norm(dim=-1)
    return torch.clamp(magnitude - reference_force, min=0.0)


def is_open(
    env: "ManagerBasedRLEnv",
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("door"),
) -> torch.Tensor:
    """Vrata otvorena preko praga. Koristi se i kao bonus i kao terminacija."""
    return _door_dof(env, asset_cfg) > threshold


def force_exceeded(
    env: "ManagerBasedRLEnv",
    limit: float,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Sigurnosni prekid epizode."""
    return tcp_wrench(env, robot_cfg)[:, :3].norm(dim=-1) > limit


# --------------------------------------------------------------------------
# Reset
# --------------------------------------------------------------------------


def reset_to_grasp(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    nominal_joint_pos: tuple[float, ...],
    position_noise: float,
    robot_cfg: SceneEntityCfg,
) -> None:
    """Postavi ruku u nominalnu konfiguraciju hvata i zatvori prste.

    §6: hvat se NE fiksira pomocnim zglobom - prsti se stvarno zatvore i silu
    nose trenjem, isto kao pri deploymentu. Ako se pri skaliranju broja
    env-ova pokaze da je ovo usko grlo, pojednostavljenje treba eksplicitno
    zapisati u rad, ne tiho uvesti ovdje.

    nominal_joint_pos: sedam kutova ruke izmjerenih iz uspjesnog hvata
    klasicnim handle_approach - PROCITAJ IH IZ LOGA I UPISI U CFG, ne pogadjaj.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    arm_ids, _ = robot.find_joints("iiwa_joint_[1-7]")
    finger_ids, _ = robot.find_joints("gripper_finger_[1-4]_joint")

    nominal = torch.tensor(nominal_joint_pos, device=env.device)
    noise = torch.rand((len(env_ids), len(arm_ids)), device=env.device) - 0.5
    arm_pos = nominal.unsqueeze(0) + 2.0 * position_noise * noise

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:, arm_ids] = arm_pos
    joint_pos[:, finger_ids] = GRIPPER_CLOSED  # pogon drzi silu hvata
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
