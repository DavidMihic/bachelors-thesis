"""door_mdp.py - funkcije koje env cfg zice u terme.

Podjela naspram door_events.py: tamo je randomizacija otpora vrata (parna uz
door_cfg.py), ovdje je sve ostalo - nagrada, terminacija, opazanje i reset
hvata.

PRIVILEGIRANE VELICINE: kut/pomak DOF-a vrata i njegov stvarni smjer gibanja
koriste se ISKLJUCIVO u nagradi i terminaciji, nikad u opazanju politike.
Isto pravilo kao u klasicnom pristupu. Nagrada smije znati istinu jer se
racuna offline; politika ne smije jer ju u stvarnosti nema.

OKVIRI: get_link_incoming_joint_force vraca silu u LOKALNOM okviru linka.
Svaka usporedba sa smjerom gibanja vrata mora ju prvo rotirati u svijet -
propust u tome daje kaznu reda 1e-6 koja izgleda kao "nema sile", a zapravo
znaci da politika trenira bez ikakve informacije o kontaktu.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from .door_cfg import DOOR_DOF_JOINT, DOOR_LEAF_BODY
from .robot_cfg import GRIPPER_CLOSED

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Link koji je dijete gripper_wrist_jointa - tu se cita transmitirana sila.
# Robot se NE konvertira s merge-joints upravo zato da ovaj link prezivi.
FT_SENSOR_BODY = "gripper_base"
TCP_BODY = "gripper_tcp"

# Geometrija hvata, izmjerena iz uspjesnog klasicnog handle_approach.
ARM_BASE_B = (0.363, -0.184, 0.7)  # iiwa_link_0 u base_link (iz URDF-a)
NOMINAL_TCP_B = (1.158, -0.287, 1.005)  # TCP pri hvatu, u base_link (iz TF-a)
DOOR_BASE_YAW = 3.14159265  # vrata gledaju natrag prema robotu

Z_AXIS = torch.tensor([0.0, 0.0, 1.0])

# Indeksi zglobova i tijela ne mijenjaju se kroz trening, a find_joints/
# find_bodies rade regex pretragu po imenima pri svakom pozivu - uz sest
# termova i 60 koraka/s to je konstantan trosak neovisan o broju env-ova.
_INDEX_CACHE: dict[tuple[int, str, str], list[int]] = {}


def _bodies(asset, key: str, pattern: str) -> list[int]:
    cache_key = (id(asset), "b", key)
    if cache_key not in _INDEX_CACHE:
        _INDEX_CACHE[cache_key] = asset.find_bodies(pattern)[0]
    return _INDEX_CACHE[cache_key]


def _joints(asset, key: str, pattern: str) -> list[int]:
    cache_key = (id(asset), "j", key)
    if cache_key not in _INDEX_CACHE:
        _INDEX_CACHE[cache_key] = asset.find_joints(pattern)[0]
    return _INDEX_CACHE[cache_key]


def _door_dof(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Pomak/kut DOF-a vrata, (num_envs,). PRIVILEGIRANO."""
    door: Articulation = env.scene[asset_cfg.name]
    joint_ids = _joints(door, "dof", DOOR_DOF_JOINT)
    return door.data.joint_pos[:, joint_ids[0]]


def _yaw_quat(angle: torch.Tensor) -> torch.Tensor:
    """Kvaternion (w, x, y, z) rotacije oko vertikale."""
    half = 0.5 * angle
    zero = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zero, zero, torch.sin(half)], dim=-1)


def _horizontal_force_w(
    env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sila na zapescu rotirana u svijet, bez vertikalne komponente, i poza
    senzora. Vraca (force_w, sensor_pos_w).

    Z se ponistava jer je tezina gripera ~15 N cisto vertikalno i inace bi
    usla u svaku kaznu kao konstanta koja nema veze s interakcijom. Vrata se
    u oba scenarija gibaju vodoravno, pa se nista bitno ne gubi. (Alternativa
    je tariranje pri resetu, kao u klasicnom pipelineu - skuplje, sa stanjem,
    bez dobitka ovdje.)
    """
    robot: Articulation = env.scene[robot_cfg.name]
    body_ids = _bodies(robot, "ft", FT_SENSOR_BODY)
    idx = body_ids[0]

    force_local = tcp_wrench(env, robot_cfg)[:, :3]
    force_w = quat_apply(robot.data.body_quat_w[:, idx], force_local).clone()
    force_w[:, 2] = 0.0
    return force_w, robot.data.body_pos_w[:, idx]


def door_motion_direction_w(
    env: "ManagerBasedRLEnv",
    tcp_pos_w: torch.Tensor,
    door_type: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Jedinicni vektor trenutnog smjera gibanja tocke hvata, u svijetu.

    PRIVILEGIRANO - racuna se iz stvarne poze vrata, ne iz procjene. Koristi
    se samo za kaznu na okomitu silu, gdje je bas i htio biti referenca:
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
    # ishodistu okvira vrata (door_dof_joint origin = 0 0 0 u URDF-u), pa je
    # radijus-vektor p_tcp - p_root, projiciran u vodoravnu ravninu.
    z = Z_AXIS.to(tcp_pos_w.device).expand_as(tcp_pos_w)
    radius = (tcp_pos_w - root_pos).clone()
    radius[:, 2] = 0.0
    tangent = torch.cross(z, radius, dim=-1)
    return tangent / (tangent.norm(dim=-1, keepdim=True) + 1e-8)


def handle_pos_w(
    env: "ManagerBasedRLEnv",
    handle_local: tuple[float, float, float],
    door_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Svjetska pozicija tocke hvata.

    Racuna se iz poze door_leafa, NE korijena artikulacije: korijen je
    door_frame i fiksan je, dok se kvaka giba s krilom. S korijenom bi ova
    funkcija mjerila koliko su se vrata otvorila umjesto koliko je hvat
    skliznuo, pa bi grasp_lost okidao cim vrata krenu.
    """
    door: Articulation = env.scene[door_cfg.name]
    body_ids = _bodies(door, "leaf", DOOR_LEAF_BODY)
    idx = body_ids[0]
    local = torch.tensor(handle_local, device=env.device).expand(env.num_envs, 3)
    return door.data.body_pos_w[:, idx] + quat_apply(
        door.data.body_quat_w[:, idx], local
    )


# --------------------------------------------------------------------------
# Opazanje
# --------------------------------------------------------------------------


def tcp_wrench(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sila i moment na zapescu, (num_envs, 6), u LOKALNOM okviru linka.

    Cita se na istom mjestu kinematickog lanca gdje tcp_wrench_estimator daje
    svoju procjenu pri deploymentu - najmanji moguci train/eval jaz za
    velicinu koja ulazi i u opazanje i u nagradu.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    body_ids = _bodies(robot, "ft", FT_SENSOR_BODY)
    forces = robot.root_physx_view.get_link_incoming_joint_force()
    return forces[:, body_ids[0], :]


def tcp_pose_b(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Poza vrha alata u frameu baze, (num_envs, 7): pozicija + kvaternion."""
    robot: Articulation = env.scene[asset_cfg.name]
    body_ids = _bodies(robot, "obs", asset_cfg.body_names[0])
    idx = body_ids[0]
    pos = robot.data.body_pos_w[:, idx] - robot.data.root_pos_w
    return torch.cat([pos, robot.data.body_quat_w[:, idx]], dim=-1)


def tcp_velocity_b(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Linearna i kutna brzina vrha alata, (num_envs, 6).

    NAPOMENA O OKVIRU: baza je fiksirana s identitetom kao orijentacijom, pa
    je okvir baze jednak svjetskom do na translaciju - a translacija ne utjece
    na brzinu. Ako robot ikad dobije rotiran spawn, i ovaj i tcp_pose_b treba
    ispraviti.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    body_ids = _bodies(robot, "obs", asset_cfg.body_names[0])
    idx = body_ids[0]
    return torch.cat(
        [robot.data.body_lin_vel_w[:, idx], robot.data.body_ang_vel_w[:, idx]], dim=-1
    )


# TODO (§5): kad constraint_estimator.py bude vektoriziran, dodaj ObsTerm koji
# vraca procijenjeni smjer ogranicenja i radijus. NE stavljaj ovdje
# privilegirani smjer kao privremenu zamjenu - to bi napravilo tocno onaj
# train/eval jaz koji izbjegavamo.


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
    izravan odgovor na razlog zasto je kruto pozicijsko pracenje luka
    zakazalo na zakretnim vratima.
    """
    force_w, tcp_pos_w = _horizontal_force_w(env, robot_cfg)
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
    force_w, _ = _horizontal_force_w(env, robot_cfg)
    return torch.clamp(force_w.norm(dim=-1) - reference_force, min=0.0)


def is_open(
    env: "ManagerBasedRLEnv",
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("door"),
) -> torch.Tensor:
    """Vrata otvorena preko praga.

    Koristi se kao bonus PO KORAKU, ne kao terminacija. Prekid epizode na
    uspjehu je politici davao degeneriranu strategiju: jednim trzajem gurni
    vrata preko praga i zavrsi, cime se izbjegne akumulirana kazna za silu -
    a hvat pritom sklizne s kvake, sto nikoga ne kosta.
    """
    return _door_dof(env, asset_cfg) > threshold


# --------------------------------------------------------------------------
# Terminacija
# --------------------------------------------------------------------------


def force_exceeded(
    env: "ManagerBasedRLEnv",
    limit: float,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Sigurnosni prekid epizode. Ista vodoravna sila kao u kazni - inace bi
    prag okidao na tezini gripera umjesto na interakciji."""
    force_w, _ = _horizontal_force_w(env, robot_cfg)
    return force_w.norm(dim=-1) > limit


def grasp_lost(
    env: "ManagerBasedRLEnv",
    max_distance: float,
    handle_local: tuple[float, float, float],
    robot_cfg: SceneEntityCfg,
    door_cfg: SceneEntityCfg = SceneEntityCfg("door"),
) -> torch.Tensor:
    """Hvat izgubljen: TCP se udaljio od sipke kvake.

    Bez ovoga klizanje s kvake nema cijenu, pa politika slobodno trza rukom i
    vrata gura udarcem umjesto vucenjem.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    tcp_ids = _bodies(robot, "tcp", TCP_BODY)
    tcp = robot.data.body_pos_w[:, tcp_ids[0]]
    handle = handle_pos_w(env, handle_local, door_cfg)
    return (tcp - handle).norm(dim=-1) > max_distance


# --------------------------------------------------------------------------
# Reset
# --------------------------------------------------------------------------


def reset_grasp_and_door(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    nominal_joint_pos: tuple[float, ...],
    delta_range: tuple[float, float],
    position_noise: float,
    handle_local: tuple[float, float, float],
    robot_cfg: SceneEntityCfg,
    door_cfg: SceneEntityCfg = SceneEntityCfg("door"),
) -> None:
    """Postavi hvat i pozu vrata uz nasumicnu rotaciju oko baze ruke.

    ZASTO ROTACIJA: iiwa_joint_1 rotira CIJELU ruku oko vertikale kroz bazu
    ruke, pa dodavanje Δ tom zglobu uz rotaciju vrata za isti Δ oko iste osi
    daje geometrijski IDENTICAN hvat. Iz jedne izmjerene konfiguracije time
    se dobiva egzaktna augmentacija preko svih smjerova iz kojih baza moze
    prici vratima.

    To nije kozmetika nego uvjet prijenosa. Pri deploymentu klasicni
    handle_approach parkira bazu gdje stigne, IK da neki joint_1, a politika
    kao opazanje dobiva pozu TCP-a u okviru baze - trenirana na jednoj
    vrijednosti, vidjela bi neviđenu raspodjelu.

    delta_range treba drzati joint_1 daleko od limita (-2.967 rad): pri
    vucenju se potrosi oko 0.3 rad, a izvorno izmjerena konfiguracija
    (-2.6804) imala je samo 0.29 rad rezerve i politika je limit potrosila -
    vrata su zbog toga stajala na 0.27 m.

    §6: hvat se NE fiksira pomocnim zglobom. Prsti se stvarno zatvore kroz
    sipku, PhysX prodor razrijesi, i hvat drzi bez koraka smirivanja -
    provjereno empirijski, pa nema pojednostavljenja koje bi trebalo braniti.

    position_noise je namjerno malen: iiwa_joint_1 ima krak do TCP-a od
    ~1.19 m, pa 0.02 rad tamo znaci 24 mm bocnog pomaka, a sipka je debela
    28 mm - polovica resetova ne bi uhvatila nista.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    door: Articulation = env.scene[door_cfg.name]
    device = env.device
    n = len(env_ids)

    delta = torch.empty((n,), device=device).uniform_(*delta_range)

    # --- ruka ---
    arm_ids, _ = robot.find_joints("iiwa_joint_[1-7]")
    finger_ids, _ = robot.find_joints("gripper_finger_[1-4]_joint")

    nominal = torch.tensor(nominal_joint_pos, device=device)
    noise = torch.rand((n, len(arm_ids)), device=device) - 0.5
    arm_pos = nominal.unsqueeze(0) + 2.0 * position_noise * noise
    arm_pos[:, 0] = arm_pos[:, 0] + delta

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:, arm_ids] = arm_pos
    joint_pos[:, finger_ids] = GRIPPER_CLOSED
    joint_vel = torch.zeros_like(joint_pos)

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)

    # --- vrata: p' = a + Rz(Δ)(p - a), yaw = yaw0 + Δ ---
    a = torch.tensor(ARM_BASE_B, device=device).expand(n, 3)
    p = torch.tensor(NOMINAL_TCP_B, device=device).expand(n, 3)
    handle_b = a + quat_apply(_yaw_quat(delta), p - a)

    door_quat = _yaw_quat(DOOR_BASE_YAW + delta)
    local = torch.tensor(handle_local, device=device).expand(n, 3)
    door_pos = handle_b - quat_apply(door_quat, local)
    door_pos = door_pos + env.scene.env_origins[env_ids]

    door.write_root_pose_to_sim(
        torch.cat([door_pos, door_quat], dim=-1), env_ids=env_ids
    )
