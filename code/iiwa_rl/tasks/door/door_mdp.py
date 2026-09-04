"""door_mdp.py - funkcije koje env cfg zice u terme.

Podjela naspram door_events.py: tamo je randomizacija otpora vrata (parna uz
door_cfg.py), ovdje je sve ostalo - nagrada, terminacija, opazanje i reset
hvata.

PRIVILEGIRANE VELICINE: kut/pomak DOF-a vrata i njegov stvarni smjer gibanja
koriste se ISKLJUCIVO u nagradi i terminaciji, nikad u opazanju politike.
Isto pravilo kao u klasicnom pristupu. Nagrada smije znati istinu jer se
racuna offline; politika ne smije jer ju u stvarnosti nema.

OKVIRI, dvije zamke:

1. get_link_incoming_joint_force vraca silu u LOKALNOM okviru linka. Svaka
   usporedba sa smjerom gibanja vrata mora ju prvo rotirati u svijet -
   propust u tome daje kaznu reda 1e-6 koja izgleda kao "nema sile", a
   zapravo znaci da politika trenira bez ikakve informacije o kontaktu.

2. Okvir baze je TIJELO base_link, a NE korijen artikulacije. Otkad robot
   ima fiktivne zglobove za pokretnu bazu, korijen je link 'world' i fiksan
   je u ishodistu env-a. Citanje root_pos_w/root_quat_w umjesto poze tijela
   dalo bi opazanje koje se uopce ne mijenja kad se baza pomakne.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_mul

from .door_cfg import DOOR_DOF_JOINT, DOOR_LEAF_BODY
from .robot_cfg import BASE_BODY, GRIPPER_OPEN, GRIPPER_CLOSED, BASE_JOINTS

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


# --------------------------------------------------------------------------
# Pomocne
# --------------------------------------------------------------------------

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


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Konjugat jedinicnog kvaterniona (w, x, y, z).

    Rucno umjesto quat_conjugate/quat_apply_inverse jer se ta imena mijenjaju
    medu verzijama Isaac Laba, a ovo je cetiri znaka koda.
    """
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


def _yaw_quat(angle: torch.Tensor) -> torch.Tensor:
    """Kvaternion (w, x, y, z) rotacije oko vertikale."""
    half = 0.5 * angle
    zero = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zero, zero, torch.sin(half)], dim=-1)


def _past_grace(env: "ManagerBasedRLEnv", grace_steps: int) -> torch.Tensor:
    """Prvih nekoliko koraka epizode terminacije ne vrijede.

    Dva razloga, oba lazni pozitivi u koraku 0:
    - prsti se pri resetu zatvaraju KROZ polugu, PhysX prodor razrijesi
      impulsom daleko iznad praga sile
    - body_pos_w nije osvjezen odmah nakon write_joint_state_to_sim, pa se
      zastarjeli TCP usporeduje s novom pozom vrata
    """
    return env.episode_length_buf > grace_steps


def _base_frame(robot: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
    """Poza tijela base_link u svijetu. Vidi zamku 2 u docstringu modula."""
    idx = _bodies(robot, "base", BASE_BODY)[0]
    return robot.data.body_pos_w[:, idx], robot.data.body_quat_w[:, idx]


def _door_dof(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Pomak/kut DOF-a vrata, (num_envs,). PRIVILEGIRANO."""
    door: Articulation = env.scene[asset_cfg.name]
    joint_ids = _joints(door, "dof", DOOR_DOF_JOINT)
    return door.data.joint_pos[:, joint_ids[0]]


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
    idx = _bodies(robot, "ft", FT_SENSOR_BODY)[0]

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

    if door_type == "sliding":
        # Klizanje je duz lokalne +Y osi okvira vrata (konvencija iz URDF-a).
        local = torch.zeros_like(tcp_pos_w)
        local[:, 1] = 1.0
        return quat_apply(door.data.root_quat_w, local)

    # Zakretna: tangenta luka je z x (p - c), gdje je c os sarke. Sarka je u
    # ishodistu okvira vrata (door_dof_joint origin = 0 0 0 u URDF-u), pa je
    # radijus-vektor p_tcp - p_root, projiciran u vodoravnu ravninu.
    z = Z_AXIS.to(tcp_pos_w.device).expand_as(tcp_pos_w)
    radius = (tcp_pos_w - door.data.root_pos_w).clone()
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
    idx = _bodies(door, "leaf", DOOR_LEAF_BODY)[0]
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
    idx = _bodies(robot, "ft", FT_SENSOR_BODY)[0]
    return robot.root_physx_view.get_link_incoming_joint_force()[:, idx, :]


def tcp_pose_b(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Poza vrha alata u frameu baze, (num_envs, 7): pozicija + kvaternion.

    Okvir baze je tijelo base_link (zamka 2). Rotira se, ne samo translatira:
    baza se moze zakretati, a bez rotacije bi politika vidjela pozu koja se
    mijenja kad se baza okrene iako se ruka naspram baze nije pomaknula.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    idx = _bodies(robot, "obs", asset_cfg.body_names[0])[0]
    base_pos, base_quat = _base_frame(robot)
    q_inv = quat_inv(base_quat)
    pos = quat_apply(q_inv, robot.data.body_pos_w[:, idx] - base_pos)
    quat = quat_mul(q_inv, robot.data.body_quat_w[:, idx])
    return torch.cat([pos, quat], dim=-1)


def tcp_velocity_b(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Linearna i kutna brzina vrha alata u frameu baze, (num_envs, 6)."""
    robot: Articulation = env.scene[asset_cfg.name]
    idx = _bodies(robot, "obs", asset_cfg.body_names[0])[0]
    _, base_quat = _base_frame(robot)
    q_inv = quat_inv(base_quat)
    return torch.cat(
        [
            quat_apply(q_inv, robot.data.body_lin_vel_w[:, idx]),
            quat_apply(q_inv, robot.data.body_ang_vel_w[:, idx]),
        ],
        dim=-1,
    )


def base_velocity(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Brzina baze (vx, vy, omega) iz fiktivnih zglobova, (num_envs, 3).

    ZASTO JE OVO U OPAZANJU: baza se giba admitancijskim zakonom, dakle iz
    razloga koji nije akcija politike. Bez ovoga politika vidi da joj se poza
    TCP-a u okviru baze mijenja i tumaci to kao posljedicu vlastite akcije,
    pa korigira - a korekcija se zbraja s gibanjem baze. To se u treningu
    ocitovalo kao nagli skok grasp_lost s 0.1 na 0.8 tocno u trenutku kad je
    politika prvi put povukla dovoljno daleko da baza uopce krene.

    Ovo NIJE privilegirana velicina: pri deploymentu je to zadani cmd_vel,
    koji inferencijski cvor ionako ima jer ga sam salje.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = [_joints(robot, f"base_{n}", n)[0] for n in BASE_JOINTS]
    return robot.data.joint_vel[:, joint_ids]


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
    """Normirani napredak DOF-a vrata. Glavni clan nagrade.

    full_travel je PRAG USPJEHA, ne puni hod vrata. Uz normiranje na puni hod
    centimetar pomaka vrijedi premalo da nadjaca kazne, pa je politika ucila
    drzati kvaku i ne dirati vrata.
    """
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


def joint_saturation_penalty(
    env: "ManagerBasedRLEnv",
    comfort_band: float,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Kazna za rad blizu zglobnih limita.

    Bez nje politika odmah ispruzi ruku do ruba radnog prostora, jer je to
    najbrzi nacin da vrata krenu. Ondje je jakobijan lose kondicioniran, fino
    upravljanje silom otpada, hvat proklizi i gripper pocne gurati polugu
    umjesto da je drzi. Cijena dolazi prekasno da bi je politika povezala s
    ispruzenjem, pa je treba naplatiti odmah.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ids = _joints(robot, "arm", "iiwa_joint_[1-7]")
    q = robot.data.joint_pos[:, ids]
    lower = robot.data.soft_joint_pos_limits[:, ids, 0]
    upper = robot.data.soft_joint_pos_limits[:, ids, 1]
    fraction = (q - lower) / (upper - lower + 1e-6)
    return ((fraction - 0.5).abs() - comfort_band).clamp(min=0.0).sum(dim=-1)


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
    grace_steps: int = 5,
) -> torch.Tensor:
    """Sigurnosni prekid epizode. Ista vodoravna sila kao u kazni - inace bi
    prag okidao na tezini gripera umjesto na interakciji."""
    force_w, _ = _horizontal_force_w(env, robot_cfg)
    return (force_w.norm(dim=-1) > limit) & _past_grace(env, grace_steps)


def grasp_lost(
    env: "ManagerBasedRLEnv",
    max_distance: float,
    handle_local: tuple[float, float, float],
    robot_cfg: SceneEntityCfg,
    door_cfg: SceneEntityCfg = SceneEntityCfg("door"),
    grace_steps: int = 5,
) -> torch.Tensor:
    """Hvat izgubljen: TCP se udaljio od sipke kvake.

    Bez ovoga klizanje s kvake nema cijenu, pa politika slobodno trza rukom i
    vrata gura udarcem umjesto vucenjem.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    tcp = robot.data.body_pos_w[:, _bodies(robot, "tcp", TCP_BODY)[0]]
    handle = handle_pos_w(env, handle_local, door_cfg)
    return ((tcp - handle).norm(dim=-1) > max_distance) & _past_grace(env, grace_steps)


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
    se dobiva egzaktna augmentacija preko smjerova iz kojih baza moze prici
    vratima.

    To nije kozmetika nego uvjet prijenosa. Pri deploymentu klasicni
    handle_approach parkira bazu gdje stigne, IK da neki joint_1, a politika
    kao opazanje dobiva pozu TCP-a u okviru baze - trenirana na jednoj
    vrijednosti, vidjela bi nevidenu raspodjelu.

    §6: hvat se NE fiksira pomocnim zglobom. Prsti se stvarno zatvore kroz
    sipku, PhysX prodor razrijesi, i hvat drzi bez koraka smirivanja -
    provjereno empirijski, pa nema pojednostavljenja koje bi trebalo braniti.

    position_noise je namjerno malen: iiwa_joint_1 ima krak do TCP-a od
    ~1.19 m, pa 0.02 rad tamo znaci 24 mm bocnog pomaka, a sipka je debela
    28 mm - polovica resetova ne bi uhvatila nista.

    BAZA: fiktivni zglobovi se vracaju na nulu, cime base_link sjeda u
    ishodiste env-a s identitetom - poza vrata dolje racuna se u tom okviru.
    To ide kroz default_joint_pos, koji za base_.*_joint drzi nulu, pa nije
    potreban zaseban upis.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    door: Articulation = env.scene[door_cfg.name]
    device = env.device
    n = len(env_ids)

    delta = torch.empty((n,), device=device).uniform_(*delta_range)

    # --- ruka, prsti i baza ---
    arm_ids = _joints(robot, "arm", "iiwa_joint_[1-7]")
    finger_ids = _joints(robot, "fingers", "gripper_finger_[1-4]_joint")

    nominal = torch.tensor(nominal_joint_pos, device=device)
    noise = torch.rand((n, len(arm_ids)), device=device) - 0.5
    arm_pos = nominal.unsqueeze(0) + 2.0 * position_noise * noise
    arm_pos[:, 0] = arm_pos[:, 0] + delta

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:, arm_ids] = arm_pos
    # Prsti se pri resetu postavljaju OTVORENI, a pogon ih zatvara kroz
    # prvih nekoliko koraka epizode. Upis zatvorenog stanja znaci da se prsti
    # instantno nadu unutar poluge; PhysX taj prodor razrjesava impulsom koji
    # doseze 130 N, a hvat u vertikali drzi svega 11.6 N - hvat je narusen
    # prije nego politika napravi ijedan potez.
    joint_pos[:, finger_ids] = GRIPPER_OPEN
    joint_vel = torch.zeros_like(joint_pos)

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # Cilj pogona je zatvoreno stanje, pa se prsti sami stisnu oko poluge.
    target = joint_pos.clone()
    target[:, finger_ids] = GRIPPER_CLOSED
    robot.set_joint_position_target(target, env_ids=env_ids)

    # --- vrata: p' = a + Rz(Δ)(p - a), yaw = yaw0 + Δ ---
    a = torch.tensor(ARM_BASE_B, device=device).expand(n, 3)
    p = torch.tensor(NOMINAL_TCP_B, device=device).expand(n, 3)
    handle_b = a + quat_apply(_yaw_quat(delta), p - a)

    door_quat = _yaw_quat(DOOR_BASE_YAW + delta)
    local = torch.tensor(handle_local, device=device).expand(n, 3)
    door_pos = handle_b - quat_apply(door_quat, local) + env.scene.env_origins[env_ids]

    door.write_root_pose_to_sim(
        torch.cat([door_pos, door_quat], dim=-1), env_ids=env_ids
    )
