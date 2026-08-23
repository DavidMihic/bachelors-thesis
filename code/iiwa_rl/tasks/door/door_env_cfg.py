"""door_env_cfg.py - env cfg za otvaranje vrata varijabilnom impedancijom.

OPSEG (§0): epizoda pocinje od stanja "gripper drzi kvaku". Prilaz i hvat
ostaju klasicni i nisu dio ovog env-a.

BAZA JE FIKSNA. Doseg ruke pri fiksnoj bazi je oko 0.17 m, pa je prag uspjeha
spusten na 0.15 m odnosno 25 stupnjeva - "vrata otvorena", ne "vrata otvorena
do kraja". To je ionako faza u kojoj se ogranicenje procjenjuje i u kojoj je
razlika izmedju klasicnog i naucenog pristupa zanimljiva. Ako se kasnije baza
doda u prostor akcije, mijenjaju se SUCCESS_* konstante i ActionsCfg, ostalo
ostaje.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils

from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions import OperationalSpaceControllerActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

import isaaclab.envs.mdp as base_mdp

from . import door_mdp as mdp
from .door_cfg import (
    REVOLUTE_CLOSER_PROBABILITY,
    REVOLUTE_CLOSER_RESISTANCE,
    REVOLUTE_DOOR_CFG,
    REVOLUTE_FREE_RESISTANCE,
    REVOLUTE_FULL_TRAVEL_RAD,
    SLIDING_DOOR_CFG,
    SLIDING_FULL_TRAVEL_M,
    SLIDING_RESISTANCE,
)
from .door_events import randomize_door_resistance
from .robot_cfg import KMR_IIWA_CFG

# --- Pragovi uspjeha, ograniceni dosegom ruke pri fiksnoj bazi ---
# Izmjereno: pri fiksnoj bazi i otvorenoj petlji vrata se otvore ~0.065 m
# prije nego ruka dodje do granice radnog prostora. Prag je postavljen ispod
# toga jer naucena politika treba imati prostora nadmasiti grubo vucenje.
# Ovo NIJE spustena ljestvica nego kinematicko ogranicenje fiksne baze - u
# radu ide kao eksplicitno ogranicenje opsega, uz izmjerenu brojku.
SUCCESS_SLIDING_M = 0.05
SUCCESS_REVOLUTE_RAD = 0.44  # 25 stupnjeva

# ~280 N je izmjereno klasicnim pristupom kao tranzijent, ne kao otpor vrata.
# Ovdje sluzi kao referenca iznad koje kaznjavamo, i kao sigurnosni prekid.
FORCE_REFERENCE_N = 280.0
FORCE_ABORT_N = 400.0

TCP_BODY = "gripper_tcp"

# Sedam kutova ruke iz uspjesnog hvata klasicnim handle_approach.
NOMINAL_GRASP_JOINT_POS = (-2.6804, -1.9776, -1.9871, 1.4265, 0.2082, 0.8742, 0.4177)

# Poza vrata izvedena IZ te konfiguracije, ne obrnuto: TCP je pri hvatu bio na
# (1.158, -0.287, 1.005) u base_link, pa su vrata postavljena tako da kvaka
# padne tocno tamo. Uz yaw 180 lokalne osi X i Y vrata gledaju u -X i -Y
# svijeta, pa se lokalni offset kvake oduzima:
#   klizna:   sipka na lokalnom (0.09, 0.65, 1.0)
#   zakretna: sredina poluge na lokalnom (0.06, 0.64, 1.0)
# Z je 0.005 umjesto 0.01 jer je kvaka bila na 1.005 m - klirens od poda je
# time upola manji, ali i dalje postoji.
SLIDING_DOOR_SPAWN = (1.248, 0.363, 0.005)
REVOLUTE_DOOR_SPAWN = (1.218, 0.353, 0.005)
DOOR_SPAWN_ROT = (0.0, 0.0, 0.0, 1.0)  # yaw 180, (w, x, y, z)


@configclass
class DoorSceneCfg(InteractiveSceneCfg):
    """Robot, vrata, pod, svjetlo. Vrata se zovu 'door' - SceneEntityCfg u
    door_events.py i door_mdp.py racuna na to ime."""

    robot = KMR_IIWA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    door = SLIDING_DOOR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Door",
        init_state=SLIDING_DOOR_CFG.init_state.replace(
            pos=SLIDING_DOOR_SPAWN, rot=DOOR_SPAWN_ROT
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0)
    )


@configclass
class ActionsCfg:
    """Varijabilna impedancija u prostoru zadatka.

    impedance_mode="variable_kp" znaci da akcija nosi i referentnu pozu i
    krutost po osi - tocno ono sto §4 trazi: politika uci KADA biti kruta a
    kada popustljiva, umjesto fiksnog pravila. Prigusenje se vodi omjerom
    prigusenja da politika ne moze zaluditi sustav u nestabilnost.

    Donja granica krutosti mora biti stvarno niska (50 N/m) - inace politika
    nema pristup popustljivom rezimu i env degenerira u kruto pozicijsko
    upravljanje, tj. tocno u ono sto je na zakretnim vratima zakazalo.
    """

    arm = OperationalSpaceControllerActionCfg(
        asset_name="robot",
        joint_names=["iiwa_joint_[1-7]"],
        body_name=TCP_BODY,
        controller_cfg=OperationalSpaceControllerCfg(
            # pose_rel: akcija je POMAK reference od trenutne poze TCP-a, ne
            # apsolutna ciljna poza. Uz pose_abs i position_scale=0.02 cilj bi
            # bio unutar 2 cm od ishodista okvira zadatka (baze), pa bi se ruka
            # trgala prema vlastitoj bazi.
            target_types=["pose_rel"],
            impedance_mode="variable_kp",
            motion_control_axes_task=[1, 1, 1, 1, 1, 1],
            inertial_dynamics_decoupling=True,
            gravity_compensation=True,
            motion_stiffness_task=300.0,  # pocetna; politika je nadjacava
            motion_damping_ratio_task=1.0,
            # Gornja granica x position_scale odredjuje maksimalnu silu:
            # 5000 x 0.05 = 250 N, tik ispod ~280 N referentne granice iz
            # klasicnog. Pri 2000 x 0.02 = 40 N vrata s trenjem preko 40 N
            # (a raspon ide do 60) bila su nerjesiva.
            motion_stiffness_limits_task=(
                50.0,
                5000.0,
            ),  # 7-DOF ruka je redundantna: bez ovoga lakat pluta i pri kontaktu
            # moze odlutati u singularitet ili zglobni limit dok TCP mirno
            # stoji. Izgleda kao nestabilna politika, a nije.
            nullspace_control="position",
            nullspace_stiffness=10.0,
            nullspace_damping_ratio=1.0,
        ),
        position_scale=0.05,  # vidi motion_stiffness_limits_task gore
        orientation_scale=0.1,
        # PROVJERI EMPIRIJSKI: akcija se mnozi ovim pa klipa na
        # motion_stiffness_limits_task. Uz scale 1.0 i izlaz politike reda
        # jedinice sve bi zavrsilo na donjoj granici i "varijabilna"
        # impedancija bi bila konstantna. Logiraj ostvarenu krutost tijekom
        # prvog treninga i podesi da raspon stvarno pokriva 50-2000.
        stiffness_scale=1000.0,
        damping_ratio_scale=1.0,
        nullspace_joint_pos_target="default",
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Samo velicine koje bi robot stvarno imao. Kut/pomak DOF-a vrata
        NIJE ovdje - isto pravilo kao u klasicnom pristupu."""

        tcp_pose = ObsTerm(
            func=mdp.tcp_pose_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=TCP_BODY)},
        )
        tcp_velocity = ObsTerm(
            func=mdp.tcp_velocity_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=TCP_BODY)},
        )
        wrench = ObsTerm(
            func=mdp.tcp_wrench,
            params={"asset_cfg": SceneEntityCfg("robot")},
            # Sum reda velicine onoga sto tcp_wrench_estimator stvarno ima
            # (±0.5 N u zraku, rezidual ~12 Nm). Bez ovoga politika uci na
            # savrsenoj sili koju pri deploymentu nema.
            noise=GaussianNoiseCfg(mean=0.0, std=0.5),
        )
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    progress = RewTerm(
        func=mdp.dof_progress,
        weight=10.0,
        params={"full_travel": SLIDING_FULL_TRAVEL_M},
    )
    perpendicular_force = RewTerm(
        func=mdp.perpendicular_force_penalty,
        weight=-0.02,
        params={"door_type": "sliding", "robot_cfg": SceneEntityCfg("robot")},
    )
    excess_force = RewTerm(
        func=mdp.total_force_penalty,
        weight=-0.05,
        params={
            "reference_force": FORCE_REFERENCE_N,
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    success = RewTerm(
        func=mdp.is_open, weight=50.0, params={"threshold": SUCCESS_SLIDING_M}
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    opened = DoneTerm(func=mdp.is_open, params={"threshold": SUCCESS_SLIDING_M})
    overforce = DoneTerm(
        func=mdp.force_exceeded,
        params={"limit": FORCE_ABORT_N, "robot_cfg": SceneEntityCfg("robot")},
    )


@configclass
class EventCfg:
    # Vrata natrag u zatvoreno. Bez ovoga ostaju otvorena iz prethodne
    # epizode, pa svaka sljedeca starta iznad praga i odmah zavrsi kao uspjeh -
    # politika dobiva nagradu besplatno i ne uci nista.
    door_state = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    door_resistance = EventTerm(
        func=randomize_door_resistance,
        mode="reset",
        params={"ranges": SLIDING_RESISTANCE},
    )
    grasp = EventTerm(
        func=mdp.reset_to_grasp,
        mode="reset",
        params={
            "nominal_joint_pos": NOMINAL_GRASP_JOINT_POS,
            # 0.005 rad; na kraku od 1.19 m do TCP-a to je ~6 mm, sto je jos
            # unutar sipke od 28 mm. Pri 0.02 rad bi pomak bio 24 mm i dobar
            # dio resetova ne bi uhvatio nista.
            "position_noise": 0.005,
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    # TODO (§4): trenje prstiju i poza vrata. Trenje prstiju vise nije
    # sporedno otkad je kvaka zavarena - sav moment sarke ide kroz njega.


@configclass
class DoorEnvCfg(ManagerBasedRLEnvCfg):
    scene: DoorSceneCfg = DoorSceneCfg(num_envs=64, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        # num_envs kreni malen i penji se mjereci VRAM (8 GB, RTX 5050).


@configclass
class DoorRevoluteEnvCfg(DoorEnvCfg):
    """Zakretna vrata. Zbog kanonskih imena u assetima razlika je samo u
    USD-u, rasponima otpora i pragu - nijedna funkcija se ne mijenja."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.door = REVOLUTE_DOOR_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Door",
            init_state=REVOLUTE_DOOR_CFG.init_state.replace(
                pos=REVOLUTE_DOOR_SPAWN, rot=DOOR_SPAWN_ROT
            ),
        )
        self.events.door_resistance.params = {
            "ranges": REVOLUTE_FREE_RESISTANCE,
            "alt_ranges": REVOLUTE_CLOSER_RESISTANCE,
            "alt_probability": REVOLUTE_CLOSER_PROBABILITY,
        }
        self.rewards.progress.params["full_travel"] = REVOLUTE_FULL_TRAVEL_RAD
        self.rewards.perpendicular_force.params["door_type"] = "revolute"
        self.rewards.success.params["threshold"] = SUCCESS_REVOLUTE_RAD
        self.terminations.opened.params["threshold"] = SUCCESS_REVOLUTE_RAD
