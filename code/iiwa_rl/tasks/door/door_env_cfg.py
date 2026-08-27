"""door_env_cfg.py - env cfg za otvaranje vrata varijabilnom impedancijom.

OPSEG (§0): epizoda pocinje od stanja "gripper drzi kvaku". Prilaz i hvat
ostaju klasicni i nisu dio ovog env-a.

BAZA JE FIKSNA. Izmjereno je da se pri fiksnoj bazi i grubom vucenju vrata
otvore oko 0.065 m prije nego ruka dodje do granice radnog prostora, pa je
prag uspjeha 0.05 m - "vrata otvorena", ne "vrata otvorena do kraja". To je
kinematicko ogranicenje, ne spustena ljestvica, i u radu ide kao eksplicitno
ogranicenje opsega uz izmjerenu brojku. Ako se baza kasnije doda u prostor
akcije, mijenjaju se SUCCESS_* konstante i ActionsCfg, ostalo ostaje.

NEMA TERMINACIJE NA USPJEHU. Prekid epizode cim vrata prijedju prag davao je
politici degeneriranu strategiju: jedan trzaj preko praga, epizoda gotova,
akumulirana kazna za silu izbjegnuta, hvat sklizne bez posljedica. Uspjeh je
zato bonus PO KORAKU za drzanje vrata otvorenima, a epizoda traje do isteka
vremena, gubitka hvata ili prekoracenja sile.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
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
from isaaclab.utils.noise import GaussianNoiseCfg

from . import door_mdp as mdp
from .door_cfg import (
    HANDLE_LOCAL_REVOLUTE,
    HANDLE_LOCAL_SLIDING,
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
SUCCESS_SLIDING_M = 0.15
SUCCESS_REVOLUTE_RAD = 0.12  # ~7 stupnjeva, isti red velicine pomaka kvake

# ~280 N je izmjereno klasicnim pristupom kao tranzijent, ne kao otpor vrata.
# Ovdje sluzi kao referenca iznad koje kaznjavamo, i kao sigurnosni prekid.
FORCE_REFERENCE_N = 280.0
FORCE_ABORT_N = 400.0

# Udaljenost TCP-a od sipke iznad koje se hvat smatra izgubljenim.
GRASP_LOST_DISTANCE_M = 0.09

TCP_BODY = "gripper_tcp"

# Ista poza hvata kao u klasicnom pokusu, ali druga grana inverzne kinematike.
# MoveIt je tamo izabrao rjesenje s laktom iza, kod kojeg su joint_1 i joint_2
# na 5% odnosno 3% raspona - politika je pri vucenju potrosila limit i vrata
# su stala na 0.27 m. Ovo rjesenje drzi sve zglobove izmedju 16% i 75%
# raspona uz istu poziciju TCP-a (do 0.1 mm) i istu orijentaciju (do 1e-4 rad).
#
# ZA DEPLOYMENT: handle_approach mora dobiti seed koji vodi na OVU granu,
# inace politika pri deploymentu starta iz konfiguracije na kojoj nije
# trenirana. To je konkretna stavka za fazu integracije.
NOMINAL_GRASP_JOINT_POS = (-0.4071, 0.9457, 0.3483, -1.4265, -0.3294, -0.7337, 1.5311)

# Zakretna vrata: ISTA tocka hvata, ali poluga lezi vodoravno umjesto
# vertikalno kao sipka kliznih, pa je gripper zakrenut za +90 stupnjeva oko
# svoje osi prilaza. Lokalna Y os gripera (ona koja lezi duz drske) time
# gleda vodoravno. Poza TCP-a je identicna do 0.1 mm; razlikuje se samo
# orijentacija. Svi zglobovi izmedju 49% i 84% raspona.
NOMINAL_GRASP_JOINT_POS_REVOLUTE = (
    0.4756,
    1.3382,
    1.9412,
    1.4265,
    -0.0489,
    0.9288,
    1.1665,
)

# Poza vrata izvedena IZ te konfiguracije, ne obrnuto: TCP je pri hvatu bio na
# (1.158, -0.287, 1.005) u base_link, pa su vrata postavljena tako da tocka
# hvata padne tocno tamo. Uz yaw 180 lokalne osi X i Y gledaju u -X i -Y
# svijeta, pa se lokalni offset kvake (HANDLE_LOCAL_*) oduzima.
SLIDING_DOOR_SPAWN = (0.174, 0.839, 0.005)
REVOLUTE_DOOR_SPAWN = (0.172, 0.808, 0.005)
DOOR_SPAWN_ROT = (-0.5646, 0.0, 0.0, 0.8253)  # yaw 180 + 68.8 stupnjeva


@configclass
class DoorSceneCfg(InteractiveSceneCfg):
    """Robot, vrata, pod, svjetlo. Vrata se MORAJU zvati 'door' - SceneEntityCfg
    u door_events.py i door_mdp.py racuna na to ime."""

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
    kada popustljiva, umjesto fiksnog pravila.

    O PARAMETRIZACIJI KRUTOSTI: OSC mnozi akciju sa stiffness_scale bez
    pomaka, pa akcija 0 daje krutost 0 koja se klipa na donju granicu. Uz
    scale 1000 i donju granicu 50 polovica Gaussove raspodjele politike lezi
    na dnu, gdje je gradijent nula - politika tada nauci biti mlitava jer je
    to najlaksi nacin da izbjegne kaznu za silu. Scale 300 uz donju granicu
    200 drzi upotrebljiv raspon unutar tipicnog izlaza politike.
    """

    arm = OperationalSpaceControllerActionCfg(
        asset_name="robot",
        joint_names=["iiwa_joint_[1-7]"],
        body_name=TCP_BODY,
        controller_cfg=OperationalSpaceControllerCfg(
            # pose_rel: akcija je POMAK reference od trenutne poze TCP-a. Uz
            # pose_abs bi cilj bio unutar position_scale od ishodista okvira
            # zadatka i ruka bi se trgala prema vlastitoj bazi.
            target_types=["pose_rel"],
            impedance_mode="variable_kp",
            motion_control_axes_task=[1, 1, 1, 1, 1, 1],
            inertial_dynamics_decoupling=True,
            gravity_compensation=True,
            motion_stiffness_task=300.0,
            motion_damping_ratio_task=1.0,
            # Gornja granica x position_scale odredjuje maksimalnu silu:
            # 5000 x 0.05 = 250 N, tik ispod referentne granice iz klasicnog.
            motion_stiffness_limits_task=(200.0, 15000.0),
            # 7-DOF ruka je redundantna: bez ovoga lakat pluta i pri kontaktu
            # moze odlutati u singularitet ili zglobni limit dok TCP mirno
            # stoji. Izgleda kao nestabilna politika, a nije.
            nullspace_control="position",
            nullspace_stiffness=10.0,
            nullspace_damping_ratio=1.0,
        ),
        # 0.015 m po koraku. Politika uzorkuje iz Gaussa sa std 1.0, dakle
        # akcija ide do ±3, pa je stvarni maksimalni skok reference ~4.5 cm na
        # 30 Hz. Pri 0.05 je bio 15 cm po koraku i eksploracija je trgala hvat
        # sa sipke prije nego je politika stigla nesto nauciti - jedina
        # strategija koja prezivi takvu eksploraciju je nulta akcija.
        position_scale=0.015,
        orientation_scale=0.1,
        stiffness_scale=300.0,
        damping_ratio_scale=1.0,
        nullspace_joint_pos_target="default",
    )
    # Prsti nisu u prostoru akcije - hvat je zatvoren cijelu epizodu (§0).


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
            # (±0.5 N u zraku). Bez ovoga politika uci na savrsenoj sili koju
            # pri deploymentu nema.
            noise=GaussianNoiseCfg(mean=0.0, std=0.5),
        )
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # Normirano na PRAG, ne na puni hod: dosezanje praga vrijedi 1.0 po
    # koraku umjesto 0.19, pa centimetar pomaka nosi 2 umjesto 0.125.
    # Uz normiranje na puni hod je vucenje bilo skuplje od mirovanja na
    # svakom koraku prije praga, a bonus na 0.15 m prerijedak da povuce
    # eksploraciju - politika je naucila drzati kvaku i ne dirati vrata.
    progress = RewTerm(
        func=mdp.dof_progress,
        weight=10.0,
        params={"full_travel": SUCCESS_SLIDING_M},
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
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.002)
    # Bonus PO KORAKU za drzanje vrata otvorenima - zato tezina 2, ne 50.
    success = RewTerm(
        func=mdp.is_open, weight=2.0, params={"threshold": SUCCESS_SLIDING_M}
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    grasp_lost = DoneTerm(
        func=mdp.grasp_lost,
        params={
            "max_distance": GRASP_LOST_DISTANCE_M,
            "handle_local": HANDLE_LOCAL_SLIDING,
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    overforce = DoneTerm(
        func=mdp.force_exceeded,
        params={"limit": FORCE_ABORT_N, "robot_cfg": SceneEntityCfg("robot")},
    )


@configclass
class EventCfg:
    # Vrata natrag u zatvoreno. Bez ovoga ostaju otvorena iz prethodne
    # epizode, pa svaka sljedeca starta iznad praga i uspjeh je besplatan.
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
        func=mdp.reset_grasp_and_door,
        mode="reset",
        params={
            "nominal_joint_pos": NOMINAL_GRASP_JOINT_POS,
            # Faza prilaza centrira robota pred vratima, pa vrata NIKAD nisu
            # bocno. Ostaje samo rezidualna greska kuta prilaska; isti raspon
            # kao spawn-yaw-jitter u build_integration_scene.py (±20°).
            "delta_range": (-0.35, 0.35),
            "position_noise": 0.005,
            "handle_local": HANDLE_LOCAL_SLIDING,
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


@configclass
class DoorRevoluteEnvCfg(DoorEnvCfg):
    """Zakretna vrata. Zbog kanonskih imena u assetima razlika je samo u
    USD-u, pozi, rasponima otpora i pragu - nijedna funkcija se ne mijenja."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.door = REVOLUTE_DOOR_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Door",
            init_state=REVOLUTE_DOOR_CFG.init_state.replace(
                pos=REVOLUTE_DOOR_SPAWN, rot=DOOR_SPAWN_ROT
            ),
        )
        self.events.grasp.params["handle_local"] = HANDLE_LOCAL_REVOLUTE
        self.events.grasp.params["nominal_joint_pos"] = NOMINAL_GRASP_JOINT_POS_REVOLUTE
        self.events.door_resistance.params = {
            "ranges": REVOLUTE_FREE_RESISTANCE,
            "alt_ranges": REVOLUTE_CLOSER_RESISTANCE,
            "alt_probability": REVOLUTE_CLOSER_PROBABILITY,
        }
        self.rewards.progress.params["full_travel"] = SUCCESS_REVOLUTE_RAD
        self.rewards.perpendicular_force.params["door_type"] = "revolute"
        self.rewards.success.params["threshold"] = SUCCESS_REVOLUTE_RAD
        self.terminations.grasp_lost.params["handle_local"] = HANDLE_LOCAL_REVOLUTE
