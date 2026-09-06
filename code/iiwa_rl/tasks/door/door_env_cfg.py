"""door_env_cfg.py - env cfg za otvaranje vrata varijabilnom impedancijom.

OPSEG (§0): epizoda pocinje od stanja "gripper drzi kvaku". Prilaz i hvat
ostaju klasicni i nisu dio ovog env-a.

BAZA SE VODI ADMITANCIJSKIM ZAKONOM (base_admittance.py), ne politikom - RL
i dalje uci samo interakciju. Koordinacija ruke i baze time NIJE naucena i
to treba tako i napisati u radu.

PRAGOVI SU IZMJERENI, NE POGODENI. Pri fiksnoj bazi je naucena politika
dosezala 0.27 m hoda kliznih vrata prije nego bi ruka potrosila zglobni
limit, pa je prag 0.15 m. Za zakretna je 0.5 rad postavljen tako da zatvarac
stvarno opterecuje sustav - pri manjem kutu je moment zatvaraca premalen da
bi razlika izmedju fiksne i varijabilne impedancije bila mjerljiva.

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
from isaaclab.envs.mdp.actions import (
    OperationalSpaceControllerActionCfg,
    JointVelocityActionCfg,
)
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
from .base_admittance import BaseAdmittanceActionCfg
from .door_cfg import (
    HANDLE_LOCAL_REVOLUTE,
    HANDLE_LOCAL_SLIDING,
    REVOLUTE_CLOSER_PROBABILITY,
    REVOLUTE_CLOSER_RESISTANCE,
    REVOLUTE_DOOR_CFG,
    REVOLUTE_FREE_RESISTANCE,
    SLIDING_DOOR_CFG,
    SLIDING_RESISTANCE,
)
from .door_events import randomize_door_resistance
from .robot_cfg import KMR_IIWA_CFG

# --- Pragovi uspjeha (vidi docstring) ---
SUCCESS_SLIDING_M = 0.6
SUCCESS_REVOLUTE_RAD = 0.5

# ~280 N je izmjereno klasicnim pristupom kao tranzijent, ne kao otpor vrata.
# Ovdje sluzi kao referenca iznad koje kaznjavamo, i kao sigurnosni prekid.
FORCE_REFERENCE_N = 280.0
FORCE_ABORT_N = 400.0

# Klizna vrata: sipka je visoka 0.25 m i gripper po njoj legitimno klizi, pa
# 0.09 prekida epizode koje nisu ispustanje. Zakretna imaju kratku polugu i
# ondje je 0.09 ispravan prag - postavlja se zasebno u DoorRevoluteEnvCfg.
GRASP_LOST_DISTANCE_M = 0.09
GRASP_LOST_DISTANCE_REVOLUTE_M = 0.09

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

# Poza vrata pri SPAWNU. reset_grasp_and_door je svejedno pregazi u prvom
# resetu, pa su ove vrijednosti bitne samo za prvi frame - ali neka budu
# tocne, jer se inace vizualna provjera kroz zero_agent cita krivo.
# Izvedene su iz NOMINAL_TCP_B uz delta = 0: vrata se postave tako da tocka
# hvata padne na TCP, a yaw 180 okrece vrata natrag prema robotu.
SLIDING_DOOR_SPAWN = (1.248, 0.363, 0.005)
REVOLUTE_DOOR_SPAWN = (1.218, 0.353, 0.005)
DOOR_SPAWN_ROT = (0.0, 0.0, 0.0, 1.0)  # yaw 180, (w, x, y, z)


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
            # 15000 x 0.015 = 225 N, tik ispod referentne granice iz klasicnog.
            motion_stiffness_limits_task=(200.0, 15000.0),
            # 7-DOF ruka je redundantna: bez ovoga lakat pluta i pri kontaktu
            # moze odlutati u singularitet ili zglobni limit dok TCP mirno
            # stoji. Izgleda kao nestabilna politika, a nije.
            nullspace_control="position",
            nullspace_stiffness=10.0,
            nullspace_damping_ratio=1.0,
        ),
        # 0.015 m po koraku. Politika uzorkuje iz Gaussa sa std 0.5, dakle
        # akcija ide do ~±1.5, pa je stvarni maksimalni skok reference ~2.2 cm
        # na 30 Hz. Pri 0.05 je bio 15 cm po koraku i eksploracija je trgala
        # hvat sa sipke prije nego je politika stigla nesto nauciti - jedina
        # strategija koja prezivi takvu eksploraciju je nulta akcija.
        position_scale=0.015,
        orientation_scale=0.1,
        stiffness_scale=300.0,
        damping_ratio_scale=1.0,
        nullspace_joint_pos_target="default",
    )

    base = BaseAdmittanceActionCfg(debug=False)


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
        base_velocity = ObsTerm(
            func=mdp.base_velocity,
            params={"asset_cfg": SceneEntityCfg("robot")},
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
    # Normirano na PRAG, ne na puni hod. Uz normiranje na puni hod (0.8 m)
    # centimetar pomaka nosi 0.125 po koraku, sto je premalo da nadjaca
    # kazne - politika je ucila drzati kvaku i ne dirati vrata, a bonus na
    # pragu bio je prerijedak da povuce eksploraciju. Ova jedna promjena
    # digla je progress s 0.006 na 24.
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
    joint_saturation = RewTerm(
        func=mdp.joint_saturation_penalty,
        weight=-2.0,
        params={"comfort_band": 0.35, "robot_cfg": SceneEntityCfg("robot")},
    )
    excess_force = RewTerm(
        func=mdp.total_force_penalty,
        weight=-0.05,
        params={
            "reference_force": FORCE_REFERENCE_N,
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    # Baza je siroka 1.08 m, pa 0.5 m od osi krila drzi platformu izvan
    # njega, a kvaka je na 0.72 m od sarke i ostaje dohvatljiva.
    base_intrusion = RewTerm(
        func=mdp.base_intrusion_penalty,
        weight=-20.0,
        params={"min_distance": 0.5, "robot_cfg": SceneEntityCfg("robot")},
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
    base_hit = DoneTerm(
        func=mdp.base_hit_leaf,
        params={"min_distance": 0.3, "robot_cfg": SceneEntityCfg("robot")},
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
            #
            # MORA biti uze od heading_tolerance u BaseAdmittanceActionCfg,
            # inace admitancijski zakon odmah ponisti nasumicni kut prilaza
            # i randomizacija nema ucinka.
            "delta_range": (-0.35, 0.35),
            "position_noise": 0.005,
            "handle_local": HANDLE_LOCAL_SLIDING,
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    # TODO (§4): trenje prstiju. Vise nije sporedno otkad je kvaka zavarena -
    # sav moment sarke ide kroz trenje prstiju na poluzi.


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
    USD-u, pozi, konfiguraciji hvata, rasponima otpora i pragu - nijedna
    funkcija se ne mijenja."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.door = REVOLUTE_DOOR_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Door",
            init_state=REVOLUTE_DOOR_CFG.init_state.replace(
                pos=REVOLUTE_DOOR_SPAWN, rot=DOOR_SPAWN_ROT
            ),
        )
        self.terminations.grasp_lost.params["handle_local"] = HANDLE_LOCAL_REVOLUTE
        self.terminations.grasp_lost.params["max_distance"] = (
            GRASP_LOST_DISTANCE_REVOLUTE_M
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
        self.actions.base.heading_joint_nominal = NOMINAL_GRASP_JOINT_POS_REVOLUTE[0]
        # Zakretna vrata tjeraju TCP po luku i mijenjaju mu orijentaciju, pa se
        # zglobovi trose brze nego kod kliznih - ondje je stroza kazna nuzna.
        # Klizna rade unutar udobnog raspona i ondje ista kazna zaustavlja
        # napredak na ~0.485 m.
        self.rewards.joint_saturation.weight = -2.0


@configclass
class DoorRevoluteLearnedBaseEnvCfg(DoorRevoluteEnvCfg):
    """Baza u prostoru akcije umjesto admitancijskog zakona.

    Akcija raste s 12 na 15: tri dodatna broja su (vx, vy, omega), dakle
    doslovno geometry_msgs/Twist. Politika time sama koordinira ruku i bazu,
    umjesto da baza reagira tek nakon sto ruci ponestane prostora.

    ZASTO USPOREDBA: s admitancijskim zakonom gibanje je sekvencijalno -
    ruka brzo okrene vrata koliko moze, pa baza sporo krene za njom. Ovdje
    se mjeri koliko se dobiva ako je koordinacija naucena. Sve ostalo je
    identicno referentnom runu.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.base = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["base_x_joint", "base_y_joint", "base_theta_joint"],
            # Uz izlaz politike reda ±3 to daje ~0.3 m/s i ~0.5 rad/s, isti
            # red velicine kao gornje granice admitancijskog zakona.
            scale={
                "base_x_joint": 0.1,
                "base_y_joint": 0.1,
                "base_theta_joint": 0.17,
            },
            use_default_offset=False,
        )


# --- Ablacija: fiksna impedancija ---------------------------------------
# Uz pose_rel je maksimalna sila ~ K * position_scale, pa jedna fiksna
# vrijednost NIJE postena usporedba: preniska ne moze razviti silu za
# zatvarac (25 Nm / 0.72 m = 35 N), previsoka se bori s ogranicenjem. Zato se
# pusta vise vrijednosti i izvjestava se NAJBOLJA - inace je usporedba
# namjestena u korist varijabilne impedancije.
#
# Uz position_scale = 0.015:
#    2000  ->  30 N max, na granici izvedivog
#    6000  ->  90 N max, srednje
#   15000  -> 225 N max, isti strop kao varijabilna (kruto upravljanje)
#
# MIJENJA SE RUCNO izmedju runova sweepa; posljednja vrijednost ostaje
# zapisana ovdje pa provjeri je prije nego pokrenes usporedbu.
FIXED_STIFFNESS_ABLATION = 15000.0


@configclass
class DoorRevoluteFixedEnvCfg(DoorRevoluteEnvCfg):
    """Kontrolna skupina: politika uci SAMO kamo pomicati referencu, ne i
    koliko biti kruta. Akcija pada s 12 na 6 brojeva; sve ostalo - nagrada,
    opazanje, randomizacija, broj iteracija - ostaje identicno, sto je i
    smisao ablacije."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm.controller_cfg.impedance_mode = "fixed"
        self.actions.arm.controller_cfg.motion_stiffness_task = FIXED_STIFFNESS_ABLATION


@configclass
class DoorRevoluteLearnedBaseFixedEnvCfg(DoorRevoluteLearnedBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.arm.controller_cfg.impedance_mode = "fixed"
        self.actions.arm.controller_cfg.motion_stiffness_task = FIXED_STIFFNESS_ABLATION


@configclass
class DoorSlidingFixedEnvCfg(DoorEnvCfg):
    """Kontrolna skupina za klizna vrata. Ista logika kao zakretna varijanta:
    politika uci samo kamo pomicati referencu, ne i koliko biti kruta."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm.controller_cfg.impedance_mode = "fixed"
        self.actions.arm.controller_cfg.motion_stiffness_task = FIXED_STIFFNESS_ABLATION


@configclass
class DoorSlidingLearnedBaseEnvCfg(DoorEnvCfg):
    """Klizna vrata s bazom u prostoru akcije.

    Zrcalna klasa DoorRevoluteLearnedBaseEnvCfg, samo nasljeduje bazicnu
    (kliznu) konfiguraciju. Skale su iste, pa su dva tipa vrata usporediva.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.base = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["base_x_joint", "base_y_joint", "base_theta_joint"],
            # Uz izlaz politike reda ±3 to daje ~0.3 m/s i ~0.5 rad/s, isti
            # red velicine kao gornje granice admitancijskog zakona.
            scale={
                "base_x_joint": 0.1,
                "base_y_joint": 0.1,
                "base_theta_joint": 0.17,
            },
            use_default_offset=False,
        )


@configclass
class DoorSlidingLearnedBaseFixedEnvCfg(DoorSlidingLearnedBaseEnvCfg):
    """Kontrolna skupina za klizna vrata s naucenom bazom."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm.controller_cfg.impedance_mode = "fixed"
        self.actions.arm.controller_cfg.motion_stiffness_task = FIXED_STIFFNESS_ABLATION
