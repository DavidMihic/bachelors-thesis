"""robot_cfg.py - Isaac Lab konfiguracija KMR iiwa robota za RL trening.

Parno uz door_cfg.py: USD nosi topologiju, gains zive ovdje.

CETIRI VAZNE RAZLIKE OD ONOGA STO JE U USD-U:

1. Ruka ide s pogonom na NULI. OSC racuna momente i pise ih kao effort
   target; da implicitni aktuator zadrzi krutost 100000 iz konverzije, PD
   pogon i OSC bi se tukli i krutost koju politika naredi ne bi imala
   ucinka. Sva krutost dolazi iz OSC-a - to je cijela poanta varijabilne
   impedancije.

2. Limiti momenta su stvarni iiwa7 (176/176/110/110/110/40/40 Nm), a ne
   300 Nm na svih sedam kako je ispalo iz konverzije. Na zapescu je 300 Nm
   7.5x previse autoriteta; politika bi ga naucila koristiti i pri
   deploymentu ne bi radila.

3. Korijen artikulacije je FIKSAN i to mora ostati. OSC racuna jakobijan i
   inercijsku matricu uz pretpostavku fiksnog korijena; s plutajucim
   korijenom kompenzacija gravitacije ispadne kriva i ruka propada -
   provjereno empirijski.

4. Baza se ipak giba, kroz tri fiktivna zgloba iz add_base_joints.py
   (world -> base_x -> base_y -> base_theta -> base_link). Korijen je sad
   link 'world' i fiksan je u ishodistu env-a; base_link je obicno TIJELO.
   Sve sto racuna u okviru baze mora zato citati pozu tijela base_link, a NE
   root_pos_w/root_quat_w - to je najlaksa greska u ovom setupu.

Baza je pozicijski upravljana: admitancijski zakon integrira svoju brzinu u
ciljnu poziciju zglobova. Velocity-only pogon bi pod reakcijom ruke puzao,
jer sila trenja vrata na bazu nema protutezu osim prigusenja.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ASSETS_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "assets"
    )
)

# Link koji predstavlja platformu. NIJE korijen artikulacije - korijen je
# 'world'. Svako racunanje u okviru baze ide preko ovog tijela.
BASE_BODY = "base_link"

# Fiktivni zglobovi iz add_base_joints.py.
BASE_JOINTS = ("base_x_joint", "base_y_joint", "base_theta_joint")

# URDF: "+q = zatvaranje prema centru za sva 4 prsta". Nula je OTVORENO.
# 0.025 a ne 0.026 jer je gripper_tcp definiran bas pri q=0.025.
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.025

# Stvarni limiti momenta KUKA LBR iiwa 7 R800, po zglobu (Nm).
IIWA7_EFFORT_LIMITS = {
    "iiwa_joint_1": 176.0,
    "iiwa_joint_2": 176.0,
    "iiwa_joint_3": 110.0,
    "iiwa_joint_4": 110.0,
    "iiwa_joint_5": 110.0,
    "iiwa_joint_6": 40.0,
    "iiwa_joint_7": 40.0,
}

# Konfiguracija ruke pri spawnu. Stvarnu pozu hvata postavlja
# reset_grasp_and_door; ovo je samo da robot ne krene iz potpuno
# ispruzenog, singularnog stanja.
DEFAULT_ARM_JOINT_POS = {
    "iiwa_joint_1": 0.0,
    "iiwa_joint_2": 0.4,
    "iiwa_joint_3": 0.0,
    "iiwa_joint_4": -1.6,
    "iiwa_joint_5": 0.0,
    "iiwa_joint_6": 1.2,
    "iiwa_joint_7": 0.0,
}

KMR_IIWA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSETS_DIR, "kmr_iiwa_full_rl.usd"),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            # NE mijenjaj u False - vidi tocku 3 u docstringu. Baza se giba
            # kroz fiktivne zglobove, ne kroz plutajuci korijen.
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            **DEFAULT_ARM_JOINT_POS,
            "gripper_finger_[1-4]_joint": GRIPPER_OPEN,
            "base_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["iiwa_joint_[1-7]"],
            # Nule su namjerne - vidi docstring, tocka 1.
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=IIWA7_EFFORT_LIMITS,
            # Mala armatura stabilizira solver kod kontakta i kosta prakticki
            # nista. Ako se pojavi jitter pri zatvorenom hvatu, ovo je prva
            # brojka koju treba dignuti, prije solver iteracija.
            armature=0.01,
        ),
        "base": ImplicitActuatorCfg(
            joint_names_expr=["base_.*_joint"],
            # Brzinski pogon: stiffness 0, sila dolazi iz prigusenja koje
            # prati zadanu brzinu. Pod reakcijom ruke baza time lagano puze -
            # pri 100 N i prigusenju 1e4 to je 0.01 m/s, zanemarivo naspram
            # zadanih 0.3 m/s. Pozicijski pogon bi bio krući, ali twist je
            # sucelje kojim se KMR stvarno vozi (/cmd_vel).
            stiffness=0.0,
            damping={
                "base_x_joint": 1.0e4,
                "base_y_joint": 1.0e4,
                "base_theta_joint": 1.0e4,
            },
            effort_limit_sim=5000.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_finger_[1-4]_joint"],
            # Ovo NIJE na nuli: prsti su pozicijski upravljani i drze hvat
            # cijelu epizodu. 20000 N/m je izmjereno, ne pogodak - pri
            # vucenju silu nose efektivno samo dva prsta, pa je efektivna
            # krutost hvata pola nominalne.
            stiffness=20000.0,
            damping=200.0,
            effort_limit_sim=600.0,
        ),
    },
)
