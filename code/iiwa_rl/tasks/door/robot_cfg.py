"""robot_cfg.py - Isaac Lab konfiguracija KMR iiwa robota za RL trening.

Parno uz door_cfg.py: USD nosi topologiju, gains zive ovdje.

DVIJE VAZNE RAZLIKE OD ONOGA STO JE U USD-U:

1. Ruka ide s pogonom na NULI. OSC racuna momente i pise ih kao effort
   target; da implicitni aktuator zadrzi krutost 100000 iz konverzije, PD
   pogon i OSC bi se tukli i krutost koju politika naredi ne bi imala
   ucinka. Sva krutost dolazi iz OSC-a - to je cijela poanta varijabilne
   impedancije.

2. Limiti momenta su stvarni iiwa7 (176/176/110/110/110/40/40 Nm), a ne
   300 Nm na svih sedam kako je ispalo iz konverzije. Na zapescu je 300 Nm
   7.5x previse autoriteta; politika bi ga naucila koristiti i pri
   deploymentu ne bi radila.

Baza je fiksirana ovdje (fix_root_link), ne u assetu - asset ostaje
upotrebljiv i za scenarij s pokretnom bazom ako se opseg kasnije prosiri.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

import os

ASSETS_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "assets"
    )
)

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

# Konfiguracija ruke pri spawnu. Stvarnu pozu hvata postavlja reset_to_grasp;
# ovo je samo da robot ne krene iz potpuno ispruzenog, singularnog stanja.
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
        usd_path=os.path.join(ASSETS_DIR, "kmr_iiwa_full.usd"),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            # Baza je fiksna (vidi docstring env cfg-a). Mijenja se ovdje ako
            # se scenarij prosiri na pokretnu bazu.
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            **DEFAULT_ARM_JOINT_POS,
            "gripper_finger_[1-4]_joint": GRIPPER_OPEN,
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
