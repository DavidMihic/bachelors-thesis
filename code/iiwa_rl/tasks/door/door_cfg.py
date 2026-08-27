"""door_cfg.py - Isaac Lab konfiguracija vrata za RL trening.

ARHITEKTURNA NAMJERA: USD nosi SAMO topologiju i geometriju. Otpor vrata
zivi ovdje i randomizira se po env-u. Zato su oba *_rl.usd assetа
konvertirana s pogonom na nuli - nikad ne podesavaj vrata rekonverzijom,
jer se tako trenirana politika veze uz konkretan fajl umjesto uz raspon.

Otpor je razdvojen na tri fizikalno razlicita clana, koja su u izvornom
assetu bila spojena u jedan pogon:

    otpor = friction                 Coulomb, konstantan
          + damping * q_dot          viskozno
          + stiffness * q            zatvarac, ogranicen effort_limitom

Jedinice su APSOLUTNE (N odnosno Nm), ne bezdimenzijski koeficijenti -
provjereno mjerenjem (measure_door_friction.py): breakaway se poklapa sa
zadanim trenjem uz omjer 1.00-1.03 kroz raspon 10-60.

Rasponi su kalibrirani prema stvarnim vratima, NE prema ~280 N izmjerenih
klasicnim pristupom - ta je brojka tranzijent akceleracije baze kroz krutu
vezu ruka<->vrata, ne otpor vrata (klizna vrata u staroj sceni imala su ~19 N
otpora pri 0.17 m). 280 N ostaje samo gornja referentna granica za kaznu na
silu u nagradi.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

# Skripte se pokrecu iz ~/IsaacLab, pa relativne putanje pokazuju u prazno.
ASSETS_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "assets"
    )
)

# Kanonska imena, identicna u oba assetа - env kod nikad ne grana po tipu vrata.
DOOR_DOF_JOINT = "door_dof_joint"
DOOR_LEAF_BODY = "door_leaf"

# Tocka hvata u lokalnom okviru vrata (prije rotacije), iz URDF-a:
#   klizna:   handle_fixed na (0.02, 0.65, 1.0), sipka na lokalnom x=0.07
#   zakretna: handle_fixed na (0.02, 0.72, 1.0), poluga na lokalnom (0.04, -0.08)
HANDLE_LOCAL_SLIDING = (0.09, 0.65, 1.0)
HANDLE_LOCAL_REVOLUTE = (0.06, 0.64, 1.0)

# Hod do kraja, iz URDF limita. Prag uspjeha NIJE ovo - ogranicen je dosegom
# ruke pri fiksnoj bazi, vidi door_env_cfg.py.
SLIDING_FULL_TRAVEL_M = 0.8
REVOLUTE_FULL_TRAVEL_RAD = 1.57


@dataclass
class DoorResistanceRanges:
    """Rasponi domain randomizacije otpora, po tipu vrata.

    Jedinice: prizmaticni zglob N/m, N*s/m, N. Rotacijski Nm/rad,
    Nm*s/rad, Nm.
    """

    stiffness: tuple[float, float]
    damping: tuple[float, float]
    friction: tuple[float, float]
    effort_limit: tuple[float, float]


# Klizna vrata nemaju povratnu oprugu - stiffness je fiksno 0. Otpor je
# kotrljanje kolica: lagana unutarnja vrata ~10 N, teska ~60 N.
SLIDING_RESISTANCE = DoorResistanceRanges(
    stiffness=(0.0, 0.0),
    damping=(5.0, 50.0),
    friction=(5.0, 60.0),
    effort_limit=(200.0, 200.0),
)

# Zakretna vrata bez zatvaraca: samo trenje sarke.
REVOLUTE_FREE_RESISTANCE = DoorResistanceRanges(
    stiffness=(0.0, 0.0),
    damping=(0.5, 5.0),
    friction=(0.5, 5.0),
    effort_limit=(200.0, 200.0),
)

# Zakretna vrata sa zatvaracem. Nizak effort_limit je namjeran: EN 3
# zatvarac daje priblizno KONSTANTAN moment, a linearna opruga ogranicena
# na 10-25 Nm ga modelira puno bolje nego cista opruga. Za usporedbu, stara
# scena je imala 100 Nm/rad BEZ smislenog capa -> 218 N na kvaci pri 90
# stupnjeva, sto nisu vrata nego opruga.
REVOLUTE_CLOSER_RESISTANCE = DoorResistanceRanges(
    stiffness=(5.0, 20.0),
    damping=(0.5, 5.0),
    friction=(0.5, 5.0),
    effort_limit=(10.0, 25.0),
)

# Vjerojatnost da zakretna vrata pri resetu dobiju zatvarac.
REVOLUTE_CLOSER_PROBABILITY = 0.5


@configclass
class DoorArticulationCfg(ArticulationCfg):
    """ArticulationCfg vrata s pogonom na nuli.

    Sve stvarne vrijednosti pisu se pri resetu iz raspona gore. Nule ovdje
    nisu previd nego invarijanta: ako env slucajno preskoci randomizaciju,
    vrata ce biti bez otpora i to se vidi odmah u nagradi, umjesto da tiho
    naslijede ono sto je zateceno u USD-u.
    """


def door_articulation_cfg(
    usd_name: str,
    prim_path: str = "{ENV_REGEX_NS}/Door",
    solver_position_iterations: int = 16,
    solver_velocity_iterations: int = 1,
) -> DoorArticulationCfg:
    """Vrata kao ArticulationCfg.

    solver_position_iterations: konverzija je stavila 32, sto je
    predimenzionirano za artikulaciju s jednim DOF-om i skalira linearno s
    num_envs. 16 je polazna vrijednost - izmjeri FPS na 8 / 16 / 32 s
    zatvorenim hvatom PRIJE nego skaliras broj env-ova.
    """
    return DoorArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, usd_name),
            activate_contact_sensors=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=solver_position_iterations,
                solver_velocity_iteration_count=solver_velocity_iterations,
                fix_root_link=True,
            ),
        ),
        init_state=DoorArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),  # env postavlja stvarnu pozu vrata
            joint_pos={DOOR_DOF_JOINT: 0.0},
            joint_vel={DOOR_DOF_JOINT: 0.0},
        ),
        actuators={
            "door_dof": ImplicitActuatorCfg(
                joint_names_expr=[DOOR_DOF_JOINT],
                stiffness=0.0,
                damping=0.0,
                # Za implicitni aktuator effort_limit_sim je ono sto zavrsi na
                # PhysX drive maxForce; effort_limit je clipping u aktuatorskom
                # modelu i ovdje ne bi imao ucinka.
                effort_limit_sim=200.0,
            ),
        },
    )


SLIDING_DOOR_CFG = door_articulation_cfg("sliding_door_rl.usd")
REVOLUTE_DOOR_CFG = door_articulation_cfg("revolute_door_rl.usd")
