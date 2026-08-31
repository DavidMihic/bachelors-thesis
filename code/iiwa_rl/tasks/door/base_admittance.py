"""base_admittance.py - baza prati ruku, ali nije u prostoru akcije.

OPSEG (§0): RL uci interakciju, ne navigaciju. Baza se zato vodi rucnim
admitancijskim zakonom umjesto da ude u akciju politike. Prostor akcije
ostaje nepromijenjen pa su sve dosadasnje politike i brojke usporedive,
ponasanje je predvidljivo, a izlaz zakona je (vx, vy, omega) - doslovno
geometry_msgs/Twist na /cmd_vel, isto sucelje kojim se KMR vec vozi.

Cijena koju treba priznati u radu: koordinacija ruke i baze NIJE naucena.

ZAKON, dvije razdvojene petlje:

  translacija - vodoravna udaljenost TCP-a od baze ruke drzi se u mrtvoj
                zoni oko nominalnog hvata (~0.80 m, izmjereno)

  rotacija    - greska kursa se NE mjeri smjerom prema TCP-u nego kutom
                zgloba iiwa_joint_1. Zakret baze prema TCP-u je losa petlja:
                rotacija baze zakrece i ruku, pa se smjer prema TCP-u mijenja
                gotovo jednako kao kurs baze i greska se ne smanjuje onoliko
                koliko zakon ocekuje - rezultat je titranje. joint_1 pak
                doslovno mjeri koliko ruka gleda ustranu u odnosu na bazu, a
                dok hvat drzi kvaku zakret baze za Δ mijenja joint_1 tocno za
                -Δ. To je cista petlja prvog reda bez sprege.

KAKO SE NAREDBA PREDAJE: brzina se integrira u ciljnu poziciju trojke
fiktivnih zglobova (base_x/base_y/base_theta iz add_base_joints.py), pa se
posalje kao position target. Ranije se umjesto toga upisivala poza korijena
preko write_root_pose_to_sim - to je teleportacija koja resetira unutarnje
stanje solvera dok zglobovi zadrze kutove, pa se ruka svaki korak nalazila u
stanju koje solver nije ocekivao i pocela bi propadati.

Cisti velocity target bi bio blizi cmd_vel semantici, ali bi baza pod
reakcijom ruke puzala: sila kojom vrata vuku bazu nema protutezu osim
prigusenja. Integrirana pozicija drzi mjesto, sto stvarni pogon takoder radi.

BRZINA JE NAMJERNO NISKA (0.3 m/s naspram 3.6 km/h iz specifikacije). Baza
koja juri unosi tranzijente kroz krutu vezu ruka<->vrata, a upravo su takvi
tranzijenti u klasicnom pristupu davali ~280 N koji nisu bili otpor vrata.
Uz to je poza TCP-a u okviru baze dio opazanja politike, pa brza baza znaci
da politika vidi pomake koje nije uzrokovala.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .robot_cfg import BASE_BODY, BASE_JOINTS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def quat_apply_batch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q (N,4) primijenjen na v (3,) ili (N,3)."""
    if v.dim() == 1:
        v = v.unsqueeze(0).expand(q.shape[0], 3)
    qw, qv = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


class BaseAdmittanceAction(ActionTerm):
    """Nulti prostor akcije - term postoji samo zbog hooka po koraku.

    apply_actions() se zove svaki fizikalni korak, sto je tocno mjesto na
    kojem zakon treba raditi. Event term s mode="reset" radi prerijetko, a
    "interval" ima vlastito vrijeme i nije vezan uz korak.
    """

    cfg: "BaseAdmittanceActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "BaseAdmittanceActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._raw = torch.zeros(self.num_envs, 0, device=self.device)
        self._arm_base_b = torch.tensor(cfg.arm_base_b, device=self.device)

        self._tcp_idx = self._asset.find_bodies(cfg.tcp_body)[0][0]
        self._base_idx = self._asset.find_bodies(BASE_BODY)[0][0]
        self._q1_idx = self._asset.find_joints(cfg.heading_joint)[0][0]
        self._arm_ids = self._asset.find_joints(cfg.arm_joint_pattern)[0]
        self._base_joint_ids = [
            self._asset.find_joints(name)[0][0] for name in BASE_JOINTS
        ]

        # Ciljna pozicija trojke [x, y, theta]. Integrira se iz brzine.
        self._target = torch.zeros(self.num_envs, 3, device=self.device)

        # Histereza: jednom pokrenuto zakretanje ide dok greska ne padne na
        # polovicu tolerancije. Bez toga zakon titra oko ruba mrtve zone.
        self._turning = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        # Politika ovom termu nista ne salje.
        pass

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # reset_grasp_and_door vraca fiktivne zglobove na nulu, pa cilj mora
        # krenuti odande - inace bi baza odmah odjurila na staru vrijednost.
        if env_ids is None:
            self._target[:] = 0.0
            self._turning[:] = False
        else:
            self._target[env_ids] = 0.0
            self._turning[env_ids] = False

    def apply_actions(self) -> None:
        asset = self._asset
        cfg = self.cfg
        dt = self._env.physics_dt

        # Okvir baze je TIJELO base_link, ne korijen artikulacije - korijen je
        # link 'world' i fiksan je u ishodistu env-a.
        base_pos = asset.data.body_pos_w[:, self._base_idx]
        base_quat = asset.data.body_quat_w[:, self._base_idx]

        # --- translacija: udaljenost TCP-a od baze ruke u mrtvu zonu ---
        arm_base = base_pos + quat_apply_batch(base_quat, self._arm_base_b)
        tcp = asset.data.body_pos_w[:, self._tcp_idx]
        delta = (tcp - arm_base)[:, :2]
        distance = delta.norm(dim=-1, keepdim=True)
        direction = delta / (distance + 1e-6)

        # ZGLOBNA ISCRPLJENOST: udaljenost TCP-a od baze ruke ne hvata sve
        # nacine na koje ruci ponestane prostora. U treningu su zakretna vrata
        # stajala na 0.29 rad uz joint_3 na 96% raspona, dok je TCP bio uredno
        # unutar mrtve zone i baza legitimno mirovala. Zato zasicenje BILO
        # KOJEG zgloba stisne gornju granicu zone: baza se primakne, ruka se
        # skupi i zglobovi se vrate prema sredini. Smjer je vec izracunat
        # (prema TCP-u), pa nije potrebna zasebna petlja.
        q = asset.data.joint_pos[:, self._arm_ids]
        lower = asset.data.soft_joint_pos_limits[:, self._arm_ids, 0]
        upper = asset.data.soft_joint_pos_limits[:, self._arm_ids, 1]
        fraction = (q - lower) / (upper - lower + 1e-6)
        saturation = ((fraction - 0.5).abs() - cfg.joint_comfort_band).clamp(min=0.0)
        saturation = saturation.max(dim=-1, keepdim=True).values

        reach_max = (cfg.reach_max - cfg.joint_reach_gain * saturation).clamp(
            min=cfg.reach_min + 0.02
        )

        too_far = (distance - reach_max).clamp(min=0.0)
        too_close = (cfg.reach_min - distance).clamp(min=0.0)
        speed = (cfg.gain_linear * (too_far - too_close)).clamp(
            -cfg.max_linear_speed, cfg.max_linear_speed
        )
        vel_xy = speed * direction

        # --- rotacija: vrati iiwa_joint_1 prema nominalnoj vrijednosti ---
        # Zakret baze za +Δ mijenja joint_1 za -Δ dok hvat drzi kvaku, pa je
        # predznak pojacanja pozitivan.
        q1_error = asset.data.joint_pos[:, self._q1_idx] - cfg.heading_joint_nominal
        magnitude = q1_error.abs()
        self._turning = torch.where(
            magnitude > cfg.heading_tolerance,
            torch.ones_like(self._turning),
            torch.where(
                magnitude < 0.5 * cfg.heading_tolerance,
                torch.zeros_like(self._turning),
                self._turning,
            ),
        )
        omega = torch.where(
            self._turning,
            (cfg.gain_angular * q1_error).clamp(
                -cfg.max_angular_speed, cfg.max_angular_speed
            ),
            torch.zeros_like(q1_error),
        )

        # --- integracija u ciljnu poziciju zglobova ---
        # vel_xy je u SVIJETU, a base_x/base_y su osi svijeta (world je
        # korijen i ne rotira), pa transformacija nije potrebna.
        self._target[:, 0] += vel_xy[:, 0] * dt
        self._target[:, 1] += vel_xy[:, 1] * dt
        self._target[:, 2] += omega * dt

        asset.set_joint_position_target(self._target, joint_ids=self._base_joint_ids)

        if cfg.debug and self._env.common_step_counter % 120 == 0:
            print(
                f"[baza] d={distance[0].item():.3f} v={speed[0].item():+.3f}"
                f" q1={asset.data.joint_pos[0, self._q1_idx].item():+.3f}"
                f" omega={omega[0].item():+.3f}"
                f" cilj=({self._target[0, 0].item():+.3f},"
                f"{self._target[0, 1].item():+.3f},{self._target[0, 2].item():+.3f})"
            )


@configclass
class BaseAdmittanceActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = BaseAdmittanceAction
    asset_name: str = "robot"

    tcp_body: str = "gripper_tcp"
    arm_base_b: tuple[float, float, float] = (0.363, -0.184, 0.7)

    # Mrtva zona vodoravne udaljenosti TCP-a od baze ruke. Nominalni hvat je
    # na ~0.80 m (izmjereno), pa zona mora biti oko te vrijednosti - inace
    # baza vec pri resetu starta izvan nje i titra oko ruba.
    reach_min: float = 0.60
    reach_max: float = 0.78

    # Kurs se mjeri kutom ovog zgloba, ne smjerom prema TCP-u (vidi docstring).
    heading_joint: str = "iiwa_joint_1"
    # Nominalna vrijednost iz konfiguracije hvata. MORA odgovarati prvom
    # elementu NOMINAL_GRASP_JOINT_POS za pripadni tip vrata.
    heading_joint_nominal: float = -0.4071
    # MORA biti sire od delta_range u reset_grasp_and_door (±0.35), inace
    # zakon odmah ponisti nasumicni kut prilaza koji je ondje namjerno uveden.
    heading_tolerance: float = 0.6

    gain_linear: float = 0.3
    gain_angular: float = 1.0
    max_linear_speed: float = 0.05
    max_angular_speed: float = 0.5

    debug: bool = False

    # Zglob se smatra udobnim unutar ±band oko sredine raspona; 0.35 znaci
    # 15%-85%. Iznad toga se mrtva zona stisce proporcionalno prekoracenju.
    arm_joint_pattern: str = "iiwa_joint_[1-7]"
    joint_comfort_band: float = 0.35
    joint_reach_gain: float = 1.5
