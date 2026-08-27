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

BAZA SE VODI KINEMATICKI - svaki korak joj se upisuju poza I brzina.
Opravdanje: baza je 390 kg naspram ruke od 24 kg, a stvarni pogon ionako
prati cmd_vel vlastitim regulatorom umjesto da slobodno reagira na sile.
Brzina se mora upisati zajedno s pozom: bez nje gravitacija akumulira
brzinu izmedju upisa poze, baza stalno "pada" dok joj se pozicija ispravlja,
i to samo po sebi proizvodi drhtanje. Iz istog razloga se Z i nagib drze
fiksnima umjesto da se preuzimaju iz trenutnog stanja.

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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def quat_apply_batch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q (N,4) primijenjen na v (3,) ili (N,3)."""
    if v.dim() == 1:
        v = v.unsqueeze(0).expand(q.shape[0], 3)
    qw, qv = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def yaw_of(q: torch.Tensor) -> torch.Tensor:
    """Kut zakreta oko vertikale iz kvaterniona (w, x, y, z)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_quat(angle: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angle
    zero = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zero, zero, torch.sin(half)], dim=-1)


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

        tcp_ids, _ = self._asset.find_bodies(cfg.tcp_body)
        self._tcp_idx = tcp_ids[0]
        joint_ids, _ = self._asset.find_joints(cfg.heading_joint)
        self._q1_idx = joint_ids[0]

        # Visina baze se drzi fiksnom umjesto da se preuzima iz stanja -
        # inace ju gravitacija polako spusta izmedju upisa.
        self._base_z = self._asset.data.default_root_state[:, 2].clone()

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
        if env_ids is None:
            self._turning[:] = False
        else:
            self._turning[env_ids] = False

    def apply_actions(self) -> None:
        asset = self._asset
        cfg = self.cfg
        dt = self._env.physics_dt

        root_pos = asset.data.root_pos_w
        root_quat = asset.data.root_quat_w
        heading = yaw_of(root_quat)

        # --- translacija: udaljenost TCP-a od baze ruke u mrtvu zonu ---
        arm_base = root_pos + quat_apply_batch(root_quat, self._arm_base_b)
        tcp = asset.data.body_pos_w[:, self._tcp_idx]
        delta = (tcp - arm_base)[:, :2]
        distance = delta.norm(dim=-1, keepdim=True)
        direction = delta / (distance + 1e-6)

        too_far = (distance - cfg.reach_max).clamp(min=0.0)
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

        # --- upis poze i brzine ---
        new_pos = torch.zeros_like(root_pos)
        new_pos[:, :2] = root_pos[:, :2] + vel_xy * dt
        new_pos[:, 2] = self._base_z
        new_quat = yaw_quat(heading + omega * dt)
        # Poza se upisuje UVIJEK. Uvjetni upis znaci da baza u mirovanju
        # postane slobodno plutajuce tijelo (fix_root_link=False) i padne
        # pod gravitacijom, povlaceci ruku za sobom. Kad zakon ne trazi
        # gibanje, upisuje se ista poza - Z i nagib se pritom drze fiksnima
        # pa se gravitacija ne akumulira.
        asset.write_root_pose_to_sim(torch.cat([new_pos, new_quat], dim=-1))
        root_vel = torch.zeros_like(asset.data.root_vel_w)
        root_vel[:, 0] = vel_xy[:, 0]
        root_vel[:, 1] = vel_xy[:, 1]
        root_vel[:, 5] = omega
        asset.write_root_velocity_to_sim(root_vel)

        if cfg.debug and self._env.common_step_counter % 120 == 0:
            print(
                f"[baza] d={distance[0].item():.3f} v={speed[0].item():+.3f}"
                f" q1={asset.data.joint_pos[0, self._q1_idx].item():+.3f}"
                f" omega={omega[0].item():+.3f} turning={bool(self._turning[0])}"
            )
            print("base_z:", self._base_z[0].item(), " root_z:", root_pos[0, 2].item())


@configclass
class BaseAdmittanceActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = BaseAdmittanceAction
    asset_name: str = "robot"

    tcp_body: str = "gripper_tcp"
    arm_base_b: tuple[float, float, float] = (0.363, -0.184, 0.7)

    # Mrtva zona vodoravne udaljenosti TCP-a od baze ruke. Nominalni hvat je
    # na ~0.80 m (izmjereno), pa zona mora biti oko te vrijednosti - inace
    # baza vec pri resetu starta izvan nje i titra oko ruba.
    reach_min: float = 0.68
    reach_max: float = 0.82

    # Kurs se mjeri kutom ovog zgloba, ne smjerom prema TCP-u (vidi docstring).
    heading_joint: str = "iiwa_joint_1"
    # Nominalna vrijednost iz konfiguracije hvata. MORA odgovarati prvom
    # elementu NOMINAL_GRASP_JOINT_POS za pripadni tip vrata.
    heading_joint_nominal: float = -0.4071
    heading_tolerance: float = 0.35  # ~20 stupnjeva

    gain_linear: float = 0.8
    gain_angular: float = 1.0
    max_linear_speed: float = 0.3
    max_angular_speed: float = 0.5

    debug: bool = False
