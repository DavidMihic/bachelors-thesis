"""Prolazak kroz vrata nakon sto ih RL politika otvori.

Politika rjesava samo interakciju s vratima. Prilaz kvaci nije u opsegu, jer
epizoda pocinje s hvatom na kvaki, a ni prolazak nije: nagrada za napredak je
linearna u stanju vrata sve do limita, pa politika nema razlog pustiti kvaku.

Modul preuzima kad su vrata dovoljno otvorena i vodi robota kroz otvor
klasicnim regulatorom. Faze:

  RL       politika upravlja, modul samo prati stanje vrata
  SETTLE   ruka se zakljuca dok hvat jos drzi, pa vrata zaustavi hvat
  RELEASE  prsti se otvore, ceka se da popusti kontakt sa sipkom
  RETREAT  TCP se povlaci po prilaznoj osi sa sipke
  BACKUP   baza se povlaci i poravnava sa sredinom otvora, ruka se sklapa
  PARK     ceka se da ruka stvarno dodje u parkirnu pozu
  DRIVE    baza vozi kroz otvor
  DONE     stoji

Sredina otvora slijedi iz poze vrata, koja je u simulatoru egzaktna, a na
stvarnom robotu je daje lidar.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply

# Parkirna poza u prostoru zglobova (iiwa_joint_1..7), ista kao u klasicnom
# pristupu: ruka se sklapa iznad i iza nosaca.
PARK_JOINT_POS = (0.0, -0.6, 0.0, -2.2, 0.0, 0.8, 0.0)

# Krutost zglobne brave. Ruka se drzi PD-om u prostoru zglobova, a ne preko
# OSC-a, jer uz pose_rel naredba nije polozajna brava nego stalna sila
# K*delta: granicna brzina je delta*sqrt(K)/(2*sqrt(m)), sto je manje od
# brzine baze, pa ruka trajno zaostaje i TCP prividno stoji u svijetu.
#
# Krutost se dize rampom jer je prijelaz s momentnog na polozajno upravljanje
# skokovit - zglobovi su do tada slobodni i odjednom dobiju PD koji ih vuce u
# parkirnu pozu. Meki pocetak to ublazi, tvrdi kraj drzi ruku cvrstom dok baza
# vozi kroz otvor.
ARM_JOINT_STIFFNESS_SOFT = 100.0
ARM_JOINT_STIFFNESS_FIRM = 3000.0
ARM_LOCK_RAMP_S = 4.0

# Prigusenje prati krutost kroz rampu; inace bi pri punoj krutosti ruka bila
# podprigusena i titrala.
ARM_DAMPING_RATIO = 0.05

# Krutost OSC-a po fazi (akcija, mnozi se sa stiffness_scale).
#
# RETREAT: ruku ondje vodi OSC, a pri donjoj granici od 200 N/m postize tek
# ~0.08 m/s, pa odmak traje predugo bez obzira sto se naredi. Baza u toj fazi
# stoji, pa veca krutost ne stvara probleme s pracenjem.
#
# Ostale faze: minimum, da OSC ne radi protiv zglobnog PD-a. Gravitacijska
# kompenzacija i dalje radi jer ne ovisi o zadanoj krutosti.
RETREAT_STIFFNESS_ACTION = 10.0
PARK_STIFFNESS_ACTION = 0.0

# Vodjenje TCP-a u RETREAT fazi: brzina reference razmjerna gresci, ogranicena
# odozgo. Cisto ogranicenje pomaka dalo bi bang-bang referencu koja preleti
# cilj i vraca se.
ARM_GAIN = 4.0  # 1/s
ARM_MAX_SPEED = 0.30  # m/s

# Prag pustanja kvake po tipu vrata, prepoznatom iz USD putanje.
#
# Zakretna, 1.745 rad (100 st.): krilo se zakrece od robota i ne ulazi u
# koridor, pa prag odreduje samo koliko je otvor cist. Pri 90 st. je krilo na
# y = 0, tek na rubu; pri 100 st. na y = -0.15.
#
# Klizna, 0.85 m: krilo klizi kroz otvor. Pri hodu d zauzima y od d do
# 0.85 + d, a baza sirine 0.63 m prolazi centrirano i zauzima y od 0.11 do
# 0.74. Pri d = 0.85 ostaje 11 cm zazora, pri d = 0.74 je dodir.
RELEASE_BY_DOOR_TYPE = (
    ("revolute", 1.745),
    ("sliding", 0.85),
)

# Terminacije su trenazni alat, a ne fizikalna ogranicenja: grasp_lost okine
# cim se TCP odmakne od kvake, a base_hit i wall_hit su podeseni za fazu
# otvaranja, gdje baza stoji ispred vrata. Pri prolasku kroz otvor od 0.87 m s
# bazom sirokom 0.63 m ostaje 0.12 m po strani, unutar njihovih tampona.
# Vrijednosti su takve da uvjet ne moze biti ispunjen.
RELAXED_TERMINATIONS = (
    ("grasp_lost", "max_distance", 1.0e6),
    ("overforce", "limit", 1.0e6),
    ("base_hit", "min_clearance", -1.0e6),
    ("wall_hit", "min_clearance", -1.0e6),
)


def _quat_inv(quat: torch.Tensor) -> torch.Tensor:
    """Inverz jedinicnog kvaterniona (w, x, y, z)."""
    return torch.cat([quat[:, :1], -quat[:, 1:]], dim=-1)


class DoorPassage:
    """Preuzima upravljanje kad su vrata dovoljno otvorena i provozi robota.

    Radi nad svim env-ovima odjednom, svaki sa svojom fazom, pa se moze
    pustiti vise env-ova i pratiti koji prolazi.
    """

    RL, SETTLE, RELEASE, RETREAT, BACKUP, PARK, DRIVE, DONE = range(8)
    PHASE_NAMES = (
        "RL",
        "SETTLE",
        "RELEASE",
        "RETREAT",
        "BACKUP",
        "PARK",
        "DRIVE",
        "DONE",
    )

    def __init__(
        self,
        env,
        release_threshold: float | None = None,
        settle_max_s: float = 2.0,
        settle_vel: float = 0.05,
        release_dwell_s: float = 1.0,
        retreat_distance: float = 0.15,
        backup_local_x: float = 0.90,
        exit_local_x: float = -1.40,
        max_speed: float = 0.30,
        max_lateral: float = 0.15,
        max_yaw_rate: float = 0.8,
        position_tolerance: float = 0.05,
        lateral_tolerance: float = 0.03,
        yaw_tolerance: float = 0.05,
        park_joint_tolerance: float = 0.15,
    ):
        """
        Args:
            release_threshold: Stanje DOF-a vrata pri kojem se pusta kvaka.
                None prepoznaje tip vrata iz USD putanje, vidi
                RELEASE_BY_DOOR_TYPE.
            settle_max_s: Najdulje cekanje da se vrata smire prije pustanja.
            settle_vel: Ispod ove brzine DOF-a vrata smatra se da su se vrata
                smirila. Bez smirivanja vrata po inerciji odu do limita i
                odbiju se natrag ispod praga.
            release_dwell_s: Cekanje nakon otvaranja prstiju. Prsti se otvaraju
                trenutno, ali kuke se sa sipke skidaju tek kad popusti kontakt.
            retreat_distance: Odmak TCP-a po prilaznoj osi. Kuke prstiju
                obuhvacaju sipku, pa samo otvaranje hvata ne oslobadja kvaku.
            backup_local_x: Udaljenost od ravnine vrata na koju se baza
                povlaci, u okviru vrata.
            exit_local_x: Dokle se vozi; negativno znaci na drugu stranu vrata.
            position_tolerance: Dopusteno odstupanje po uzduznoj osi prije
                prelaska iz BACKUP u PARK.
            lateral_tolerance: Dopusteno bocno odstupanje od sredine otvora.
                Zajedno s yaw_tolerance mora ostati znatno ispod 0.12 m, koliko
                je zazora po strani pri prolasku.
            yaw_tolerance: Dopusteno odstupanje kursa. Uz polovicu duljine baze
                od 0.54 m, 0.05 rad daje jos 2.7 cm bocnog zamaha.
            park_joint_tolerance: Dopusteno odstupanje po zglobu prije nego
                baza krene kroz otvor.
        """
        self.env = env
        self.settle_max_s = settle_max_s
        self.settle_vel = settle_vel
        self.release_dwell_s = release_dwell_s
        self.retreat_distance = retreat_distance
        self.backup_local_x = backup_local_x
        self.exit_local_x = exit_local_x
        self.max_speed = max_speed
        self.max_lateral = max_lateral
        self.max_yaw_rate = max_yaw_rate
        self.position_tolerance = position_tolerance
        self.lateral_tolerance = lateral_tolerance
        self.yaw_tolerance = yaw_tolerance
        self.park_joint_tolerance = park_joint_tolerance

        unwrapped = env.unwrapped
        self.device = unwrapped.device
        self.n_env = unwrapped.num_envs
        self.robot = unwrapped.scene["robot"]
        self.door = unwrapped.scene["door"]

        from iiwa_rl.tasks.door.door_cfg import DOOR_DOF_JOINT
        from iiwa_rl.tasks.door.robot_cfg import BASE_BODY, GRIPPER_OPEN

        self.gripper_open = GRIPPER_OPEN
        self.dof_idx = self.door.find_joints(DOOR_DOF_JOINT)[0][0]
        self.base_idx = self.robot.find_bodies(BASE_BODY)[0][0]
        self.tcp_idx = self.robot.find_bodies("gripper_tcp")[0][0]
        self.finger_ids = self.robot.find_joints("gripper_finger_[1-4]_joint")[0]
        self.arm_ids = self.robot.find_joints("iiwa_joint_[1-7]")[0]
        self.park_joint_pos = torch.tensor(PARK_JOINT_POS, device=self.device)

        self.release_threshold = release_threshold or self._detect_release(unwrapped)

        # Sredina otvora iz iste konfiguracije prepreka koju koristi kazna za
        # zid, pa se dvije definicije otvora ne mogu raziei.
        obstacles = unwrapped.cfg.rewards.wall_intrusion.params["obstacles"]
        self.opening_y = 0.5 * (
            max(obstacles[0][1], obstacles[0][2])
            + min(obstacles[1][1], obstacles[1][2])
        )

        scale = unwrapped.cfg.actions.base.scale
        self.base_scale = torch.tensor(
            [scale["base_x_joint"], scale["base_y_joint"], scale["base_theta_joint"]],
            device=self.device,
        )
        self.position_scale = unwrapped.cfg.actions.arm.position_scale
        self.dt = unwrapped.step_dt

        self.phase = torch.full(
            (self.n_env,), self.RL, dtype=torch.long, device=self.device
        )
        self.retreat_target_w = torch.zeros(self.n_env, 3, device=self.device)
        self.settle_steps = torch.zeros(self.n_env, device=self.device)
        self.dwell_steps = torch.zeros(self.n_env, device=self.device)
        self.locked = torch.zeros(self.n_env, dtype=torch.bool, device=self.device)
        self.locked_joint_pos = torch.zeros(
            self.n_env, len(self.arm_ids), device=self.device
        )
        self.lock_steps = torch.zeros(self.n_env, device=self.device)

        self._relax_terminations(unwrapped)

    def _detect_release(self, unwrapped) -> float:
        """Prag pustanja iz tipa vrata prepoznatog u USD putanji."""
        usd_path = str(unwrapped.cfg.scene.door.spawn.usd_path).lower()
        for name, threshold in RELEASE_BY_DOOR_TYPE:
            if name in usd_path:
                print(f"[passage] vrata: {name}, prag pustanja {threshold}")
                return threshold
        raise ValueError(
            f"Tip vrata nije prepoznat iz '{usd_path}'. "
            "Zadaj prag rucno preko release_threshold."
        )

    def _relax_terminations(self, unwrapped) -> None:
        manager = unwrapped.termination_manager
        for name, key, value in RELAXED_TERMINATIONS:
            if name not in manager.active_terms:
                continue
            cfg = manager.get_term_cfg(name)
            if key in cfg.params:
                cfg.params[key] = value
                print(f"[passage] terminacija '{name}' opustena ({key} = {value:g})")

    def _lock_arm(self, env_ids: torch.Tensor, target: torch.Tensor) -> None:
        """Zakljucaj ruku u prostoru zglobova na zadanu pozu.

        Aktuatori ruke su konfigurirani na nultu krutost i prigusenje jer OSC
        pise momente. Pojacanja postavlja _update_lock_gains rampom; ovdje se
        zadaje samo cilj.
        """
        if env_ids.numel() == 0:
            return
        self.locked_joint_pos[env_ids] = target
        self.locked[env_ids] = True
        self.lock_steps[env_ids] = 0.0

    def _unlock_arm(self, env_ids: torch.Tensor) -> None:
        """Vrati ruku u momentno upravljanje."""
        if env_ids.numel() == 0:
            return
        self.robot.write_joint_stiffness_to_sim(
            0.0, joint_ids=self.arm_ids, env_ids=env_ids
        )
        self.robot.write_joint_damping_to_sim(
            0.0, joint_ids=self.arm_ids, env_ids=env_ids
        )
        self.locked[env_ids] = False

    def _update_lock_gains(self) -> None:
        """Podigni krutost brave s meke na tvrdu kroz ARM_LOCK_RAMP_S."""
        locked_ids = self.locked.nonzero(as_tuple=False).flatten()
        if locked_ids.numel() == 0:
            return
        self.lock_steps[locked_ids] += 1.0
        ratio = (self.lock_steps[locked_ids] * self.dt / ARM_LOCK_RAMP_S).clamp(
            0.0, 1.0
        )
        stiffness = ARM_JOINT_STIFFNESS_SOFT + ratio * (
            ARM_JOINT_STIFFNESS_FIRM - ARM_JOINT_STIFFNESS_SOFT
        )
        stiffness = stiffness.unsqueeze(-1).expand(-1, len(self.arm_ids))
        self.robot.write_joint_stiffness_to_sim(
            stiffness, joint_ids=self.arm_ids, env_ids=locked_ids
        )
        self.robot.write_joint_damping_to_sim(
            stiffness * ARM_DAMPING_RATIO, joint_ids=self.arm_ids, env_ids=locked_ids
        )

    def _door_dof(self) -> torch.Tensor:
        return self.door.data.joint_pos[:, self.dof_idx].abs()

    def _yaw_of(self, quat: torch.Tensor) -> torch.Tensor:
        axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.n_env, 3)
        forward = quat_apply(quat, axis)
        return torch.atan2(forward[:, 1], forward[:, 0])

    def _heading_error(self, base_quat: torch.Tensor) -> torch.Tensor:
        """Odstupanje kursa od smjera prolaska, a to je -x okvira vrata.

        Robot je na +x strani, gdje strsi i kvaka, pa je isti kurs ispravan i
        pri povlacenju i pri voznji: u BACKUP fazi se vozi unatrag, ne okrece.
        """
        axis = torch.tensor([-1.0, 0.0, 0.0], device=self.device).expand(self.n_env, 3)
        heading = quat_apply(self.door.data.root_quat_w, axis)
        error = torch.atan2(heading[:, 1], heading[:, 0]) - self._yaw_of(base_quat)
        return torch.atan2(torch.sin(error), torch.cos(error))

    def _line_command(
        self, target_local_x: float, base_pos: torch.Tensor, base_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Brzine baze za voznju po sredisnjoj crti otvora.

        Uzduzna i bocna komponenta reguliraju se odvojeno u okviru vrata, pa
        se bocno odstupanje ponistava kroz cijelu voznju i prolaz je uvijek
        okomit. Zakon "idi prema tocki" bi vozio dijagonalno i sjekao
        dovratnik.

        Vraca (naredba, preostalo po uzduznoj osi, bocno odstupanje, greska
        kursa).
        """
        rel = base_pos - self.door.data.root_pos_w
        local = quat_apply(_quat_inv(self.door.data.root_quat_w), rel)

        along = target_local_x - local[:, 0]
        cross = self.opening_y - local[:, 1]

        vel_local = torch.zeros(self.n_env, 3, device=self.device)
        vel_local[:, 0] = (2.0 * along).clamp(-self.max_speed, self.max_speed)
        vel_local[:, 1] = (2.0 * cross).clamp(-self.max_lateral, self.max_lateral)
        vel_world = quat_apply(self.door.data.root_quat_w, vel_local)

        yaw_error = self._heading_error(base_quat)
        omega = (2.0 * yaw_error).clamp(-self.max_yaw_rate, self.max_yaw_rate)

        command = torch.zeros(self.n_env, 3, device=self.device)
        command[:, 0] = vel_world[:, 0]
        command[:, 1] = vel_world[:, 1]
        command[:, 2] = omega
        return command, along.abs(), cross.abs(), yaw_error

    def reset(self, dones: torch.Tensor) -> None:
        """Vrati resetirane env-ove u RL fazu i otpusti bravu.

        Bez ovoga modul nastavlja pisati otvorene prste i zakljucane zglobove i
        nakon reseta, pa sljedeca epizoda krene bez hvata.
        """
        if not dones.any():
            return
        done_ids = dones.bool().nonzero(as_tuple=False).flatten()
        self._unlock_arm(done_ids)
        self.phase[done_ids] = self.RL

    def override(self, actions: torch.Tensor) -> torch.Tensor:
        """Prepise akcije za env-ove koji vise nisu u RL fazi."""
        actions = actions.clone()

        base_pos = self.robot.data.body_pos_w[:, self.base_idx]
        base_quat = self.robot.data.body_quat_w[:, self.base_idx]
        tcp_w = self.robot.data.body_pos_w[:, self.tcp_idx]

        # Ruka se zakljucava dok hvat jos drzi, pa vrata zaustavi hvat, a ne
        # trenje. Prsti ostaju zatvoreni do RELEASE.
        starting = (self.phase == self.RL) & (
            self._door_dof() >= self.release_threshold
        )
        if starting.any():
            ids = starting.nonzero(as_tuple=False).flatten()
            self._lock_arm(ids, self.robot.data.joint_pos[ids][:, self.arm_ids])
            self.phase[starting] = self.SETTLE
            self.settle_steps[starting] = self.settle_max_s / self.dt

        active = self.phase != self.RL
        if not active.any():
            return actions

        # Prsti nisu u prostoru akcije, pa se ciljevi pisu izravno.
        opening = active & (self.phase != self.SETTLE)
        opening_ids = opening.nonzero(as_tuple=False).flatten()
        if opening_ids.numel() > 0:
            self.robot.set_joint_position_target(
                torch.full(
                    (opening_ids.numel(), len(self.finger_ids)),
                    self.gripper_open,
                    device=self.device,
                ),
                joint_ids=self.finger_ids,
                env_ids=opening_ids,
            )

        settling = self.phase == self.SETTLE
        if settling.any():
            self.settle_steps[settling] -= 1.0
            door_vel = self.door.data.joint_vel[:, self.dof_idx].abs()
            settled = settling & (
                (door_vel < self.settle_vel) | (self.settle_steps <= 0.0)
            )
            self.phase[settled] = self.RELEASE
            self.dwell_steps[settled] = self.release_dwell_s / self.dt

        waiting = self.phase == self.RELEASE
        if waiting.any():
            self.dwell_steps[waiting] -= 1.0
            leaving = waiting & (self.dwell_steps <= 0.0)
            if leaving.any():
                # Brava se otpusta jer u RETREAT fazi ruku vodi OSC, a
                # zakljucani zglobovi bi mu nulirali pomak reference.
                self._unlock_arm(leaving.nonzero(as_tuple=False).flatten())
                # Odmak ide po prilaznoj osi hvataljke (TCP +Z), pa se kuke
                # skidaju sa sipke umjesto da je vuku ustranu. Racuna se tek
                # sad, iz orijentacije koju hvataljka ima nakon smirivanja.
                axis_z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(
                    self.n_env, 3
                )
                approach = quat_apply(
                    self.robot.data.body_quat_w[:, self.tcp_idx], axis_z
                )
                self.retreat_target_w[leaving] = (
                    tcp_w - approach * self.retreat_distance
                )[leaving]
                self.phase[leaving] = self.RETREAT

        # Baza se racuna prije ruke jer naredba ulazi u kompenzaciju reference.
        command = torch.zeros(self.n_env, 3, device=self.device)

        backing = self.phase == self.BACKUP
        if backing.any():
            cmd, along, cross, yaw_error = self._line_command(
                self.backup_local_x, base_pos, base_quat
            )
            command[backing] = cmd[backing]
            arrived = (
                (along < self.position_tolerance)
                & (cross < self.lateral_tolerance)
                & (yaw_error.abs() < self.yaw_tolerance)
            )
            self.phase[backing & arrived] = self.PARK

        driving = self.phase == self.DRIVE
        if driving.any():
            cmd, along, _, _ = self._line_command(
                self.exit_local_x, base_pos, base_quat
            )
            command[driving] = cmd[driving]
            self.phase[driving & (along < 0.15)] = self.DONE

        # OSC vodi ruku samo u RETREAT fazi. U zakljucanim fazama je cilj
        # trenutna poza, sto daje nulti pomak reference, pa dva regulatora ne
        # rade jedan protiv drugoga.
        target_w = torch.where(
            (self.phase == self.RETREAT).unsqueeze(-1), self.retreat_target_w, tcp_w
        )
        delta_w = target_w - tcp_w

        velocity = (ARM_GAIN * delta_w).clamp(-ARM_MAX_SPEED, ARM_MAX_SPEED)
        velocity[:, 0:2] += command[:, 0:2]
        velocity = torch.where(
            self.locked.unsqueeze(-1), torch.zeros_like(velocity), velocity
        )

        # pose_rel ocekuje pomak u okviru zadatka. Task frame nije zadan, pa je
        # to korijen artikulacije: fiksan i nezakrenut, dakle svjetski pomak.
        step = velocity * self.dt
        actions[active, 0:3] = (step / self.position_scale).clamp(-1.0, 1.0)[active]
        actions[active, 3:6] = 0.0
        if actions.shape[1] > 9:  # variable_kp: 6 poze + 6 krutosti + 3 baze
            osc_stiffness = torch.where(
                self.phase == self.RETREAT,
                torch.full_like(delta_w[:, 0], RETREAT_STIFFNESS_ACTION),
                torch.full_like(delta_w[:, 0], PARK_STIFFNESS_ACTION),
            )
            actions[active, 6:12] = osc_stiffness[active].unsqueeze(-1)

        # Brava ide odmah na parkirnu pozu, pa se ruka sklapa istovremeno s
        # povlacenjem baze: zglobni ciljevi su apsolutni, pa gibanje baze
        # sklapanje ne remeti.
        retreated = (self.phase == self.RETREAT) & (delta_w.norm(dim=-1) < 0.04)
        if retreated.any():
            ids = retreated.nonzero(as_tuple=False).flatten()
            self._lock_arm(ids, self.park_joint_pos.expand(ids.numel(), -1))
            self.phase[retreated] = self.BACKUP

        parking = self.phase == self.PARK
        if parking.any():
            joint_error = (
                (self.robot.data.joint_pos[:, self.arm_ids] - self.locked_joint_pos)
                .abs()
                .max(dim=-1)
                .values
            )
            self.phase[parking & (joint_error < self.park_joint_tolerance)] = self.DRIVE

        self._update_lock_gains()

        # Ciljevi brave se pisu svaki korak: set_joint_position_target puni
        # pufer koji write_data_to_sim prazni.
        locked_ids = self.locked.nonzero(as_tuple=False).flatten()
        if locked_ids.numel() > 0:
            self.robot.set_joint_position_target(
                self.locked_joint_pos[locked_ids],
                joint_ids=self.arm_ids,
                env_ids=locked_ids,
            )

        actions[active, -3:] = (command / self.base_scale)[active]
        return actions

    def summary(self) -> str:
        counts = [int((self.phase == i).sum()) for i in range(len(self.PHASE_NAMES))]
        return "  ".join(f"{n}={c}" for n, c in zip(self.PHASE_NAMES, counts))
