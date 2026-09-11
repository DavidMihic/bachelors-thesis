"""door_passage.py - faza prolaska kroz vrata nakon sto ih RL politika otvori.

RL politika rjesava SAMO interakciju s vratima: prilaz kvaci nije u opsegu
(epizoda pocinje s hvatom na kvaki), a ni prolazak nije - politika nema razlog
pustiti kvaku jer je progress linearan u kutu vrata sve do limita.

Ovaj modul preuzima kad su vrata dovoljno otvorena i vodi robota kroz otvor
klasicnim regulatorom:

  RL       politika upravlja, modul samo gleda DOF vrata
  RELEASE  prsti se otvore i ceka se da popusti kontakt sa sipkom
  RETREAT  TCP se povlaci po prilaznoj osi sa sipke
  BACKUP   baza se povlaci od vrata i poravnava sa sredinom otvora
  PARK     ruka se podize iznad baze
  DRIVE    baza vozi ravno kroz otvor
  DONE     stoji

ZASTO KLASICNI REGULATOR: prolazak je navigacija kroz poznatu geometriju -
sredina otvora slijedi iz poze vrata, koja je u simulatoru egzaktna, a na
stvarnom robotu je daje lidar (klasicni pristup je ondje nalazi unutar ~1 cm).

ZASTO SU TERMINACIJE OPUSTENE: grasp_lost, base_hit, wall_hit i overforce su
trenazni alat koji oblikuje ucenje, a ne fizikalna ogranicenja. Bez opustanja
grasp_lost okine cim se TCP odmakne od kvake i epizoda zavrsi prije nego
prolazak uopce pocne. base_hit i wall_hit su usto podeseni za fazu otvaranja,
gdje baza stoji ispred vrata - pri prolasku kroz otvor od 0.87 m s bazom
sirokom 0.63 m ostaje 0.12 m po strani, sto je unutar njihovih tampona.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply

# Ciljna poza TCP-a u okviru baze tijekom parkiranja, (x, y, z).
#
# Iznad nosaca ruke, koji je na (0.363, -0.184, 0.70) (kmr_iiwa.urdf.xacro).
# Visina 1.45 m je 0.75 m iznad nosaca, unutar dosega od 0.80 m, i znatno
# ispod nadvratnika na 2.0 m. Ruka je time skupljena nad vlastitim tlocrtom
# pa ne strsi u dovratnik ni pri zakretu baze.
PARK_TCP_IN_BASE = (0.363, -0.184, 1.45)

# Krutost koju saljemo nakon pustanja (akcija, mnozi se sa stiffness_scale=300).
# 2.0 -> 600 N/m: dovoljno da ruka dodje u pozu i ostane ondje dok se baza
# giba, a daleko od granice gdje bi trzaj pri pokretanju stvarao velike sile.
PARK_STIFFNESS_ACTION = 2.0

# Zakon vodjenja ruke: brzina reference je razmjerna gresci, ogranicena
# odozgo.
#
# ZASTO NE CISTO OGRANICENJE POMAKA: uz clamp je referenca bang-bang - juri
# punom brzinom, preleti cilj i vraca se, sto se vidi kao tresenje. Razmjerni
# zakon uspori pri prilasku i slegne se bez oscilacije.
ARM_GAIN = 1.5  # 1/s
ARM_MAX_SPEED = 0.30  # m/s

# Terminacije koje se opustaju kad je prolazak ukljucen: (naziv, parametar,
# vrijednost). Vrijednosti su odabrane tako da uvjet nikad ne moze biti
# ispunjen, umjesto da se termovi uklanjaju iz managera.
_RELAXED_TERMINATIONS = (
    ("grasp_lost", "max_distance", 1.0e6),
    ("overforce", "limit", 1.0e6),
    ("base_hit", "min_clearance", -1.0e6),
    ("wall_hit", "min_clearance", -1.0e6),
)


class DoorPassage:
    """Preuzima upravljanje kad su vrata dovoljno otvorena i provozi robota.

    Radi nad svim env-ovima odjednom, svaki sa svojom fazom, pa se u play.py
    moze pustiti vise env-ova i gledati koji prolazi.
    """

    RL, RELEASE, RETREAT, BACKUP, PARK, DRIVE, DONE = 0, 1, 2, 3, 4, 5, 6
    _NAMES = ("RL", "RELEASE", "RETREAT", "BACKUP", "PARK", "DRIVE", "DONE")

    def __init__(
        self,
        env,
        release_threshold: float,
        release_dwell_s: float = 1.0,
        retreat_distance: float = 0.10,
        backup_local_x: float = 1.60,
        exit_local_x: float = -1.40,
        max_speed: float = 0.12,
        max_yaw_rate: float = 0.8,
        position_tolerance: float = 0.12,
        yaw_tolerance: float = 0.10,
        park_tolerance: float = 0.12,
    ):
        """
        release_threshold  stanje DOF-a vrata pri kojem se pusta kvaka: kut u
                           rad za zakretna (1.745 = 100 st.), hod u m za
                           klizna. Krilo se zakrece na stranu SUPROTNU od
                           robota (os +z, krilo od +y prema -x), pa ne ulazi u
                           koridor kojim baza prolazi - prag odreduje samo
                           koliko je otvor cist.
        release_dwell_s    koliko se ceka nakon otvaranja prstiju prije nego
                           ruka krene. Prsti se otvaraju trenutno, ali kuke se
                           sa sipke skidaju tek kad popusti kontakt; bez stanke
                           ruka krene prerano i povuce vrata natrag sa sobom.
        retreat_distance   koliko se TCP povlaci po prilaznoj osi. Kuke prstiju
                           OBUHVACAJU sipku, pa samo otvaranje hvata ne
                           oslobadja kvaku; bez odmaka kuka zapne i povuce
                           vrata natrag na robota.
        backup_local_x     udaljenost od ravnine vrata na koju se baza povlaci
                           prije parkiranja ruke i voznje, u okviru vrata
        exit_local_x       dokle se vozi; negativno = prosao na drugu stranu
        max_speed          NAJVAZNIJA postavka za mirnocu ruke. Uz pose_rel i
                           kriticno prigusenje OSC-a ustaljena brzina koju ruka
                           postigne je ~ e*sqrt(K)/(2*sqrt(m)); pri K=600 i
                           gresci od 1 cm to je ~0.06 m/s, pri K=3000 ~0.13.
                           Iznad toga ruka trajno zaostaje i oscilira, a
                           dizanje ARM_MAX_SPEED to samo pogorsa. Izmjereno:
                           pri 0.35 m/s greska je rasla s 0.014 na 0.465 m.
        """
        self.env = env
        self.release_threshold = release_threshold
        self.release_dwell_s = release_dwell_s
        self.retreat_distance = retreat_distance
        self.backup_local_x = backup_local_x
        self.exit_local_x = exit_local_x
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.position_tolerance = position_tolerance
        self.yaw_tolerance = yaw_tolerance
        self.park_tolerance = park_tolerance

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
        # Cilj odmicanja (svijet) i nosiva poza ruke (okvir baze), po env-u.
        self.retreat_target_w = torch.zeros(self.n_env, 3, device=self.device)
        self.carry_local = torch.zeros(self.n_env, 3, device=self.device)
        # Preostali koraci stanke nakon otvaranja prstiju, po env-u.
        self.dwell_steps = torch.zeros(self.n_env, device=self.device)

        self._relax_terminations(unwrapped)

    # ---------------------------------------------------------------- priprema

    def _relax_terminations(self, unwrapped) -> None:
        manager = unwrapped.termination_manager
        for name, key, value in _RELAXED_TERMINATIONS:
            if name not in manager.active_terms:
                continue
            cfg = manager.get_term_cfg(name)
            if key in cfg.params:
                cfg.params[key] = value
                print(f"[passage] terminacija '{name}' opustena ({key} = {value:g})")

    # ---------------------------------------------------------------- pomocno

    def _door_dof(self) -> torch.Tensor:
        return self.door.data.joint_pos[:, self.dof_idx].abs()

    def _yaw_of(self, quat: torch.Tensor) -> torch.Tensor:
        axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.n_env, 3)
        forward = quat_apply(quat, axis)
        return torch.atan2(forward[:, 1], forward[:, 0])

    def _waypoint_w(self, local_x: float) -> torch.Tensor:
        local = torch.tensor([local_x, self.opening_y, 0.0], device=self.device).expand(
            self.n_env, 3
        )
        return self.door.data.root_pos_w + quat_apply(self.door.data.root_quat_w, local)

    def _heading_error(self, base_quat: torch.Tensor) -> torch.Tensor:
        """Odstupanje kursa od smjera prolaska (-x okvira vrata).

        Robot je na +x strani (kvaka strsi na +x, HANDLE_LOCAL ima x > 0) i
        izlazi na -x, pa je isti kurs ispravan i pri povlacenju i pri voznji -
        u BACKUP fazi se vozi unatrag, ne okrece.
        """
        axis = torch.tensor([-1.0, 0.0, 0.0], device=self.device).expand(self.n_env, 3)
        heading = quat_apply(self.door.data.root_quat_w, axis)
        error = torch.atan2(heading[:, 1], heading[:, 0]) - self._yaw_of(base_quat)
        return torch.atan2(torch.sin(error), torch.cos(error))

    def _base_command(
        self, target_w: torch.Tensor, base_pos: torch.Tensor, base_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Brzine baze prema tocki. Vraca (naredba, udaljenost, greska kursa).

        Fiktivni zglobovi base_x i base_y translatiraju po osima KORIJENA
        artikulacije, koji je fiksan u ishodistu env-a i nezakrenut - pa su
        naredbe izravno u svjetskim osima, bez obzira na zakret baze.
        """
        error = target_w[:, :2] - base_pos[:, :2]
        distance = error.norm(dim=-1)
        speed = (2.0 * distance).clamp(max=self.max_speed)
        velocity = error / distance.clamp(min=1e-6).unsqueeze(-1) * speed.unsqueeze(-1)

        yaw_error = self._heading_error(base_quat)
        omega = (2.0 * yaw_error).clamp(-self.max_yaw_rate, self.max_yaw_rate)

        command = torch.zeros(self.n_env, 3, device=self.device)
        command[:, 0] = velocity[:, 0]
        command[:, 1] = velocity[:, 1]
        command[:, 2] = omega
        return command, distance, yaw_error

    # ------------------------------------------------------------------ glavno

    def reset(self, dones: torch.Tensor) -> None:
        """Vrati resetirane env-ove u RL fazu.

        BEZ OVOGA modul nastavlja pisati otvorene prste i nakon reseta: grasp
        event zatvori hvat, a modul ga u istom koraku otvori, pa svaka sljedeca
        epizoda krene bez hvata i nijedna vrata se vise ne otvore.
        """
        if dones.any():
            self.phase[dones.bool()] = self.RL

    def override(self, actions: torch.Tensor) -> torch.Tensor:
        """Prepise akcije za env-ove koji vise nisu u RL fazi."""
        actions = actions.clone()

        base_pos = self.robot.data.body_pos_w[:, self.base_idx]
        base_quat = self.robot.data.body_quat_w[:, self.base_idx]
        tcp_w = self.robot.data.body_pos_w[:, self.tcp_idx]

        # -- RL -> RETREAT
        starting = (self.phase == self.RL) & (
            self._door_dof() >= self.release_threshold
        )
        if starting.any():
            # Odmak ide po PRILAZNOJ osi hvataljke (TCP +Z; gripper.xacro:
            # hvat se zatvara duz X, sipka lezi duz Y), pa se kuke skidaju sa
            # sipke umjesto da je vuku ustranu.
            axis_z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(
                self.n_env, 3
            )
            approach = quat_apply(self.robot.data.body_quat_w[:, self.tcp_idx], axis_z)
            self.retreat_target_w[starting] = (
                tcp_w - approach * self.retreat_distance
            )[starting]
            self.phase[starting] = self.RELEASE
            self.dwell_steps[starting] = self.release_dwell_s / self.dt

        active = self.phase != self.RL
        if not active.any():
            return actions

        active_ids = active.nonzero(as_tuple=False).flatten()

        # -- prsti ostaju otvoreni. Nisu u prostoru akcije (hvat postavlja
        # event pri resetu), pa ih se pise izravno i nista ih ne prepisuje.
        self.robot.set_joint_position_target(
            torch.full(
                (active_ids.numel(), len(self.finger_ids)),
                self.gripper_open,
                device=self.device,
            ),
            joint_ids=self.finger_ids,
            env_ids=active_ids,
        )

        # -- stanka: prsti su otvoreni, ruka jos miruje
        waiting = self.phase == self.RELEASE
        if waiting.any():
            self.dwell_steps[waiting] -= 1.0
            self.phase[waiting & (self.dwell_steps <= 0.0)] = self.RETREAT

        # -- baza po fazama. Racuna se PRIJE ruke jer naredba baze ulazi u
        # unaprijednu kompenzaciju reference ruke.
        command = torch.zeros(self.n_env, 3, device=self.device)

        backing = self.phase == self.BACKUP
        if backing.any():
            cmd, distance, yaw_error = self._base_command(
                self._waypoint_w(self.backup_local_x), base_pos, base_quat
            )
            command[backing] = cmd[backing]
            arrived = (distance < self.position_tolerance) & (
                yaw_error.abs() < self.yaw_tolerance
            )
            self.phase[backing & arrived] = self.PARK

        driving = self.phase == self.DRIVE
        if driving.any():
            cmd, distance, _ = self._base_command(
                self._waypoint_w(self.exit_local_x), base_pos, base_quat
            )
            command[driving] = cmd[driving]
            self.phase[driving & (distance < 0.15)] = self.DONE

        # -- cilj ruke po fazama
        park_w = base_pos + quat_apply(
            base_quat,
            torch.tensor(PARK_TCP_IN_BASE, device=self.device).expand(self.n_env, 3),
        )
        # Ciljevi u BACKUP i DRIVE fazi vezani su za BAZU, ne za svijet. Da su
        # vezani za svijet, ruka bi ostala stajati dok se baza giba i razvukla
        # bi se do zglobnih limita - sto se i vidjelo kao vucenje TCP-a.
        carry_w = base_pos + quat_apply(base_quat, self.carry_local)

        target_w = torch.where(
            (self.phase == self.RETREAT).unsqueeze(-1), self.retreat_target_w, park_w
        )
        target_w = torch.where(
            (self.phase == self.BACKUP).unsqueeze(-1), carry_w, target_w
        )
        # Tijekom stanke ruka stoji: cilj je poza u kojoj vec jest.
        target_w = torch.where(
            (self.phase == self.RELEASE).unsqueeze(-1), tcp_w, target_w
        )

        delta_w = target_w - tcp_w

        # Razmjerna brzina reference prema cilju, plus unaprijedna kompenzacija
        # brzine baze. Bez kompenzacije referenca zaostaje za bazom tocno za
        # v/ARM_GAIN (pri 0.25 m/s to je 17 cm), pa ruka trajno vuce unatrag.
        velocity = (ARM_GAIN * delta_w).clamp(-ARM_MAX_SPEED, ARM_MAX_SPEED)
        velocity[:, 0:2] = velocity[:, 0:2] + command[:, 0:2]

        # pose_rel ocekuje pomak u okviru zadatka; task frame nije zadan, pa je
        # to korijen artikulacije - fiksan i nezakrenut, dakle svjetski pomak.
        step = velocity * self.dt
        actions[active, 0:3] = (step / self.position_scale).clamp(-1.0, 1.0)[active]
        actions[active, 3:6] = 0.0
        if actions.shape[1] > 9:  # variable_kp: 6 poze + 6 krutosti + 3 baze
            actions[active, 6:12] = PARK_STIFFNESS_ACTION

        # -- prijelazi koji ovise o ruci (nakon sto je delta_w izracunata)
        reach = delta_w.norm(dim=-1)
        retreating = self.phase == self.RETREAT
        just_retreated = retreating & (reach < 0.04)
        if just_retreated.any():
            # Zapamti pozu ruke u okviru baze da je BACKUP moze drzati.
            local = quat_apply(_quat_inv(base_quat), (tcp_w - base_pos))
            self.carry_local[just_retreated] = local[just_retreated]
            self.phase[just_retreated] = self.BACKUP

        parking = self.phase == self.PARK
        self.phase[parking & (reach < self.park_tolerance)] = self.DRIVE

        actions[active, -3:] = (command / self.base_scale)[active]
        return actions

    def summary(self) -> str:
        counts = [int((self.phase == i).sum()) for i in range(len(self._NAMES))]
        return "  ".join(f"{n}={c}" for n, c in zip(self._NAMES, counts))


def _quat_inv(quat: torch.Tensor) -> torch.Tensor:
    """Inverz jedinicnog kvaterniona (w, x, y, z)."""
    return torch.cat([quat[:, :1], -quat[:, 1:]], dim=-1)
