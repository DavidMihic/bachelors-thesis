"""door_task_node.py - vodi cijeli zadatak otvaranja vrata, od prilaska do
prolaska kroz otvor.

Lanac:
    prilazak i poravnanje -> hvat kvake -> otvaranje -> prolazak

Tip vrata se prepoznaje SAM, iz razmaka dvaju tagova na kvaki: okomit vektor
znaci klizna sipka, vodoravan znaci zakretna poluga (vidi detect_vertical_bar).
Nema parametra kojim bi se tip zadavao - robot to zakljuci iz percepcije, a iz
istog podatka slijedi i koja se faza otvaranja pokrece.

    klizna   -> open_sliding, pa pass_through_door
    zakretna -> open_revolute (prolazak nije izveden, vidi nize)

Prilazak: P regulator vodi bazu na standoff udaljenost duz NORMALE vrata,
izvedene iz orijentacije taga. Bearing-only prilazak (gledaj u tag) ne
garantira okomitost - holonomna baza moze zadovoljiti "tag je tocno ispred" iz
beskonacno mnogo smjerova, ovisno samo o putanji prilaska.

Faze otvaranja i prolaska su obicne funkcije (run) u svojim modulima, pa se
svaka moze pokrenuti i zasebno preko `ros2 run` za otklanjanje gresaka. One
stvaraju vlastite nodove, ali ne diraju rclpy.init/shutdown - kontekst je
ovdje vec inicijaliziran.

Node se vrti u zasebnoj niti, a glavna nit vodi zadatak sekvencijalno: faze su
blokirajuce (cekaju izvrsavanje trajektorija) pa ne mogu zivjeti u timer
callbacku.

Pokretanje:
    ros2 run kmr_iiwa_task door_task_node
"""

import json
import math
import time
import numpy as np
from enum import Enum, auto

import threading

import rclpy
from geometry_msgs.msg import Twist
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from kmr_iiwa_task.geometry import quat_rotate_vector
from kmr_iiwa_task.handle_approach import run_grasp_sequence, spin_node_forever
from kmr_iiwa_task.open_revolute import run as run_open_revolute
from kmr_iiwa_task.open_sliding import run as run_open_sliding
from kmr_iiwa_task.pass_through_door import run as run_pass_through


class Phase(Enum):
    WAITING_FOR_TAG = auto()
    APPROACHING = auto()
    BASE_LOCKED = auto()
    GRASPING = auto()
    GRASPED = auto()
    FAILED = auto()


def apply_speed_floor(
    value: float, limit: float, floor: float, error_abs: float, tolerance: float
) -> float:
    """Ogranici komandu na [-limit, limit], uz minimalnu magnitudu (floor) dok
    je greska izvan tolerance. Bez poda regulator asimptotski padne ispod
    praga statickog trenja baze i nikad ne stigne do cilja. Unutar tolerance
    vraca nulu, jer bi dalje guranje samo izbacilo bazu natrag van nje."""
    if error_abs <= tolerance or value == 0.0:
        return 0.0
    magnitude = max(min(abs(value), limit), floor)
    return math.copysign(magnitude, value)


class DoorTaskNode(Node):
    def __init__(self):
        super().__init__("kmr_door_task")

        # --- TF ---
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("door_tag_frame", "door_tag_center")

        # --- Izlaz ---
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("control_rate_hz", 20.0)

        # --- Cilj prilaska ---
        self.declare_parameter("standoff_distance_m", 1.35)

        # --- P regulator ---
        self.declare_parameter("kp_x", 0.6)
        self.declare_parameter("kp_y", 0.6)
        self.declare_parameter("kp_yaw", 1.2)

        # --- Settle kriterij (debounce protiv suma) ---
        self.declare_parameter("pos_tolerance_m", 0.03)
        self.declare_parameter("yaw_tolerance_rad", 0.04)
        self.declare_parameter("settle_ticks", 10)

        # --- Limiti brzine ---
        self.declare_parameter("max_linear_speed_mps", 0.3)
        self.declare_parameter("max_angular_speed_radps", 0.5)
        self.declare_parameter("min_linear_speed_mps", 0.12)
        self.declare_parameter("min_angular_speed_radps", 0.13)

        # --- Logiranje ---
        self.declare_parameter("log_path", "/tmp/kmr_door_task_log.jsonl")

        self.base_frame = self.get_parameter("base_frame").value
        self.door_tag_frame = self.get_parameter("door_tag_frame").value
        self.standoff = self.get_parameter("standoff_distance_m").value
        self.kp_x = self.get_parameter("kp_x").value
        self.kp_y = self.get_parameter("kp_y").value
        self.kp_yaw = self.get_parameter("kp_yaw").value
        self.pos_tol = self.get_parameter("pos_tolerance_m").value
        self.yaw_tol = self.get_parameter("yaw_tolerance_rad").value
        self.settle_ticks_required = self.get_parameter("settle_ticks").value
        self.max_lin = self.get_parameter("max_linear_speed_mps").value
        self.max_ang = self.get_parameter("max_angular_speed_radps").value
        self.min_lin = self.get_parameter("min_linear_speed_mps").value
        self.min_ang = self.get_parameter("min_angular_speed_radps").value
        self.log_path = self.get_parameter("log_path").value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_vel_pub = self.create_publisher(
            Twist, self.get_parameter("cmd_vel_topic").value, 10
        )

        self.phase = Phase.WAITING_FOR_TAG
        self.settle_counter = 0
        self.approach_start_time = None
        self._base_locked_logged = False
        self._last_tag_stamp_ns = None

        rate = self.get_parameter("control_rate_hz").value
        self.timer = self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f"kmr_door_task pokrenut. Faza={self.phase.name}. "
            f"Cekam TF {self.base_frame} -> {self.door_tag_frame}..."
        )

    # ------------------------------------------------------------------ #
    # Dispatch

    def _tick(self):
        if self.phase == Phase.WAITING_FOR_TAG:
            self._tick_waiting_for_tag()
        elif self.phase == Phase.APPROACHING:
            self._tick_approaching()
        elif self.phase == Phase.BASE_LOCKED:
            self._tick_base_locked()

    # ------------------------------------------------------------------ #
    # Faze

    def _tick_waiting_for_tag(self):
        transform = self._get_fresh_tag_transform()
        if transform is None:
            self.get_logger().warn(
                f"Nema TF za {self.door_tag_frame} - cekam...",
                throttle_duration_sec=2.0,
            )
            return

        self.phase = Phase.APPROACHING
        self.approach_start_time = time.monotonic()
        self.get_logger().info("Tag pronadjen. Pocinjem prilazak.")
        self._drive_toward_tag(transform)

    def _tick_approaching(self):
        transform = self._get_fresh_tag_transform()
        if transform is None:
            self.get_logger().warn(
                f"Izgubljen TF za {self.door_tag_frame} tijekom prilaska - "
                "zaustavljam bazu dok se tag ne vrati.",
                throttle_duration_sec=2.0,
            )
            return
        self._drive_toward_tag(transform)

    def _tick_base_locked(self):
        # Baza se vise ne mice - namjerno se ne publisha nista, ni nule.
        if not self._base_locked_logged:
            self.get_logger().info("Baza zakljucana na stajnoj tocki.")
            self._base_locked_logged = True

    # ------------------------------------------------------------------ #
    # Pomocne metode

    def _lookup_tag(self):
        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame, self.door_tag_frame, Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _get_fresh_tag_transform(self):
        """Vrati svjez TF (base_frame -> door_tag_frame) ili None ako je
        nedostupan ili zastario. U oba slucaja kad vraca None, vec je
        publishala nulti Twist i resetirala settle_counter - pozivatelj
        samo treba prekinuti obradu ovog ticka (i po zelji dodatno logirati
        specifican razlog)."""
        transform = self._lookup_tag()
        if transform is None:
            self._publish_zero_twist()
            self.settle_counter = 0
            return None

        stamp_ns = Time.from_msg(transform.header.stamp).nanoseconds
        if self._last_tag_stamp_ns is not None and stamp_ns == self._last_tag_stamp_ns:
            self._publish_zero_twist()
            self.settle_counter = 0
            self.get_logger().warn(
                f"TF za {self.door_tag_frame} nije osvjezen (stamp identican "
                "proslom ticku - zastarjela detekcija, npr. tag izvan FOV-a) - "
                "zaustavljam bazu dok ne stigne svjeza detekcija.",
                throttle_duration_sec=2.0,
            )
            return None
        self._last_tag_stamp_ns = stamp_ns
        return transform

    def _drive_toward_tag(self, transform):
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        q = transform.transform.rotation

        # Normala vrata (izlazi iz plohe, prema robotu) - ista projekcija
        # tag Z-osi na vodoravnu ravninu kao u add_door_collision.py
        # (build_vertical_panel_orientation), jer tag ima nagib/roll sum
        # koji bi inace pokvario racun. Bearing-only pristup (stara
        # verzija) ne garantira okomitost - holonomna baza moze
        # zadovoljiti "tag je tocno ispred" iz beskonacno mnogo smjerova,
        # ovisno samo o putanji prilaska.
        nx, ny, _ = quat_rotate_vector([q.x, q.y, q.z, q.w], [0.0, 0.0, 1.0])
        norm = math.hypot(nx, ny)
        if norm < 1e-6:
            # Degenerirano (tag gleda gotovo okomito gore/dolje) - fallback
            # na stari bearing pristup radije nego da robot stane.
            bearing = math.atan2(ty, tx)
            nx, ny = -math.cos(bearing), -math.sin(bearing)
            norm = 1.0
        nx, ny = nx / norm, ny / norm

        # Ciljna tocka: standoff duz PRAVE normale, ne "gdje god trenutno
        # gledam tag". Ciljni yaw: robot okrenut USUPROT normali (u vrata).
        err_x = tx + self.standoff * nx
        err_y = ty + self.standoff * ny
        err_yaw = math.atan2(-ny, -nx)

        lin_x = apply_speed_floor(
            self.kp_x * err_x, self.max_lin, self.min_lin, abs(err_x), self.pos_tol
        )
        lin_y = apply_speed_floor(
            self.kp_y * err_y, self.max_lin, self.min_lin, abs(err_y), self.pos_tol
        )
        ang_z = apply_speed_floor(
            self.kp_yaw * err_yaw,
            self.max_ang,
            self.min_ang,
            abs(err_yaw),
            self.yaw_tol,
        )

        twist = Twist()
        twist.linear.x = lin_x
        twist.linear.y = lin_y
        twist.angular.z = ang_z
        self.cmd_vel_pub.publish(twist)

        within_tolerance = (
            abs(err_x) < self.pos_tol
            and abs(err_y) < self.pos_tol
            and abs(err_yaw) < self.yaw_tol
        )
        self.settle_counter = self.settle_counter + 1 if within_tolerance else 0

        if self.settle_counter >= self.settle_ticks_required:
            self._lock_base(err_x, err_y, err_yaw)

    def _lock_base(self, err_x, err_y, err_yaw):
        self._publish_zero_twist()
        elapsed = time.monotonic() - self.approach_start_time
        self.get_logger().info(
            f"Stajna tocka dosegnuta za {elapsed:.2f}s "
            f"(err_x={err_x:+.3f}m, err_y={err_y:+.3f}m, "
            f"err_yaw={math.degrees(err_yaw):+.2f}deg). Baza zakljucana."
        )
        self._append_log(
            {
                "event": "approach_complete",
                "elapsed_sec": elapsed,
                "final_err_x_m": err_x,
                "final_err_y_m": err_y,
                "final_err_yaw_rad": err_yaw,
            }
        )
        self.phase = Phase.BASE_LOCKED

    def _publish_zero_twist(self):
        self.cmd_vel_pub.publish(Twist())

    def detect_vertical_bar(self, timeout_sec=5.0):
        """Tip kvake iz geometrije, bez parametra.

        Vektor izmedju dva taga na kvaki je OKOMIT kod klizne sipke (tagovi na
        z=-0.11 i z=+0.11 u sliding_door.urdf) i VODORAVAN kod zakretne poluge
        (tagovi na y=-0.015 i y=-0.145 u revolute_door.urdf). Razlika je
        jednoznacna, pa robot tip moze ocitati sam.

        Vraca True za okomitu sipku, False za vodoravnu polugu, None ako se
        tagovi ne mogu ocitati.
        """
        deadline = time.time() + timeout_sec
        while time.time() < deadline and rclpy.ok():
            try:
                a = self.tf_buffer.lookup_transform(
                    self.base_frame, "handle_tag_a", Time()
                ).transform.translation
                b = self.tf_buffer.lookup_transform(
                    self.base_frame, "handle_tag_b", Time()
                ).transform.translation
            except (
                LookupException,
                ConnectivityException,
                ExtrapolationException,
            ):
                time.sleep(0.1)
                continue

            v = np.array([b.x - a.x, b.y - a.y, b.z - a.z])
            n = float(np.linalg.norm(v))
            if n < 0.05:
                time.sleep(0.1)
                continue

            v = v / n
            vertical = abs(v[2]) > 0.7
            tip = (
                "OKOMITA sipka (klizna vrata)"
                if vertical
                else "VODORAVNA poluga (zakretna vrata)"
            )
            self.get_logger().info(
                f"Tip kvake iz geometrije: {tip} - os {np.round(v, 2)}, "
                f"razmak tagova {n*1000:.0f} mm"
            )
            self._append_log(
                {
                    "event": "handle_type_detected",
                    "vertical_bar": bool(vertical),
                    "axis": [float(c) for c in v],
                    "tag_spacing_m": n,
                }
            )
            return vertical
        return None

    def _append_log(self, entry: dict):
        entry = {"stamp_unix": time.time(), **entry}
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as exc:
            self.get_logger().warn(f"Ne mogu upisati log ({self.log_path}): {exc}")


def main():
    rclpy.init()
    node = DoorTaskNode()
    callback_group = ReentrantCallbackGroup()

    # Node se vrti u zasebnoj niti, a glavna nit vodi zadatak sekvencijalno.
    # Faza hvata je blokirajuca (ceka izvrsavanje trajektorija), pa ne moze
    # zivjeti u timer callbacku - odatle ova podjela.
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    threading.Thread(
        target=spin_node_forever, args=(node, executor), daemon=True
    ).start()

    try:
        while rclpy.ok() and node.phase != Phase.BASE_LOCKED:
            time.sleep(0.1)

        # Regulator baze se gasi prije hvata: run_grasp_sequence sam primice
        # bazu preko /cmd_vel, pa bi dva izvora naredbi na istom topicu
        # radila jedan protiv drugoga.
        node.timer.cancel()
        node.phase = Phase.GRASPING
        node.get_logger().info("Pocinjem hvat kvake.")

        vertical_bar = node.detect_vertical_bar()
        if vertical_bar is None:
            node.get_logger().error(
                "Ne mogu ocitati oba taga na kvaki (handle_tag_a/handle_tag_b) - "
                "ne znam tip kvake, prekidam."
            )
            node.phase = Phase.FAILED
            node._append_log({"event": "handle_type_detect_failed"})
            return

        ok = run_grasp_sequence(
            node,
            node.tf_buffer,
            callback_group,
            vertical_bar=vertical_bar,
        )
        node.phase = Phase.GRASPED if ok else Phase.FAILED
        node.get_logger().info("Kvaka uhvacena." if ok else "Hvat kvake nije uspio.")
        node._append_log({"event": "grasp_complete", "success": bool(ok)})

        if not ok:
            return

        if vertical_bar:
            node.get_logger().info("=== Otvaram klizna vrata ===")
            run_open_sliding()
            node.get_logger().info("=== Prolazim kroz vrata ===")
            run_pass_through()
        else:
            node.get_logger().info("=== Otvaram zakretna vrata ===")
            run_open_revolute()
            # Prolazak je izveden samo za klizna vrata: ondje robot cijelo
            # vrijeme ostaje s iste strane zida, pa mu otvor na kraju ostane
            # bocno i do njega se dolazi bocnim gibanjem. Kod zakretnih vrata
            # robot krilo vuce prema sebi i putanja kroz otvor je bitno
            # drugacija, pa bi trebala zasebna izvedba.
            node.get_logger().info(
                "Prolazak kroz zakretna vrata nije izveden - zadatak zavrsava "
                "nakon otvaranja."
            )

        node._append_log(
            {"event": "task_complete", "vertical_bar": bool(vertical_bar)}
        )
        node.get_logger().info("=== ZADATAK ZAVRSEN ===")

        while rclpy.ok():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
