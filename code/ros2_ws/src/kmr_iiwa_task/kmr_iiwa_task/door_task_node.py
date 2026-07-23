"""
door_task_node.py - state-machine node za zadatak otvaranja vrata. Ova verzija
implementira samo fazu prilaska: baza se dovede na fiksnu stajnu udaljenost i
kut ispred vrata, koristeci door_tag_center kao vizualnu referencu, pa se
zakljuca kad regulacija konvergira. Naredne faze (prilazak kvaci, hvat,
provjera brave, procjena ogranicenja gibanja, impedancijsko otvaranje) nisu
jos implementirane.

Nasumican spawn baze rjesava se odvojeno, u build_integration_scene.py
(--randomize-spawn), prije nego se scena uopce ucita - ne ovdje.

Napomena o pristupu - bearing umjesto poravnanja orijentacije:
Regulator ne pokusava poravnati orijentaciju baze s orijentacijom
door_tag_center transforma, jer bi to zahtijevalo pretpostavku o konvenciji
osi koju apriltag_ros koristi za tag frame koja nije potvrdjena. Umjesto toga
koristi cisto pozicijski "bearing": atan2(ty, tx) u base_link frameu, kut
izmedju robotove naprijed-osi i smjera prema tagu. Ovo je dovoljno da baza
zavrsi priblizno kvadratno ispred vrata, sto je dovoljno za domet ruke u
sljedecoj fazi.

TF lanac: ne treba dodatni glue kod - add_camera_ros_graph.py vec publisha
kompletno kinematicko stablo (base_link -> ... -> camera_color_optical_frame)
preko ROS2PublishTransformTree, a apriltag_ros dodaje door_tag_center na isti
tf2 tree (isti princip kao u handle_pose_fusion.py). Ovaj node samo radi
lookup_transform(base_frame, door_tag_frame, Time()) - najnoviji dostupan.

Sigurnosna napomena: cmd_vel_bridge.py primjenjuje zadnju primljenu Twist
poruku svaki fizicki korak - nema failsafe timeouta. Zato ovaj node mora
eksplicitno publishati nulti Twist cim TF postane nedostupan ili zastario
(ne smije se osloniti na "prestani publishati pa ce robot stati"), i mora
prestati publishati bilo sto, ukljucujuci nule, nakon zakljucavanja baze -
vidi _tick_base_locked().

Sucelje:
  PUB  /cmd_vel (geometry_msgs/Twist), u base_link frameu (isto kao
       cmd_vel_bridge.py ocekuje)
  Koristi TF (base_frame -> door_tag_frame), bez direktne pretplate na
  apriltag_ros topice (isti razlog kao u handle_pose_fusion.py).

Logiranje: pri prelasku APPROACHING -> BASE_LOCKED, upisuje jedan JSON red
u --log-path (default /tmp/kmr_door_task_log.jsonl) s vremenom prilaska i
finalnim greskama.

Pokretanje:
    ros2 run kmr_iiwa_task door_task_node
    ros2 run kmr_iiwa_task door_task_node --ros-args \
        -p standoff_distance_m:=1.2 -p door_tag_frame:=door_tag_center
"""

import json
import math
import time
from enum import Enum, auto

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class Phase(Enum):
    WAITING_FOR_TAG = auto()
    APPROACHING = auto()
    BASE_LOCKED = auto()
    # Naredne faze (nisu jos implementirane):
    #   APPROACHING_HANDLE    - IK prema /perception/handle_pose
    #   GRASPING              - zatvaranje grippera na kvaci
    #   ATTACHED              - fiksni joint nakon potvrdjenog kontakta
    #   PROBING               - provjera postoji li brava
    #   UNLOCKING             - otkljucavanje ako brava postoji
    #   ESTIMATING_CONSTRAINT - rekurzivni fit kruznice/pravca za gibanje vrata
    #   OPENING               - impedancijsko povlacenje/klizanje
    #   DONE


def apply_speed_floor(
    value: float, limit: float, floor: float, error_abs: float, tolerance: float
) -> float:
    """Ogranici komandu na [-limit, limit]. Dok je pripadna greska (error_abs)
    izvan tolerance, garantiraj minimalnu magnitudu (floor) kad je komanda
    nenulta - sprjecava da regulator asimptotski padne ispod praga koji
    svladava staticko trenje baze i nikad stvarno ne stigne do cilja. Kad je
    greska vec unutar tolerance, vraca nulu - dalje guranje bi samo izbacilo
    bazu natrag van tolerancije."""
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
        self.declare_parameter("standoff_distance_m", 1.2)

        # --- P regulator ---
        self.declare_parameter("kp_x", 0.6)
        self.declare_parameter("kp_y", 0.6)
        self.declare_parameter("kp_yaw", 1.2)

        # --- Settle kriterij (debounce protiv suma pri konvergenciji) ---
        self.declare_parameter("pos_tolerance_m", 0.05)
        self.declare_parameter("yaw_tolerance_rad", 0.08)
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
        self._locked_stub_logged = False
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
        # Baza se vise ne mice do kraja zadatka - namjerno ne publishamo bas
        # nista ovdje (cak ni nule), da se ne otvori slucajni prozor gdje
        # neka buduca izmjena doda logiku prije ovog returna i preskoci ga.
        if not self._locked_stub_logged:
            self.get_logger().info(
                "Baza zakljucana na stajnoj tocki. Naredne faze (attach, "
                "probing, otkljucavanje, procjena ogranicenja, impedancija) "
                "nisu jos implementirane u ovom nodu."
            )
            self._locked_stub_logged = True

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

        bearing = math.atan2(ty, tx)
        err_x = tx - self.standoff
        err_y = ty
        err_yaw = bearing

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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
