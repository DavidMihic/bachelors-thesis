"""pass_through_door.py - nakon otvaranja vrata: pusti kvaku, odmakni se,
parkiraj ruku, poravnaj se s otvorom i prodji kroz njega.

Pokrece se nakon open_sliding (ili open_revolute). U tom trenutku robot vise
NIJE ispred otvora - kod kliznih vrata se pomaknuo skoro metar u stranu dok je
vukao krilo, pa mu je otvor bocno, ne ispred.

DETEKCIJA OTVORA
Tocke iz /scan ostaju u redoslijedu skena, pa je praznina u tom nizu praznina
u prostoru gledano sa senzora. Trazi se praznina sirine bliske
DOORWAY_WIDTH_M izmedju dva niza tocaka. Izmjereno na mirujucem robotu:
srediste stabilno unutar ~1 cm kroz minutu, sirina 0.85-0.86 m naspram
stvarnih 0.85. Pred zatvorenim vratima ispravno ne nalazi nista.

Raniji pristup - trazenje jednog ruba dovratnika - nije bio upotrebljiv: rub
nadjen u 44% ciklusa uz rasap od 600 mm. Praznina poznate sirine ima dva
ogranicenja umjesto jednog, pa ju je puno teze zamijeniti s necim drugim.

FAZE
  1. RELEASE - otvori gripper, pa se BAZOM odmakni unatrag. Gripper je nakon
     otpustanja jos oko sipke; put do parkirne poze vodi kroz nju, pa bi ruka
     odgurnula vrata. Odmicanje bazom je jednostavnije od planiranja pomaka
     rukom i nema rizika da putanja prodje kroz kvaku.
  2. PARK   - ruka u parkirnu pozu (istu koju salje full_stack launch). Bez
     toga gripper strsi u ravnini vrata i zapinje pri prolasku.
  3. ALIGN  - bocno gibanje (holonomna baza, linear.y) dok otvor ne dodje
     ravno ispred. Baza se NE rotira, pa ostaje okomita na vrata.
  4. DRIVE  - ravno naprijed kroz otvor, dok prijedjeni put ne premasi
     PASS_DISTANCE_M ili dok /scan ne javi prepreku preblizu.

BRZINE: baza postize samo dio naredjenog. Uzduzno oko 45%, bocno oko 11% -
sto odgovara specifikaciji KMR-a (3.6 naspram 2.0 km/h) uz prag statickog
trenja koji pri malim brzinama udara jace. Naredbe su zato postavljene znatno
iznad zeljene brzine, a vremenska ogranicenja su velikodusna.

Preduvjet: vrata su otvorena, robot drzi ili je upravo pustio kvaku,
arm_controller i /scan rade.

Pokrece se iz door_task_node (funkcija run) ili zasebno preko
`ros2 run kmr_iiwa_task pass_through_door`.
"""

import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from kmr_iiwa_task.geometry import quat_rotate_vector, wrap_pi

JOINT_NAMES = [f"iiwa_joint_{i}" for i in range(1, 8)]
# Ista poza koju full_stack.launch.py salje pri dizanju kontrolera.
PARK_POSE = [0.0, -0.6, 0.0, -2.2, 0.0, 0.8, 0.0]
PARK_TIME_SEC = 4

LIDAR_FRAME = "lidar_link"
SELF_OCCLUSION_HALF_ANGLE_DEG = 94.0

# --- Detekcija otvora ---
DOORWAY_WIDTH_M = 0.85
WIDTH_TOL_M = 0.35
GAP_MIN_M = 0.30
MAX_RANGE_M = 6.0

# --- Odmicanje od kvake (baza unatrag) ---
RETREAT_SPEED_MPS = 0.15
RETREAT_TIME_SEC = 3.0

# --- Poravnavanje (bocno, bez rotacije) ---
ALIGN_SPEED_MPS = 0.22  # bocno se postize ~11% naredjenog, pa naredba mora
# biti znatno veca od zeljene brzine
ALIGN_TOL_M = 0.05
ALIGN_TIMEOUT_SEC = 150.0

# --- Prolazak ---
DRIVE_SPEED_MPS = 0.22
# Prolazak se mjeri iz percepcije: cx je udaljenost do sredista otvora. Kad
# padne ispod PASS_CX_M, ravnina zida je iza prednjeg ruba baze. Integracija
# naredbe se ne koristi kao kriterij - visestruko precjenjuje.
PASS_CX_M = -0.30
DRIVE_TIMEOUT_SEC = 120.0
LOST_DOORWAY_STEPS = 20  # koliko ciklusa bez otvora znaci da smo prosli
PASS_MARGIN_M = 0.10  # koliko straznji rub baze prolazi iza ravnine zida
BASE_SPEED_EFFICIENCY = 0.45  # baza uzduzno postize ~45% naredjenog

# --- Sigurnost ---
KMR_LENGTH_M = 1.08
KMR_WIDTH_M = 0.63
FRONT_X_MIN_M = KMR_LENGTH_M / 2.0
HARD_STOP_M = 0.12
CORRIDOR_HALF_WIDTH_M = KMR_WIDTH_M / 2.0 + 0.05

PUBLISH_PERIOD_SEC = 0.02
CONTROL_PERIOD_SEC = 0.05


def run():
    node = Node("pass_through_door")

    cmd_vel_pub = node.create_publisher(Twist, "/cmd_vel", 10)
    gripper_pub = node.create_publisher(Float32, "/gripper_cmd", 10)
    traj_pub = node.create_publisher(
        JointTrajectory, "/arm_controller/joint_trajectory", 10
    )

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    state = {"doorway": None, "too_close": False, "have_scan": False}
    lidar_tf = {"v": None}

    def _ensure_tf():
        if lidar_tf["v"] is not None:
            return True
        try:
            tf = tf_buffer.lookup_transform("base_link", LIDAR_FRAME, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return False
        t, q = tf.transform.translation, tf.transform.rotation
        lidar_tf["v"] = (np.array([t.x, t.y, t.z]), (q.x, q.y, q.z, q.w))
        return True

    def _on_scan(msg: LaserScan):
        if not _ensure_tf():
            return
        t, q = lidar_tf["v"]
        mask = math.radians(SELF_OCCLUSION_HALF_ANGLE_DEG)
        limit = min(msg.range_max, MAX_RANGE_M)

        pts = []
        too_close = False
        angle = msg.angle_min
        for r in msg.ranges:
            a = angle
            angle += msg.angle_increment
            if not math.isfinite(r) or r < msg.range_min or r > limit:
                continue
            if abs(wrap_pi(a)) > mask:
                continue
            x_l, y_l = r * math.cos(a), r * math.sin(a)
            x_b, y_b, _ = quat_rotate_vector(q, [x_l, y_l, 0.0])
            x_b += t[0]
            y_b += t[1]
            pts.append((x_b, y_b))
            if (
                abs(y_b) < CORRIDOR_HALF_WIDTH_M
                and FRONT_X_MIN_M < x_b < FRONT_X_MIN_M + HARD_STOP_M
            ):
                too_close = True

        state["too_close"] = too_close
        state["have_scan"] = True

        best = None
        for i in range(len(pts) - 1):
            p, n = pts[i], pts[i + 1]
            w = math.hypot(n[0] - p[0], n[1] - p[1])
            if w > GAP_MIN_M and abs(w - DOORWAY_WIDTH_M) < WIDTH_TOL_M:
                if best is None or abs(w - DOORWAY_WIDTH_M) < abs(
                    best[0] - DOORWAY_WIDTH_M
                ):
                    best = (w, 0.5 * (p[0] + n[0]), 0.5 * (p[1] + n[1]))
        state["doorway"] = best

    node.create_subscription(LaserScan, "/scan", _on_scan, 10)

    cmd = {"vx": 0.0, "vy": 0.0}
    stop_flag = {"v": False}

    def publisher_loop():
        """cmd_vel_bridge primjenjuje zadnju primljenu poruku svaki fizicki
        korak i nema failsafe timeout, pa naredbe salje zasebna nit u stalnom
        ritmu - neujednacen ritam znaci trzajno gibanje."""
        while rclpy.ok() and not stop_flag["v"]:
            tw = Twist()
            tw.linear.x = cmd["vx"]
            tw.linear.y = cmd["vy"]
            cmd_vel_pub.publish(tw)
            time.sleep(PUBLISH_PERIOD_SEC)

    def shutdown(msg=None, error=False):
        cmd["vx"] = 0.0
        cmd["vy"] = 0.0
        time.sleep(0.2)
        stop_flag["v"] = True
        time.sleep(0.1)
        cmd_vel_pub.publish(Twist())
        time.sleep(0.5)
        if msg:
            (node.get_logger().error if error else node.get_logger().info)(msg)
        executor.shutdown()
        time.sleep(0.2)
        node.destroy_node()

    # Vlastiti izvrsavac, ne globalni - vidi isti komentar u open_sliding.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    threading.Thread(target=publisher_loop, daemon=True).start()

    node.get_logger().info("Cekam /scan...")
    while not state["have_scan"] and rclpy.ok():
        time.sleep(0.1)

    # --- Faza 1: pusti kvaku i odmakni se od nje ---
    node.get_logger().info("Otvaram gripper...")
    for _ in range(20):
        gripper_pub.publish(Float32(data=0.0))
        time.sleep(0.1)

    node.get_logger().info("Odmicem se od kvake...")
    cmd["vx"] = -RETREAT_SPEED_MPS
    time.sleep(RETREAT_TIME_SEC)
    cmd["vx"] = 0.0
    time.sleep(0.5)

    # --- Faza 2: parkiraj ruku ---
    node.get_logger().info("Parkiram ruku...")
    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES
    pt = JointTrajectoryPoint()
    pt.positions = list(PARK_POSE)
    pt.time_from_start.sec = PARK_TIME_SEC
    traj.points = [pt]
    traj_pub.publish(traj)
    time.sleep(PARK_TIME_SEC + 1.5)
    node.get_logger().info("Ruka parkirana.")

    # --- Faza 3: bocno poravnavanje s otvorom (bez rotacije) ---
    node.get_logger().info("Poravnavam se s otvorom (bocno)...")
    t0 = time.monotonic()
    aligned = False
    while rclpy.ok() and time.monotonic() - t0 < ALIGN_TIMEOUT_SEC:
        d = state["doorway"]
        if d is None:
            cmd["vy"] = 0.0
            node.get_logger().warn(
                "Otvor nije vidljiv - stojim.", throttle_duration_sec=2.0
            )
            time.sleep(CONTROL_PERIOD_SEC)
            continue

        _, cx, cy = d
        if abs(cy) < ALIGN_TOL_M:
            cmd["vy"] = 0.0
            aligned = True
            node.get_logger().info(f"Poravnat: otvor na ({cx:+.2f}, {cy:+.2f}) m.")
            break
        cmd["vy"] = math.copysign(ALIGN_SPEED_MPS, cy)
        node.get_logger().info(
            f"otvor ({cx:+.2f}, {cy:+.2f}) -> vy={cmd['vy']:+.2f}",
            throttle_duration_sec=2.0,
        )
        time.sleep(CONTROL_PERIOD_SEC)

    cmd["vy"] = 0.0
    time.sleep(0.5)

    if not aligned:
        shutdown("Poravnavanje nije uspjelo - ne ulazim.", error=True)
        return

    # --- Faza 4: ravno naprijed kroz otvor ---
    node.get_logger().info("Prolazim kroz otvor...")
    t0 = time.monotonic()
    lost = 0
    last_cx = None
    outcome = "vrijeme isteklo"
    while rclpy.ok() and time.monotonic() - t0 < DRIVE_TIMEOUT_SEC:
        if state["too_close"]:
            outcome = "prepreka preblizu - zaustavljam"
            break

        cmd["vx"] = DRIVE_SPEED_MPS
        d = state["doorway"]
        if d is None:
            lost += 1
            if lost >= LOST_DOORWAY_STEPS:
                # Otvor je nestao iz vidnog polja (rubovi su izasli iz maske
                # samozaklona), ne znaci da smo prosli. Zadnji vidjeni cx je
                # udaljenost do ravnine zida; do nje treba dodati jos pola
                # duljine baze da i straznji rub prodje, plus marza.
                extra = (last_cx or 0.0) + KMR_LENGTH_M / 2.0 + PASS_MARGIN_M
                secs = extra / (DRIVE_SPEED_MPS * BASE_SPEED_EFFICIENCY)
                node.get_logger().info(
                    f"Otvor izasao iz vidnog polja na cx={last_cx:+.2f} m - "
                    f"vozim jos {secs:.0f} s da straznji rub prodje."
                )
                t_extra = time.monotonic()
                while rclpy.ok() and time.monotonic() - t_extra < secs:
                    if state["too_close"]:
                        break
                    cmd["vx"] = DRIVE_SPEED_MPS
                    time.sleep(CONTROL_PERIOD_SEC)
                outcome = "prosao"
                break
        else:
            lost = 0
            _, cx, cy = d
            last_cx = cx
            node.get_logger().info(
                f"otvor na cx={cx:+.2f} m", throttle_duration_sec=2.0
            )
            if cx < PASS_CX_M:
                outcome = f"prosao (otvor iza, cx={cx:+.2f} m)"
                break
        time.sleep(CONTROL_PERIOD_SEC)

    cmd["vx"] = 0.0
    node.get_logger().info("=== SAZETAK ===")
    node.get_logger().info(f"  ishod: {outcome}")
    shutdown()

    cmd["vx"] = 0.0
    node.get_logger().info("=== SAZETAK ===")
    node.get_logger().info(f"  ishod: {outcome}")
    shutdown()


def main():
    """Samostalno pokretanje. Kad se faza poziva iz door_task_node, koristi se
    run() - kontekst je ondje vec inicijaliziran."""
    rclpy.init()
    try:
        run()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
