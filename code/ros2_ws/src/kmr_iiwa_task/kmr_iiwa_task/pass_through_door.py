"""
pass_through_door.py - nakon otvaranja vrata robot pusta kvaku, parkira ruku,
poravnava se s otvorom i prolazi kroz njega.

Kod kliznih vrata robot se dok je vukao krilo pomaknuo skoro metar u stranu,
pa mu otvor na kraju nije ispred nego bocno. Do njega dolazi bocnim gibanjem,
bez rotacije, cime ostaje okomit na vrata.

Otvor se nalazi iz /scan: tocke ostaju u redoslijedu skena, pa je praznina u
tom nizu praznina u prostoru gledano sa senzora. Trazi se praznina sirine
bliske DOORWAY_WIDTH_M. Trazenje jednog ruba dovratnika se pokazalo
neupotrebljivim (nadjen u 44% ciklusa, rasap 600 mm); praznina poznate sirine
ima dva ogranicenja umjesto jednog.

Prijedjeni put se mjeri odometrijom. Baza postize samo dio naredjene brzine
(oko 45% uzduzno, 11% bocno), pa je racunanje puta iz naredbe i vremena
neupotrebljivo.

Preduvjet: vrata su otvorena, robot drzi ili je upravo pustio kvaku,
arm_controller i /scan rade, cmd_vel_bridge vrti OdomPublisher.

Pokrece se iz door_task_node (funkcija run) ili zasebno preko
`ros2 run kmr_iiwa_task pass_through_door`.
"""

import math
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from kmr_iiwa_task.geometry import quat_rotate_vector, wrap_pi

JOINT_NAMES = [f"iiwa_joint_{i}" for i in range(1, 8)]
PARK_POSE = [0.0, -0.6, 0.0, -2.2, 0.0, 0.8, 0.0]  # ista koju salje full_stack
PARK_TIME_SEC = 4

LIDAR_FRAME = "lidar_link"
# Prednji uglovi kucista su tangentne tocke sjene pod +-95.44 deg; malo uze da
# se pojas oko ruba sigurno odbaci.
SELF_OCCLUSION_HALF_ANGLE_DEG = 94.0

DOORWAY_WIDTH_M = 0.85
WIDTH_TOL_M = 0.35
GAP_MIN_M = 0.30
MAX_RANGE_M = 6.0

RETREAT_DISTANCE_M = 0.20  # odmicanje od kvake prije parkiranja ruke
RETREAT_SPEED_MPS = 0.15

ALIGN_SPEED_MPS = 0.22
ALIGN_TOL_M = 0.05
ALIGN_TIMEOUT_SEC = 150.0

DRIVE_SPEED_MPS = 0.22
PASS_CX_M = -0.30  # srediste otvora iza ove tocke znaci da je zid prosao
PASS_MARGIN_M = 0.10  # koliko straznji rub baze prolazi iza ravnine zida
LOST_DOORWAY_STEPS = 20
DRIVE_TIMEOUT_SEC = 120.0

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
    odom = {"xy": None}
    lidar_tf = {"v": None}

    def _on_odom(msg: Odometry):
        odom["xy"] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

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
    node.create_subscription(Odometry, "/odom", _on_odom, 10)

    cmd = {"vx": 0.0, "vy": 0.0}
    stop_flag = {"v": False}

    def publisher_loop():
        """cmd_vel_bridge primjenjuje zadnju primljenu poruku svaki fizicki
        korak i nema failsafe timeout, pa naredbe idu u stalnom ritmu."""
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

    def drive_distance(target_m, speed):
        """Vozi naprijed dok odometrija ne pokaze target_m. Vraca prijedjeni
        put, ili None ako je prekinuto zbog prepreke."""
        p0 = odom["xy"].copy()
        while rclpy.ok():
            if state["too_close"]:
                cmd["vx"] = 0.0
                return None
            done = float(np.linalg.norm(odom["xy"] - p0))
            if done >= target_m:
                cmd["vx"] = 0.0
                return done
            cmd["vx"] = speed
            time.sleep(CONTROL_PERIOD_SEC)
        cmd["vx"] = 0.0
        return None

    # rclpy.spin bez izricitog izvrsavaca koristi globalni, pa bi ga druga faza
    # vrtjela istovremeno iz svoje niti.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    threading.Thread(target=publisher_loop, daemon=True).start()

    node.get_logger().info("Cekam /scan i /odom...")
    t_wait = time.monotonic()
    while (not state["have_scan"] or odom["xy"] is None) and rclpy.ok():
        time.sleep(0.1)
        if time.monotonic() - t_wait > 15.0:
            shutdown("Nema /scan ili /odom - prekidam.", error=True)
            return

    # Gripper se otvara vise puta jer se jedna poruka poslana odmah po
    # stvaranju publishera izgubi dok DDS ne uspostavi vezu.
    node.get_logger().info("Otvaram gripper...")
    for _ in range(20):
        gripper_pub.publish(Float32(data=0.0))
        time.sleep(0.1)

    # Gripper je nakon otpustanja jos oko sipke, a put do parkirne poze vodi
    # kroz nju - zato se prvo odmakne cijela baza.
    node.get_logger().info("Odmicem se od kvake...")
    drive_distance(RETREAT_DISTANCE_M, -RETREAT_SPEED_MPS)
    time.sleep(0.5)

    node.get_logger().info("Parkiram ruku...")
    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES
    pt = JointTrajectoryPoint()
    pt.positions = list(PARK_POSE)
    pt.time_from_start.sec = PARK_TIME_SEC
    traj.points = [pt]
    traj_pub.publish(traj)
    time.sleep(PARK_TIME_SEC + 1.5)

    node.get_logger().info("Poravnavam se s otvorom...")
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

    node.get_logger().info("Prolazim kroz otvor...")
    p_drive_start = odom["xy"].copy()
    t0 = time.monotonic()
    lost = 0
    last_cx = None
    outcome = "vrijeme isteklo"
    while rclpy.ok() and time.monotonic() - t0 < DRIVE_TIMEOUT_SEC:
        if state["too_close"]:
            outcome = "prepreka preblizu"
            break

        cmd["vx"] = DRIVE_SPEED_MPS
        d = state["doorway"]
        if d is None:
            lost += 1
            if lost >= LOST_DOORWAY_STEPS:
                # Rubovi otvora su izasli iz maske samozaklona, sto ne znaci da
                # smo prosli. Zadnji vidjeni cx je udaljenost do ravnine zida;
                # do nje treba jos pola duljine baze i marza.
                need = (last_cx or 0.0) + KMR_LENGTH_M / 2.0 + PASS_MARGIN_M
                node.get_logger().info(
                    f"Otvor izvan vidnog polja na cx={last_cx:+.2f} m - "
                    f"vozim jos {need:.2f} m."
                )
                done = drive_distance(need, DRIVE_SPEED_MPS)
                outcome = "prosao" if done is not None else "prepreka preblizu"
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
    travelled = float(np.linalg.norm(odom["xy"] - p_drive_start))
    node.get_logger().info("=== SAZETAK ===")
    node.get_logger().info(f"  ishod: {outcome}")
    node.get_logger().info(f"  prosao naprijed: {travelled*1000:.0f} mm")
    shutdown()


def main():
    rclpy.init()
    try:
        run()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
