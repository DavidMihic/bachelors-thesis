"""
door_open.py - otvara vrata i istovremeno procjenjuje ogranicenje gibanja,
koristeci samo velicine koje bi imao i stvarni robot: pozu gripera iz
kinematike i procijenjeni wrench na TCP-u. Kut zgloba vrata se ne cita.

Nakon cvrstog hvata pretpostavlja se da su griper i kvaka kruto spojeni po
ORIJENTACIJI. Ta je pretpostavka jaca nego za poziciju: hvat klizi duz
poluge, sto mijenja poziciju gripera u odnosu na kvaku, ali ne i njegov
nagib. Zato se sve izvodi iz zakreta, ne iz pomaka.

Zakret gripera oko vertikale jednak je zakretu vrata:
  - zakret ostaje ~0 uz znatan pomak  -> prizmaticno ogranicenje
  - zakret raste s pomakom            -> rotacijsko ogranicenje
Tip vrata se time prepoznaje sam.

Kod rotacijskih vrata, za tocku kruto vezanu uz vrata vrijedi
    p_i - c = R_z(theta_i) * (p_0 - c)
gdje je c polozaj osi sarke u vodoravnoj ravnini. Preurediti u
    (I - R_z(theta_i)) c = p_i - R_z(theta_i) p_0
i rijesiti po c najmanjim kvadratima preko svih uzoraka. Tangenta luka u
trenutnoj tocki je z x (p - c), i to je smjer sljedeceg koraka - petlja se
time zatvara: procijeni, pomakni se po procjeni, ponovno procijeni.

Faza istrazivanja na pocetku ne pretpostavlja tip vrata nego proba cetiri
vodoravna smjera po jedan korak i bira onaj s najvise gibanja po jedinici
sile.

Preduvjet: gripper drzi kvaku; tcp_wrench_estimator radi.

Pokretanje:
    ros2 run kmr_iiwa_task door_open
"""

import json
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Empty, Float32
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from kmr_iiwa_task.add_door_collision import quat_rotate_vector, COLLISION_OBJECT_ID

JOINT_NAMES = [f"iiwa_joint_{i}" for i in range(1, 8)]

EXPLORE_STEP_M = 0.004  # probni korak po smjeru u fazi istrazivanja
STEP_M = 0.006  # korak tijekom otvaranja
MAX_OPEN_STEPS = 40
FORCE_ABORT_N = 40.0
SETTLE_SEC = 0.4
RETARE_EVERY = 6  # tare vrijedi samo lokalno, treba ga osvjezavati

# Ispod ovog zakreta (I - R) je lose uvjetovana i rjesenje za os eksplodira.
MIN_ANGLE_FOR_FIT_RAD = math.radians(0.15)
PRISMATIC_ANGLE_RAD = math.radians(0.4)
PRISMATIC_MIN_DISP_M = 0.010

LOG_PATH = "/tmp/kmr_door_open.json"
Z_AXIS = np.array([0.0, 0.0, 1.0])


def quat_conj(q):
    x, y, z, w = q
    return [-x, -y, -z, w]


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def yaw_between(q0, q1):
    """Zakret oko vertikale od poze q0 do q1, u radijanima. Z komponenta
    vektora rotacije relativnog kvaterniona - tocno za cistu rotaciju oko z,
    dobra aproksimacija za male zakrete oko drugih osi."""
    qr = quat_mul(q1, quat_conj(q0))
    x, y, z, w = qr
    if w < 0.0:  # uvijek uzmi kraci luk
        x, y, z, w = -x, -y, -z, -w
    v = math.sqrt(x * x + y * y + z * z)
    if v < 1e-12:
        return 0.0
    return 2.0 * math.atan2(v, w) * (z / v)


def fit_rotation_center(p0_xy, samples):
    """samples = [(theta_i, p_i_xy), ...]. Vrati (c_xy, rezidual) ili
    (None, None) ako nema dovoljno upotrebljivih uzoraka."""
    rows, rhs = [], []
    for th, p in samples:
        if abs(th) < MIN_ANGLE_FOR_FIT_RAD:
            continue
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        rows.append(np.eye(2) - R)
        rhs.append(p - R @ p0_xy)
    if len(rows) < 2:
        return None, None
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    residual = float(np.linalg.norm(A @ sol - b) / math.sqrt(len(b)))
    return sol, residual


def main():
    rclpy.init()
    node = Node("door_open")
    cb = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name="base_link",
        end_effector_name="gripper_tcp",
        group_name="iiwa_arm",
        callback_group=cb,
    )
    moveit2.max_velocity = 0.05
    moveit2.max_acceleration = 0.05

    # Wrench i griper na zasebnom nodu i izvrsavacu - MoveIt2-ova pozadinska
    # aktivnost inace gladuje obicne pretplate na istom nodu.
    sensor_node = Node("door_open_sensor")
    wrench = {"f": None}

    def _on_wrench(msg: WrenchStamped):
        wrench["f"] = np.array(
            [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]
        )

    sensor_node.create_subscription(
        WrenchStamped, "/estimation/tcp_wrench", _on_wrench, 10
    )
    tare_pub = sensor_node.create_publisher(Empty, "/estimation/tare", 10)
    gripper_pub = sensor_node.create_publisher(Float32, "/gripper_cmd", 10)

    sensor_exec = SingleThreadedExecutor()
    sensor_exec.add_node(sensor_node)
    threading.Thread(target=sensor_exec.spin, daemon=True).start()

    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    def tcp_pose():
        for _ in range(50):
            try:
                t = tf_buffer.lookup_transform(
                    "base_link", "gripper_tcp", rclpy.time.Time()
                )
                tr, r = t.transform.translation, t.transform.rotation
                return np.array([tr.x, tr.y, tr.z]), [r.x, r.y, r.z, r.w]
            except (LookupException, ConnectivityException, ExtrapolationException):
                time.sleep(0.1)
        return None, None

    def goto(p, q):
        moveit2.move_to_pose(position=list(p), quat_xyzw=list(q), cartesian=True)
        moveit2.wait_until_executed()

    node.get_logger().info("Cekam TF i wrench...")
    p_start, q_start = tcp_pose()
    if p_start is None:
        node.get_logger().error("Nema TF-a za gripper_tcp - prekidam.")
        rclpy.shutdown()
        return
    while wrench["f"] is None and rclpy.ok():
        time.sleep(0.1)

    node.get_logger().info("Stiscem griper i drzim ga zatvorenim.")
    gripper_pub.publish(Float32(data=1.0))
    time.sleep(1.0)

    node.get_logger().info("Micem door_panel kolizijski objekt - zastita je sila.")
    moveit2.remove_collision_object(COLLISION_OBJECT_ID)
    time.sleep(0.5)

    # Osi iz orijentacije gripera (gripper.xacro: +Z prsti/prilaz,
    # +X zatvaranje, +Y duz sipke).
    ax_approach = np.array(quat_rotate_vector(q_start, [0.0, 0.0, 1.0]))
    ax_closing = np.array(quat_rotate_vector(q_start, [1.0, 0.0, 0.0]))

    log = {"explore": {}, "steps": []}

    # ---------- Faza 1: istrazivanje, bez pretpostavke o tipu vrata ----------
    node.get_logger().info("=== Faza istrazivanja ===")
    best = (None, -1e9)
    for name, axis in (
        ("approach+", ax_approach),
        ("approach-", -ax_approach),
        ("closing+", ax_closing),
        ("closing-", -ax_closing),
    ):
        goto(p_start, q_start)
        time.sleep(SETTLE_SEC)
        tare_pub.publish(Empty())
        time.sleep(SETTLE_SEC)

        goto(p_start + axis * EXPLORE_STEP_M, q_start)
        time.sleep(SETTLE_SEC)
        p, _ = tcp_pose()
        f = wrench["f"]
        if p is None or f is None:
            continue
        achieved = float(np.dot(p - p_start, axis))
        f_along = float(np.dot(f, axis))
        score = achieved / (1.0 + abs(f_along))
        log["explore"][name] = {
            "achieved_m": achieved,
            "force_N": f_along,
            "score": score,
        }
        node.get_logger().info(
            f"  {name}: pomak {achieved*1000:+.2f}mm, sila {f_along:+.1f}N, ocjena {score:.5f}"
        )
        if score > best[1]:
            best = (axis.copy(), score)

    goto(p_start, q_start)
    time.sleep(SETTLE_SEC)
    if best[0] is None:
        node.get_logger().error("Istrazivanje nije dalo nijedan smjer - prekidam.")
        rclpy.shutdown()
        return
    direction = best[0]
    node.get_logger().info(f"Odabran smjer otvaranja: {np.round(direction,3)}")

    # ---------- Faza 2: otvaranje uz procjenu ----------
    node.get_logger().info("=== Otvaranje uz procjenu ogranicenja ===")
    p_ref, q_ref = tcp_pose()
    tare_pub.publish(Empty())
    time.sleep(SETTLE_SEC)

    samples = []  # (theta, p_xy) u odnosu na p_ref/q_ref
    center = None
    verdict = "NEPOZNATO"

    for step in range(1, MAX_OPEN_STEPS + 1):
        target = None
        p_now, q_now = tcp_pose()
        if p_now is None:
            break

        if center is not None:
            # tangenta luka u trenutnoj tocki, predznak prema dosadasnjem smjeru
            r = np.array([p_now[0] - center[0], p_now[1] - center[1], 0.0])
            tang = np.cross(Z_AXIS, r)
            n = np.linalg.norm(tang)
            if n > 1e-9:
                tang = tang / n
                if np.dot(tang, direction) < 0:
                    tang = -tang
                direction = tang
        target = p_now + direction * STEP_M

        goto(target, q_now)
        time.sleep(SETTLE_SEC)

        p_new, q_new = tcp_pose()
        f = wrench["f"]
        if p_new is None or f is None:
            break

        theta = yaw_between(q_ref, q_new)
        disp = float(np.linalg.norm(p_new - p_ref))
        f_along = float(np.dot(f, direction))
        samples.append((theta, np.array([p_new[0], p_new[1]])))

        center, resid = fit_rotation_center(np.array([p_ref[0], p_ref[1]]), samples)
        radius = (
            float(np.linalg.norm(np.array([p_new[0], p_new[1]]) - center))
            if center is not None
            else None
        )

        if abs(theta) < PRISMATIC_ANGLE_RAD and disp > PRISMATIC_MIN_DISP_M:
            verdict = "PRIZMATICNO"
            center = None
        elif center is not None:
            verdict = "ROTACIJSKO"

        node.get_logger().info(
            f"korak {step:2d}: pomak {disp*1000:6.1f}mm  zakret {math.degrees(theta):+6.2f}deg  "
            f"sila {f_along:+6.1f}N  -> {verdict}"
            + (
                f"  os=({center[0]:.3f},{center[1]:.3f}) r={radius:.3f}m rez={resid*1000:.1f}mm"
                if center is not None
                else ""
            )
        )
        log["steps"].append(
            {
                "step": step,
                "disp_m": disp,
                "theta_rad": theta,
                "force_along_N": f_along,
                "verdict": verdict,
                "center_xy": (
                    None if center is None else [float(center[0]), float(center[1])]
                ),
                "radius_m": radius,
                "fit_residual_m": resid,
            }
        )

        if abs(f_along) > FORCE_ABORT_N:
            node.get_logger().warn(
                f"Sila {f_along:.1f}N preko praga - prekidam otvaranje."
            )
            break
        if step % RETARE_EVERY == 0:
            # tare vrijedi samo lokalno; osvjezi ga i referentnu pozu
            tare_pub.publish(Empty())
            time.sleep(SETTLE_SEC)

    node.get_logger().info("=== SAZETAK ===")
    node.get_logger().info(f"  zakljucak: {verdict}")
    if log["steps"]:
        last = log["steps"][-1]
        node.get_logger().info(
            f"  ukupno: pomak {last['disp_m']*1000:.1f}mm, zakret {math.degrees(last['theta_rad']):.2f}deg"
        )
        if last["center_xy"]:
            node.get_logger().info(
                f"  procijenjena os: {np.round(last['center_xy'],3)}, radijus {last['radius_m']:.3f}m"
            )

    with open(LOG_PATH, "w") as fh:
        json.dump(log, fh, indent=2)
    node.get_logger().info(f"Detalji u {LOG_PATH}")

    sensor_exec.shutdown()
    sensor_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
