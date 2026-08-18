"""
door_probe.py - procjena ogranicenja gibanja i krutosti okoline iz mjerenja
sile i momenta, dok gripper vec drzi kvaku.

Iz uhvacene poze rade se mali, silom-nadzirani pomaci u sest smjerova (+-3
osi okvira gripera). Za svaki smjer biljezi se ostvareni pomak i porast sile,
a smjerovi se klasificiraju po prividnoj krutosti (sila po ostvarenom
pomaku). Omjer ostvareno/naredjeno nije upotrebljiv kao kriterij jer robot s
krutim pozicijskim upravljanjem progura kamo mu se kaze, dok krutost razdvaja
smjerove kroz tri reda velicine.

Osi se uzimaju iz orijentacije gripera (gripper.xacro: +Z prsti/prilaz, +X
zatvaranje, +Y duz sipke), a ne iz door_tag_center, koji ima oko 20 stupnjeva
odstupanja - dovoljno da guranje "duz slobodne osi" dobije veliku komponentu
u plocu vrata i izmjeri se kao krutost koja ne postoji.

Interpretacija krutosti: pogoni zglobova imaju krutost 100000, pa je u
ogranicenim smjerovima izmjerena vrijednost dominantno krutost robota i
hvata, ne okoline. Okolina se moze procijeniti samo ako je mekša od njih, pa
je rezultat u tim smjerovima bolje citati kao "ovdje nema gibanja" nego kao
apsolutnu krutost vrata.

door_panel kolizijski objekt se na pocetku mice, jer probing namjerno gura
prema vratima. Zastita od sudara je sila, ne kolizijski model.

Preduvjet: gripper drzi kvaku i ostaje zatvoren; tcp_wrench_estimator radi.

Pokretanje:
    ros2 run kmr_iiwa_task door_probe
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
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from kmr_iiwa_task.add_door_collision import quat_rotate_vector, COLLISION_OBJECT_ID

JOINT_NAMES = [f"iiwa_joint_{i}" for i in range(1, 8)]

STEP_M = 0.003  # velicina jednog koraka
MAX_STEPS = 4  # najvise koraka po smjeru
FORCE_ABORT_N = 100.0  # prekid smjera cim sila predje ovo
SETTLE_SEC = 0.4  # da se sila smiri nakon koraka
FREE_STIFFNESS_N_PER_M = 1000.0  # ispod = slobodan smjer
BLOCKED_STIFFNESS_N_PER_M = 5000.0  # iznad = ogranicen

LOG_PATH = "/tmp/kmr_door_probe.json"


def main():
    rclpy.init()
    node = Node("door_probe")
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

    # Wrench na zasebnom nodu i izvrsavacu - MoveIt2-ova pozadinska aktivnost
    # inace gladuje obicne pretplate na istom nodu.
    sensor_node = Node("door_probe_sensor")
    wrench = {"f": None}

    def _on_wrench(msg: WrenchStamped):
        wrench["f"] = np.array(
            [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]
        )

    sensor_node.create_subscription(
        WrenchStamped, "/estimation/tcp_wrench", _on_wrench, 10
    )
    tare_pub = sensor_node.create_publisher(Empty, "/estimation/tare", 10)

    sensor_exec = SingleThreadedExecutor()
    sensor_exec.add_node(sensor_node)
    sensor_thread = threading.Thread(target=sensor_exec.spin, daemon=True)
    sensor_thread.start()

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
                tr = t.transform.translation
                r = t.transform.rotation
                return np.array([tr.x, tr.y, tr.z]), [r.x, r.y, r.z, r.w]
            except (LookupException, ConnectivityException, ExtrapolationException):
                time.sleep(0.1)
        return None, None

    node.get_logger().info("Cekam TF i wrench...")

    start_pos, grasp_quat = tcp_pose()
    if start_pos is None:
        node.get_logger().error("Nema TF-a za gripper_tcp - prekidam.")
        rclpy.shutdown()
        return

    # Osi iz orijentacije gripera, ne iz door_tag_center - vidi docstring.
    axes = {
        "approach": np.array(quat_rotate_vector(grasp_quat, [0.0, 0.0, 1.0])),
        "closing": np.array(quat_rotate_vector(grasp_quat, [1.0, 0.0, 0.0])),
        "along_bar": np.array(quat_rotate_vector(grasp_quat, [0.0, 1.0, 0.0])),
    }

    while wrench["f"] is None and rclpy.ok():
        time.sleep(0.1)

    node.get_logger().info(
        "Micem door_panel kolizijski objekt - zastita je sila, ne kolizijski model."
    )
    moveit2.remove_collision_object(COLLISION_OBJECT_ID)
    time.sleep(0.5)

    node.get_logger().info(f"Polazna TCP pozicija: {np.round(start_pos, 4)}")

    def goto(p, quat):
        moveit2.move_to_pose(position=list(p), quat_xyzw=list(quat), cartesian=True)
        moveit2.wait_until_executed()

    results = {}
    for name, axis in axes.items():
        for sign in (+1.0, -1.0):
            label = f"{name}{'+' if sign > 0 else '-'}"
            direction = sign * axis

            goto(start_pos, grasp_quat)
            time.sleep(SETTLE_SEC)
            tare_pub.publish(Empty())
            time.sleep(SETTLE_SEC)

            samples = []
            aborted = False
            for step in range(1, MAX_STEPS + 1):
                target = start_pos + direction * (STEP_M * step)
                goto(target, grasp_quat)
                time.sleep(SETTLE_SEC)

                actual, _ = tcp_pose()
                f = wrench["f"]
                if actual is None or f is None:
                    break
                achieved = float(np.dot(actual - start_pos, direction))
                f_along = float(np.dot(f, direction))
                samples.append(
                    {
                        "commanded_m": STEP_M * step,
                        "achieved_m": achieved,
                        "force_along_N": f_along,
                        "force_norm_N": float(np.linalg.norm(f)),
                    }
                )
                node.get_logger().info(
                    f"  {label} korak {step}: naredjeno {STEP_M*step*1000:.1f}mm, "
                    f"ostvareno {achieved*1000:+.2f}mm, sila {f_along:+.1f}N"
                )
                if abs(f_along) > FORCE_ABORT_N:
                    node.get_logger().warn(
                        f"  {label}: sila preko praga - prekid smjera."
                    )
                    aborted = True
                    break

            if samples:
                last = samples[-1]
                ratio = (
                    last["achieved_m"] / last["commanded_m"]
                    if last["commanded_m"]
                    else 0.0
                )

                # Prividna krutost: sila po ostvarenom pomaku. Glavni kriterij
                # klasifikacije; omjer se biljezi samo za zapis.
                stiffness = None
                if abs(last["achieved_m"]) > 1e-4:
                    stiffness = last["force_along_N"] / last["achieved_m"]

                if stiffness is None or abs(stiffness) > BLOCKED_STIFFNESS_N_PER_M:
                    verdict = "OGRANICEN"
                elif abs(stiffness) < FREE_STIFFNESS_N_PER_M:
                    verdict = "SLOBODAN"
                else:
                    verdict = "DJELOMICAN"

                results[label] = {
                    "verdict": verdict,
                    "ratio": ratio,
                    "aborted": aborted,
                    "stiffness_N_per_m": stiffness,
                    "samples": samples,
                }

    goto(start_pos, grasp_quat)

    node.get_logger().info("=== SAZETAK ===")
    for label, r in results.items():
        node.get_logger().info(
            f"  {label:12s} {r['verdict']:12s} omjer={r['ratio']:+.2f}"
        )

    with open(LOG_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    node.get_logger().info(f"Detalji spremljeni u {LOG_PATH}")

    sensor_exec.shutdown()
    sensor_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
