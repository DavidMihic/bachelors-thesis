"""
door_pull_base_arc.py - otvara zakretna vrata gibanjem baze po luku, uz
ukocenu ruku i zatvoren gripper.

Kod zakretnih vrata kvaka se giba po kruznici oko sarke i pritom se zakrece
zajedno s krilom. Baza zato mora istovremeno translatirati po tangenti i
rotirati za isti kut. Buduci da se giba kao kruto tijelo zajedno s vratima,
oboje odredjuje jedan parametar - zakret vrata - pa za mali zakret dtheta
baza rotira za dtheta i translatira po tangenti za r * dtheta.

Rotaciju nosi baza, a ne ruka. Ruka nema doseg za puni zakret vrata iz
uhvacene poze, a cim se pomakne, mijenja se konfiguracija zglobova i tare
procjenitelja sile prestaje vrijediti. Rotacija baze oko vertikale ne mijenja
odnos ruke prema gravitaciji, pa tare prezivi cijelu voznju.

Rotacija mora postojati od PRVOG trenutka. Dok baza samo translatira, poluga
se uvrce unutar hvata i sila naraste na vise stotina njutna prije nego se
vrata primjetno otvore.

OS SARKE se odredjuje geometrijski, bez probnog poteza. Ranija zamisao je
bila procijeniti je iz zakreta gripera, kao u door_open.py, ali ondje se
pomice RUKA - ovdje je ruka ukocena, pa je TF base_link -> gripper_tcp
konstantan po definiciji i zakret gripera je uvijek nula bez obzira koliko
se vrata otvore.

Umjesto toga: door_tag_center je na sredini krila, krilo je siroko 0.85 m, pa
je sarka 0.425 m od taga u ravnini vrata, na suprotnu stranu od kvake.
Smjer se projicira na ravninu vrata (normala je os prilaza gripera) jer kvaka
strsi iz plohe, pa bi bez projekcije procjena promasila oko 85 mm.

Preciznost nije kriticna: simulacija pokazuje da i 100 mm greske u osi samo
usporava napredak (80 -> 71 stupnjeva), jer vrata sama drze geometriju.

Tangenta se racuna iz svjeze ocitane poze taga, ne integrira, pa se greska ne
akumulira.

Naredbe salje zasebna nit u stalnom ritmu. cmd_vel_bridge primjenjuje zadnju
primljenu poruku svaki fizicki korak i nema failsafe timeout, pa neujednacen
ritam znaci trzajno gibanje i skokove sile.

Preduvjet: door_task_node je uhvatio kvaku (vertical_bar:=false) i miruje;
tcp_wrench_estimator radi.

Pokretanje:
    ros2 run kmr_iiwa_task door_pull_base_arc
"""

import json
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, WrenchStamped
from rclpy.node import Node
from std_msgs.msg import Empty, Float32
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# --- Geometrija vrata (iz revolute_door.urdf) ---
TAG_TO_HINGE_M = 0.425  # door_tag_center je na sredini krila sirokog 0.85

# --- Profil ---
CRUISE_OMEGA_RADPS = 0.25  # kutna brzina vrata
ACCEL_SEC = 2.0  # bez rampe kretanje iz mirovanja udara u krutu vezu
DECEL_SEC = 1.5
TARGET_ANGLE_RAD = math.radians(60.0)

PUBLISH_PERIOD_SEC = 0.02
CONTROL_PERIOD_SEC = 0.05

# --- Sanity granice procjene ---
MIN_RADIUS_M = 0.30
MAX_RADIUS_M = 1.20

# --- Sigurnosni prekidi ---
FORCE_ABORT_N = 700.0
FORCE_SPIKE_STEPS = 3
STALE_ABORT_STEPS = 10

LOG_PATH = "/tmp/kmr_door_pull_arc.json"


def quat_axes(q):
    """Vrati (x_os, z_os) orijentacije q u base_link okviru."""
    x, y, z, w = q
    x_axis = np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ]
    )
    z_axis = np.array(
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ]
    )
    return x_axis, z_axis


def main():
    rclpy.init()
    node = Node("door_pull_base_arc")

    wrench = {"f": None}

    def _on_wrench(msg: WrenchStamped):
        wrench["f"] = np.array(
            [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]
        )

    node.create_subscription(WrenchStamped, "/estimation/tcp_wrench", _on_wrench, 10)
    cmd_vel_pub = node.create_publisher(Twist, "/cmd_vel", 10)
    gripper_pub = node.create_publisher(Float32, "/gripper_cmd", 10)
    tare_pub = node.create_publisher(Empty, "/estimation/tare", 10)

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    def lookup(frame):
        try:
            t = tf_buffer.lookup_transform("base_link", frame, rclpy.time.Time())
            tr, r = t.transform.translation, t.transform.rotation
            stamp = rclpy.time.Time.from_msg(t.header.stamp).nanoseconds
            return np.array([tr.x, tr.y, tr.z]), (r.x, r.y, r.z, r.w), stamp
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None, None, None

    cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    stop_flag = {"v": False}

    def publisher_loop():
        while rclpy.ok() and not stop_flag["v"]:
            gripper_pub.publish(Float32(data=1.0))
            tw = Twist()
            tw.linear.x = cmd["vx"]
            tw.linear.y = cmd["vy"]
            tw.angular.z = cmd["wz"]
            cmd_vel_pub.publish(tw)
            time.sleep(PUBLISH_PERIOD_SEC)

    def stop_base():
        cmd["vx"] = cmd["vy"] = cmd["wz"] = 0.0

    node.get_logger().info("Cekam TF i wrench...")
    p_tcp, q_tcp = None, None
    while (p_tcp is None or wrench["f"] is None) and rclpy.ok():
        p_tcp, q_tcp, _ = lookup("gripper_tcp")
        time.sleep(0.1)

    p_tag, _, _ = lookup("door_tag_center")
    while p_tag is None and rclpy.ok():
        p_tag, _, _ = lookup("door_tag_center")
        time.sleep(0.1)

    gripper_pub.publish(Float32(data=1.0))
    time.sleep(1.0)
    tare_pub.publish(Empty())
    time.sleep(0.5)
    f0 = wrench["f"]
    node.get_logger().info(
        f"Sila nakon tare, prije gibanja: {float(np.linalg.norm(f0)):.1f} N"
    )

    # --- Os sarke, geometrijski ---
    _, z_axis = quat_axes(q_tcp)
    normal = np.array([z_axis[0], z_axis[1]])
    n = np.linalg.norm(normal)
    if n < 1e-6:
        node.get_logger().error("Os prilaza je vertikalna - ne mogu odrediti normalu.")
        rclpy.shutdown()
        return
    normal = normal / n

    d = p_tag[:2] - p_tcp[:2]
    d = d - np.dot(d, normal) * normal  # projekcija na ravninu vrata
    nd = np.linalg.norm(d)
    if nd < 1e-6:
        node.get_logger().error("Kvaka i tag se poklapaju u ravnini vrata - prekidam.")
        rclpy.shutdown()
        return
    center = p_tag[:2] + TAG_TO_HINGE_M * (d / nd)
    radius = float(np.linalg.norm(p_tcp[:2] - center))

    if not (MIN_RADIUS_M <= radius <= MAX_RADIUS_M):
        node.get_logger().error(
            f"Procijenjeni radijus {radius:.3f} m je izvan raspona "
            f"[{MIN_RADIUS_M}, {MAX_RADIUS_M}] - prekidam."
        )
        rclpy.shutdown()
        return

    # Smjer otvaranja: kvaka se pri otvaranju giba prema plohi vrata, dakle
    # duz osi prilaza gripera. Predznak rotacije bira se tako da tangenta
    # pokazuje u tom smjeru.
    rad_vec = p_tcp[:2] - center
    tang = np.array([-rad_vec[1], rad_vec[0]]) / radius
    sign = 1.0 if float(np.dot(tang, normal)) > 0.0 else -1.0

    node.get_logger().info(
        f"Os sarke: ({center[0]:.3f}, {center[1]:.3f}), radijus {radius:.3f} m, "
        f"smjer {sign:+.0f}"
    )

    log = {
        "center_xy": [float(center[0]), float(center[1])],
        "radius_m": radius,
        "sign": sign,
        "run": [],
    }

    # --- Voznja po luku ---
    total_time = (
        ACCEL_SEC
        + DECEL_SEC
        + max(
            0.0,
            (TARGET_ANGLE_RAD - CRUISE_OMEGA_RADPS * (ACCEL_SEC + DECEL_SEC) * 0.5)
            / CRUISE_OMEGA_RADPS,
        )
    )
    node.get_logger().info(
        f"Luk do {math.degrees(TARGET_ANGLE_RAD):.0f} deg, trajanje {total_time:.1f} s"
    )

    threading.Thread(target=publisher_loop, daemon=True).start()

    t_start = time.monotonic()
    last_stamp = None
    stale_count = 0
    force_spike_count = 0
    aborted = None

    while rclpy.ok():
        now = time.monotonic()
        elapsed = now - t_start
        if elapsed >= total_time:
            break

        p_tag_now, _, stamp = lookup("door_tag_center")
        f = wrench["f"]
        fmag = float(np.linalg.norm(f)) if f is not None else 0.0

        if stamp is not None and stamp == last_stamp:
            stale_count += 1
        else:
            stale_count = 0
            last_stamp = stamp

        # Trapez po kutnoj brzini
        up = min(1.0, elapsed / ACCEL_SEC) if ACCEL_SEC > 0 else 1.0
        down = min(1.0, (total_time - elapsed) / DECEL_SEC) if DECEL_SEC > 0 else 1.0
        omega = CRUISE_OMEGA_RADPS * min(up, down) * sign

        # Os sarke se osvjezava iz svjeze poze taga. Sve je izrazeno u
        # base_link, a taj se okvir giba, pa pocetna procjena zastarijeva.
        if p_tag_now is not None:
            d_now = p_tag_now[:2] - p_tcp[:2]
            d_now = d_now - np.dot(d_now, normal) * normal
            nd_now = np.linalg.norm(d_now)
            if nd_now > 1e-6:
                center = p_tag_now[:2] + TAG_TO_HINGE_M * (d_now / nd_now)

        cmd["vx"] = float(omega * center[1])
        cmd["vy"] = float(-omega * center[0])
        cmd["wz"] = float(omega)

        log["run"].append(
            {
                "t": elapsed,
                "omega": omega,
                "vx": cmd["vx"],
                "vy": cmd["vy"],
                "center_xy": [float(center[0]), float(center[1])],
                "force_N": fmag,
            }
        )

        if stale_count > STALE_ABORT_STEPS:
            aborted = "TF door_tag_center zastario"
            break
        if fmag > FORCE_ABORT_N:
            force_spike_count += 1
        else:
            force_spike_count = 0
        if force_spike_count >= FORCE_SPIKE_STEPS:
            aborted = f"sila {fmag:.0f}N kroz vise uzoraka"
            break

        time.sleep(CONTROL_PERIOD_SEC)

    stop_base()
    time.sleep(0.5)
    stop_flag["v"] = True
    time.sleep(0.1)
    cmd_vel_pub.publish(Twist())
    time.sleep(1.0)

    node.get_logger().info("=== SAZETAK ===")
    node.get_logger().info(
        f"  os sarke: ({center[0]:.3f}, {center[1]:.3f}), radijus {radius:.3f} m"
    )
    if log["run"]:
        forces = [r["force_N"] for r in log["run"]]
        node.get_logger().info(
            f"  sila: prosjek {sum(forces)/len(forces):.0f} N, najveca {max(forces):.0f} N"
        )
        node.get_logger().info(f"  koraka: {len(log['run'])}")
    if aborted:
        node.get_logger().warn(f"  prekinuto: {aborted}")
    node.get_logger().info(
        "  PROVJERI zakret u Isaac Simu: hinge_joint -> Raw USD properties -> "
        "state:angular:physics:position"
    )

    log["summary"] = {"aborted": aborted}
    with open(LOG_PATH, "w") as fh:
        json.dump(log, fh, indent=2)
    node.get_logger().info(f"Detalji u {LOG_PATH}")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
