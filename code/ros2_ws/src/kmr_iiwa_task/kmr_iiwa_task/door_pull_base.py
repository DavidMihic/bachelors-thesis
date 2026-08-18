"""
door_pull_base.py - otvara klizna vrata gibanjem baze po trapeznom profilu
brzine, uz ukocenu ruku i zatvoren gripper.

Hod vrata (0.8 m) premasuje doseg ruke, pa gibanje nosi baza, a ruka samo
prenosi silu. Ruku ne treba posebno ukociti - arm_controller drzi zadnju
tocku trajektorije s krutoscu pogona 100000, pa je vec kruta dok joj se nista
ne salje. Time je i gravitacijski moment konstantan, pa tare procjenitelja
sile vrijedi kroz cijelu voznju.

Naredbe salje zasebna nit u stalnom ritmu. cmd_vel_bridge primjenjuje zadnju
primljenu poruku svaki fizicki korak i nema failsafe timeout, pa neujednacen
ritam znaci trzajno gibanje, a trzaj kroz krutu vezu daje skokove sile od
vise stotina njutna.

Brzina se ne regulira po sili. Sila uvijek naraste kad baza krece, sto je
normalno pri vucenju, pa je svaki takav regulator uvodio zastajkivanje. Vrata
k tome imaju gotovo konstantan otpor (jointFriction 2, pogon 100 N/m). Sila
se prati samo kao sigurnosni prekid.

Smjer otvaranja se odredjuje geometrijski: kvaka je blize onom rubu krila
prema kojem se vrata otvaraju, a door_tag_center je na sredini krila.

Prijedjeni put je integracija zadane brzine i vise je orijentacijski nego
tocan. Stvarno otvaranje provjeri u Isaac Simu: slide_joint -> Raw USD
properties -> state:linear:physics:position.

Preduvjet: door_task_node je uhvatio kvaku i miruje; tcp_wrench_estimator
radi.

Pokretanje:
    ros2 run kmr_iiwa_task door_pull_base
"""

import json
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, WrenchStamped
from rclpy.node import Node
from std_msgs.msg import Empty, Float32
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# --- Trapezni profil ---
CRUISE_SPEED_MPS = 0.22  # iznad ovoga sila naglo raste
ACCEL_SEC = 1.5
DECEL_SEC = 1.5
TARGET_DISTANCE_M = 0.50

# --- Slanje naredbi ---
PUBLISH_PERIOD_SEC = 0.02  # 50 Hz, iz zasebne niti

# --- Sigurnosni prekidi (NE regulacija) ---
FORCE_ABORT_N = 600.0
FORCE_SPIKE_STEPS = 3
LAG_ABORT_M = 0.08
STALE_ABORT_STEPS = 10

CONTROL_PERIOD_SEC = 0.05

LOG_PATH = "/tmp/kmr_door_pull_base.json"


def quat_rotate_vector(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array(
        [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ]
    )


def profile_duration():
    """Trajanje trapeza za TARGET_DISTANCE_M. Rampe prijedju po pola svoje
    pune brzine, pa zajedno daju (ACCEL+DECEL)/2 * v puta."""
    ramp_distance = CRUISE_SPEED_MPS * (ACCEL_SEC + DECEL_SEC) * 0.5
    if TARGET_DISTANCE_M <= ramp_distance:
        return 2.0 * (TARGET_DISTANCE_M / CRUISE_SPEED_MPS)
    cruise_time = (TARGET_DISTANCE_M - ramp_distance) / CRUISE_SPEED_MPS
    return ACCEL_SEC + cruise_time + DECEL_SEC


def trapezoid_speed(elapsed, total):
    """Trapezni profil po vremenu. Usporavanje po preostaloj udaljenosti daje
    eksponencijalni rep koji nikad ne dosegne cilj."""
    if elapsed >= total:
        return 0.0
    up = min(1.0, elapsed / ACCEL_SEC) if ACCEL_SEC > 0 else 1.0
    down = min(1.0, (total - elapsed) / DECEL_SEC) if DECEL_SEC > 0 else 1.0
    return CRUISE_SPEED_MPS * min(up, down)


def main():
    rclpy.init()
    node = Node("door_pull_base")

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
        """Vrati (pozicija, kvaternion, stamp_ns) ili (None, None, None)."""
        try:
            t = tf_buffer.lookup_transform("base_link", frame, rclpy.time.Time())
            tr, r = t.transform.translation, t.transform.rotation
            stamp = rclpy.time.Time.from_msg(t.header.stamp).nanoseconds
            return np.array([tr.x, tr.y, tr.z]), (r.x, r.y, r.z, r.w), stamp
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None, None, None

    # --- Nit koja salje naredbe u stalnom ritmu ---
    cmd = {"speed": 0.0, "dir": np.zeros(3)}
    stop_flag = {"v": False}
    publish_count = {"n": 0}

    def publisher_loop():
        while rclpy.ok() and not stop_flag["v"]:
            gripper_pub.publish(Float32(data=1.0))
            tw = Twist()
            tw.linear.x = float(cmd["dir"][0]) * cmd["speed"]
            tw.linear.y = float(cmd["dir"][1]) * cmd["speed"]
            cmd_vel_pub.publish(tw)
            publish_count["n"] += 1
            time.sleep(PUBLISH_PERIOD_SEC)

    node.get_logger().info("Cekam TF i wrench...")
    p_tcp, q_tcp = None, None
    while (p_tcp is None or wrench["f"] is None) and rclpy.ok():
        p_tcp, q_tcp, _ = lookup("gripper_tcp")
        time.sleep(0.1)

    gripper_pub.publish(Float32(data=1.0))
    time.sleep(1.0)
    tare_pub.publish(Empty())
    time.sleep(0.5)

    # Os klizanja: os zatvaranja prstiju (lokalni X gripera), projicirana na
    # vodoravnu ravninu. Kod okomite sipke ta os lezi u ravnini vrata.
    x_axis = quat_rotate_vector(q_tcp, [1.0, 0.0, 0.0])
    slide = np.array([x_axis[0], x_axis[1], 0.0])
    n = np.linalg.norm(slide)
    if n < 1e-6:
        node.get_logger().error(
            "Os zatvaranja je gotovo vertikalna - ne mogu odrediti smjer klizanja."
        )
        rclpy.shutdown()
        return
    slide = slide / n
    node.get_logger().info(f"Os klizanja: {np.round(slide, 3)}")

    log = {"slide_axis": slide.tolist(), "run": []}

    p_tag_start, _, tag_stamp = lookup("door_tag_center")
    if p_tag_start is None:
        node.get_logger().error("Nema door_tag_center - ne mogu odrediti smjer.")
        rclpy.shutdown()
        return
    p_tag_start = p_tag_start[:2].copy()
    offset = float(np.dot(p_tcp[:2] - p_tag_start, slide[:2]))
    sign = 1.0 if offset > 0.0 else -1.0
    direction = sign * slide
    log["open_direction_sign"] = sign
    node.get_logger().info(
        f"Smjer otvaranja iz geometrije: {sign:+.0f} "
        f"(kvaka je {offset*1000:+.0f} mm od sredine krila duz osi klizanja)"
    )

    # --- Voznja ---
    total_time = profile_duration()
    node.get_logger().info(
        f"Trapez: rampa {ACCEL_SEC}s -> {CRUISE_SPEED_MPS} m/s -> "
        f"{TARGET_DISTANCE_M*1000:.0f} mm, trajanje {total_time:.1f} s"
    )

    cmd["dir"] = direction
    threading.Thread(target=publisher_loop, daemon=True).start()

    travelled = 0.0
    t_start = time.monotonic()
    t_last = t_start
    last_tag_stamp = tag_stamp
    stale_count = 0
    force_spike_count = 0
    max_loop_period = 0.0
    aborted = None

    while rclpy.ok():
        f = wrench["f"]
        fmag = float(np.linalg.norm(f)) if f is not None else 0.0

        p_tag3, _, tag_stamp = lookup("door_tag_center")
        p_tag = None if p_tag3 is None else p_tag3[:2].copy()

        if tag_stamp is not None and tag_stamp == last_tag_stamp:
            stale_count += 1
        else:
            stale_count = 0
            last_tag_stamp = tag_stamp

        lag = (
            float(np.linalg.norm(p_tag - p_tag_start))
            if (p_tag is not None and p_tag_start is not None)
            else 0.0
        )

        now = time.monotonic()
        elapsed = now - t_start
        if elapsed >= total_time:
            break

        speed = trapezoid_speed(elapsed, total_time)
        cmd["speed"] = speed  # nit sama salje u stalnom ritmu

        loop_period = now - t_last
        max_loop_period = max(max_loop_period, loop_period)
        travelled += speed * loop_period
        t_last = now

        log["run"].append(
            {
                "travelled_m": travelled,
                "speed_mps": speed,
                "tag_lag_m": lag,
                "force_N": fmag,
                "stale_count": stale_count,
                "loop_period_s": loop_period,
                "tag_pos_xy": (
                    None if p_tag is None else [float(p_tag[0]), float(p_tag[1])]
                ),
            }
        )

        if stale_count > STALE_ABORT_STEPS:
            aborted = "detekcija taga zastarjela - guramo naslijepo"
            break
        if lag > LAG_ABORT_M:
            aborted = f"zaostajanje {lag*1000:.0f}mm - hvat je vjerojatno popustio"
            break
        if fmag > FORCE_ABORT_N:
            force_spike_count += 1
        else:
            force_spike_count = 0
        if force_spike_count >= FORCE_SPIKE_STEPS:
            aborted = f"sila {fmag:.0f}N kroz vise uzoraka"
            break

        time.sleep(CONTROL_PERIOD_SEC)

    cmd["speed"] = 0.0
    time.sleep(0.2)
    stop_flag["v"] = True
    time.sleep(0.1)
    cmd_vel_pub.publish(Twist())
    time.sleep(1.0)

    p_tag_end3, _, _ = lookup("door_tag_center")
    p_tag_end = None if p_tag_end3 is None else p_tag_end3[:2]
    final_lag = (
        float(np.linalg.norm(p_tag_end - p_tag_start))
        if (p_tag_end is not None and p_tag_start is not None)
        else None
    )
    opened = travelled - (final_lag or 0.0)
    publish_hz = publish_count["n"] / max(1e-6, time.monotonic() - t_start)

    node.get_logger().info("=== SAZETAK ===")
    node.get_logger().info(f"  baza presla (zadano): {travelled*1000:.0f} mm")
    if final_lag is not None:
        node.get_logger().info(f"  pomak taga u base_link: {final_lag*1000:.0f} mm")
        node.get_logger().info(f"  procijenjeno otvaranje: {opened*1000:.0f} mm")
    if log["run"]:
        forces = [r["force_N"] for r in log["run"]]
        node.get_logger().info(
            f"  sila: prosjek {sum(forces)/len(forces):.0f} N, najveca {max(forces):.0f} N"
        )
        node.get_logger().info(
            f"  koraka: {len(log['run'])}, najduzi ciklus petlje "
            f"{max_loop_period*1000:.0f} ms"
        )
    node.get_logger().info(f"  cmd_vel objavljen prosjecno {publish_hz:.0f} Hz")
    if aborted:
        node.get_logger().warn(f"  prekinuto: {aborted}")
    node.get_logger().info(
        "  PROVJERI stvarno otvaranje u Isaac Simu: slide_joint -> "
        "Raw USD properties -> state:linear:physics:position"
    )

    log["summary"] = {
        "travelled_m": travelled,
        "final_tag_lag_m": final_lag,
        "opened_m": opened,
        "max_loop_period_s": max_loop_period,
        "publish_hz": publish_hz,
        "aborted": aborted,
    }
    with open(LOG_PATH, "w") as fh:
        json.dump(log, fh, indent=2)
    node.get_logger().info(f"Detalji u {LOG_PATH}")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
