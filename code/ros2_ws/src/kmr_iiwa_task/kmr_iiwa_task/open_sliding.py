"""
open_sliding.py - otvara klizna vrata gibanjem baze po trapeznom profilu
brzine, uz ukocenu ruku i zatvoren gripper.

Hod vrata premasuje doseg ruke, pa gibanje nosi baza, a ruka samo prenosi
silu. Ruku ne treba posebno ukociti - arm_controller drzi zadnju tocku
trajektorije s krutoscu pogona 100000, pa je vec kruta dok joj se nista ne
salje. Time je i gravitacijski moment konstantan, pa tare procjenitelja sile
vrijedi kroz cijelu voznju.

Naredbe salje zasebna nit u stalnom ritmu. cmd_vel_bridge primjenjuje zadnju
primljenu poruku svaki fizicki korak i nema failsafe timeout, pa neujednacen
ritam znaci trzajno gibanje, a trzaj kroz krutu vezu daje skokove sile od
vise stotina njutna.

Brzina se ne regulira po sili. Sila uvijek naraste kad baza krece, sto je
normalno pri vucenju, pa je svaki takav regulator uvodio zastajkivanje. Vrata
k tome imaju gotovo konstantan otpor. Sila se prati samo kao sigurnosni
prekid.

Smjer otvaranja se odredjuje geometrijski: kvaka je blize onom rubu krila
prema kojem se vrata otvaraju, a door_tag_center je na sredini krila.

KRITERIJ ZAVRSETKA CITA STANJE ZGLOBA IZ SIMULATORA
Otvorenost se uzima s /ground_truth/door_joint, koji objavljuje
door_gt_publisher iz Isaac Sima - dakle NIJE izvedena iz percepcije. To
odstupa od ostatka sustava, gdje robot sve zakljucuje iz senzora.

Razlog: prijedjeni put baze nije rijesen. Integracija zadane brzine
precjenjuje visestruko - izmjereno protiv ground trutha, baza postize oko
40% naredjenog pomaka, a uzrok je ostao neutvrdjen i nakon sto su iskljuceni
pogon zgloba vrata (krutost i prigusenje na nuli), trenje s podom, viskozno
prigusenje baze i usporena simulacija (mjereno 59 fizickih koraka/s, dakle
realno vrijeme). Odometrija iz lidara preko ruba dovratnika pokusana je i
odbacena: rub je nadjen u 44% ciklusa uz rasap od 600 mm, dok je stvarni
pomak baze bio 120 mm.

Preduvjet: door_task_node je uhvatio kvaku i miruje; tcp_wrench_estimator
radi; cmd_vel_bridge vrti door_gt_publisher.

Pokrece se iz door_task_node (funkcija run) ili zasebno preko
`ros2 run kmr_iiwa_task open_sliding`.
"""

import json
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, WrenchStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from kmr_iiwa_task.geometry import quat_rotate_vector

# --- Trapezni profil ---
CRUISE_SPEED_MPS = 0.20  # iznad ovoga sila naglo raste
ACCEL_SEC = 3.0
DECEL_SEC = 1.5
TARGET_DISTANCE_M = 2.50  # profil mora trajati DULJE nego sto treba: stvarni
# zavrsetak odredjuje TARGET_SLIDE_M, a baza postize samo dio naredjenog pomaka

# Otvorenost pri kojoj se voznja zaustavlja. Cita se IZ SIMULATORA (vidi
# docstring) preko /ground_truth/door_joint.
TARGET_SLIDE_M = 0.90

# Voznja se ponavlja u vise prolaza BEZ ponovnog hvata. Prolaz koji ne donese
# barem MIN_PASS_PROGRESS_M smatra se zaglavljenim i ciklus staje.
MAX_PASSES = 4
MIN_PASS_PROGRESS_M = 0.02
PASS_PAUSE_SEC = 1.5

# --- Slanje naredbi ---
PUBLISH_PERIOD_SEC = 0.02  # 50 Hz, iz zasebne niti

# --- Sigurnosni prekidi (NE regulacija) ---
FORCE_ABORT_N = 1400.0
FORCE_SPIKE_STEPS = 3
LAG_ABORT_M = 0.08
STALE_ABORT_STEPS = 10

CONTROL_PERIOD_SEC = 0.05

LOG_PATH = "/tmp/open_sliding.json"


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


def run():
    node = Node("open_sliding")

    wrench = {"f": None}

    def _on_wrench(msg: WrenchStamped):
        wrench["f"] = np.array(
            [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]
        )

    # Stvarna otvorenost vrata iz simulatora - kriterij zavrsetka (vidi docstring).
    gt = {"slide_m": None}

    def _on_gt_joint(msg: JointState):
        if msg.position:
            gt["slide_m"] = float(msg.position[0])

    node.create_subscription(WrenchStamped, "/estimation/tcp_wrench", _on_wrench, 10)
    node.create_subscription(JointState, "/ground_truth/door_joint", _on_gt_joint, 10)
    cmd_vel_pub = node.create_publisher(Twist, "/cmd_vel", 10)
    gripper_pub = node.create_publisher(Float32, "/gripper_cmd", 10)

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    # Vlastiti izvrsavac, ne globalni: rclpy.spin(node) bez izricitog
    # izvrsavaca koristi GLOBALNI, pa bi ga druga faza (pass_through_door)
    # vrtjela istovremeno iz svoje niti - "generator already executing".
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

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

    node.get_logger().info("Cekam TF, wrench i /ground_truth/door_joint...")
    p_tcp, q_tcp = None, None
    t_wait = time.monotonic()
    while (p_tcp is None or wrench["f"] is None) and rclpy.ok():
        p_tcp, q_tcp, _ = lookup("gripper_tcp")
        time.sleep(0.1)
    while gt["slide_m"] is None and rclpy.ok():
        time.sleep(0.1)
        if time.monotonic() - t_wait > 15.0:
            node.get_logger().error(
                "Nema /ground_truth/door_joint - je li door_gt_publisher aktivan u "
                "cmd_vel_bridgeu? Bez njega nema kriterija zavrsetka."
            )
            executor.shutdown()
            node.destroy_node()
            return

    gripper_pub.publish(Float32(data=1.0))
    time.sleep(1.0)
    # tare_pub.publish(Empty())
    # time.sleep(0.5)

    # Os klizanja: os zatvaranja prstiju (lokalni X gripera), projicirana na
    # vodoravnu ravninu. Kod okomite sipke ta os lezi u ravnini vrata.
    x_axis = quat_rotate_vector(q_tcp, [1.0, 0.0, 0.0])
    slide = np.array([x_axis[0], x_axis[1], 0.0])
    n = np.linalg.norm(slide)
    if n < 1e-6:
        node.get_logger().error(
            "Os zatvaranja je gotovo vertikalna - ne mogu odrediti smjer klizanja."
        )
        executor.shutdown()
        node.destroy_node()
        return
    slide = slide / n
    node.get_logger().info(f"Os klizanja: {np.round(slide, 3)}")

    log = {"slide_axis": slide.tolist(), "run": []}

    p_tag_start, _, tag_stamp = lookup("door_tag_center")
    if p_tag_start is None:
        node.get_logger().error("Nema door_tag_center - ne mogu odrediti smjer.")
        executor.shutdown()
        node.destroy_node()
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
    last_tag_stamp = tag_stamp
    max_loop_period = 0.0
    aborted = None

    def _drive_one_pass():
        """Jedan trapezni prolaz. Vraca kad profil istekne, kad je cilj
        dosegnut ili kad je postavljen razlog prekida."""
        nonlocal travelled, t_last, last_tag_stamp, stale_count
        nonlocal force_spike_count, max_loop_period, aborted
        while rclpy.ok():
            f = wrench["f"]
            fmag = float(np.linalg.norm(f)) if f is not None else 0.0

            p_tag3, _, tag_stamp = lookup("door_tag_center")
            p_tag = None if p_tag3 is None else p_tag3[:2].copy()

            # Smjer klizanja se racuna SVAKI ciklus. Ako se baza usput zakrene, a
            # smjer ostane zamrznut iz pocetka, naredba vise ne ide duz vrata nego
            # dijelom U njih - sila tada raste kako se zakret gomila.
            _, q_tcp_now, _ = lookup("gripper_tcp")
            if q_tcp_now is not None:
                x_now = quat_rotate_vector(q_tcp_now, [1.0, 0.0, 0.0])
                s_now = np.array([x_now[0], x_now[1], 0.0])
                n_now = np.linalg.norm(s_now)
                if n_now > 1e-6:
                    cmd["dir"] = sign * (s_now / n_now)

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

            slide = gt["slide_m"]
            if slide is not None and abs(slide) >= TARGET_SLIDE_M:
                break

            now = time.monotonic()
            elapsed = now - t_start
            if elapsed >= total_time:
                break  # normalan kraj prolaza, ne prekid - vanjska petlja odlucuje

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
                    "gt_slide_m": gt["slide_m"],
                    "stale_count": stale_count,
                    "loop_period_s": loop_period,
                    "tag_pos_xy": (
                        None if p_tag is None else [float(p_tag[0]), float(p_tag[1])]
                    ),
                    "dir_xy": [float(cmd["dir"][0]), float(cmd["dir"][1])],
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

    log["passes"] = []
    t_run_start = time.monotonic()

    # Hod vrata premasuje ono sto baza stigne prijeci u jednom prolazu, pa se
    # voznja ponavlja BEZ ponovnog hvata - gripper drzi kvaku cijelo vrijeme,
    # baza se samo zaustavi i ponovno krene s pocetka profila.
    for pass_idx in range(MAX_PASSES):
        slide_before = abs(gt["slide_m"] or 0.0)
        if slide_before >= TARGET_SLIDE_M:
            break
        if pass_idx > 0:
            node.get_logger().info(
                f"--- prolaz {pass_idx + 1}: vrata na {slide_before*1000:.0f} mm, "
                f"nastavljam BEZ ponovnog hvata ---"
            )
        t_start = time.monotonic()
        t_last = t_start
        stale_count = 0
        force_spike_count = 0
        _drive_one_pass()

        slide_after = abs(gt["slide_m"] or 0.0)
        progress = slide_after - slide_before
        log["passes"].append(
            {
                "pass": pass_idx + 1,
                "slide_before_m": slide_before,
                "slide_after_m": slide_after,
                "progress_m": progress,
            }
        )
        node.get_logger().info(
            f"    prolaz {pass_idx + 1}: +{progress*1000:.0f} mm -> ukupno "
            f"{slide_after*1000:.0f} mm"
        )

        # zaustavi bazu izmedju prolaza da se napetost sklopa smiri
        cmd["speed"] = 0.0
        time.sleep(PASS_PAUSE_SEC)

        if aborted is not None:
            break
        if slide_after >= TARGET_SLIDE_M:
            break
        if progress < MIN_PASS_PROGRESS_M:
            aborted = (
                f"prolaz {pass_idx + 1} donio samo {progress*1000:.0f} mm - "
                "vrata se vise ne pomicu"
            )
            break

    if aborted is None:
        final_slide = abs(gt["slide_m"] or 0.0)
        if final_slide < TARGET_SLIDE_M:
            aborted = (
                f"iscrpljeno {MAX_PASSES} prolaza na {final_slide*1000:.0f} mm "
                f"(cilj {TARGET_SLIDE_M*1000:.0f} mm)"
            )

    t_start = t_run_start  # publish_hz u sazetku racuna od pocetka cijele voznje

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
    opened = abs(gt["slide_m"]) if gt["slide_m"] is not None else None
    publish_hz = publish_count["n"] / max(1e-6, time.monotonic() - t_start)

    node.get_logger().info("=== SAZETAK ===")
    if opened is not None:
        uspjeh = opened >= TARGET_SLIDE_M
        node.get_logger().info(f"  prolaza: {len(log['passes'])}")
        node.get_logger().info(
            f"  vrata otvorena: {opened*1000:.0f} mm  "
            f"({'CILJ DOSEGNUT' if uspjeh else 'ispod cilja'})"
        )
    node.get_logger().info(
        f"  baza presla (integracija naredbe, NIJE mjereno): {travelled*1000:.0f} mm"
    )
    if final_lag is not None:
        node.get_logger().info(
            f"  klizanje hvata (pomak taga u base_link): {final_lag*1000:.0f} mm"
        )
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
    log["summary"] = {
        "opened_m": opened,
        "target_slide_m": TARGET_SLIDE_M,
        "passes": len(log["passes"]),
        "travelled_m": travelled,
        "final_tag_lag_m": final_lag,
        "max_loop_period_s": max_loop_period,
        "publish_hz": publish_hz,
        "aborted": aborted,
    }
    with open(LOG_PATH, "w") as fh:
        json.dump(log, fh, indent=2)
    node.get_logger().info(f"Detalji u {LOG_PATH}")

    executor.shutdown()
    time.sleep(0.2)
    node.destroy_node()


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
