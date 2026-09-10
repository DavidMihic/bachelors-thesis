"""open_revolute.py - otvaranje zakretnih vrata: baza vozi RAVNO naprijed (bez
rotacije), ruka prati kvaku malim Cartesian koracima oko procijenjene osi
sarke. Zavrsava kad su vrata otvorena dovoljno da platforma moze proci.

ARHITEKTURA
Stari pristup (baza rotira i translatira po tocnom luku, ruka ukocena) vozio je
bazu u desni zid. Baza sad ide samo ravno, cime heading ostaje okomit na vrata -
sto je i uvjet da robot poslije moze proci kroz otvor.

Mjereno protiv ground trutha: vrata se otvore 3-6 stupnjeva po koraku, dok ruka
komandira 1. Lanac rame->ruka->gripper->kvaka je gotovo krut, pa gibanje baze
samo po sebi zakrece vrata; kutni korak ruke sluzi da meta ostane malo ISPRED
kvake (bez toga nema sile povlacenja), ne kao glavni pogon.

KRITERIJ ZAVRSETKA
Najuzi prolaz kroz djelomicno otvorena vrata je udaljenost vrha krila do
suprotnog dovratnika: 2 * 0.85 * sin(theta/2). Za platformu siroku 0.63 m to
znaci:
    40 deg -> 0.58 m  ne prolazi
    45 deg -> 0.65 m  prolazi, 2 cm marze
    50 deg -> 0.72 m  prolazi, 9 cm marze
Iznad ~47 stupnjeva hvat pocinje popustati (kvaka klizi iz gripera), pa je
TARGET_DOOR_ANGLE_DEG postavljen na 45 - najmanji kut koji jos omogucuje
prolaz, dosegnut prije nego hvat popusti.

SIDRENJE NA TAG (sve iz percepcije, nista iz simulatora)
- POZICIJA: expected_tcp = p_tag + R(q_tag) * handle_offset, gdje je
  handle_offset izmjeren jednom pri hvatu. Cilj se racuna od te tocke, ne od
  trenutnog p_tcp - inace je cilj samoreferentan i klizanje kvake se ne moze
  ni ispraviti ni izmjeriti.
- ORIJENTACIJA: expected_quat = tag_offset_quat * q_tag, isti princip.
- SARKA: fiksni vektor TAG_TO_HINGE_VEC u okviru taga. Raniji pristup (smjer iz
  vektora gripper->tag, normala iz orijentacije gripera) ovisio je o p_tcp pa je
  mjerio i samu sebe; greska protiv GT-a mu je izmedju runova skakala 4-46 mm.
  Ovako je 1-9 mm.

SIGURNOST
- door_panel i lidar_wall_* kolizijski objekti se brisu na pocetku - zastita je
  sila (isti obrazac kao door_open.py).
- baza: provjera koridora izravno iz /scan (baza ne prolazi kroz MoveIt).
- slip > SLIP_ABORT_M -> NEUSPJEH (kvaka klizi iz hvata).
- sila > FORCE_ABORT_N kroz vise uzoraka, ili promasaj mete -> NEUSPJEH.

GROUND TRUTH (samo log, nikad u regulaciji)
Pretplata na /ground_truth/door_joint i /ground_truth/hinge_pose koje objavljuje
door_gt_publisher iz Isaac Sima. Sluzi za validaciju procjene, ne za upravljanje.
Ako se ne objavljuje, polja u logu ostaju null i sve ostalo radi normalno.

Preduvjet: door_task_node je uhvatio kvaku (vertical_bar:=false) i miruje;
tcp_wrench_estimator radi.

Pokretanje:
    ros2 run kmr_iiwa_task open_revolute
"""

import json
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist, WrenchStamped
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Empty, Float32
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from kmr_iiwa_task.add_door_collision import quat_rotate_vector, COLLISION_OBJECT_ID
from kmr_iiwa_task.handle_approach import quat_angle_between

JOINT_NAMES = [f"iiwa_joint_{i}" for i in range(1, 8)]

# --- Geometrija vrata (iz revolute_door.urdf) ---
DOOR_LEAF_WIDTH_M = 0.85
TAG_TO_HINGE_M = 0.425
# Vektor od sredista taga do osi sarke, U OKVIRU TAGA. door_tag_center stoji na
# xyz="0.021 0.425 1.0" u door_leaf, s rpy="0 1.5708 0" - pa su osi taga:
#   tagX -> -leafZ,  tagY -> +leafY (sirina krila),  tagZ -> +leafX (normala).
# Clan -0.021 je odmak taga OD PLOHE krila; bez njega procjena sarke promasuje
# za 21 mm duz normale vrata (izmjereno protiv GT-a).
TAG_TO_HINGE_VEC = [0.0, -TAG_TO_HINGE_M, -0.021]

# --- Cilj: dovoljno otvoreno da platforma prodje (vidi docstring) ---
TARGET_DOOR_ANGLE_DEG = 45.0

# --- Baza: samo ravno naprijed ---
BASE_CRUISE_MPS = 0.20
BASE_ACCEL_SEC = 1.0
PUBLISH_PERIOD_SEC = 0.02

# --- Ruka ---
ARM_STEP_DEG = 1.0  # FIKSNO. Korak izveden iz izmjerenog prirasta vrata je
# pozitivna povratna veza: naredba ruci pomice vrata, veci izmjereni pomak daje
# jos vecu naredbu. Izmjereno - vrata su u tri koraka otisla s 1.3 na 34.7
# stupnjeva, sila 1289 N, ruka na rubu dosega.
ARM_ACCEL_SEC = 0.5  # ruka mora dosegnuti puni korak PRIJE nego baza dosegne
# puni cruise (BASE_ACCEL_SEC=1.0) - inace baza gura naprijed dok ruka tek
# "budi se", i napetost odmah skoci vrlo visoko
SETTLE_SEC = 0.3
MIN_RADIUS_M = 0.30
MAX_RADIUS_M = 1.20

# --- /scan provjera koridora ispred baze ---
KMR_LENGTH_M = 1.08
KMR_WIDTH_M = 0.63
FRONT_X_MIN_M = KMR_LENGTH_M / 2.0
LOOKAHEAD_MIN_M = 0.6
LOOKAHEAD_MARGIN_M = 0.5
HARD_STOP_M = 0.15
CORRIDOR_HALF_WIDTH_M = KMR_WIDTH_M / 2.0 + 0.10
SELF_OCCLUSION_HALF_ANGLE_DEG = 94.0  # geometrijski izvod: senzor na
# (0.57, 0, 0.08) u base_link, prednji uglovi kucista pod +-95.44 deg; malo uze
# da se pojas oko ruba sigurno odbaci
LIDAR_FRAME = "lidar_link"

# --- Sigurnosni prekidi ---
FORCE_ABORT_N = 700.0
FORCE_SPIKE_STEPS = 3
TRACKING_HARD_FAIL_M = 0.015
TRACKING_FAIL_ERROR_M = 0.03
TRACKING_FAIL_STEPS = 3
SLIP_ABORT_M = 0.035  # koliko gripper smije odstupiti od mjesta na kvaki na
# kojem je bio pri hvatu (mjereno preko taga, neovisno o tome kamo ruku
# saljemo). Ovo JEST klizanje, za razliku od tracking_error koji mjeri samo je
# li ruka stigla tamo kamo smo je poslali.
MAX_STEPS = 200
RETARE_EVERY = 6
RETARE_MAX_FORCE_N = 50.0  # ne tariraj pod opterecenjem - tada tariranje
# "izbrise" stvarnu silu u novu nulu umjesto da nulira samo gravitacijsku i
# konfiguracijsku pristranost

LIDAR_WALL_ID_RANGE = range(20)

LOG_PATH = "/tmp/open_revolute.json"


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def quat_conj(q):
    x, y, z, w = q
    return [-x, -y, -z, w]


def _wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def main():
    rclpy.init()
    node = Node("open_revolute")
    cb = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name="base_link",
        end_effector_name="gripper_tcp",
        group_name="iiwa_arm",
        callback_group=cb,
    )
    moveit2.max_velocity = 0.08
    moveit2.max_acceleration = 0.08

    # Wrench/gripper/tare/scan/cmd_vel na zasebnom nodu i izvrsavacu - MoveIt2
    # inace gladuje obicne pretplate na istom nodu (isti razlog kao door_open).
    sensor_node = Node("open_revolute_sensors")
    wrench = {"f": None}
    scan_state = {"clear": False, "too_close": False, "have_scan": False}
    door_geom = {"center_y": None, "width_sign": None}
    scan_params = {"lookahead_m": LOOKAHEAD_MIN_M}
    # SAMO ZA VALIDACIJU - nikad ne ulazi u regulacijsku petlju.
    gt = {"angle_rad": None, "hinge_xy": None}

    def _on_wrench(msg: WrenchStamped):
        wrench["f"] = np.array(
            [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]
        )

    def _on_gt_joint(msg: JointState):
        if msg.position:
            gt["angle_rad"] = float(msg.position[0])

    def _on_gt_hinge(msg: PoseStamped):
        gt["hinge_xy"] = np.array([msg.pose.position.x, msg.pose.position.y])

    tf_buffer = Buffer()
    TransformListener(tf_buffer, sensor_node)

    lidar_to_base = {"t": None, "q": None}

    def _ensure_lidar_tf():
        if lidar_to_base["t"] is not None:
            return True
        try:
            tf = tf_buffer.lookup_transform("base_link", LIDAR_FRAME, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return False
        t, q = tf.transform.translation, tf.transform.rotation
        lidar_to_base["t"] = np.array([t.x, t.y, t.z])
        lidar_to_base["q"] = (q.x, q.y, q.z, q.w)
        return True

    def _on_scan(msg: LaserScan):
        if not _ensure_lidar_tf():
            return
        t, q = lidar_to_base["t"], lidar_to_base["q"]
        mask_rad = math.radians(SELF_OCCLUSION_HALF_ANGLE_DEG)
        too_close = False
        center_y = door_geom["center_y"]
        width_sign = door_geom["width_sign"]
        clear = center_y is not None and width_sign is not None
        y_lo = y_hi = None
        if center_y is not None and width_sign is not None:
            if width_sign > 0:
                y_lo = center_y + 0.05
                y_hi = center_y + DOOR_LEAF_WIDTH_M - 0.05
            else:
                y_lo = center_y - (DOOR_LEAF_WIDTH_M - 0.05)
                y_hi = center_y - 0.05

        angle = msg.angle_min
        for r in msg.ranges:
            a = angle
            angle += msg.angle_increment
            if not math.isfinite(r) or r < msg.range_min or r > msg.range_max:
                continue
            if abs(_wrap_pi(a)) > mask_rad:
                continue
            x_l, y_l = r * math.cos(a), r * math.sin(a)
            x_b, y_b, _ = quat_rotate_vector(q, [x_l, y_l, 0.0])
            x_b += t[0]
            y_b += t[1]

            if (
                abs(y_b) < CORRIDOR_HALF_WIDTH_M
                and FRONT_X_MIN_M < x_b < FRONT_X_MIN_M + HARD_STOP_M
            ):
                too_close = True

            if FRONT_X_MIN_M < x_b < FRONT_X_MIN_M + scan_params["lookahead_m"]:
                if center_y is not None and y_lo < y_b < y_hi:
                    clear = False

        scan_state["clear"] = clear
        scan_state["too_close"] = too_close
        scan_state["have_scan"] = True

    sensor_node.create_subscription(
        WrenchStamped, "/estimation/tcp_wrench", _on_wrench, 10
    )
    sensor_node.create_subscription(LaserScan, "/scan", _on_scan, 10)
    sensor_node.create_subscription(
        JointState, "/ground_truth/door_joint", _on_gt_joint, 10
    )
    sensor_node.create_subscription(
        PoseStamped, "/ground_truth/hinge_pose", _on_gt_hinge, 10
    )
    tare_pub = sensor_node.create_publisher(Empty, "/estimation/tare", 10)
    gripper_pub = sensor_node.create_publisher(Float32, "/gripper_cmd", 10)
    cmd_vel_pub = sensor_node.create_publisher(Twist, "/cmd_vel", 10)

    sensor_exec = SingleThreadedExecutor()
    sensor_exec.add_node(sensor_node)
    threading.Thread(target=sensor_exec.spin, daemon=True).start()

    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

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

    def tag_pose():
        try:
            t = tf_buffer.lookup_transform(
                "base_link", "door_tag_center", rclpy.time.Time()
            )
            tr, r = t.transform.translation, t.transform.rotation
            return np.array([tr.x, tr.y, tr.z]), [r.x, r.y, r.z, r.w]
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None, None

    def estimate_center(p_tag_now, q_tag_now):
        """Os sarke iz poze taga, preko fiksnog vektora u okviru taga."""
        v = np.array(quat_rotate_vector(q_tag_now, list(TAG_TO_HINGE_VEC)))
        return (p_tag_now + v)[:2]

    def door_angle_from_tag(p_tag_now, center_now):
        """Kut smjera sarka->tag u vodoravnoj ravnini. Razlika u odnosu na
        pocetnu vrijednost je otvorenost vrata."""
        v = p_tag_now[:2] - center_now
        return math.atan2(v[1], v[0])

    node.get_logger().info("Cekam TF, wrench i /scan...")
    p_tcp, q_tcp = tcp_pose()
    while (
        p_tcp is None or wrench["f"] is None or not scan_state["have_scan"]
    ) and rclpy.ok():
        time.sleep(0.1)
        p_tcp, q_tcp = tcp_pose()

    p_tag, q_tag = tag_pose()
    while p_tag is None and rclpy.ok():
        time.sleep(0.1)
        p_tag, q_tag = tag_pose()

    scan_params["lookahead_m"] = max(
        LOOKAHEAD_MIN_M, float(p_tag[0]) - FRONT_X_MIN_M + LOOKAHEAD_MARGIN_M
    )

    gripper_pub.publish(Float32(data=1.0))
    time.sleep(1.0)
    tare_pub.publish(Empty())
    time.sleep(0.5)
    node.get_logger().info(
        f"Sila nakon tare, prije gibanja: {float(np.linalg.norm(wrench['f'])):.1f} N"
    )

    node.get_logger().info("Micem door_panel i lidar_wall_* - zastita je sila.")
    moveit2.remove_collision_object(COLLISION_OBJECT_ID)
    for i in LIDAR_WALL_ID_RANGE:
        moveit2.remove_collision_object(f"lidar_wall_{i}")
    time.sleep(0.5)

    center0 = estimate_center(p_tag, q_tag)
    door_geom["width_sign"] = 1.0 if (p_tag[1] - center0[1]) >= 0.0 else -1.0
    door_geom["center_y"] = float(center0[1])

    radius0 = float(np.linalg.norm(p_tcp[:2] - center0))
    if not (MIN_RADIUS_M <= radius0 <= MAX_RADIUS_M):
        node.get_logger().error(
            f"Procijenjeni radijus {radius0:.3f}m izvan "
            f"[{MIN_RADIUS_M},{MAX_RADIUS_M}] - prekidam."
        )
        rclpy.shutdown()
        return

    # Smjer otvaranja: tangenta na luk, u smjeru u kojem gripper prilazi vratima.
    z_tcp = np.array(quat_rotate_vector(q_tcp, [0.0, 0.0, 1.0]))
    normal0 = np.array([z_tcp[0], z_tcp[1]])
    normal0 = normal0 / np.linalg.norm(normal0)
    rad_vec0 = p_tcp[:2] - center0
    tang0 = np.array([-rad_vec0[1], rad_vec0[0]]) / radius0
    sign = 1.0 if float(np.dot(tang0, normal0)) > 0.0 else -1.0

    # Fiksni odnos gripper<->tag, izmjeren jednom dok hvat sigurno drzi.
    tag_offset_quat = quat_mul(q_tcp, quat_conj(q_tag))
    handle_offset = np.array(quat_rotate_vector(quat_conj(q_tag), list(p_tcp - p_tag)))

    door_theta0 = door_angle_from_tag(p_tag, center0)

    node.get_logger().info(
        f"Os sarke: ({center0[0]:.3f}, {center0[1]:.3f}), radijus {radius0:.3f}m, "
        f"smjer {sign:+.0f}. Cilj: {TARGET_DOOR_ANGLE_DEG:.0f} deg."
    )

    stop_flag = {"v": False}

    def base_publisher_loop():
        t0 = time.monotonic()
        while rclpy.ok() and not stop_flag["v"]:
            elapsed = time.monotonic() - t0
            ramp = min(1.0, elapsed / BASE_ACCEL_SEC) if BASE_ACCEL_SEC > 0 else 1.0
            gripper_pub.publish(Float32(data=1.0))
            tw = Twist()
            tw.linear.x = BASE_CRUISE_MPS * ramp
            cmd_vel_pub.publish(tw)
            time.sleep(PUBLISH_PERIOD_SEC)

    threading.Thread(target=base_publisher_loop, daemon=True).start()

    log = {
        "center0_xy": [float(center0[0]), float(center0[1])],
        "radius0_m": radius0,
        "sign": sign,
        "target_angle_deg": TARGET_DOOR_ANGLE_DEG,
        "run": [],
    }

    def goto(p, q):
        moveit2.move_to_pose(position=list(p), quat_xyzw=list(q), cartesian=True)
        moveit2.wait_until_executed()

    t_start = time.monotonic()
    force_spike_count = 0
    tracking_fail_count = 0
    steps_since_retare = 0
    outcome = None
    step = 0
    door_angle_deg = 0.0

    while rclpy.ok() and step < MAX_STEPS:
        step += 1
        steps_since_retare += 1
        elapsed = time.monotonic() - t_start

        if scan_state["too_close"]:
            outcome = "NEUSPJEH: prepreka preblizu ispred baze"
            break

        p_tcp_now, q_tcp_now = tcp_pose()
        p_tag_now, q_tag_now = tag_pose()
        if p_tcp_now is None or p_tag_now is None:
            outcome = "NEUSPJEH: izgubljen TF gripper_tcp/door_tag_center"
            break

        # GT snimamo U ISTOM TRENUTKU kad i procjenu - gt se osvjezava
        # asinkrono, a do upisa u log prodje cijeli goto().
        gt_hinge_snapshot = gt["hinge_xy"]
        gt_angle_snapshot = gt["angle_rad"]

        center = estimate_center(p_tag_now, q_tag_now)
        door_geom["width_sign"] = 1.0 if (p_tag_now[1] - center[1]) >= 0.0 else -1.0
        door_geom["center_y"] = float(center[1])

        door_angle_deg = math.degrees(
            abs(_wrap_pi(door_angle_from_tag(p_tag_now, center) - door_theta0))
        )
        if door_angle_deg >= TARGET_DOOR_ANGLE_DEG:
            outcome = f"USPJEH: vrata otvorena {door_angle_deg:.1f} deg"
            break

        gt_angle_deg = (
            math.degrees(gt_angle_snapshot) if gt_angle_snapshot is not None else None
        )
        center_error_mm = (
            float(np.linalg.norm(center - gt_hinge_snapshot)) * 1000
            if gt_hinge_snapshot is not None
            else None
        )

        ramp = min(1.0, elapsed / ARM_ACCEL_SEC) if ARM_ACCEL_SEC > 0 else 1.0
        dtheta = sign * math.radians(ARM_STEP_DEG) * ramp

        # Gdje bi gripper TREBAO biti prema svjezem tagu (a ne gdje trenutno
        # jest). Bez ovoga je cilj samoreferentan: ako kvaka klizne, cilj klizne
        # s njom i ruka nikad ne dobije naredbu da se vrati.
        expected_tcp = p_tag_now + np.array(
            quat_rotate_vector(q_tag_now, list(handle_offset))
        )
        slip = float(np.linalg.norm(p_tcp_now - expected_tcp))

        rad_vec = expected_tcp[:2] - center
        rad_norm = np.linalg.norm(rad_vec)
        if rad_norm > 1e-6:
            rad_vec = rad_vec / rad_norm * radius0
        c, s = math.cos(dtheta), math.sin(dtheta)
        R = np.array([[c, -s], [s, c]])
        target_xy = center + R @ rad_vec
        target = np.array([target_xy[0], target_xy[1], expected_tcp[2]])

        expected_quat = quat_mul(tag_offset_quat, q_tag_now)
        half = dtheta / 2.0
        dq = [0.0, 0.0, math.sin(half), math.cos(half)]
        target_quat = quat_mul(dq, expected_quat)

        goto(target, target_quat)
        time.sleep(SETTLE_SEC)

        p_after, q_after = tcp_pose()
        f = wrench["f"]
        fmag = float(np.linalg.norm(f)) if f is not None else 0.0
        tracking_error = (
            float(np.linalg.norm(p_after - target)) if p_after is not None else None
        )
        angle_error = (
            quat_angle_between(q_after, target_quat) if q_after is not None else None
        )

        log["run"].append(
            {
                "step": step,
                "t": elapsed,
                "door_angle_deg": door_angle_deg,
                "center_xy": [float(center[0]), float(center[1])],
                "tracking_error_m": tracking_error,
                "angle_error_deg": (
                    math.degrees(angle_error) if angle_error is not None else None
                ),
                "force_N": fmag,
                "slip_mm": slip * 1000,
                "gt_door_angle_deg": gt_angle_deg,
                "center_error_mm": center_error_mm,
            }
        )

        if tracking_error is not None:
            gt_str = f"{gt_angle_deg:.1f}" if gt_angle_deg is not None else "n/a"
            ce_str = f"{center_error_mm:.0f}" if center_error_mm is not None else "n/a"
            node.get_logger().info(
                f"korak {step:3d}: vrata {door_angle_deg:5.1f}deg  "
                f"greska={tracking_error*1000:.1f}mm  klizanje={slip*1000:.1f}mm  "
                f"sila={fmag:.0f}N  gt={gt_str}deg  sarka_err={ce_str}mm"
            )
        else:
            node.get_logger().warn(f"korak {step:3d}: TF izgubljen nakon pomaka")

        if slip > SLIP_ABORT_M:
            outcome = f"NEUSPJEH: kvaka klizi iz hvata ({slip*1000:.0f}mm od hvatista)"
            break

        if tracking_error is not None and tracking_error > TRACKING_HARD_FAIL_M:
            outcome = (
                f"NEUSPJEH: jedan korak promasio metu za {tracking_error*1000:.0f}mm"
            )
            break
        if tracking_error is not None and tracking_error > TRACKING_FAIL_ERROR_M:
            tracking_fail_count += 1
        else:
            tracking_fail_count = 0
        if tracking_fail_count >= TRACKING_FAIL_STEPS:
            outcome = "NEUSPJEH: ruka ne moze dalje pratiti kvaku"
            break

        if fmag > FORCE_ABORT_N:
            force_spike_count += 1
        else:
            force_spike_count = 0
        if force_spike_count >= FORCE_SPIKE_STEPS:
            outcome = f"NEUSPJEH: sila {fmag:.0f}N kroz vise uzoraka"
            break

        if steps_since_retare >= RETARE_EVERY and fmag < RETARE_MAX_FORCE_N:
            tare_pub.publish(Empty())
            time.sleep(SETTLE_SEC)
            steps_since_retare = 0

    if outcome is None:
        outcome = "NEUSPJEH: dosegnut MAX_STEPS bez ishoda"

    stop_flag["v"] = True
    time.sleep(0.1)
    cmd_vel_pub.publish(Twist())
    time.sleep(0.5)

    passage_m = 2.0 * DOOR_LEAF_WIDTH_M * math.sin(math.radians(door_angle_deg) / 2.0)
    node.get_logger().info("=== SAZETAK ===")
    node.get_logger().info(f"  ishod: {outcome}")
    node.get_logger().info(f"  koraka: {step}")
    node.get_logger().info(
        f"  vrata otvorena: {door_angle_deg:.1f} deg -> prolaz {passage_m:.2f} m "
        f"(platforma {KMR_WIDTH_M:.2f} m)"
    )
    if log["run"]:
        forces = [r["force_N"] for r in log["run"]]
        node.get_logger().info(
            f"  sila: prosjek {sum(forces)/len(forces):.0f}N, "
            f"najveca {max(forces):.0f}N"
        )

    log["summary"] = {
        "outcome": outcome,
        "steps": step,
        "door_angle_deg": door_angle_deg,
        "passage_width_m": passage_m,
    }
    with open(LOG_PATH, "w") as fh:
        json.dump(log, fh, indent=2)
    node.get_logger().info(f"Detalji u {LOG_PATH}")

    executor.shutdown()
    sensor_exec.shutdown()
    time.sleep(0.2)
    node.destroy_node()
    sensor_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
