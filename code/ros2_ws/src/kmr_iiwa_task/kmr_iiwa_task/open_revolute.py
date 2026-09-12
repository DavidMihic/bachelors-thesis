"""open_revolute.py - otvaranje zakretnih vrata.

Do BASE_STOP_ANGLE_DEG baza vozi RAVNO naprijed (bez rotacije, cime heading
ostaje okomit na vrata) i nosi vecinu otvaranja: izmjereno protiv ground
trutha, vrata se otvore oko 5 deg po koraku dok ruka komandira 1. Lanac
rame->ruka->gripper->kvaka je gotovo krut, pa gibanje baze samo po sebi zakrece
vrata; kutni korak ruke sluzi da meta ostane malo ISPRED kvake (bez toga nema
sile povlacenja), ne kao glavni pogon.

Iznad tog kuta ruci ponestaje dosega - orijentacija gripera na poluzi je
fiksna, pa je stvarni radni prostor puno uzi nego sama udaljenost od ramena.
Zato se dalje IZMJENJUJU faze: ruka otvara dok baza stoji, pa baza vozi dok
ruka samo drzi kvaku i time se presloziti u povoljniju konfiguraciju. Prijelaz
ide po STVARNOM ispruzenju ruke (udaljenost kvake od ramena, REACH_MIN/MAX), ne
po vremenu - vremenska faza je proizvoljna, a limitirajuca velicina je doseg.
Cilj je 100 deg, isti prag uspjeha koji koristi RL grana.

SIDRENJE NA TAG (sve iz percepcije, nista iz simulatora)
- POZICIJA: expected_tcp = p_tag + R(q_tag) * handle_offset, gdje je
  handle_offset izmjeren jednom pri hvatu. Cilj se racuna od te tocke, ne od
  trenutnog p_tcp - inace je cilj samoreferentan i klizanje kvake se ne moze
  ni ispraviti ni izmjeriti.
- ORIJENTACIJA: expected_quat = tag_offset_quat * q_tag, isti princip.
- SARKA: fiksni vektor TAG_TO_HINGE_VEC u okviru taga. Raniji pristup (smjer iz
  vektora gripper->tag, normala iz orijentacije gripera) ovisio je o p_tcp pa je
  mjerio i samu sebe: greska protiv GT-a skakala mu je 4-46 mm. Ovako je 1-9 mm.

TARA: ovaj node NE tarira procjenitelj sile. Tara se postavlja u
handle_approach dok je gripper jos OTVOREN - stiskanje prstiju unosi oko 1400 N
u ocitanje bez ikakvog gibanja ruke.

SIGURNOST
- door_panel i lidar_wall_* kolizijski objekti se brisu na pocetku - zastita je
  sila (isti obrazac kao door_open.py).
- baza: provjera koridora izravno iz /scan (baza ne prolazi kroz MoveIt).
- slip > SLIP_ABORT_M -> NEUSPJEH (kvaka klizi iz hvata).
- sila > FORCE_ABORT_N kroz vise uzoraka, ili promasaj mete -> NEUSPJEH.

GROUND TRUTH (samo log, nikad u regulaciji)
Pretplata na /ground_truth/door_joint i /ground_truth/hinge_pose koje objavljuje
door_gt_publisher iz Isaac Sima. Sluzi za validaciju procjene, ne za
upravljanje. Ako se ne objavljuje, polja u logu ostaju null i sve ostalo radi.

Preduvjet: door_task_node je uhvatio kvaku i miruje; tcp_wrench_estimator radi.

Pokrece se iz door_task_node (funkcija run) ili zasebno preko
`ros2 run kmr_iiwa_task open_revolute`.
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

from kmr_iiwa_task.add_door_collision import COLLISION_OBJECT_ID
from kmr_iiwa_task.geometry import (
    quat_angle_between,
    quat_conj,
    quat_mul,
    quat_rotate_vector,
    wrap_pi,
)

JOINT_NAMES = [f"iiwa_joint_{i}" for i in range(1, 8)]

# --- Geometrija vrata (iz revolute_door.urdf) ---
DOOR_LEAF_WIDTH_M = 0.85
TAG_TO_HINGE_M = 0.425
# Vektor od sredista taga do osi sarke, U OKVIRU TAGA. door_tag_center stoji na
# xyz="0.021 0.425 1.0" u door_leaf, s rpy="0 1.5708 0" - pa su osi taga:
#   tagX -> -leafZ,  tagY -> +leafY (sirina krila),  tagZ -> +leafX (normala).
# Clan -0.021 je odmak taga od sredine krila (vrata 40 mm, pola je 20 mm plus 1 mm debljina taga);
# bez njega procjena sarke promasuje
# za 21 mm duz normale vrata (izmjereno protiv GT-a).
TAG_TO_HINGE_VEC = [0.0, -TAG_TO_HINGE_M, -0.021]

# --- Baza: samo ravno naprijed, bez rotacije ---
BASE_CRUISE_MPS = 0.20
BASE_ACCEL_SEC = 1.0
BASE_DECEL_SEC = 2.0  # zaustavljanje je sporije od kretanja: baza do zakljucavanja
# nosi vecinu otvaranja (vrata idu ~5 deg po koraku dok ruka komandira 1), pa
# nagli prekid trgne cijeli lanac
PUBLISH_PERIOD_SEC = 0.02

# --- Naizmjenicne faze iznad BASE_STOP_ANGLE_DEG ---
# Ispod tog kuta baza vozi kontinuirano i nosi vecinu otvaranja. Iznad njega
# ruci ponestaje dosega: orijentacija gripera na poluzi je fiksna, pa je stvarni
# radni prostor puno uzi nego sama udaljenost od ramena. Zato se izmjenjuju
# faze - ruka otvara dok baza stoji, pa baza vozi dok ruka samo drzi kvaku i
# time se presloziti u povoljniju konfiguraciju.
TARGET_DOOR_ANGLE_DEG = 50.0
BASE_STOP_ANGLE_DEG = 35.0
BASE_PHASE_MPS = 0.10  # u fazama baza gura ruku koja vec drzi kvaku pod
# fiksnom orijentacijom - ista brzina koja je u cruise fazi bila u redu ovdje
# istrgne polugu iz stiska
BASE_PHASE_ACCEL_SEC = 2.5
# Faze se prebacuju po STVARNOM ispruzenju ruke, ne po vremenu: limitirajuca
# velicina je udaljenost kvake od ramena, a ne koliko je sekundi baza vozila.
# Histereza sprjecava titranje oko jednog praga.
REACH_MAX_M = 0.86  # iznad ovoga ruka je pri kraju dosega -> baza vozi
REACH_MIN_M = 0.79  # ispod ovoga ruka opet ima prostora -> baza staje
SHOULDER_FALLBACK_XY = (0.363, -0.184)  # iiwa_mount_x/y iz kmr_iiwa.urdf.xacro

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
TRACKING_HARD_FAIL_M = 0.030
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

USE_GT_HINGE = False  # EKSPERIMENT: umjesto procjene iz taga koristi stvarnu
# poziciju sarke iz simulatora. Sluzi da se provjeri je li greska procjene
# (izmjereno 5-8 mm, tj. ~0.7 deg pri r=0.65) uzrok zaglavljivanja. Za normalan
# rad mora ostati False - robot inace ne rjesava zadatak iz percepcije.


def run():
    node = Node("open_revolute")
    cb = ReentrantCallbackGroup()

    # MoveIt2 loggira "Joint states are not available yet!" pri svakom pozivu.
    # Vlastiti node znaci da se moze utisati bez diranja nasih poruka.
    moveit_node = Node("open_revolute_moveit")
    rclpy.logging.set_logger_level(
        "open_revolute_moveit", rclpy.logging.LoggingSeverity.ERROR
    )

    moveit2 = MoveIt2(
        node=moveit_node,
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
    stop_flag = {"v": False}
    base_cmd = {"target": BASE_CRUISE_MPS, "current": 0.0}
    phase = {"mode": "cruise"}
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
            if abs(wrap_pi(a)) > mask_rad:
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
    executor.add_node(moveit_node)
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
        node.destroy_node()

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
        last = time.monotonic()
        while rclpy.ok() and not stop_flag["v"]:
            now = time.monotonic()
            dt = now - last
            last = now
            tgt = base_cmd["target"]
            cur = base_cmd["current"]
            ramp_sec = BASE_ACCEL_SEC if tgt > cur else BASE_DECEL_SEC
            if phase["mode"] != "cruise" and tgt > cur:
                ramp_sec = BASE_PHASE_ACCEL_SEC
            step = BASE_CRUISE_MPS / ramp_sec * dt if ramp_sec > 0 else abs(tgt - cur)
            base_cmd["current"] = (
                min(tgt, cur + step) if cur < tgt else max(tgt, cur - step)
            )
            gripper_pub.publish(Float32(data=1.0))
            tw = Twist()
            tw.linear.x = base_cmd["current"]
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

    try:
        _tf = tf_buffer.lookup_transform("base_link", "iiwa_link_0", rclpy.time.Time())
        shoulder_xy = np.array(
            [_tf.transform.translation.x, _tf.transform.translation.y]
        )
    except (LookupException, ConnectivityException, ExtrapolationException):
        node.get_logger().warn("Nema TF za iiwa_link_0 - koristim nominalni mount.")
        shoulder_xy = np.array(SHOULDER_FALLBACK_XY)

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

        # Provjera vrijedi samo dok baza kontinuirano vozi. U fazama je ono
        # ispred sama vrata koja otvaramo, sto nije prepreka.
        if phase["mode"] == "cruise" and scan_state["too_close"]:
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

        if USE_GT_HINGE and gt_hinge_snapshot is not None:
            center = gt_hinge_snapshot.copy()

        door_geom["width_sign"] = 1.0 if (p_tag_now[1] - center[1]) >= 0.0 else -1.0
        door_geom["center_y"] = float(center[1])

        door_angle_deg = math.degrees(
            abs(wrap_pi(door_angle_from_tag(p_tag_now, center) - door_theta0))
        )
        reach = float(np.linalg.norm(p_tcp_now[:2] - shoulder_xy))

        if phase["mode"] == "cruise" and door_angle_deg >= BASE_STOP_ANGLE_DEG:
            phase["mode"] = "arm"
            base_cmd["target"] = 0.0
            node.get_logger().info(
                f"Vrata na {door_angle_deg:.1f} deg - prelazim na naizmjenicne "
                f"faze, cilj {TARGET_DOOR_ANGLE_DEG:.0f} deg."
            )
        elif phase["mode"] == "arm" and reach > REACH_MAX_M:
            phase["mode"] = "drive"
            base_cmd["target"] = BASE_PHASE_MPS
        elif phase["mode"] == "drive" and reach < REACH_MIN_M:
            phase["mode"] = "arm"
            base_cmd["target"] = 0.0
            node.get_logger().info(f"  doseg {reach:.2f} m -> ruka otvara dalje")

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

        # U fazi voznje ruka i dalje dobiva cilj iz svjezeg taga, samo bez
        # pomaka naprijed - drzi kvaku dok se baza primice.
        ramp = min(1.0, elapsed / ARM_ACCEL_SEC) if ARM_ACCEL_SEC > 0 else 1.0
        step_deg = 0.0 if phase["mode"] == "drive" else ARM_STEP_DEG
        dtheta = sign * math.radians(step_deg) * ramp

        # Gdje bi gripper TREBAO biti prema svjezem tagu (a ne gdje trenutno
        # jest). Bez ovoga je cilj samoreferentan: ako kvaka klizne, cilj klizne
        # s njom i ruka nikad ne dobije naredbu da se vrati.
        expected_tcp = p_tag_now + np.array(
            quat_rotate_vector(q_tag_now, list(handle_offset))
        )
        slip = float(np.linalg.norm(p_tcp_now - expected_tcp))

        if phase["mode"] == "drive":
            # Baza vozi, ruka samo drzi kvaku. Cilj je tocno expected_tcp - bez
            # projekcije na luk, jer se u ovoj fazi i center i expected_tcp
            # pomicu s bazom, pa projekcija vuce poziciju po luku dok
            # orijentacija stoji. Ta razlika zavrce gripper oko z na poluzi.
            target = expected_tcp.copy()
        else:
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
                "phase": phase["mode"],
                "reach_m": reach,
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
                f"doseg={reach:.2f}m  "
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

    node.get_logger().info("=== SAZETAK ===")
    node.get_logger().info(f"  ishod: {outcome}")
    node.get_logger().info(f"  koraka: {step}")
    node.get_logger().info(f"  vrata otvorena: {door_angle_deg:.1f} deg")
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
    }
    with open(LOG_PATH, "w") as fh:
        json.dump(log, fh, indent=2)
    node.get_logger().info(f"Detalji u {LOG_PATH}")

    executor.shutdown()
    sensor_exec.shutdown()
    time.sleep(0.2)
    node.destroy_node()
    moveit_node.destroy_node()
    sensor_node.destroy_node()


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
