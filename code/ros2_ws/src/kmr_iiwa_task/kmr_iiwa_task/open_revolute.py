"""
open_revolute.py - otvaranje zakretnih vrata.

Baza se giba po tangenti luka kvake, bez rotacije. Ruka pritom koraca po luku
oko osi sarke.

Zakretanje baze - i u mjestu i kao kruzenje oko sarke, da rame ide naprijed -
probano je i odbaceno: rotacija platforme ima mrtvu zonu od 0.36 rad/s
(izmjereno), pa krece s barem 20 deg/s i trgne krutu vezu ruka-kvaka. Hvat je
pucao u istom koraku u kojem bi zakretanje pocelo, bez obzira na prag i rampu.

Guranje u lokalni +x ima komponentu duz poluge koja raste kao sin(kut vrata) -
na 50 deg je 77% sile usmjereno prema rubu krila, trenje to ne zadrzava i
gripper klizi. Tangenta je okomita na polugu, pa te komponente nema. Zakretanje
uz to primice rame kvaki bez trosenja bocnog prostora, kojeg izmedju zidova ima
svega oko 11 cm.

Rotacija baze ima prag i zasicenje: izmjereno, postignuto = 0.562 * naredba -
0.204 [rad/s], ispod 0.36 rad/s se ne mice uopce. Naredba se skalira inverzom.

SVE IZ PERCEPCIJE, BEZ TAGA TIJEKOM OTVARANJA
Tag se cita samo pri hvatu, jer nakon nekog kuta izlazi iz vidnog polja kamere.
Dalje:
- SARKA se fiksira na hvatu i prati kroz odometriju. Nepomicna je u svijetu, pa
  je dovoljno oduzeti prijedjeni put baze.
- KUT VRATA je smjer sarka->TCP. Dok hvat drzi, gripper je na kvaki, pa je to
  ujedno smjer krila.
- ORIJENTACIJA gripera je poza pri hvatu zakrenuta za izmjereni kut vrata.
  Mjereno, ne akumulirano - akumulacija je ranije driftala.
- KLIZANJE se mjeri kao radijalno odstupanje: koliko je gripper dalje od sarke
  nego pri hvatu.

Tijekom otvaranja nema /scan provjere. Krilo koje robot vuce i zidovi kroz koje
prolazi stalno su u prozoru provjere, a baza se usput zakrece pa prozor vise ne
gleda u otvor. Zastita je sila.

TARA: ovaj node NE tarira procjenitelj sile. Tara se postavlja u
handle_approach dok je gripper jos OTVOREN - stiskanje prstiju unosi oko 1400 N
u ocitanje bez ikakvog gibanja ruke.

Ground truth (/ground_truth/*) se cita samo za log, nikad u regulaciji.

Preduvjet: door_task_node je uhvatio kvaku i miruje; tcp_wrench_estimator radi;
cmd_vel_bridge vrti OdomPublisher.

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
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Empty, Float32
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from kmr_iiwa_task.add_door_collision import COLLISION_OBJECT_ID
from kmr_iiwa_task.geometry import (
    quat_angle_between,
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
BASE_MAX_MPS = 0.33
PUBLISH_PERIOD_SEC = 0.02

TARGET_DOOR_ANGLE_DEG = 70.0

# --- Gibanje baze ---
# Baza se giba po TANGENTI luka kvake, ne po lokalnoj x osi. Guranje u +x ima
# komponentu duz poluge koja raste kao sin(kut vrata) - na 50 deg je 77% sile
# usmjereno prema rubu krila, trenje to ne zadrzava i gripper klizi. Izmjereno:
# prirast radijalnog klizanja po koraku podijeljen sa sin(kuta) je konstantan
# (~6), sto je potpis upravo tog smjera sile. Tangenta je okomita na polugu, pa
# te komponente nema.
#
# Bocni pomak je ogranicen jer baza vozi izmedju zidova: MAX_LATERAL_M je
# mjereno odometrijom od pocetne poze, i preko te granice ostaje samo x.
REACH_TARGET_M = 0.45
REACH_GAIN = 0.8  # m/s po metru greske
REACH_DEADBAND_M = 0.03
MAX_LATERAL_M = 0.30
BASE_RAMP_MPS2 = 0.15  # Naredba translacije se mijenja postupno. Bez toga
# skoci na punu vrijednost cim reach_err prijede mrtvu zonu, a pucanje hvata se
# u svim runovima dogadjalo tocno u koraku nakon sto baza prvi put krene.

BASE_RAMP_DECEL_MPS2 = 1.5  # zaustavljanje je brze od kretanja: pri
# usporavanju nema trzaja kojeg bi rampa stitila, a sporo zaustavljanje pusti
# krilo da se otme
YAW_RAMP_DECEL_RADPS2 = 3.0

# Baza se tijekom otvaranja NE zakrece (vidi docstring). Rampa na wz ostaje
# jer je publisher petlja dijeli s translacijom.
YAW_RAMP_RADPS2 = 0.5
SHOULDER_FALLBACK_XY = (0.363, -0.184)  # iiwa_mount_x/y iz kmr_iiwa.urdf.xacro

# --- Ruka ---
# Korak ruke je proporcionalan PROTEKLOM VREMENU, jer baza vozi kontinuirano a
# ciklus traje 0.4-2.1 s ovisno o tome koliko MoveIt planira. S fiksnim korakom
# ruka u duljem ciklusu zaostane za onim sto je baza odvezla.
#
# Nije isto kao raniji pokusaj da se korak izvede iz IZMJERENOG prirasta vrata:
# ondje je nastala povratna veza (naredba pomice vrata, veci pomak daje vecu
# naredbu, vrata su u tri koraka otisla s 1.3 na 34.7 stupnjeva uz 1289 N).
# Vrijeme robot ne moze ubrzati svojim djelovanjem, pa te veze nema.
ARM_STEP_DEG_PER_SEC = 1.5
ARM_STEP_MAX_DEG = 4.0  # gornja granica po koraku
ARM_ACCEL_SEC = 0.5  # blazi start, da napetost ne skoci u prvim koracima
SETTLE_SEC = 0.3
MIN_CYCLE_SEC = 0.0  # bez dopune - korak ruke se skalira vremenom
MONITOR_PERIOD_SEC = 0.05  # 20 Hz. Sve se prije mjerilo jednom po ciklusu, a
# ciklus traje 0.4-2.1 s jer ga odreduje MoveIt - za to vrijeme baza vozi i
# vrata se otvore 15-18 stupnjeva, pa su prag i prekidi uvijek kasnili.
MIN_RADIUS_M = 0.30
MAX_RADIUS_M = 1.20

# --- Sigurnosna udaljenost iz /scan ---
# Umjesto trake ispred baze gleda se NAJBLIZA tocka u punom vidnom polju, jer
# se baza tijekom otvaranja zakrece pa traka vise ne pokazuje kamo ide. Krilo
# koje robot vuce se izuzima.
LIDAR_FRAME = "lidar_link"
SELF_OCCLUSION_HALF_ANGLE_DEG = 94.0
LEAF_MASK_M = 0.15
LEAF_MASK_SKIP_M = 0.25  # maska krila krece tek ovoliko od sarke
NEAREST_STOP_M = 0.08
KMR_LENGTH_M = 1.08
KMR_WIDTH_M = 0.63

# --- Sigurnosni prekidi ---
FORCE_ABORT_N = 700.0
FORCE_SPIKE_STEPS = 3
TRACKING_HARD_FAIL_M = 0.100  # Kad baza zakrece, ruka jedan ciklus zaostane i
# bez da je hvat stradao - izmjereno, promasaj od 70-90 mm uz rad_odst ispod
# 4 mm. Stvarno klizanje hvata pokriva SLIP_ABORT_M, sudar pokriva sila.
TRACKING_FAIL_ERROR_M = 0.03
TRACKING_FAIL_STEPS = 3
SLIP_ABORT_M = 0.020  # koliko gripper smije odstupiti od mjesta na kvaki na
# kojem je bio pri hvatu (mjereno preko taga, neovisno o tome kamo ruku
# saljemo). Ovo JEST klizanje, za razliku od tracking_error koji mjeri samo je
# li ruka stigla tamo kamo smo je poslali.
MAX_STEPS = 200
RETARE_EVERY = 6
RETARE_MAX_FORCE_N = 50.0  # ne tariraj pod opterecenjem - tada tariranje
# "izbrise" stvarnu silu u novu nulu u
# mjesto da nulira samo gravitacijsku i
# konfiguracijsku pristranost

PIVOT_X_M = 0.54  # sredina prednje strane baze
PIVOT_Y_M = 0.0
PIVOT_ANGLE_DEG = 20.0
PIVOT_RATE_RADPS = 0.25
PIVOT_TIMEOUT_SEC = 30.0

LIDAR_WALL_ID_RANGE = range(20)

LOG_PATH = "/tmp/open_revolute.json"


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
    moveit2.max_velocity = 0.15
    moveit2.max_acceleration = 0.15

    # Wrench/gripper/tare/scan/cmd_vel na zasebnom nodu i izvrsavacu - MoveIt2
    # inace gladuje obicne pretplate na istom nodu (isti razlog kao door_open).
    sensor_node = Node("open_revolute_sensors")
    wrench = {"f": None}
    stop_flag = {"v": False}
    base_cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    # SAMO ZA VALIDACIJU - nikad ne ulazi u regulacijsku petlju.
    gt = {"angle_rad": None, "hinge_xy": None, "handle": None}
    scan = {"nearest": None, "have": False}
    leaf_ref = {"seg": None}
    lidar_tf = {"v": None}
    odom = {"xy": None, "yaw": None}

    def _on_odom(msg: Odometry):
        odom["xy"] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        odom["yaw"] = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def _on_wrench(msg: WrenchStamped):
        wrench["f"] = np.array(
            [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]
        )

    def _on_gt_joint(msg: JointState):
        if msg.position:
            gt["angle_rad"] = float(msg.position[0])

    def _on_gt_hinge(msg: PoseStamped):
        gt["hinge_xy"] = np.array([msg.pose.position.x, msg.pose.position.y])

    def _on_gt_handle(msg: PoseStamped):
        gt["handle"] = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )

    tf_buffer = Buffer()
    TransformListener(tf_buffer, sensor_node)

    def _ensure_lidar_tf():
        if lidar_tf["v"] is not None:
            return True
        try:
            tf = tf_buffer.lookup_transform("base_link", LIDAR_FRAME, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return False
        t, q = tf.transform.translation, tf.transform.rotation
        lidar_tf["v"] = (np.array([t.x, t.y, t.z]), (q.x, q.y, q.z, q.w))
        return True

    def _on_leaf(x, y):
        """Lezi li tocka na krilu koje robot vuce? Krilo je duzina sarka->vrh,
        u smjeru gripera."""
        seg = leaf_ref["seg"]
        if seg is None:
            return False
        h, tip = seg
        # Maska krece TEK OD LEAF_MASK_SKIP_M od sarke. Dovratnik je tocno na
        # sarki, pa bi ga maska koja krece od nje izbrisala zajedno s krilom -
        # i robot bi se u njega zabio bez upozorenja.
        dseg = tip - h
        nseg = float(np.linalg.norm(dseg))
        if nseg > 1e-6:
            h = h + dseg / nseg * LEAF_MASK_SKIP_M
        d = tip - h
        n2 = float(np.dot(d, d))
        if n2 < 1e-9:
            return False
        u = float(np.clip(np.dot(np.array([x, y]) - h, d) / n2, 0.0, 1.0))
        return float(np.linalg.norm(np.array([x, y]) - (h + u * d))) < LEAF_MASK_M

    def _on_scan(msg: LaserScan):
        """Najbliza tocka u punom vidnom polju, izuzev krila. Za razliku od
        trake ispred baze, ovo vrijedi i kad je baza zakrenuta."""
        if not _ensure_lidar_tf():
            return
        t, q = lidar_tf["v"]
        mask = math.radians(SELF_OCCLUSION_HALF_ANGLE_DEG)
        nearest = None
        angle = msg.angle_min
        for r in msg.ranges:
            a = angle
            angle += msg.angle_increment
            if not math.isfinite(r) or r < msg.range_min or r > msg.range_max:
                continue
            if abs(wrap_pi(a)) > mask:
                continue
            x_l, y_l = r * math.cos(a), r * math.sin(a)
            x_b, y_b, _ = quat_rotate_vector(q, [x_l, y_l, 0.0])
            x_b += t[0]
            y_b += t[1]
            if _on_leaf(x_b, y_b):
                continue
            d = _dist_to_base(x_b, y_b)
            if nearest is None or d < nearest:
                nearest = d
        scan["nearest"] = nearest
        scan["have"] = True

    sensor_node.create_subscription(LaserScan, "/scan", _on_scan, 10)
    sensor_node.create_subscription(
        WrenchStamped, "/estimation/tcp_wrench", _on_wrench, 10
    )
    sensor_node.create_subscription(
        JointState, "/ground_truth/door_joint", _on_gt_joint, 10
    )
    sensor_node.create_subscription(
        PoseStamped, "/ground_truth/hinge_pose", _on_gt_hinge, 10
    )
    sensor_node.create_subscription(
        PoseStamped, "/ground_truth/handle_pose", _on_gt_handle, 10
    )
    sensor_node.create_subscription(Odometry, "/odom", _on_odom, 10)
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

    node.get_logger().warn("Cekam TF, wrench, /scan i /odom...")
    p_tcp, q_tcp = tcp_pose()
    while (p_tcp is None or wrench["f"] is None or not scan["have"]) and rclpy.ok():
        time.sleep(0.1)
        p_tcp, q_tcp = tcp_pose()

    p_tag, q_tag = tag_pose()
    while p_tag is None and rclpy.ok():
        time.sleep(0.1)
        p_tag, q_tag = tag_pose()

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

    # Orijentacija gripera pri hvatu - referenca za ciljnu orijentaciju.
    q_tcp_grasp = list(q_tcp)

    # Ista referenca, ali prema STVARNOJ kvaki iz simulatora. Sluzi da se
    # izmjereno klizanje razdvoji od greske procjene taga: slip se racuna preko
    # q_tag, a greska od 2 deg u orijentaciji taga daje oko 8 mm laznog
    # klizanja pri |handle_offset| = 0.22 m.
    gt_handle_ref = None if gt["handle"] is None else (p_tcp - gt["handle"]).copy()

    # Radijus mjeren protiv STVARNE sarke. Ako je radius0 (mjeren protiv
    # procijenjene) veci, cilj na luku stalno stoji malo predaleko od sarke, pa
    # ruka svaki korak vuce gripper prema van duz poluge.
    gt_radius0 = (
        None
        if gt["hinge_xy"] is None
        else float(np.linalg.norm(p_tcp[:2] - gt["hinge_xy"]))
    )
    if gt_radius0 is not None:
        node.get_logger().info(
            f"Radijus: procijenjen {radius0:.4f} m, stvarni {gt_radius0:.4f} m, "
            f"razlika {(radius0 - gt_radius0)*1000:+.1f} mm"
        )

    # Sarka je nepomicna u svijetu: fiksiramo je na hvatu i dalje pratimo kroz
    # odometriju. Time otvaranje ne ovisi o tagu, koji nakon nekog kuta izlazi
    # iz vidnog polja kamere.
    # Sarka u odom okviru. base_link se pod odom okvirom i pomice i ZAKRECE,
    # pa pretvorba mora uracunati i rotaciju - bez nje procjena odluta cim baza
    # pocne zakretati (izmjereno, greska je narasla do 836 mm).
    def base_to_odom(p_base):
        c, sn = math.cos(odom["yaw"]), math.sin(odom["yaw"])
        return odom["xy"] + np.array(
            [c * p_base[0] - sn * p_base[1], sn * p_base[0] + c * p_base[1]]
        )

    def odom_to_base(p_odom):
        c, sn = math.cos(odom["yaw"]), math.sin(odom["yaw"])
        d = p_odom - odom["xy"]
        return np.array([c * d[0] + sn * d[1], -sn * d[0] + c * d[1]])

    hinge_odom = (
        None if (odom["xy"] is None or odom["yaw"] is None) else base_to_odom(center0)
    )

    node.get_logger().info(
        f"hinge_odom: {hinge_odom}, odom xy={odom['xy']} yaw={odom['yaw']}"
    )

    def hinge_now():
        """Sarka u base_link, iz fiksne poze u odom okviru."""
        if hinge_odom is None or odom["xy"] is None or odom["yaw"] is None:
            return center0
        return odom_to_base(hinge_odom)

    # Kut vrata se racuna iz smjera sarka->TCP. Dok hvat drzi, gripper je na
    # kvaki, pa je to ujedno i smjer krila.
    def door_dir_angle(p_tcp_xy=None):
        """Kut smjera sarka->TCP u ODOM okviru.

        Racuna se u odom, ne u base_link: baza tijekom kruzenja rotira, pa bi
        se u base_link vlastita rotacija mjerila kao otvorenost vrata i prag bi
        se dosegao prerano, bez da su se vrata stvarno pomakla.
        """
        p_xy = p_tcp_now_ref["v"][:2] if p_tcp_xy is None else p_tcp_xy
        if hinge_odom is None or odom["xy"] is None or odom["yaw"] is None:
            v = p_xy - hinge_now()
        else:
            v = base_to_odom(p_xy) - hinge_odom
        return math.atan2(v[1], v[0])

    p_tcp_now_ref = {"v": p_tcp}
    door_theta0 = door_dir_angle()

    node.get_logger().info(
        f"Os sarke: ({center0[0]:.3f}, {center0[1]:.3f}), radijus {radius0:.3f}m, "
        f"smjer {sign:+.0f}. Cilj: {TARGET_DOOR_ANGLE_DEG:.0f} deg."
    )

    stop_flag = {"v": False}

    wz_state = {"cur": 0.0, "t": time.monotonic()}
    v_state = {"vx": 0.0, "vy": 0.0}

    def base_publisher_loop():
        """Naredbe idu u stalnom ritmu - cmd_vel_bridge primjenjuje zadnju
        primljenu poruku svaki fizicki korak i nema failsafe timeout."""
        while rclpy.ok() and not stop_flag["v"]:
            gripper_pub.publish(Float32(data=1.0))
            now_p = time.monotonic()
            dt_p = now_p - wz_state["t"]
            wz_state["t"] = now_p
            tw = Twist()
            # Zaustavljanje ide brzom rampom, kretanje sporom.
            for axis in ("vx", "vy"):
                tgt_v, cur_v = base_cmd[axis], v_state[axis]
                rate = (
                    BASE_RAMP_DECEL_MPS2 if abs(tgt_v) < abs(cur_v) else BASE_RAMP_MPS2
                )
                step_v = rate * dt_p
                v_state[axis] = (
                    min(tgt_v, cur_v + step_v)
                    if cur_v < tgt_v
                    else max(tgt_v, cur_v - step_v)
                )
            tw.linear.x = v_state["vx"]
            tw.linear.y = v_state["vy"]

            # Rampa na zakretu, da naredba ne skoci iz nule preko praga.
            tgt = base_cmd["wz"]
            cur = wz_state["cur"]
            yaw_rate = YAW_RAMP_DECEL_RADPS2 if abs(tgt) < abs(cur) else YAW_RAMP_RADPS2
            step_p = yaw_rate * dt_p
            wz_state["cur"] = (
                min(tgt, cur + step_p) if cur < tgt else max(tgt, cur - step_p)
            )
            tw.angular.z = wz_state["cur"]

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

    odom_start = None if odom["xy"] is None else odom["xy"].copy()

    # Dijeljeno stanje izmedju nadzorne niti i glavne petlje. Bez brave -
    # citanja i pisanja su pojedinacni floatovi, pa je najgore sto se moze
    # dogoditi da glavna petlja procita vrijednost staru 50 ms.
    mon = {
        "center": center0,
        "door_deg": 0.0,
        "reach": radius0,
        "radial_dev": 0.0,
        "p_tcp": p_tcp,
        "lateral": 0.0,
        "abort": None,
        "stop": False,
    }

    def monitor_loop():
        """Mjeri i regulira bazu na 20 Hz. Cim neki uvjet okine, ODMAH
        zaustavlja bazu - cekanje na glavnu petlju znacilo bi da baza jos
        sekundu i pol gura nakon sto je trebala stati."""
        while rclpy.ok() and not mon["stop"]:
            p_now, _ = tcp_pose()
            if p_now is None:
                time.sleep(MONITOR_PERIOD_SEC)
                continue

            c = hinge_now()
            radial_dir_m = p_now[:2] - c
            rd_m = float(np.linalg.norm(radial_dir_m))
            door_deg = math.degrees(
                abs(wrap_pi(door_dir_angle(p_now[:2]) - door_theta0))
            )
            to_h = p_now[:2] - shoulder_xy
            reach_m = float(np.linalg.norm(to_h))
            radial_dev_m = rd_m - radius0
            lat = 0.0
            if odom["xy"] is not None and odom_start is not None:
                lat = float((odom["xy"] - odom_start)[1])

            mon["center"] = c
            mon["door_deg"] = door_deg
            mon["reach"] = reach_m
            mon["radial_dev"] = radial_dev_m
            mon["p_tcp"] = p_now
            mon["lateral"] = lat

            # --- prekidi ---
            stop_now = None
            if door_deg >= TARGET_DOOR_ANGLE_DEG:
                stop_now = f"USPJEH: vrata otvorena {door_deg:.1f} deg"
            elif abs(radial_dev_m) > SLIP_ABORT_M:
                stop_now = (
                    f"NEUSPJEH: gripper klizi po poluzi "
                    f"({radial_dev_m*1000:+.0f}mm od hvatista)"
                )
            elif scan["nearest"] is not None and scan["nearest"] < NEAREST_STOP_M:
                stop_now = (
                    f"NEUSPJEH: prepreka na {scan['nearest']*1000:.0f} mm od baze"
                )

            if stop_now is not None and mon["abort"] is None:
                base_cmd["vx"] = base_cmd["vy"] = base_cmd["wz"] = 0.0
                mon["abort"] = stop_now
                try:
                    moveit2.cancel_execution()
                except Exception:
                    pass
                time.sleep(MONITOR_PERIOD_SEC)
                continue

            if mon["abort"] is not None:
                time.sleep(MONITOR_PERIOD_SEC)
                continue

            # --- regulator baze ---
            # Baza se giba po TANGENTI luka kvake. Guranje u lokalni +x ima
            # komponentu duz poluge koja raste kao sin(kut vrata) - na 50 deg
            # je 77% sile usmjereno prema rubu krila i gripper klizi. Tangenta
            # je okomita na polugu, pa te komponente nema.
            #
            # Kruzenje oko sarke (baza istovremeno rotira i translatira, da
            # rame ide naprijed) je probano i odbaceno: rotacija platforme ima
            # mrtvu zonu od 0.36 rad/s, pa svako zakretanje krece s barem
            # 20 deg/s i trgne krutu vezu ruka-kvaka. Hvat je pucao u istom
            # koraku u kojem bi kruzenje pocelo, bez obzira na prag i rampu.
            reach_err_m = reach_m - REACH_TARGET_M
            if rd_m > 1e-6 and abs(reach_err_m) > REACH_DEADBAND_M:
                u_rad_m = radial_dir_m / rd_m
                tangent_m = sign * np.array([-u_rad_m[1], u_rad_m[0]])
                v_m = float(
                    np.clip(REACH_GAIN * reach_err_m, -BASE_MAX_MPS, BASE_MAX_MPS)
                )
                vx_m, vy_m = v_m * tangent_m[0], v_m * tangent_m[1]

                # Bocna granica se provjerava u ODOM okviru - zidovi su
                # nepomicni u svijetu, a baza se moze zakrenuti.
                if odom["yaw"] is not None:
                    cy_m, sy_m = math.cos(odom["yaw"]), math.sin(odom["yaw"])
                    v_odom_y = sy_m * vx_m + cy_m * vy_m
                    if abs(lat) >= MAX_LATERAL_M and lat * v_odom_y > 0.0:
                        v_odom_x = cy_m * vx_m - sy_m * vy_m
                        vx_m = cy_m * v_odom_x
                        vy_m = -sy_m * v_odom_x
                base_cmd["vx"], base_cmd["vy"] = float(vx_m), float(vy_m)
            else:
                base_cmd["vx"] = base_cmd["vy"] = 0.0
            base_cmd["wz"] = 0.0

            time.sleep(MONITOR_PERIOD_SEC)

    threading.Thread(target=monitor_loop, daemon=True).start()

    t_start = time.monotonic()
    last_cycle_s = 1.0  # procjena za prvi korak
    force_spike_count = 0
    tracking_fail_count = 0
    steps_since_retare = 0
    outcome = None
    step = 0
    door_angle_deg = 0.0

    while rclpy.ok() and step < MAX_STEPS:
        step += 1
        steps_since_retare += 1
        cycle_start = time.monotonic()
        elapsed = cycle_start - t_start

        p_tcp_now, q_tcp_now = tcp_pose()
        if p_tcp_now is None:
            outcome = "NEUSPJEH: izgubljen TF gripper_tcp"
            break
        p_tcp_now_ref["v"] = p_tcp_now

        # Sve mjereno dolazi iz nadzorne niti (20 Hz), ne iz ovog ciklusa.
        if mon["abort"] is not None:
            outcome = mon["abort"]
            break

        center = mon["center"]
        door_angle_deg = mon["door_deg"]
        reach = mon["reach"]
        radial_dev = mon["radial_dev"]
        lateral = mon["lateral"]
        radial_dir = p_tcp_now[:2] - center
        rd = float(np.linalg.norm(radial_dir))

        # Tag se cita samo za log - nakon nekog kuta izlazi iz vidnog polja, pa
        # se otvaranje na njega vise ne oslanja.
        p_tag_now, q_tag_now = tag_pose()

        # GT se snima nakon sto je procjena osvjezena, da se usporeduju
        # vrijednosti iz istog koraka.
        gt_hinge_snapshot = gt["hinge_xy"]
        gt_angle_deg = (
            math.degrees(gt["angle_rad"]) if gt["angle_rad"] is not None else None
        )
        center_error_mm = (
            float(np.linalg.norm(center - gt_hinge_snapshot)) * 1000
            if gt_hinge_snapshot is not None
            else None
        )
        gt_reach = (
            float(np.linalg.norm(p_tcp_now[:2] - gt_hinge_snapshot))
            if gt_hinge_snapshot is not None
            else None
        )

        # Krilo za masku: od sarke do vrha, u smjeru gripera.
        to_tcp = p_tcp_now[:2] - center
        n_tcp = float(np.linalg.norm(to_tcp))
        if n_tcp > 1e-6:
            leaf_ref["seg"] = (center, center + to_tcp / n_tcp * 0.85)

        ramp = min(1.0, elapsed / ARM_ACCEL_SEC) if ARM_ACCEL_SEC > 0 else 1.0
        step_deg = min(ARM_STEP_MAX_DEG, ARM_STEP_DEG_PER_SEC * last_cycle_s)
        dtheta = sign * math.radians(step_deg) * ramp

        # Cilj je tocka na luku oko sarke, radijusa izmjerenog pri hvatu.
        # Sarka dolazi iz odometrije, ne iz taga.
        if rd > 1e-6:
            rad_vec = radial_dir / rd * radius0
        else:
            rad_vec = radial_dir
        c, sn = math.cos(dtheta), math.sin(dtheta)
        R = np.array([[c, -sn], [sn, c]])
        target_xy = center + R @ rad_vec
        target = np.array([target_xy[0], target_xy[1], p_tcp_now[2]])

        # Orijentacija: poza pri hvatu zakrenuta za IZMJERENI kut vrata plus
        # korak. Mjereno, ne akumulirano - akumulacija je ranije driftala.
        total = sign * math.radians(door_angle_deg) + dtheta
        half = total / 2.0
        target_quat = quat_mul([0.0, 0.0, math.sin(half), math.cos(half)], q_tcp_grasp)

        gt_slip = None
        if gt["handle"] is not None and gt_handle_ref is not None:
            gt_slip = float(np.linalg.norm((p_tcp_now - gt["handle"]) - gt_handle_ref))

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
                "cycle_s": time.monotonic() - cycle_start,
                "step_deg": step_deg,
                "door_angle_deg": door_angle_deg,
                "reach_m": reach,
                "base_vx": base_cmd["vx"],
                "base_vy": base_cmd["vy"],
                "base_wz": base_cmd["wz"],
                "base_wz_out": wz_state["cur"],
                "base_v_out": [v_state["vx"], v_state["vy"]],
                "lateral_m": lateral,
                "nearest_m": scan["nearest"],
                "center_xy": [float(center[0]), float(center[1])],
                "tracking_error_m": tracking_error,
                "angle_error_deg": (
                    math.degrees(angle_error) if angle_error is not None else None
                ),
                "force_N": fmag,
                "radial_dev_mm": radial_dev * 1000,
                "gt_slip_mm": None if gt_slip is None else gt_slip * 1000,
                "gt_radius_m": gt_reach,
                "radius_err_mm": (
                    None if gt_reach is None else (gt_reach - radius0) * 1000
                ),
                "gt_door_angle_deg": gt_angle_deg,
                "center_error_mm": center_error_mm,
            }
        )

        if tracking_error is not None:
            gt_str = f"{gt_angle_deg:.1f}" if gt_angle_deg is not None else "n/a"
            ce_str = f"{center_error_mm:.0f}" if center_error_mm is not None else "n/a"
            gt_slip_str = "n/a" if gt_slip is None else f"{gt_slip*1000:.1f}"
            near_str = (
                "n/a" if scan["nearest"] is None else f"{scan['nearest']*1000:.0f}mm"
            )
            node.get_logger().info(
                f"korak {step:3d}: vrata {door_angle_deg:5.1f}deg  "
                f"doseg={reach:.2f}m  greska={tracking_error*1000:.1f}mm  "
                f"rad_odst={radial_dev*1000:+.1f}mm (gt klizanje {gt_slip_str})  "
                f"sila={fmag:.0f}N  gt={gt_str}deg  sarka_err={ce_str}mm  "
                f"r_err={'n/a' if gt_reach is None else f'{(gt_reach - radius0)*1000:+.1f}'}mm  "
                f"bocno={lateral*1000:+.0f}mm  najblize={near_str}"
                f"wz={base_cmd['wz']:+.2f}/{wz_state['cur']:+.2f}  "
            )
        else:
            node.get_logger().warn(f"korak {step:3d}: TF izgubljen nakon pomaka")

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

        if MIN_CYCLE_SEC > 0.0:
            rest = MIN_CYCLE_SEC - (time.monotonic() - cycle_start)
            if rest > 0:
                time.sleep(rest)
        last_cycle_s = time.monotonic() - cycle_start

    if outcome is None:
        outcome = "NEUSPJEH: dosegnut MAX_STEPS bez ishoda"

    if outcome is not None and outcome.startswith("USPJEH"):
        # Nadzorna nit u ovom trenutku vise ne regulira bazu (abort je
        # postavljen), ali je gasimo da sigurno ne prepise base_cmd.
        mon["stop"] = True
        time.sleep(0.2)

        yaw0 = odom["yaw"]
        if yaw0 is not None:
            target = yaw0 + math.radians(PIVOT_ANGLE_DEG)
            node.get_logger().info(
                f"Zakrecem {PIVOT_ANGLE_DEG:+.0f} deg oko ({PIVOT_X_M}, {PIVOT_Y_M})."
            )
            t_piv = time.monotonic()
            while rclpy.ok() and time.monotonic() - t_piv < PIVOT_TIMEOUT_SEC:
                err = wrap_pi(target - odom["yaw"])
                if abs(err) < math.radians(1.5):
                    break
                w = math.copysign(PIVOT_RATE_RADPS, err)
                # Kruzenje oko tocke c: v = w x (0 - c)
                base_cmd["vx"] = w * PIVOT_Y_M
                base_cmd["vy"] = -w * PIVOT_X_M
                base_cmd["wz"] = math.copysign((abs(w) + 0.204) / 0.562, w)
                time.sleep(0.05)
            base_cmd["vx"] = base_cmd["vy"] = base_cmd["wz"] = 0.0
            node.get_logger().info(
                f"Zakret gotov (greska {math.degrees(wrap_pi(target - odom['yaw'])):+.1f} deg)."
            )
            time.sleep(0.5)

    mon["stop"] = True
    base_cmd["vx"] = base_cmd["vy"] = base_cmd["wz"] = 0.0
    time.sleep(0.5)  # pusti da rampa spusti naredbe na nulu
    stop_flag["v"] = True
    time.sleep(0.1)
    cmd_vel_pub.publish(Twist())
    time.sleep(0.5)

    node.get_logger().info("=== SAZETAK ===")
    node.get_logger().info(f"  ishod: {outcome}")
    node.get_logger().info(f"  koraka: {step}")
    # Konacni kut se ocitava nakon sto se sve smiri - vrijednost iz zadnjeg
    # koraka je snimljena dok je gibanje jos trajalo.
    time.sleep(1.5)
    p_final, _ = tcp_pose()
    if p_final is not None:
        p_tcp_now_ref["v"] = p_final
        door_angle_deg = math.degrees(abs(wrap_pi(door_dir_angle() - door_theta0)))
    gt_final = math.degrees(gt["angle_rad"]) if gt["angle_rad"] is not None else None
    node.get_logger().info(
        f"  vrata otvorena: {door_angle_deg:.1f} deg"
        + (f"  (ground truth {gt_final:.1f} deg)" if gt_final is not None else "")
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


def _dist_to_base(x, y):
    """Udaljenost tocke do ruba pravokutne baze, u base_link."""
    dx = max(abs(x) - KMR_LENGTH_M / 2.0, 0.0)
    dy = max(abs(y) - KMR_WIDTH_M / 2.0, 0.0)
    return math.hypot(dx, dy)


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
