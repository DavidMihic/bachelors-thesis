"""
handle_approach.py - prilaz i hvat kvake preko /perception/handle_pose.
Redoslijed: citanje i usrednjavanje percepcije -> primicanje baze uz
rekonstrukciju poze kvake -> kolizijski objekt vrata -> ready poza ->
pre-grasp -> uron -> bocna korekcija -> hvat.

Kvaka se ne fiksira na gripper. Fiksni joint bi onemogucio mjerenje sile i
momenta, sto je dio zadatka.

Primicanje baze: standoff u door_task_node je velik (~1.35 m) da mali tagovi
na kvaki ostanu u vidnom polju, jer se inace orijentacija hvata raspadne. Na
toj udaljenosti kvaka je izvan dosega ruke, pa se baza nakon ocitanja
primakne. Ocitanje prezivi pomak jer se poza kvake rekonstruira iz njezinog
invarijantnog odnosa prema velikom tagu na vratima: oboje je kruto na istim
vratima, a veliki tag ostaje pouzdan i izbliza. Percepcija time zamjenjuje
odometriju, koju baza nema.

Konvencija iz handle_pose_fusion.py:
  - pozicija = poloviste dva taga (otprilike centar drske)
  - Y-os poruke = duz drske (baseline izmedju tagova)
  - Z-os poruke = od povrsine vrata prema kameri

Konvencija iz gripper.xacro:
  - gripper_tcp lokalni +Z = smjer prstiju, +X = smjer zatvaranja,
    +Y = duz sipke
  - gripper_tcp je os sipke kad je zahvacena, pa je grasp cilj tocno
    handle_pose pozicija

Ciljna orijentacija je orijentacija iz handle_pose rotirana 180 stupnjeva oko
vlastite Y-osi, sto slijedi iz usporedbe te dvije konvencije
(R_gripper = R_handle @ diag(-1,1,-1)). Time gripper prilazi prstima prema
drsci, a sipka zavrsi duz gripperovog Y.

Preduvjet: Isaac Sim, ros2_control, move_group i gripper_bridge rade.

Pokretanje:
    ros2 run kmr_iiwa_task handle_approach
"""

import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from kmr_iiwa_task.add_door_collision import (
    build_vertical_panel_orientation,
    quat_rotate_vector,
    rotmat_to_quat,
    DOOR_PANEL_SIZE,
    TAG_TO_PANEL_CENTER_OFFSET,
    COLLISION_PADDING_DEPTH_M,
    COLLISION_PADDING_WIDTH_HEIGHT_M,
    COLLISION_OBJECT_ID,
)

JOINT_NAMES = [
    "iiwa_joint_1",
    "iiwa_joint_2",
    "iiwa_joint_3",
    "iiwa_joint_4",
    "iiwa_joint_5",
    "iiwa_joint_6",
    "iiwa_joint_7",
]

# Ruka skupljena uz sebe (lakat jako savijen), zapesce zakrenuto tako da
# gripper gleda PREMA NAPRIJED, prema vratima. Pretpostavka je opravdana jer
# door_task_node bazu dovodi manje-vise okomito na vrata. Time je prijelaz u
# pre-grasp cisto priblizavanje bez preorijentacije - a preorijentacija je
# bila uzrok divljih putanja izmedju ready i pre-grasp poze. Skupljena ruka
# ujedno manje zaklanja kameru i tagove na kvaki.
#
# Zadnji zglob je +pi/2 jer je zadana meta KLIZNA vrata, gdje je sipka
# OKOMITA - gripper mora biti zarotiran 90 stupnjeva oko vlastite osi alata
# u odnosu na zakretna vrata, gdje je poluga vodoravna. Zglob 7 rotira bas
# oko te osi, pa se tip vrata mijenja samo tom vrijednoscu.
READY_POSE = [0.0, 0.45, 0.0, -1.9, 0.0, -0.8, 1.5708]

# Odmak pre-grasp tocke unatrag duz Z-osi handle_pose. Kratak, jer kraca
# pravocrtna dionica ima manju sansu da computeCartesianPath naidje na
# singularnost usput i odustane.
STANDOFF_M = 0.06

# Primicanje baze nakon ocitanja - vidi docstring.
REAPPROACH_DELTA_M = 0.18
REAPPROACH_TOLERANCE_M = 0.01
REAPPROACH_KP = 0.6
REAPPROACH_MIN_SPEED = 0.12  # prag statickog trenja baze
REAPPROACH_MAX_SPEED = 0.25
REAPPROACH_TIMEOUT_SEC = 20.0

# Trajanje prikupljanja uzoraka za usrednjavanje percepcije.
SAMPLE_WINDOW_SEC = 2.5

# Grasp cilj je os sipke, a gripper_tcp je po dizajnu bas ta tocka.
GRASP_STANDOFF_M = 0

# Ciljana frakcija zatvorenosti za dobar hvat. Gripper je dimenzioniran tako
# da se oko sipke Ø28 zatvori gotovo do kraja, uz mali preklop, pa je hvat na
# pola hoda znak da sipka nije sjela u utor.
GOOD_GRASP_MIN_FRACTION = 0.90

# Ako prvi pokusaj stane ispod ovoga, prsti su se zaustavili gotovo odmah i
# pretraga se prekida. Iskljuceno (-1.0) jer door_panel kolizijski objekt vec
# stiti od guranja vrata pri korekcijama.
EARLY_ABORT_FRACTION = -1.0

# Koliko puta pokusati "otvori, pomakni, zatvori ponovno" prije odustajanja.
MAX_GRASP_ATTEMPTS = 6

# Pomak po pokusaju pretrage, duz osi prilaza.
RETRY_ADVANCE_M = 0.002

# 180 stupnjeva oko Y, u (x,y,z,w) redoslijedu - vidi docstring.
Y_180 = (0.0, 1.0, 0.0, 0.0)

GRIPPER_STALL_TIMEOUT_SEC = 10.0

# Uzastopnih True ocitanja /gripper_stalled prije nego se stall prihvati kao
# stvaran, a ne prolazan sumni trzaj.
STALL_DEBOUNCE_COUNT = 3

# Ako je gripper_tcp dalje od grasp cilja od ovoga, gripper se ne zatvara.
# wait_until_executed() ne baca gresku kad izvrsavanje trajektorije ne uspije,
# pa je ova provjera jedini nacin da se to uhvati.
MAX_GRASP_POSITION_ERROR_M = 0.03

# Koliko puta ponoviti pravocrtni uron prije odustajanja.
MAX_CARTESIAN_RETRIES = 4

# Ruka zna stici na tocnu poziciju ali nagnuta, a nagnut gripper ne moze
# zatvoriti utor oko sipke. Zato uz pozicijsku ide i provjera orijentacije.
MAX_ORIENTATION_ERROR_RAD = math.radians(5.0)

# Pretraga smije i unatrag, ograniceno - optimum je ponekad plici od
# procijenjene mete.
MIN_DEPTH_OFFSET_M = -0.004

# Kod kliznih vrata sipka je okomita, sto je poznato iz modela, dok je
# procjena nagiba iz malih tagova nepouzdana. Zato se iz percepcije uzima
# samo smjer prema vratima, a nagib se prisiljava na vodoravan. Za zakretna
# vrata, gdje je poluga vodoravna, ovo treba iskljuciti.
FORCE_HORIZONTAL_APPROACH = True

# Prsti 1,2 su na +X strani baze (ox=+0.035), prsti 3,4 na -X (ox=-0.035).
# Razlika njihovih zaustavnih pozicija daje pomak kvake duz osi zatvaranja.
FINGER_SIDE_A = ["gripper_finger_1_joint", "gripper_finger_2_joint"]  # +X
FINGER_SIDE_B = ["gripper_finger_3_joint", "gripper_finger_4_joint"]  # -X
MAX_LATERAL_CORRECTIONS = 6
LATERAL_TOLERANCE_M = 0.0015


def math_dist(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def quat_angle_between(q1, q2):
    """Kut najkrace rotacije izmedju dvije orijentacije, u radijanima.
    Apsolutna vrijednost skalarnog produkta jer q i -q predstavljaju istu
    rotaciju (dvostruko pokrivanje)."""
    d = abs(sum(a * b for a, b in zip(q1, q2)))
    return 2.0 * math.acos(min(1.0, d))


def quat_multiply(q1, q2):
    """Hamilton produkt, oba u (x,y,z,w) redoslijedu. q_total = q1 (x) q2 -
    q2 se primjenjuje kao DODATNA LOKALNA rotacija nakon q1 (standardna
    konvencija za body-frame kompoziciju)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quat_z_axis(q):
    """Vrati Z-os (treci stupac rotacijske matrice) direktno iz kvaterniona
    (x,y,z,w), kompaktna standardna formula - provjereno protiv
    handle_pose_fusion.py-evog quat_to_rotmat()."""
    x, y, z, w = q
    return np.array(
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ]
    )


def quat_conj(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def tf_compose(pa, qa, pb, qb):
    """Slozi transformacije: A->B pa B->C daje A->C."""
    p = np.array(pa) + np.array(quat_rotate_vector(qa, list(pb)))
    return p, quat_multiply(qa, qb)


def tf_inverse(p, q):
    qi = quat_conj(q)
    return -np.array(quat_rotate_vector(qi, list(p))), qi


def orient_from_handle(q_handle, logger=None):
    """Iz orijentacije kvake izvedi (z_axis, q_gripper) za prilaz.

    Uz FORCE_HORIZONTAL_APPROACH os prilaza se projicira na vodoravnu ravninu
    i orijentacija se ponovno gradi oko poznate okomite sipke, pa se iz
    percepcije zadrzava samo smjer prema vratima. Bez toga gripper kopira
    nagib kvake i ulazi ukoso.
    """
    z_axis = quat_z_axis(q_handle)
    if not FORCE_HORIZONTAL_APPROACH:
        return z_axis, quat_multiply(q_handle, Y_180)

    z_flat = np.array([z_axis[0], z_axis[1], 0.0])
    n = np.linalg.norm(z_flat)
    if n < 1e-6:
        if logger is not None:
            logger.warn(
                "Os prilaza je gotovo vertikalna - ne mogu je projicirati, "
                "koristim sirovu orijentaciju iz percepcije."
            )
        return z_axis, quat_multiply(q_handle, Y_180)

    z_axis = z_flat / n
    y_axis = np.array([0.0, 0.0, 1.0])  # sipka je okomita
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    q_flat = rotmat_to_quat([x_axis, y_axis, z_axis])
    return z_axis, quat_multiply(q_flat, Y_180)


def compute_target(hp_x, hp_y, hp_z, z_axis, q_gripper, standoff):
    position = [
        hp_x + standoff * float(z_axis[0]),
        hp_y + standoff * float(z_axis[1]),
        hp_z + standoff * float(z_axis[2]),
    ]
    return position, list(q_gripper)


def collect_average_pose(samples, window_sec, wait_timeout_sec=None):
    """Ocisti samples listu, ceka prvu poruku (do wait_timeout_sec ako je
    zadano, inace beskonacno), prikupi jos window_sec, uprosjeci poziciju i
    orijentaciju (isti sign-korigirani kvaternionski prosjek). Vraca
    (hp_x, hp_y, hp_z, q_handle) ili None ako ni jedna poruka nije stigla
    unutar wait_timeout_sec."""
    samples.clear()
    waited = 0.0
    while len(samples) == 0 and rclpy.ok():
        if wait_timeout_sec is not None and waited >= wait_timeout_sec:
            return None
        time.sleep(0.1)
        waited += 0.1

    time.sleep(window_sec)

    positions = np.array(
        [[s.pose.position.x, s.pose.position.y, s.pose.position.z] for s in samples]
    )
    quats = np.array(
        [
            [
                s.pose.orientation.x,
                s.pose.orientation.y,
                s.pose.orientation.z,
                s.pose.orientation.w,
            ]
            for s in samples
        ]
    )

    avg_position = positions.mean(axis=0)

    ref = quats[0]
    for i in range(len(quats)):
        if np.dot(quats[i], ref) < 0:
            quats[i] = -quats[i]
    avg_quat = quats.mean(axis=0)
    avg_quat /= np.linalg.norm(avg_quat)

    hp_x, hp_y, hp_z = avg_position
    return hp_x, hp_y, hp_z, tuple(avg_quat)


def run_grasp_sequence(node, tf_buffer, callback_group):
    """Odradi cijeli hvat kvake na VEC POKRENUTOM i VEC SPINANOM nodu.

    Pozivatelj je duzan prije poziva:
      - pokrenuti rclpy i stvoriti node,
      - dodati node u izvrsavac i vrtjeti ga u zasebnoj niti,
      - stvoriti tf_buffer s TransformListenerom na tom nodu.
    Ovako je ista logika upotrebljiva i samostalno i iz door_task_node, bez
    dupliciranja koda.

    Vraca True ako je hvat postignut, inace False.
    """
    moveit2 = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name="base_link",
        end_effector_name="gripper_tcp",
        group_name="iiwa_arm",
        callback_group=callback_group,
    )

    # Gripper na zasebnom nodu i izvrsavacu - MoveIt2-ova pozadinska aktivnost
    # inace gladuje obicne pretplate na istom nodu.
    gripper_node = Node("handle_approach_gripper_monitor")
    gripper_pub = gripper_node.create_publisher(Float32, "/gripper_cmd", 10)
    cmd_vel_pub = gripper_node.create_publisher(Twist, "/cmd_vel", 10)
    stalled = {"value": False, "consecutive_true": 0}
    state = {"value": None}
    finger_pos = {"A": None, "B": None}

    def _on_isaac_states(msg):
        pos = dict(zip(msg.name, msg.position))
        try:
            finger_pos["A"] = float(np.mean([pos[n] for n in FINGER_SIDE_A]))
            finger_pos["B"] = float(np.mean([pos[n] for n in FINGER_SIDE_B]))
        except KeyError:
            pass

    gripper_node.create_subscription(
        JointState, "/isaac_joint_states", _on_isaac_states, 10
    )

    def _on_stalled(msg):
        # Debounce, ne obican latch: prvi True zna biti prolazan trzaj blizu
        # pocetka zatvaranja, pa bi petlja prestala pratiti prerano i
        # zabiljezila gotovo nultu frakciju iako se gripper jos zatvara.
        if msg.data:
            stalled["consecutive_true"] += 1
            if stalled["consecutive_true"] >= STALL_DEBOUNCE_COUNT:
                stalled["value"] = True
        else:
            stalled["consecutive_true"] = 0

    def _on_state(msg):
        state["value"] = msg.data

    gripper_node.create_subscription(Bool, "/gripper_stalled", _on_stalled, 10)
    gripper_node.create_subscription(Float32, "/gripper_state", _on_state, 10)

    gripper_executor = SingleThreadedExecutor()
    gripper_executor.add_node(gripper_node)
    gripper_thread = threading.Thread(target=gripper_executor.spin, daemon=True)
    gripper_thread.start()

    # --- Korak 0: citaj i usrednji /perception/handle_pose PRIJE bilo kakvog
    # pokreta ruke - ready poza (korak 1) proteze ruku naprijed i MOZE
    # zakloniti tagove kameri, pa citanje mora biti prvo, ne nakon toga. ---
    samples = []
    node.create_subscription(
        PoseStamped,
        "/perception/handle_pose",
        samples.append,
        10,
        callback_group=callback_group,
    )

    node.get_logger().info("Cekam /perception/handle_pose...")
    hp_x, hp_y, hp_z, q_handle = collect_average_pose(samples, SAMPLE_WINDOW_SEC)
    node.get_logger().info(
        f"Uprosjecena pozicija=({hp_x:.3f}, {hp_y:.3f}, {hp_z:.3f}), "
        f"orijentacija={q_handle}"
    )

    z_axis, q_gripper = orient_from_handle(q_handle, node.get_logger())

    # --- Korak 0.5: primakni bazu, uz rekonstrukciju poze kvake ---
    # Standoff je namjerno velik da mali tagovi na kvaki ostanu vidljivi, ali
    # je kvaka na toj udaljenosti izvan dosega ruke. Primicemo se regulirajuci
    # po VELIKOM tagu (pouzdan i izbliza), a pozu kvake rekonstruiramo iz
    # njezinog invarijantnog odnosa prema tom tagu - oboje je kruto na istim
    # vratima. Bez toga bi pomak baze obesmislio ocitanje, jer je handle_pose
    # izrazen u base_link, a baza nema odometriju.
    def read_door_tag():
        try:
            t = tf_buffer.lookup_transform(
                "base_link", "door_tag_center", rclpy.time.Time()
            )
            tr, r = t.transform.translation, t.transform.rotation
            return np.array([tr.x, tr.y, tr.z]), (r.x, r.y, r.z, r.w)
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None, None

    node.get_logger().info("Cekam door_tag_center za rekonstrukciju...")
    p_tag, q_tag = None, None
    while p_tag is None and rclpy.ok():
        p_tag, q_tag = read_door_tag()
        time.sleep(0.1)

    p_tag_inv, q_tag_inv = tf_inverse(p_tag, q_tag)
    p_rel, q_rel = tf_compose(
        p_tag_inv, q_tag_inv, np.array([hp_x, hp_y, hp_z]), q_handle
    )
    node.get_logger().info(f"Kvaka u okviru velikog taga: {np.round(p_rel, 4)}")

    d_start = float(np.linalg.norm(p_tag[:2]))
    d_target = d_start - REAPPROACH_DELTA_M
    node.get_logger().info(
        f"Primicem bazu: {d_start:.3f} m -> {d_target:.3f} m do velikog taga"
    )

    t_begin = time.monotonic()
    while rclpy.ok():
        p_tag_now, _ = read_door_tag()
        if p_tag_now is None:
            time.sleep(0.05)
            continue
        err = float(np.linalg.norm(p_tag_now[:2])) - d_target
        if abs(err) < REAPPROACH_TOLERANCE_M:
            break
        if time.monotonic() - t_begin > REAPPROACH_TIMEOUT_SEC:
            node.get_logger().warn(
                "Timeout pri primicanju baze - nastavljam s postignutim."
            )
            break
        v = REAPPROACH_KP * err
        v = math.copysign(
            min(max(abs(v), REAPPROACH_MIN_SPEED), REAPPROACH_MAX_SPEED), v
        )
        tw = Twist()
        tw.linear.x = v
        cmd_vel_pub.publish(tw)
        time.sleep(0.05)

    cmd_vel_pub.publish(Twist())
    time.sleep(1.5)  # neka se baza smiri prije novog ocitanja

    p_tag2, q_tag2 = None, None
    while p_tag2 is None and rclpy.ok():
        p_tag2, q_tag2 = read_door_tag()
        time.sleep(0.1)

    p_handle_new, q_handle_new = tf_compose(p_tag2, q_tag2, p_rel, q_rel)
    hp_x, hp_y, hp_z = p_handle_new
    q_handle = q_handle_new
    z_axis, q_gripper = orient_from_handle(q_handle, node.get_logger())
    node.get_logger().info(
        f"Kvaka rekonstruirana nakon pomaka: ({hp_x:.3f}, {hp_y:.3f}, {hp_z:.3f})"
    )

    # --- Korak 0.6: vrata kao kolizijski objekt, tek sad nakon primicanja
    # baze. Racuna se iz door_tag_center izrazenog u base_link, pa bi ga pomak
    # baze ucinio zastarjelim da je dodan ranije. ---
    node.get_logger().info(
        "Cekam TF base_link -> door_tag_center (za kolizijski objekt vrata)..."
    )
    door_tag_transform = None
    while door_tag_transform is None and rclpy.ok():
        try:
            door_tag_transform = tf_buffer.lookup_transform(
                "base_link", "door_tag_center", rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            pass

    dt = door_tag_transform.transform.translation
    dq = door_tag_transform.transform.rotation
    door_tag_pos = [dt.x, dt.y, dt.z]
    door_tag_quat = [dq.x, dq.y, dq.z, dq.w]

    door_panel_quat = build_vertical_panel_orientation(door_tag_quat)
    door_depth_axis = quat_rotate_vector(door_panel_quat, [1.0, 0.0, 0.0])
    door_panel_center = [
        door_tag_pos[i] + TAG_TO_PANEL_CENTER_OFFSET[0] * door_depth_axis[i]
        for i in range(3)
    ]
    door_padded_dims = [
        DOOR_PANEL_SIZE[0] + COLLISION_PADDING_DEPTH_M,
        DOOR_PANEL_SIZE[1] + COLLISION_PADDING_WIDTH_HEIGHT_M,
        DOOR_PANEL_SIZE[2] + COLLISION_PADDING_WIDTH_HEIGHT_M,
    ]

    moveit2.add_collision_primitive(
        id=COLLISION_OBJECT_ID,
        primitive_type=SolidPrimitive.BOX,
        dimensions=door_padded_dims,
        position=door_panel_center,
        quat_xyzw=door_panel_quat,
    )
    node.get_logger().info(
        f"Kolizijski objekt vrata dodan (svjeze, ova sesija): centar={door_panel_center}"
    )

    # --- Korak 1: ready poza. Tek sad, kad je cilj vec izracunat iz cistog
    # ocitanja, jer pruzena ruka moze zakloniti tagove kameri. ---
    node.get_logger().info(f"Saljem ready pozu: {READY_POSE}")
    moveit2.move_to_configuration(READY_POSE)
    moveit2.wait_until_executed()

    # Sigurnosno usporavanje za sve pokrete blizu vrata.
    moveit2.max_velocity = 0.05
    moveit2.max_acceleration = 0.05

    # --- Korak 2: pre-grasp ---
    pre_grasp_position, pre_grasp_quat = compute_target(
        hp_x, hp_y, hp_z, z_axis, q_gripper, STANDOFF_M
    )
    node.get_logger().info(
        f"Pre-grasp cilj: pozicija={pre_grasp_position}, orijentacija={pre_grasp_quat}"
    )

    def read_pre_grasp_error():
        """Vrati (pozicijska_greska_m, kutna_greska_rad, ok)."""
        try:
            actual = tf_buffer.lookup_transform(
                "base_link", "gripper_tcp", rclpy.time.Time()
            )
            at = actual.transform.translation
            ar = actual.transform.rotation
            return (
                math_dist((at.x, at.y, at.z), pre_grasp_position),
                quat_angle_between((ar.x, ar.y, ar.z, ar.w), pre_grasp_quat),
                True,
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            node.get_logger().warn(f"Ne mogu procitati TF gripper_tcp ({exc}).")
            return 0.0, 0.0, False

    # wait_until_executed() ne baca gresku kad planiranje ili izvrsavanje ne
    # uspije, pa bi bez ove provjere skripta nastavila dalje i kad se ruka
    # uopce nije pomaknula.
    for pre_grasp_attempt in range(3):
        moveit2.move_to_pose(
            position=pre_grasp_position, quat_xyzw=pre_grasp_quat, cartesian=False
        )
        moveit2.wait_until_executed()
        pg_error, pg_angle, pg_tf_ok = read_pre_grasp_error()
        node.get_logger().info(
            f"Nakon pre-grasp pokusaja {pre_grasp_attempt}, "
            f"odstupanje={pg_error:.3f}m, nagib={math.degrees(pg_angle):.1f}deg"
        )
        if (
            pg_error <= MAX_GRASP_POSITION_ERROR_M
            and pg_angle <= MAX_ORIENTATION_ERROR_RAD
        ):
            break
    else:
        node.get_logger().error(
            f"Pre-grasp NIJE dosegnut nakon 3 pokusaja (odstupanje={pg_error:.3f}m, "
            f"nagib={math.degrees(pg_angle):.1f}deg) - odustajem, ne nastavljam "
            "na grasp-dive iz krive bazne poze."
        )
        gripper_executor.shutdown()
        gripper_node.destroy_node()
        gripper_thread.join(timeout=2.0)
        return False

    node.get_logger().info("Pre-grasp dosegnut.")

    # --- Korak 2.5: svjeze ocitanje izbliza, gdje je AprilTag procjena
    # preciznija. Ako tagovi odavde nisu vidljivi, koristi se staro. ---
    node.get_logger().info("Pokusavam svjeze ocitanje na pre-grasp tocki...")
    fresh = collect_average_pose(samples, SAMPLE_WINDOW_SEC, wait_timeout_sec=1.5)
    if fresh is not None:
        hp_x, hp_y, hp_z, q_handle = fresh
        z_axis, q_gripper = orient_from_handle(q_handle, node.get_logger())
        node.get_logger().info(
            f"Svjeza pozicija=({hp_x:.3f}, {hp_y:.3f}, {hp_z:.3f}), "
            f"orijentacija={q_handle} - koristim ovo za grasp cilj."
        )
    else:
        node.get_logger().warn(
            "Nema svjezeg ocitanja s pre-grasp tocke (mozda zaklon) - "
            "koristim staro (izdaleka) ocitanje za grasp cilj."
        )

    # --- Korak 3: uron do kvake, pravocrtno. Pravocrtna interpolacija po
    # definiciji ne moze rotirati usput, dok RRT u ovom stisnutom prostoru
    # bira putanje koje dovedu gripper ukoso. ---
    grasp_position, grasp_quat = compute_target(
        hp_x, hp_y, hp_z, z_axis, q_gripper, GRASP_STANDOFF_M
    )
    node.get_logger().info(
        f"Grasp cilj: pozicija={grasp_position}, orijentacija={grasp_quat}"
    )

    def read_gripper_tcp_error():
        """Vrati (pozicijska_greska_m, kutna_greska_rad, ok) u odnosu na grasp
        cilj, ili (0.0, 0.0, False) ako TF trenutno nije citljiv."""
        try:
            actual = tf_buffer.lookup_transform(
                "base_link", "gripper_tcp", rclpy.time.Time()
            )
            at = actual.transform.translation
            ar = actual.transform.rotation
            return (
                math_dist((at.x, at.y, at.z), grasp_position),
                quat_angle_between((ar.x, ar.y, ar.z, ar.w), grasp_quat),
                True,
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            node.get_logger().warn(f"Ne mogu procitati TF gripper_tcp ({exc}).")
            return 0.0, 0.0, False

    # Bez RRT fallbacka: RRT ovdje zna birati putanju koja udari u vrata.
    # Umjesto toga se ista pravocrtna metoda ponavlja - planiranje nije
    # deterministicko, pa ponovljeni pokusaj cesto uspije.
    for grasp_dive_attempt in range(MAX_CARTESIAN_RETRIES):
        moveit2.move_to_pose(
            position=grasp_position, quat_xyzw=grasp_quat, cartesian=True
        )
        moveit2.wait_until_executed()
        error, angle, tf_ok = read_gripper_tcp_error()
        node.get_logger().info(
            f"Nakon cartesian=True pokusaja {grasp_dive_attempt}, "
            f"odstupanje={error:.3f}m, nagib={math.degrees(angle):.1f}deg"
        )
        if not tf_ok or (
            error <= MAX_GRASP_POSITION_ERROR_M and angle <= MAX_ORIENTATION_ERROR_RAD
        ):
            break

    node.get_logger().info("Grasp poza dosegnuta.")

    # --- Sigurnosna provjera: je li ruka STVARNO stigla na izracunati cilj?
    # wait_until_executed() ne baca gresku ni kad izvrsavanje trajektorije
    # ne uspije (npr. STATUS_ABORTED) - bez ove provjere, skripta bi slijepo
    # nastavila i zatvorila gripper gdje god je ruka zavrsila, ne nuzno na
    # cilju. Isti TF obrazac koji koristimo cijeli razgovor. ---
    if error > MAX_GRASP_POSITION_ERROR_M or angle > MAX_ORIENTATION_ERROR_RAD:
        node.get_logger().error(
            f"Grasp poza nije dosegnuta (odstupanje={error:.3f}m, "
            f"nagib={math.degrees(angle):.1f}deg; dopusteno "
            f"{MAX_GRASP_POSITION_ERROR_M}m i "
            f"{math.degrees(MAX_ORIENTATION_ERROR_RAD):.0f}deg). "
            "NE zatvaram gripper na krivoj pozi."
        )
        gripper_executor.shutdown()
        gripper_node.destroy_node()
        gripper_thread.join(timeout=2.0)
        return False

    # --- Korak 4: pomakni malo naprijed, stisni, otpusti, ponovi - pozicija
    # se AKUMULIRA (ne resetira na originalnu grasp_position izmedju
    # pokusaja), tako da ukupni hod naprijed ostane malen i kontroliran.
    # Ciljamo DOBAR hvat (frakcija blizu 1.0), ne bilo koji stall -
    # gripper.xacro dizajnerski komentar: "q=0.025 = POTPUNO ZATVOREN
    # (namjerni preklop ~1.5mm sa sipkom Ø28 u centru)" - nasa sipka je bas
    # Ø28 (radius 0.014, iz sliding_door.urdf), pa je DIZAJNIRANO ponasanje
    # da se ispravno uhvacena sipka zatvori gotovo do kraja, ne na pola
    # hoda. Stall na nizoj frakciji = prsti stali prerano ILI smo vec
    # presli sipku - ne znamo unaprijed koji smjer je ispravan, pa
    # koristimo hill-climbing pretragu ispod (usporedi sa proslim
    # pokusajem, okreni smjer ako se pogorsalo). ---
    node.get_logger().info("Zatvaram gripper...")

    # --- Bocna korekcija duz osi zatvaranja prstiju (lokalni X gripera).
    # Smjer i iznos slijede iz razlike zaustavnih pozicija dviju strana:
    # strana koja stane ranije je ona prema kojoj je kvaka pomaknuta. ---
    gripper_x = np.array(quat_rotate_vector(grasp_quat, [1.0, 0.0, 0.0]))

    for lat in range(MAX_LATERAL_CORRECTIONS):
        stalled["value"] = False
        stalled["consecutive_true"] = 0
        gripper_pub.publish(Float32(data=1.0))
        waited = 0.0
        while (
            not stalled["value"] and waited < GRIPPER_STALL_TIMEOUT_SEC and rclpy.ok()
        ):
            time.sleep(0.1)
            waited += 0.1

        qa, qb = finger_pos["A"], finger_pos["B"]
        if qa is None or qb is None:
            node.get_logger().warn(
                "Nema /isaac_joint_states - preskacem bocnu korekciju."
            )
            break

        d = (qb - qa) / 2.0
        node.get_logger().info(
            f"Bocna asimetrija: strana A={qa:.4f} B={qb:.4f} -> pomak {d*1000:+.1f} mm"
        )
        if abs(d) <= LATERAL_TOLERANCE_M:
            node.get_logger().info("Bocno poravnato.")
            break

        gripper_pub.publish(Float32(data=0.0))
        time.sleep(0.5)
        grasp_position = [grasp_position[j] + d * float(gripper_x[j]) for j in range(3)]
        node.get_logger().info(f"Pomicem bocno na {grasp_position}")
        moveit2.move_to_pose(
            position=grasp_position, quat_xyzw=grasp_quat, cartesian=True
        )
        moveit2.wait_until_executed()

    # depth_offset je pomak od originalne grasp_position duz osi prilaza,
    # pozitivno prema kvaki. Ogranicen je s MIN_DEPTH_OFFSET_M jer povlacenje
    # unatrag zna lazno pokazati visu frakciju - prsti se zatvore u prazno.
    depth_offset = 0.0
    current_position = list(grasp_position)
    direction = 1.0  # +1 = dublje (prema -Z_axis), -1 = natrag (prema +Z_axis)
    best_fraction = -1.0
    best_position = list(grasp_position)
    prev_fraction = None
    good_grasp = False
    attempt = 0
    while attempt <= MAX_GRASP_ATTEMPTS and rclpy.ok():
        stalled["value"] = False
        stalled["consecutive_true"] = 0
        gripper_pub.publish(Float32(data=1.0))

        waited = 0.0
        while (
            not stalled["value"] and waited < GRIPPER_STALL_TIMEOUT_SEC and rclpy.ok()
        ):
            time.sleep(0.1)
            waited += 0.1

        if not stalled["value"]:
            node.get_logger().warn(
                f"Gripper NIJE stalled nakon {GRIPPER_STALL_TIMEOUT_SEC}s "
                f"(pokusaj {attempt}) - vjerojatno posve promasen hvat."
            )
            break

        fraction = state["value"] if state["value"] is not None else 0.0
        node.get_logger().info(
            f"Stalled nakon {waited:.1f}s na frakciji {fraction:.3f} (pokusaj {attempt})"
        )

        if attempt == 0 and fraction < EARLY_ABORT_FRACTION:
            node.get_logger().error(
                f"PRVI pokusaj je stao na frakciji {fraction:.3f} (< {EARLY_ABORT_FRACTION}) - "
                "prsti su se zaustavili GOTOVO ODMAH, prije bilo kakvog stvarnog "
                "zatvaranja. To je znak da je gripper dotaknuo NESTO POGRESNO "
                "(npr. plosnata strana vrata, ne sipka kroz kuku), ne 'malo "
                "prekratko/predaleko' - daljnje sitne korekcije pozicije "
                "vjerojatno nece pomoci i mogu dalje gurati vrata. Odustajem "
                "ODMAH umjesto pokusavanja hill-climbinga."
            )
            gripper_pub.publish(Float32(data=0.0))
            break

        if fraction > best_fraction:
            best_fraction = fraction
            best_position = list(current_position)

        if fraction >= GOOD_GRASP_MIN_FRACTION:
            good_grasp = True
            node.get_logger().info(
                "Frakcija blizu 1.0 - izgleda kao dobar, dubok hvat."
            )
            break

        if attempt == MAX_GRASP_ATTEMPTS:
            node.get_logger().warn(
                f"Frakcija {fraction:.3f} i dalje ispod {GOOD_GRASP_MIN_FRACTION} nakon "
                f"{MAX_GRASP_ATTEMPTS} pokusaja - odustajem, hvat vjerojatno plitak/promasen. "
                f"Najbolja vidjena frakcija: {best_fraction:.3f}."
            )
            break

        # Hill-climbing: ako se frakcija POGORSALA naspram proslog pokusaja,
        # smjer je bio krivi - okreni ga i vrati se na najbolju dosad
        # vidjenu poziciju prije nastavka u novom smjeru. Ako se poboljsala
        # (ili je ovo prvi pokusaj), nastavi u istom smjeru.
        if prev_fraction is not None and fraction < prev_fraction:
            direction *= -1.0
            current_position = list(best_position)
            node.get_logger().info(
                f"Frakcija se pogorsala (bila {prev_fraction:.3f}) - okrecem smjer "
                f"i vracam se na najbolju poziciju (frakcija {best_fraction:.3f})."
            )
        prev_fraction = fraction

        depth_offset += direction * RETRY_ADVANCE_M
        if depth_offset < MIN_DEPTH_OFFSET_M:
            depth_offset = MIN_DEPTH_OFFSET_M
            node.get_logger().info(
                "Pretraga je dosegla najplicu dopustenu tocku - dalje unatrag "
                "ne idem."
            )

        node.get_logger().info(
            f"Otvaram, pomicem na dubinu {depth_offset*1000:.1f}mm od originalne mete, "
            "pokusavam ponovno..."
        )
        gripper_pub.publish(Float32(data=0.0))
        time.sleep(0.5)

        current_position = [
            grasp_position[j] - depth_offset * float(z_axis[j]) for j in range(3)
        ]
        moveit2.move_to_pose(
            position=current_position, quat_xyzw=grasp_quat, cartesian=True
        )
        moveit2.wait_until_executed()

        attempt += 1

    if good_grasp:
        node.get_logger().info("Dobar hvat postignut.")
    else:
        node.get_logger().warn(
            f"Dobar hvat NIJE postignut. Najbolja vidjena frakcija: {best_fraction:.3f} "
            f"na poziciji {best_position}."
        )

    gripper_executor.shutdown()
    gripper_node.destroy_node()
    gripper_thread.join(timeout=2.0)
    return good_grasp


def spin_node_forever(node, executor):
    """Vrti izvrsavac otporno na iznimke. NE koristi executor.spin izravno -
    jedna neuhvacena iznimka iz pymoveit2/rclpy internog action clienta tiho
    ubije nit, nakon cega se nijedan callback vise ne poziva."""
    while rclpy.ok():
        try:
            executor.spin_once(timeout_sec=0.1)
        except Exception as exc:
            node.get_logger().error(f"Executor spin iznimka (nastavljam): {exc}")


def main():
    rclpy.init()
    node = Node("handle_approach")
    callback_group = ReentrantCallbackGroup()

    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    threading.Thread(
        target=spin_node_forever, args=(node, executor), daemon=True
    ).start()

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    run_grasp_sequence(node, tf_buffer, callback_group)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
