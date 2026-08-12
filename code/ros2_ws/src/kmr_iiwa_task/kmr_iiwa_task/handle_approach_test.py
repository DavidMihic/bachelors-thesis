"""
handle_approach_test.py - puni test prilaska i hvata kvake preko
/perception/handle_pose: ready poza -> citanje/usrednjavanje percepcije ->
pre-grasp -> grasp-dive -> zatvaranje grippera. Attach (fiksni joint) jos
NIJE implementiran - to je sljedeci korak nakon ovoga.

Konvencija iz handle_pose_fusion.py (provjereno u izvoru):
  - pozicija = poloviste dva taga (otprilike centar drske)
  - Y-os poruke = duz drske (baseline izmedju tagova)
  - Z-os poruke = "gleda prema" - od povrsine vrata PREMA kameri/robotu

Konvencija iz gripper.xacro (provjereno u izvoru):
  - gripper_base/gripper_tcp lokalni +Z = smjer prstiju (alat gleda "naprijed")
  - hvat se zatvara duz lokalnog X, sipka/kvaka lezi duz lokalnog Y
  - gripper_tcp referentna tocka = os sipke KAD JE ZAHVACENA (q=0.026) - znaci
    grasp cilj = TOCNO handle_pose pozicija

Da gripper ispravno prilazi (prstima PREMA drsci, ne od nje) i da sipka
zavrsi duz gripperovog Y (kako dizajn ocekuje), ciljna orijentacija =
orijentacija iz handle_pose ROTIRANA 180 stupnjeva oko VLASTITE Y-osi
(flips predznak X i Z, Y ostaje) - izvedeno usporedbom dvije konvencije
gore (R_gripper = R_handle @ diag(-1,1,-1)), ne pretpostavka.

READY_POSE: poslana NAKON citanja/usrednjavanja percepcije (ne prije - ready
poza proteze ruku naprijed i MOZE zakloniti tagove kameri, pa citanje mora
biti prvo, dok je ruka jos u pocetnoj/cistoj konfiguraciji), a PRIJE
pokreta prema pre-grasp cilju, da RRT planer krece iz
razumne (ne rubne/nulte) konfiguracije - smanjuje sansu za neprirodno
"skvrcena" rjesenja. Trenutna vrijednost potvrdjena vizualno kao
"u redu, radi", ALI je pruzena previse naprijed po korisnikovoj ocjeni -
TODO: pomaknuti end effector u ready pozi ~15-30cm prema -X kad se nadje
vrijeme, nije prioritet sad.

Keep-out zona za zaklon kamere je isprobana i namjerno UKLONJENA (vidi
prijasnju verziju/razgovor) - blokirala je prevelik dio prostora. Trenutna
arhitektura (jedno usrednjeno ocitanje PRIJE pokreta, svi pokreti nakon
toga "slijepi") to ionako ne treba za sad.

Isaac Sim, ros2_control_test.launch.py, move_group.launch.py moraju raditi,
kao i gripper_driver.py (za /gripper_cmd, /gripper_stalled).

Pokretanje:
    ros2 run kmr_iiwa_task handle_approach_test
"""

import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
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

# Poslano prije citanja percepcije - vidi napomenu u docstringu.
READY_POSE = [0.0, 0.6, 0.0, -1.2, 0.0, 1.0, 0.0]

# Koliko unatrag (duz Z-osi handle_pose) za pre-grasp tocku. Bio 0.15 dok
# je padding u dubini bio debeo (0.15) - sad kad je COLLISION_PADDING_DEPTH_M
# tanak (0.02), ne treba tako velik standoff da izbjegnemo vlastitu
# sigurnosnu zonu. Smanjen na 0.06 - kraci grasp-dive povecava sansu da
# cartesian=True uspije (kraca pravocrtna dionica = manja sansa da
# computeCartesianPath naidje na singularnost/granicu usput i odustane).
STANDOFF_M = 0.06

# Koliko dugo prikupljati uzorke za usrednjavanje (i za prvo i za drugo,
# svjeze ocitanje na pre-grasp tocki).
SAMPLE_WINDOW_SEC = 2.5

# Grasp cilj = TOCNO os sipke (gripper_tcp je po dizajnu definiran kao os
# sipke kad je zahvacena, vidi gripper.xacro).
GRASP_STANDOFF_M = 0

# Ciljana frakcija zatvorenosti za DOBAR hvat (vidi Korak 4) -
# gripper.xacro dizajnerski komentar: "q=0.025 = POTPUNO ZATVOREN (namjerni
# preklop ~1.5mm sa sipkom Ø28 u centru)" - nasa sipka je bas Ø28 (radius
# 0.014, iz sliding_door.urdf "handle" linka), pa je 1.0 (potpuno zatvoren,
# uz otpor) DIZAJNIRANI ocekivani ishod za ispravno uhvacenu sipku, ne
# nesto na pola hoda. Malo popustanja (0.82, ne bas 1.0) za simulacijsku
# netocnost.
GOOD_GRASP_MIN_FRACTION = 0.82

# Ako PRVI pokusaj stane ispod ovoga, prsti su se zaustavili gotovo odmah -
# ranije (prije kolizijskog objekta vrata) ovo je bio signal za trenutni
# odustanak, jer je hill-climbing pretraga mogla gurati vrata pri
# pokusajima korekcije. Sad kad door_panel kolizijski objekt stiti od toga,
# taj rizik je puno manji - postavljeno na -1.0 (efektivno iskljuceno) da
# hill-climbing pretraga stvarno dobije priliku raditi ono za sto je
# napravljena, umjesto da se uvijek presijece prije prvog koraka.
EARLY_ABORT_FRACTION = -1.0

# Koliko puta pokusati "otvori, pomakni dublje, zatvori ponovno" prije
# odustajanja (timeout mehanizam koji sprjecava beskonacno pokusavanje).
MAX_GRASP_ATTEMPTS = 6

# Koliko dublje (duz -Z_axis, prema sipci) pomaknuti po pokusaju - mnozi se
# s brojem pokusaja (attempt+1), pa svaki sljedeci pokusaj ide malo dalje
# od proslog. PLACEHOLDER, podesi empirijski.
RETRY_ADVANCE_M = 0.002

# 180 stupnjeva oko Y, u (x,y,z,w) redoslijedu - vidi obrazlozenje u docstringu.
Y_180 = (0.0, 1.0, 0.0, 0.0)

GRIPPER_STALL_TIMEOUT_SEC = 10.0

# Koliko uzastopnih True /gripper_stalled ocitanja (svaki ~0.1s) treba
# prije nego prihvatimo stall kao stvaran, ne prolazan sumni trzaj.
STALL_DEBOUNCE_COUNT = 3

# Ako je gripper_tcp dalje od izracunatog grasp cilja od ovoga (npr. zbog
# STATUS_ABORTED izvrsavanja trajektorije), NE zatvaraj gripper - vidi
# provjeru prije Koraka 4.
MAX_GRASP_POSITION_ERROR_M = 0.03

# Koliko puta ponoviti cartesian=True grasp-dive pokusaj prije odustajanja
# (bez RRT fallbacka - vidi Korak 3).
MAX_CARTESIAN_RETRIES = 4

# Prsti 1,2 su na +X strani baze (ox=+0.035), prsti 3,4 na -X (ox=-0.035).
# Razlika njihovih zaustavnih pozicija daje pomak kvake duz osi zatvaranja.
FINGER_SIDE_A = ["gripper_finger_1_joint", "gripper_finger_2_joint"]  # +X
FINGER_SIDE_B = ["gripper_finger_3_joint", "gripper_finger_4_joint"]  # -X
MAX_LATERAL_CORRECTIONS = 3
LATERAL_TOLERANCE_M = 0.0015


def math_dist(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


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


def main():
    rclpy.init()
    node = Node("handle_approach_test")
    callback_group = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name="base_link",
        end_effector_name="gripper_tcp",
        group_name="iiwa_arm",
        callback_group=callback_group,
    )

    executor = MultiThreadedExecutor(4)
    executor.add_node(node)

    # Sve pretplate/publisheri koje ce nam trebati - NAMJERNO stvoreni ovdje,
    # prije nego executor nit krene (vidi _spin_forever ispod). Otkriveno
    # Gripper monitoring na POTPUNO ODVOJENOM nodu + izvrsavacu - dokazano
    # (izoliranim testom) da MoveIt2-ova vlastita pozadinska aktivnost
    # (action feedback/status na istom nodu/executoru kao nase pretplate)
    # gladuje nase jednostavne pretplate na istom dijeljenom setupu.
    # Zaseban Node s SingleThreadedExecutor-om, isti obrazac kao izolirani
    # test koji je radio besprijekorno.
    gripper_node = Node("handle_approach_gripper_monitor")
    gripper_pub = gripper_node.create_publisher(Float32, "/gripper_cmd", 10)
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
        # DEBOUNCE, ne obican latch - otkriveno da jednostavan latch (prvi
        # True odmah prihvacen) hvata i RANE, SUMNE trzajeve blizu pocetka
        # zatvaranja (frakcija ~0.002), uzrokujuci da nasa while petlja
        # prestane pratiti prerano i ispise laznu, gotovo-nula frakciju -
        # dok gripper u stvarnosti nastavlja zatvarati dalje i stvarno
        # stane puno kasnije (npr. 0.556 potvrdjeno direktnim echo-om).
        # Sad trazimo STALL_DEBOUNCE_COUNT UZASTOPNIH True ocitanja (svaki
        # ~0.1s, iz gripper_bridge check_period_sec) prije nego prihvatimo
        # stall kao stvaran - jedan prolazni trzaj se filtrira, stvaran
        # mehanicki zastoj (koji ostaje stabilno True) prolazi.
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

    def _spin_forever():
        # NE koristi executor.spin direktno - jedna neuhvacena iznimka
        # (vidjeli smo "Failed to get number of ready entities for action
        # client" od pymoveit2/rclpy internog action clienta) tiho ubije
        # ovu nit. Ova petlja hvata iznimke po iteraciji i nastavlja.
        while rclpy.ok():
            try:
                executor.spin_once(timeout_sec=0.1)
            except Exception as exc:
                node.get_logger().error(f"Executor spin iznimka (nastavljam): {exc}")

    executor_thread = threading.Thread(target=_spin_forever, daemon=True)
    executor_thread.start()

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    # --- Korak -1: dodaj vrata kao kolizijski objekt u planning scenu,
    # SVJEZE u OVOJ sesiji (ne oslanjaj se na odvojeno, ranije pokrenutu
    # add_door_collision - to je bio zaseban rucni korak koji je mogao
    # ostati zastario ako se scena resetirala izmedju pokretanja, uzrokujuci
    # kolizijski objekt koji stoji na krivom mjestu naspram stvarne scene).
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

    z_axis = quat_z_axis(q_handle)
    q_gripper = quat_multiply(q_handle, Y_180)

    # --- Korak 1: ready poza - SAD kad vec imamo izracunat cilj iz cistog
    # ocitanja, sigurno je pomaknuti ruku (RRT planer krece iz razumne
    # konfiguracije umjesto iz nulte/rubne). ---
    node.get_logger().info(f"Saljem ready pozu: {READY_POSE}")
    moveit2.move_to_configuration(READY_POSE)
    moveit2.wait_until_executed()

    # Sigurnosno usporavanje - PRIJE pre-grasp poteza sad, ne tek nakon
    # njega. Otkad je STANDOFF_M smanjen na 0.03, i pre-grasp cilj je blizu
    # vrata (bio je siguran na 12cm, sad je na 3cm) - ranije je usporavanje
    # dolazilo tek nakon pre-grasp poteza, kad je to imalo smisla na 12cm.
    # NAPOMENA - isti oprez kao i za set_path_orientation_constraint: nisam
    # mogao provjeriti tocan naziv svojstva izravno iz izvora, samo iz
    # referentnog vodica. Ako baci AttributeError, provjeri:
    #   python3 -c "from pymoveit2 import MoveIt2; print([a for a in dir(MoveIt2) if 'veloc' in a.lower() or 'accel' in a.lower()])"
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
        try:
            actual = tf_buffer.lookup_transform(
                "base_link", "gripper_tcp", rclpy.time.Time()
            )
            at = actual.transform.translation
            return math_dist((at.x, at.y, at.z), pre_grasp_position), True
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            node.get_logger().warn(f"Ne mogu procitati TF gripper_tcp ({exc}).")
            return 0.0, False

    # NAPOMENA - ova provjera je nedostajala do sad (grasp-dive je vec imao
    # slicnu, pre-grasp ne). wait_until_executed() ne baca gresku ni kad
    # planiranje/izvrsavanje ne uspije (isti STATUS_ABORTED/INVALID_MOTION_PLAN
    # obrazac kao vise puta ranije) - bez ove provjere, skripta bi ispisala
    # "Pre-grasp dosegnut" i nastavila dalje CAK I DA SE RUKA UOPCE NIJE
    # POMAKNULA (npr. ostala u ready pozi). Otkriveno na revolute vratima -
    # razlog jos nije poznat (geometrija vrata za koliziju je provjereno
    # identicna sliding/revolute, nije to), ali provjera barem sprjecava
    # tihi nastavak na krivoj bazi bez obzira na uzrok.
    for pre_grasp_attempt in range(3):
        moveit2.move_to_pose(
            position=pre_grasp_position, quat_xyzw=pre_grasp_quat, cartesian=False
        )
        moveit2.wait_until_executed()
        pg_error, pg_tf_ok = read_pre_grasp_error()
        node.get_logger().info(
            f"Nakon pre-grasp pokusaja {pre_grasp_attempt}, odstupanje od cilja={pg_error:.3f}m"
        )
        if pg_error <= MAX_GRASP_POSITION_ERROR_M:
            break
    else:
        node.get_logger().error(
            f"Pre-grasp NIJE dosegnut nakon 3 pokusaja (zadnje odstupanje={pg_error:.3f}m) - "
            "odustajem, ne nastavljam na grasp-dive iz krive bazne poze."
        )
        gripper_executor.shutdown()
        gripper_node.destroy_node()
        rclpy.shutdown()
        executor_thread.join(timeout=2.0)
        gripper_thread.join(timeout=2.0)
        return

    node.get_logger().info("Pre-grasp dosegnut.")

    # --- Korak 2.5: pokusaj SVJEZE ocitanje ovdje, izbliza (~STANDOFF_M od
    # kvake) - AprilTag procjena je precznija na manjoj udaljenosti (veci
    # prividni tag u slici = manji relativni utjecaj piksel-suma na kut).
    # Fallback na staro ocitanje ako tagovi odavde nisu vidljivi (zaklon) -
    # ne cekaj beskonacno. ---
    node.get_logger().info("Pokusavam svjeze ocitanje na pre-grasp tocki...")
    fresh = collect_average_pose(samples, SAMPLE_WINDOW_SEC, wait_timeout_sec=1.5)
    if fresh is not None:
        hp_x, hp_y, hp_z, q_handle = fresh
        z_axis = quat_z_axis(q_handle)
        q_gripper = quat_multiply(q_handle, Y_180)
        node.get_logger().info(
            f"Svjeza pozicija=({hp_x:.3f}, {hp_y:.3f}, {hp_z:.3f}), "
            f"orijentacija={q_handle} - koristim ovo za grasp cilj."
        )
    else:
        node.get_logger().warn(
            "Nema svjezeg ocitanja s pre-grasp tocke (mozda zaklon) - "
            "koristim staro (izdaleka) ocitanje za grasp cilj."
        )

    # --- Korak 3: grasp-dive (kraci uron, koristi svjeze ocitanje iz koraka
    # 2.5 ako je bilo dostupno, inace staro).
    #
    # Isprobali smo vise kombinacija za ovaj kratki, ali prostorno stisnuti
    # pokret (blizu vrata/kvake):
    #   - cartesian=False (RRT): pouzdano STIGNE, ali slobodno rotira usput
    #     ("dodje sa strane umjesto okomito", zapinje o gripper).
    #   - cartesian=False + path orientation constraint: constraint sprijeci
    #     rotaciju, ALI RRT u ovom stisnutom prostoru ne moze naci NIJEDAN
    #     put koji panduje kolizije I ostane unutar tolerancije - probano na
    #     0.3 i 0.5 rad, oboje INVALID_MOTION_PLAN.
    #   - cartesian=True (pravocrtna interpolacija): NE MOZE rotirati usput
    #     po definiciji (linearno interpolira i poziciju i orijentaciju), ali
    #     ranije je znao ili "sletjeti" na krivu lokaciju ili STATUS_ABORTED
    #     kad pravocrtni put nije kolizijski slobodan.
    #
    # Sad kad imamo TF sigurnosnu provjeru (ispod) koja hvata OBA loša
    # ishoda bez obzira na metodu, probamo cartesian=True PRVI (nema
    # rotacije po prirodi stvari), s cartesian=False (bez constrainta) kao
    # fallback ako cartesian=True ne uspije stici. ---
    grasp_position, grasp_quat = compute_target(
        hp_x, hp_y, hp_z, z_axis, q_gripper, GRASP_STANDOFF_M
    )
    node.get_logger().info(
        f"Grasp cilj: pozicija={grasp_position}, orijentacija={grasp_quat}"
    )

    def read_gripper_tcp_error():
        """Vrati (error_m, ok) - udaljenost gripper_tcp od grasp_position, ili
        (0.0, False) ako TF trenutno nije citljiv (ne blokiraj u tom slucaju)."""
        try:
            actual = tf_buffer.lookup_transform(
                "base_link", "gripper_tcp", rclpy.time.Time()
            )
            at = actual.transform.translation
            return math_dist((at.x, at.y, at.z), grasp_position), True
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            node.get_logger().warn(f"Ne mogu procitati TF gripper_tcp ({exc}).")
            return 0.0, False

    # NAMJERNO nema RRT (cartesian=False) fallbacka ovdje - probano, ali
    # RRT zna birati divlju/rotirajucu putanju kroz ovaj prostor koja
    # POVREMENO fizicki udari u vrata/kvaku. Umjesto toga, PONOVI istu
    # sigurnu metodu (cartesian=True) nekoliko puta - planiranje nije
    # savrseno deterministicko, ponovljeni pokusaj ima realnu sansu uspjeti
    # cak i kad je prvi pao, bez rizika koji RRT fallback nosi.
    for grasp_dive_attempt in range(MAX_CARTESIAN_RETRIES):
        moveit2.move_to_pose(
            position=grasp_position, quat_xyzw=grasp_quat, cartesian=True
        )
        moveit2.wait_until_executed()
        error, tf_ok = read_gripper_tcp_error()
        node.get_logger().info(
            f"Nakon cartesian=True pokusaja {grasp_dive_attempt}, odstupanje od cilja={error:.3f}m"
        )
        if not tf_ok or error <= MAX_GRASP_POSITION_ERROR_M:
            break

    node.get_logger().info("Grasp poza dosegnuta.")

    # --- Sigurnosna provjera: je li ruka STVARNO stigla na izracunati cilj?
    # wait_until_executed() ne baca gresku ni kad izvrsavanje trajektorije
    # ne uspije (npr. STATUS_ABORTED) - bez ove provjere, skripta bi slijepo
    # nastavila i zatvorila gripper gdje god je ruka zavrsila, ne nuzno na
    # cilju. Isti TF obrazac koji koristimo cijeli razgovor. ---
    if error > MAX_GRASP_POSITION_ERROR_M:
        node.get_logger().error(
            f"Gripper_tcp je {error:.3f}m od cilja (> {MAX_GRASP_POSITION_ERROR_M}m) - "
            "vjerojatno izvrsavanje trajektorije nije uspjelo (STATUS_ABORTED ili slicno). "
            "NE zatvaram gripper na krivoj poziciji."
        )
        gripper_executor.shutdown()
        gripper_node.destroy_node()
        rclpy.shutdown()
        executor_thread.join(timeout=2.0)
        gripper_thread.join(timeout=2.0)
        return

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

    # --- Bocna korekcija duz osi zatvaranja prstiju ---
    # Nije ista os kao GRASP_STANDOFF_M (to je dubina, normala vrata).
    # Ova os je lokalni X gripera. Smjer i iznos citamo iz razlike
    # zaustavnih pozicija dviju strana - strana koja stane RANIJE je strana
    # prema kojoj je kvaka pomaknuta.
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

    # depth_offset prati koliko smo se pomaknuli OD ORIGINALNE grasp_position
    # duz -Z_axis (pozitivno = dublje prema kvaki). CLAMPANO na >=0 - ne
    # dopusti pretrazi da ide DALJE UNATRAG od originalne, percepcijom
    # izracunate mete. Otkriveno empirijski: povlacenje unatrag zna
    # LAZNO pokazati visu frakciju (prsti se zatvore skoro do kraja jer
    # NEMA NICEGA da ih zaustavi - promasaj, ne dobar hvat), pa bi
    # hill-climbing bez ovog ogranicenja bjezao sve dalje u prazninu
    # slijedeci tu laznu "poboljsanu" frakciju.
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
        if depth_offset < 0.0:
            depth_offset = 0.0
            node.get_logger().info(
                "Pretraga bi otisla dalje unatrag od originalne mete - "
                "ogranicavam na originalnu poziciju (ne bjezi u prazninu)."
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

    node.get_logger().info(
        "Gotovo (hvat zavrsen, attach - fiksni joint - jos NIJE implementiran)."
    )

    gripper_executor.shutdown()
    gripper_node.destroy_node()
    rclpy.shutdown()
    executor_thread.join(timeout=2.0)
    gripper_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
