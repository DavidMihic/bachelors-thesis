"""pass_through_door.py - nakon otvaranja vrata: pusti kvaku, odmakni se,
parkiraj ruku, poravnaj se s otvorom i prodji kroz njega.

Pokrece se nakon open_sliding (ili open_revolute). U tom trenutku robot vise
NIJE ispred otvora - kod kliznih vrata se pomaknuo skoro metar u stranu dok je
vukao krilo, pa mu je otvor bocno, ne ispred.

DETEKCIJA OTVORA
Tocke iz /scan ostaju u redoslijedu skena, pa je praznina u tom nizu praznina
u prostoru gledano sa senzora. Trazi se praznina sirine bliske
DOORWAY_WIDTH_M izmedju dva niza tocaka. Izmjereno na mirujucem robotu:
srediste stabilno unutar ~1 cm kroz minutu, sirina 0.85-0.86 m naspram
stvarnih 0.85. Pred zatvorenim vratima ispravno ne nalazi nista.

Raniji pristup - trazenje jednog ruba dovratnika - nije bio upotrebljiv: rub
nadjen u 44% ciklusa uz rasap od 600 mm. Praznina poznate sirine ima dva
ogranicenja umjesto jednog, pa ju je puno teze zamijeniti s necim drugim.

FAZE
  1. RELEASE - otvori gripper, pa se BAZOM odmakni unatrag. Gripper je nakon
     otpustanja jos oko sipke; put do parkirne poze vodi kroz nju, pa bi ruka
     odgurnula vrata. Odmicanje bazom je jednostavnije od planiranja pomaka
     rukom i nema rizika da putanja prodje kroz kvaku.
  2. PARK   - ruka u parkirnu pozu (istu koju salje full_stack launch). Bez
     toga gripper strsi u ravnini vrata i zapinje pri prolasku.
  3. ALIGN  - bocno gibanje (holonomna baza, linear.y) dok otvor ne dodje
     ravno ispred. Baza se NE rotira, pa ostaje okomita na vrata.
  4. DRIVE  - ravno naprijed kroz otvor, dok prijedjeni put ne premasi
     PASS_DISTANCE_M ili dok /scan ne javi prepreku preblizu.

BRZINE: baza postize samo dio naredjenog. Uzduzno oko 45%, bocno oko 11% -
sto odgovara specifikaciji KMR-a (3.6 naspram 2.0 km/h) uz prag statickog
trenja koji pri malim brzinama udara jace. Naredbe su zato postavljene znatno
iznad zeljene brzine, a vremenska ogranicenja su velikodusna.

Kod zakretnih vrata baza nakon otvaranja ostane blizu zida i zakrenuta, pa se
prije prolaska odmakne (back_off_m) i ispravi okomito na zid. Uz to krilo strsi
u otvor, pa robot kroz njega prolazi dijagonalno, zaobilazeci ga (steer=True).

Kod kliznih vrata nista od toga ne treba: baza ne rotira, stoji dalje, a krilo
se povlaci u stranu i ne strsi u prolaz. Ondje robot nakon poravnanja vozi
RAVNO - zaobilazenje bi samo unosilo titranje po y.

Preduvjet: vrata su otvorena, robot drzi ili je upravo pustio kvaku,
arm_controller i /scan rade.

Pokrece se iz door_task_node (funkcija run) ili zasebno preko
`ros2 run kmr_iiwa_task pass_through_door`.
"""

import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from kmr_iiwa_task.geometry import quat_rotate_vector, wrap_pi

JOINT_NAMES = [f"iiwa_joint_{i}" for i in range(1, 8)]
# Ista poza koju full_stack.launch.py salje pri dizanju kontrolera.
PARK_POSE = [0.0, -0.6, 0.0, -2.2, 0.0, 0.8, 0.0]
PARK_TIME_SEC = 4

LIDAR_FRAME = "lidar_link"
SELF_OCCLUSION_HALF_ANGLE_DEG = 94.0

# --- Detekcija otvora ---
DOORWAY_WIDTH_M = 0.85
WIDTH_TOL_M = 0.35
GAP_MIN_M = 0.30
WALL_INLIER_M = 0.06  # koliko tocka smije odstupati od pravca zida
WALL_MIN_POINTS = 30
RANSAC_ITERS = 60
WALL_DIR_MIN_Y = 0.7
LEAF_INLIER_M = 0.05
LEAF_MIN_POINTS = 8
LEAF_RANGE_M = 1.5

# --- Odbojno polje (prolazak kroz zakretna vrata) ---
REPULSE_RANGE_M = 0.45  # domet polja
REPULSE_LOOKAHEAD_M = 0.45  # koliko ispred robota se polje racuna
REPULSE_X_LIMIT_M = 0.90  # izvan ovoga se tocke ne broje
REPULSE_ACTIVE_CX_M = 1.10  # polje se ukljucuje tek kad je otvor blize od ovoga
REPULSE_LAT_GAIN = 0.20
REPULSE_YAW_GAIN = 0.40  # vise = robot se vise odmakne od krila, ali
# time i blize desnom zidu; prolaz je uzak pa oba nisu istovremeno moguca
MEMORY_CAP = 20000
MEMORY_CELL_M = 0.05

# Tamponska zona oko krila. Krilo pri 70 deg strsi u otvor - slobodni
# pravocrtni koridor je tada 0.56 m, a platforma je 0.63. Robot ga zato
# zaobilazi bocno dok prolazi.
LEAF_BUFFER_M = 0.20
LEAF_AVOID_MPS = 0.25
# Ispod CLEAR_ACT_M se robot aktivno odmice od blize strane. Sama usporedba
# strana nije dovoljna: dok je prilazio otvoru razlika je stajala na 53-64 mm,
# ispod mrtve zone, pa korekcije nije bilo sve dok lijeva strana nije pala na
# 89 mm - a tada je vec bilo prekasno.
CLEAR_ACT_M = 0.25
CLEAR_TARGET_M = 0.20  # zeljeni razmak od blize strane
CLEAR_GAIN = 1.5
CLEAR_BALANCE_TOL_M = 0.02
GATE_LAT_GAIN = 0.8
GATE_YAW_GAIN = 0.8
# Zakret je dvopolozajan po prirodi: ispod 0.36 rad/s baza se ne mice, pa se i
# najmanja greska pretvara u naredbu te velicine. Bez histereze i filtriranja
# regulator prebaci preko cilja pa se vraca, u kratkim ciklusima.
GATE_YAW_ON_RAD = 0.10  # iznad ove greske se zakret UKLJUCUJE (~6 deg)
GATE_YAW_OFF_RAD = 0.04  # ispod ove se ISKLJUCUJE (~2 deg)
GATE_FILTER_ALPHA = 0.3  # nize = jace izgladeno
BALANCE_GAIN = 0.6
CLEARANCE_STOP_M = 0.04  # udaljenost do RUBA baze, ne do sredista
MAX_RANGE_M = 6.0

# --- Odmicanje od kvake (baza unatrag) ---
RETREAT_M = 0.08  # koliko se TCP povuce od kvake prije parkiranja ruke

# Kod zakretnih vrata baza ostane blizu zida i zakrenuta od otvaranja, pa se
# prije prolaska mora odmaknuti i ispraviti. Kod kliznih ne rotira i stoji
# dalje, pa joj to ne treba - iznos dolazi kao argument.
STRAIGHTEN_TOL_RAD = 0.03
STRAIGHTEN_GAIN = 1.2
STRAIGHTEN_TIMEOUT_SEC = 60.0
ALIGN_GAIN = 1.2  # proporcionalno umjesto bang-bang, da ne titra oko cilja
STRAIGHTEN_TOL_RAD = 0.03
STRAIGHTEN_GAIN = 1.2
STRAIGHTEN_TIMEOUT_SEC = 60.0
ALIGN_MIN_MPS = 0.18  # ispod ovoga se baza bocno ne mice (prag trenja)

# --- Poravnavanje (bocno, bez rotacije) ---
ALIGN_SPEED_MPS = 0.22  # bocno se postize ~11% naredjenog, pa naredba mora
# biti znatno veca od zeljene brzine
ALIGN_TOL_M = 0.05
ALIGN_TIMEOUT_SEC = 150.0

# --- Prolazak ---
DRIVE_SPEED_MPS = 0.22
# Prolazak se mjeri iz percepcije: cx je udaljenost do sredista otvora. Kad
# padne ispod PASS_CX_M, ravnina zida je iza prednjeg ruba baze. Integracija
# naredbe se ne koristi kao kriterij - visestruko precjenjuje.
PASS_CX_M = -0.30
DRIVE_TIMEOUT_SEC = 120.0
LOST_DOORWAY_STEPS = 20  # koliko ciklusa bez otvora znaci da smo prosli
PASS_MARGIN_M = 0.10
PASS_STRAIGHT_M = 2.00  # put kroz otvor kod zakretnih vrata

# Pracenje pravca krila: greska je zbroj odstojanja i kursa, kao kod klasicnog
# pracenja zida. D clan sad ima smisla jer je referenca glatka - na najblizoj
# tocki je derivacija bila besmislena.
LEAF_FOLLOW_DIST_M = 0.30
LEAF_FOLLOW_DIST_GAIN = 1.5
LEAF_KP = 1.0
LEAF_KD = 0.3
LEAF_YAW_DEADBAND = 0.06
WALL_LIMIT_M = 0.12  # ispod ovoga desni zid nadjacava pracenje krila
WALL_LIMIT_GAIN = 4.0  # koliko straznji rub baze prolazi iza ravnine zida

# --- Sigurnost ---
KMR_LENGTH_M = 1.08
KMR_WIDTH_M = 0.63

PUBLISH_PERIOD_SEC = 0.02
CONTROL_PERIOD_SEC = 0.05


def _fit_wall(pts):
    """Najjaci pravac kroz tocke skena, RANSAC-om. Vraca (tocka, smjer,
    pristalice).

    Oba zida leze u istoj ravnini, pa daju jedan pravac s puno pristalica.
    Otvoreno krilo stoji pod kutom i ispada iz te skupine - bez toga se
    praznina trazila i preko tocaka na krilu, sto je davalo lazne otvore.
    """
    n = len(pts)
    if n < WALL_MIN_POINTS:
        return None
    arr = np.asarray(pts)
    best = (0, None, None)
    rng = np.random.default_rng(0)
    for _ in range(RANSAC_ITERS):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        p0, p1 = arr[i], arr[j]
        d = p1 - p0
        dn = float(np.linalg.norm(d))
        if dn < 0.3:  # preblizu jedna drugoj, smjer je nepouzdan
            continue
        d = d / dn
        # Zid je okomit na robota, pa mu smjer mora biti pretezno po y.
        # Bez toga RANSAC zna uhvatiti KRILO: kad je robot blizu, zid se vidi
        # pod ostrim kutom i daje malo tocaka, a krilo je odmah ispred i gusto
        # uzorkovano - pa ispadne "najjaci pravac", zidne tocke se proglase
        # krilom i izbjegavanje radi naopako.
        if abs(d[1]) < WALL_DIR_MIN_Y:
            continue
        nrm = np.array([-d[1], d[0]])
        dist = np.abs((arr - p0) @ nrm)
        cnt = int(np.count_nonzero(dist < WALL_INLIER_M))
        if cnt > best[0]:
            best = (cnt, p0, d)
    if best[1] is None or best[0] < WALL_MIN_POINTS:
        return None
    p0, d = best[1], best[2]
    nrm = np.array([-d[1], d[0]])
    inl = arr[np.abs((arr - p0) @ nrm) < WALL_INLIER_M]
    return p0, d, inl


def _merge(old, new, cell=MEMORY_CELL_M, cap=MEMORY_CAP):
    """Dodaj nove tocke u memoriju, sazete u prostornu resetku.

    Ranije se pamtilo zadnjih N tocaka, a svaki sken ih doda nekoliko stotina -
    pa je memorija drzala samo zadnja dva-tri skena i sve vidjeno pri prilasku
    bi ispalo. Robot bi tada mislio da je strana koju je upravo prosao slobodna
    (izmjereno: desni dovratnik uz bok prikazivan kao 1000+ mm) i strugao bi po
    njoj. S resetkom se svaka celija pamti jednom, pa memorija ostaje mala a
    nista se ne gubi.
    """
    m = new if old is None or len(old) == 0 else np.vstack([old, new])
    keys = np.round(m / cell).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    m = m[np.sort(idx)]
    if len(m) > cap:
        m = m[-cap:]
    return m


def _min_dist_and_y(pts_base):
    """Najmanja udaljenost do RUBA baze, uz (x, y) te tocke. None ako nema
    tocaka. Vraca (d, x, y)."""
    best = None
    for q in pts_base:
        d = _dist_to_base(float(q[0]), float(q[1]))
        if best is None or d < best[0]:
            best = (d, float(q[0]), float(q[1]))
    return best


def _gate_target(left, right):
    """Iz najblize lijeve i desne tocke odredi najuzi prolaz.

    Te dvije tocke su vrh krila i suprotni dovratnik - spojnica medju njima je
    mjesto kroz koje robot mora proci. Iz nje slijedi CILJNA orijentacija
    (okomica na spojnicu) i CILJNA bocna pozicija (poloviste). Oboje su
    apsolutni ciljevi, pa se greska ne gomila kao kad se zadaje samo kutna
    brzina.

    Vraca (yaw_err, lateral_err) ili None.
    """
    if left is None or right is None:
        return None
    a = np.array([left[1], left[2]])
    b = np.array([right[1], right[2]])
    gate = b - a
    span = float(np.linalg.norm(gate))
    if span < 0.3 or span > 2.0:
        return None
    n = np.array([gate[1], -gate[0]])
    if n[0] < 0.0:
        n = -n
    yaw_err = math.atan2(n[1], n[0])
    mid = 0.5 * (a + b)
    return yaw_err, float(mid[1])


def _dist_to_base(x, y):
    """Udaljenost tocke do RUBA pravokutne baze, u base_link."""
    dx = max(abs(x) - KMR_LENGTH_M / 2.0, 0.0)
    dy = max(abs(y) - KMR_WIDTH_M / 2.0, 0.0)
    return math.hypot(dx, dy)


def _fit_leaf(pts_base, wall_fit):
    """Pravac kroz tocke KRILA, RANSAC-om.

    Krilo su tocke koje nisu na zidu. Pravac kroz njih je kontinuirana
    referenca - za razliku od najblize tocke, koja skace s krila na dovratnik i
    natrag, pa je ciljni kut treperio od -22 do +14 stupnjeva izmedju ciklusa.

    Vraca (tocka, smjer) ili None.
    """
    if wall_fit is None or len(pts_base) < LEAF_MIN_POINTS:
        return None
    p0w, dw, _ = wall_fit
    nw = np.array([-dw[1], dw[0]])
    arr = np.asarray(pts_base)
    cand = arr[np.abs((arr - p0w) @ nw) >= WALL_INLIER_M]
    # Samo ono ispred robota i u dometu - straznje tocke i daleki zid nisu krilo.
    cand = cand[(cand[:, 0] > 0.0) & (np.linalg.norm(cand, axis=1) < LEAF_RANGE_M)]
    n = len(cand)
    if n < LEAF_MIN_POINTS:
        return None
    best = (0, None, None)
    rng = np.random.default_rng(1)
    for _ in range(RANSAC_ITERS):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        a, b = cand[i], cand[j]
        d = b - a
        dn = float(np.linalg.norm(d))
        if dn < 0.2:
            continue
        d = d / dn
        nrm = np.array([-d[1], d[0]])
        cnt = int(np.count_nonzero(np.abs((cand - a) @ nrm) < LEAF_INLIER_M))
        if cnt > best[0]:
            best = (cnt, a, d)
    if best[1] is None or best[0] < LEAF_MIN_POINTS:
        return None
    return best[1], best[2]


def _repulsion(pts_base):
    """Zbirni odbojni doprinos svih zapamcenih tocaka, racunat za polozaj
    REPULSE_LOOKAHEAD_M ISPRED robota.

    Zbroj po stotinama tocaka mijenja se glatko kako se robot pomice, za razliku
    od odluke na temelju jedne najblize tocke (skakala s krila na dovratnik, kut
    je treperio -22 do +14 stupnjeva) ili pravca kroz tocke krila (u stvarnoj
    sceni kurs je isao -16 do -44 stupnjeva).

    Racuna se ispred robota jer se oko trenutnog polozaja doprinosi s lijeva i
    zdesna ponistavaju dok je robot izmedu prepreka - a krilo strsi ispred, ne
    bocno.

    Vraca prosjecnu bocnu komponentu (pozitivno = gura ulijevo).
    """
    fy = 0.0
    cnt = 0
    for q in pts_base:
        x, y = float(q[0]), float(q[1])
        xa = x - REPULSE_LOOKAHEAD_M
        if abs(xa) > REPULSE_X_LIMIT_M:
            continue  # daleki dio zida bi inace nadglasao krilo
        d = _dist_to_base(xa, y)
        if d >= REPULSE_RANGE_M:
            continue
        n = math.hypot(xa, y)
        if n < 1e-6:
            continue
        w = (REPULSE_RANGE_M - d) / REPULSE_RANGE_M
        fy -= w * w * y / n
        cnt += 1
    return 0.0 if cnt == 0 else fy / cnt


def _find_doorway_from(fit):
    """Otvor u zidu: praznina sirine bliske DOORWAY_WIDTH_M medju tockama koje
    leze NA ZIDU. Vraca (sirina, cx, cy, rub_a, rub_b) ili None."""
    if fit is None:
        return None
    p0, d, inl = fit
    # Poredaj zidne tocke duz zida, pa trazi prazninu medu njima.
    order = np.argsort((inl - p0) @ d)
    wall = inl[order]
    best = None
    for i in range(len(wall) - 1):
        a, b = wall[i], wall[i + 1]
        w = float(np.linalg.norm(b - a))
        if w > GAP_MIN_M and abs(w - DOORWAY_WIDTH_M) < WIDTH_TOL_M:
            if best is None or abs(w - DOORWAY_WIDTH_M) < abs(
                best[0] - DOORWAY_WIDTH_M
            ):
                best = (
                    w,
                    0.5 * float(a[0] + b[0]),
                    0.5 * float(a[1] + b[1]),
                    a,
                    b,
                )
    return best


def run(back_off_m=0.0, steer=False):
    node = Node("pass_through_door")

    cmd_vel_pub = node.create_publisher(Twist, "/cmd_vel", 10)
    gripper_pub = node.create_publisher(Float32, "/gripper_cmd", 10)
    traj_pub = node.create_publisher(
        JointTrajectory, "/arm_controller/joint_trajectory", 10
    )

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    # MoveIt2 loggira "Joint states are not available yet!" pri svakom pozivu,
    # pa dobiva vlastiti node koji se moze utisati.
    moveit_node = Node("pass_through_moveit")
    rclpy.logging.set_logger_level(
        "pass_through_moveit", rclpy.logging.LoggingSeverity.ERROR
    )
    moveit_executor = MultiThreadedExecutor(2)
    moveit_executor.add_node(moveit_node)
    threading.Thread(target=moveit_executor.spin, daemon=True).start()

    moveit2 = MoveIt2(
        node=moveit_node,
        joint_names=JOINT_NAMES,
        base_link_name="base_link",
        end_effector_name="gripper_tcp",
        group_name="iiwa_arm",
        callback_group=ReentrantCallbackGroup(),
    )
    moveit2.max_velocity = 0.1
    moveit2.max_acceleration = 0.1

    def tcp_pose():
        try:
            t = tf_buffer.lookup_transform(
                "base_link", "gripper_tcp", rclpy.time.Time()
            )
            tr, r = t.transform.translation, t.transform.rotation
            return np.array([tr.x, tr.y, tr.z]), [r.x, r.y, r.z, r.w]
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None, None

    mem = {"pts": None}
    state = {
        "doorway": None,
        "too_close": False,
        "have_scan": False,
        "nearest": None,
        "wall_fit": None,
        "scan_base": None,
        "leaf_near": None,
    }
    odom = {"xy": None, "yaw": None}

    def _on_odom(msg: Odometry):
        odom["xy"] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        odom["yaw"] = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def base_to_odom(p_base):
        c, sn = math.cos(odom["yaw"]), math.sin(odom["yaw"])
        return (
            odom["xy"]
            + np.array(
                [
                    c * p_base[:, 0] - sn * p_base[:, 1],
                    sn * p_base[:, 0] + c * p_base[:, 1],
                ]
            ).T
        )

    def odom_to_base(p_odom):
        c, sn = math.cos(odom["yaw"]), math.sin(odom["yaw"])
        d = p_odom - odom["xy"]
        return np.array([c * d[:, 0] + sn * d[:, 1], -sn * d[:, 0] + c * d[:, 1]]).T

    lidar_tf = {"v": None}

    def _ensure_tf():
        if lidar_tf["v"] is not None:
            return True
        try:
            tf = tf_buffer.lookup_transform("base_link", LIDAR_FRAME, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return False
        t, q = tf.transform.translation, tf.transform.rotation
        lidar_tf["v"] = (np.array([t.x, t.y, t.z]), (q.x, q.y, q.z, q.w))
        return True

    def _on_scan(msg: LaserScan):
        if not _ensure_tf():
            return
        t, q = lidar_tf["v"]
        mask = math.radians(SELF_OCCLUSION_HALF_ANGLE_DEG)
        limit = min(msg.range_max, MAX_RANGE_M)

        pts = []
        too_close = False
        angle = msg.angle_min
        for r in msg.ranges:
            a = angle
            angle += msg.angle_increment
            if not math.isfinite(r) or r < msg.range_min or r > limit:
                continue
            if abs(wrap_pi(a)) > mask:
                continue
            x_l, y_l = r * math.cos(a), r * math.sin(a)
            x_b, y_b, _ = quat_rotate_vector(q, [x_l, y_l, 0.0])
            x_b += t[0]
            y_b += t[1]
            pts.append((x_b, y_b))

        # Udaljenost do RUBA baze, ne do sredista: tlocrt je pravokutnik
        # 1.08 x 0.63, pa kruznica oko sredista ili traka ispred ne opisuju
        # gdje robot stvarno moze proci - pogotovo kad je zakrenut.
        fit = _fit_wall(pts)
        leaf_pts = []
        if fit is not None:
            p0, d, inl = fit
            nrm = np.array([-d[1], d[0]])
            # Normala usmjerena PREMA robotu (ishodiste base_link).
            if float(np.dot(nrm, -p0)) < 0.0:
                nrm = -nrm
            arr = np.asarray(pts)
            signed = (arr - p0) @ nrm
            # Krilo se otvara prema robotu, pa lezi s NJEGOVE strane ravnine
            # zida. Bez tog uvjeta u "krilo" upadaju i tocke iza otvora, sum na
            # rubovima vidnog polja i dijelovi suprotnog zida - pa korekcija
            # gura robota u zid.
            leaf_pts = arr[signed >= WALL_INLIER_M]

        nearest = None
        for x_b, y_b in pts:
            dd = _dist_to_base(x_b, y_b)
            if nearest is None or dd < nearest:
                nearest = dd
        too_close = nearest is not None and nearest < CLEARANCE_STOP_M

        leaf_near = None
        for q_pt in leaf_pts:
            dd = _dist_to_base(float(q_pt[0]), float(q_pt[1]))
            if leaf_near is None or dd < leaf_near[0]:
                leaf_near = (dd, float(q_pt[1]))

        # Sve tocke se pamte u ODOM okviru, BEZ klasifikacije na zid i krilo.
        # Za prolazak je vazno samo koliko ima mjesta lijevo a koliko desno -
        # je li prepreka krilo ili dovratnik ne mijenja nista. Razdvajanje se
        # pokazalo nepouzdanim: dovratnici i suprotni zid upadali su u "krilo",
        # a memorija se gomilala pa je jedan los fit trajno kvario skup.
        if odom["xy"] is not None and odom["yaw"] is not None and pts:
            mem["pts"] = _merge(mem["pts"], base_to_odom(np.asarray(pts)))

        state["wall_fit"] = fit
        state["scan_base"] = np.asarray(pts) if pts else None
        state["too_close"] = too_close
        state["nearest"] = nearest
        state["leaf_near"] = leaf_near
        state["have_scan"] = True
        state["doorway"] = _find_doorway_from(fit)

    node.create_subscription(LaserScan, "/scan", _on_scan, 10)
    node.create_subscription(Odometry, "/odom", _on_odom, 10)

    cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    stop_flag = {"v": False}

    def publisher_loop():
        """cmd_vel_bridge primjenjuje zadnju primljenu poruku svaki fizicki
        korak i nema failsafe timeout, pa naredbe salje zasebna nit u stalnom
        ritmu - neujednacen ritam znaci trzajno gibanje."""
        while rclpy.ok() and not stop_flag["v"]:
            tw = Twist()
            tw.linear.x = cmd["vx"]
            tw.linear.y = cmd["vy"]
            tw.angular.z = cmd["wz"]
            cmd_vel_pub.publish(tw)
            time.sleep(PUBLISH_PERIOD_SEC)

    gate_filt = {"yaw": None, "lat": None, "on": False}

    def gate_smooth(gate):
        """Izgladi ciljni kut i bocnu gresku, pa odluci je li zakret aktivan.

        Najbliza tocka zna skociti s jedne prepreke na drugu, pa sirovi ciljni
        kut treperi. Histereza sprjecava ukljucivanje i iskljucivanje oko istog
        praga.
        """
        if gate is None:
            gate_filt["on"] = False
            return None
        ye, le = gate
        for key, val in (("yaw", ye), ("lat", le)):
            prev = gate_filt[key]
            gate_filt[key] = (
                val if prev is None else prev + GATE_FILTER_ALPHA * (val - prev)
            )
        ysm = gate_filt["yaw"]
        if gate_filt["on"]:
            if abs(ysm) < GATE_YAW_OFF_RAD:
                gate_filt["on"] = False
        elif abs(ysm) > GATE_YAW_ON_RAD:
            gate_filt["on"] = True
        return ysm, gate_filt["lat"], gate_filt["on"]

    def odom_pose():
        """(x, y, yaw) iz odometrije, ili None."""
        if odom["xy"] is None or odom["yaw"] is None:
            return None
        return float(odom["xy"][0]), float(odom["xy"][1]), float(odom["yaw"])

    def move_lateral_to(line_p, line_d, tol=0.03, timeout=60.0):
        """Bocno se pomakni dok srediste baze ne dodje na pravac (line_p, line_d)
        u odom okviru. Jedan jednosmjeran pokret: greska monotono pada, pa nema
        sto oscilirati."""
        n = np.array([-line_d[1], line_d[0]])
        t0 = time.monotonic()
        while rclpy.ok() and time.monotonic() - t0 < timeout:
            if state["too_close"]:
                cmd["vy"] = 0.0
                return False
            pose = odom_pose()
            if pose is None:
                time.sleep(CONTROL_PERIOD_SEC)
                continue
            err = float(np.dot(np.array(pose[:2]) - line_p, n))
            if abs(err) < tol:
                cmd["vy"] = 0.0
                node.get_logger().info(f"Na pravcu (greska {err*1000:+.0f} mm).")
                return True
            # Greska je u odom okviru, naredba u base_link.
            c, sn = math.cos(pose[2]), math.sin(pose[2])
            v_odom = -math.copysign(ALIGN_SPEED_MPS, err) * n
            cmd["vy"] = float(-sn * v_odom[0] + c * v_odom[1])
            cmd["vx"] = float(c * v_odom[0] + sn * v_odom[1])
            node.get_logger().info(
                f"bocno: greska {err*1000:+.0f} mm", throttle_duration_sec=1.0
            )
            time.sleep(CONTROL_PERIOD_SEC)
        cmd["vx"] = cmd["vy"] = 0.0
        return False

    def turn_to(target_yaw, tol=0.03, timeout=60.0):
        """Zakreni se na ciljni kurs i stani. Jedan pokret, jedno zaustavljanje."""
        t0 = time.monotonic()
        while rclpy.ok() and time.monotonic() - t0 < timeout:
            pose = odom_pose()
            if pose is None:
                time.sleep(CONTROL_PERIOD_SEC)
                continue
            err = math.atan2(
                math.sin(target_yaw - pose[2]), math.cos(target_yaw - pose[2])
            )
            if abs(err) < tol:
                cmd["wz"] = 0.0
                node.get_logger().info(
                    f"Zakrenut (greska {math.degrees(err):+.1f} deg)."
                )
                return True
            want = float(np.clip(STRAIGHTEN_GAIN * err, -0.35, 0.35))
            cmd["wz"] = float(math.copysign((abs(want) + 0.204) / 0.562, want))
            time.sleep(CONTROL_PERIOD_SEC)
        cmd["wz"] = 0.0
        return False

    def clearances():
        """Najmanja udaljenost do ruba baze, zasebno LIJEVO (+y) i DESNO (-y).

        Racuna se iz memorije prebacene u base_link, pa vrijedi i kad lidar
        prepreku vise ne vidi - a to se dogodi bas kad joj je robot najblizi,
        jer je senzor na prednjoj strani.
        """
        out = {"left": None, "right": None, "base": None}
        if odom["xy"] is None or odom["yaw"] is None:
            return out
        pts_m = mem["pts"]
        if pts_m is None or len(pts_m) == 0:
            return out
        base = odom_to_base(pts_m)
        out["base"] = base
        # BEZ filtriranja po x. Dovratnik uz bok robota, koji je upravo prosao
        # prednjim rubom, i dalje je opasan za straznji kut. Filtar po x ga je
        # izbacivao iz racuna cim mu x padne ispod -L/2, pa je "lijevo" naglo
        # skakalo (izmjereno 166 -> 319 mm u jednom ciklusu) i robot je skretao
        # ravno u njega. _dist_to_base ionako daje velike vrijednosti za ono
        # sto je stvarno daleko.
        if len(base) == 0:
            return out
        left = base[base[:, 1] >= 0.0]
        right = base[base[:, 1] < 0.0]
        if len(left):
            out["left"] = _min_dist_and_y(left)
        if len(right):
            out["right"] = _min_dist_and_y(right)
        return out

    def drive_distance(target_m, speed, avoid=False):
        """Vozi dok odometrija ne pokaze target_m. Vraca prijedjeni put, ili
        None ako je prekinuto zbog prepreke. Put se MJERI - racunanje iz brzine
        i vremena promasuje jer baza postize samo dio naredjenog.

        S avoid=True usput izbjegava krilo i blago zakrece od njega. Bez toga
        zadnja dionica (kad otvor vise nije vidljiv) ide slijepo, a upravo je
        tada robot najblize krilu."""
        if odom["xy"] is None:
            return None
        p0 = odom["xy"].copy()
        while rclpy.ok():
            if state["too_close"]:
                cmd["vx"] = cmd["vy"] = cmd["wz"] = 0.0
                return None
            done = float(np.linalg.norm(odom["xy"] - p0))
            if done >= target_m:
                cmd["vx"] = cmd["vy"] = cmd["wz"] = 0.0
                return done
            cmd["vx"] = speed
            if avoid:
                clr = clearances()
                lf, rt = clr["left"], clr["right"]
                g = gate_smooth(_gate_target(lf, rt))
                if g is not None:
                    ye, le, on = g
                    cmd["vy"] = float(
                        np.clip(GATE_LAT_GAIN * le, -ALIGN_SPEED_MPS, ALIGN_SPEED_MPS)
                    )
                    if on:
                        ww = float(np.clip(GATE_YAW_GAIN * ye, -0.3, 0.3))
                        cmd["wz"] = math.copysign((abs(ww) + 0.204) / 0.562, ww)
                    else:
                        cmd["wz"] = 0.0
                    node.get_logger().info(
                        f"lijevo={lf[0]*1000:.0f}mm desno={rt[0]*1000:.0f}mm "
                        f"kut={math.degrees(ye):+.1f}deg",
                        throttle_duration_sec=1.0,
                    )
                else:
                    cmd["vy"] = cmd["wz"] = 0.0
            time.sleep(CONTROL_PERIOD_SEC)
        cmd["vx"] = cmd["vy"] = cmd["wz"] = 0.0
        return None

    def shutdown(msg=None, error=False):
        cmd["vx"] = cmd["vy"] = cmd["wz"] = 0.0
        time.sleep(0.2)
        stop_flag["v"] = True
        time.sleep(0.1)
        cmd_vel_pub.publish(Twist())
        time.sleep(0.5)
        if msg:
            (node.get_logger().error if error else node.get_logger().info)(msg)
        executor.shutdown()
        moveit_executor.shutdown()
        time.sleep(0.2)
        node.destroy_node()
        moveit_node.destroy_node()

    # Vlastiti izvrsavac, ne globalni - vidi isti komentar u open_sliding.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    threading.Thread(target=publisher_loop, daemon=True).start()

    node.get_logger().info("Cekam /scan i /odom...")
    t_wait = time.monotonic()
    while (not state["have_scan"] or odom["xy"] is None) and rclpy.ok():
        time.sleep(0.1)
        if time.monotonic() - t_wait > 15.0:
            shutdown("Nema /scan ili /odom - prekidam.", error=True)
            return

    # --- Faza 1: pusti kvaku i odmakni se od nje ---
    node.get_logger().info("Otvaram gripper...")
    for _ in range(20):
        gripper_pub.publish(Float32(data=0.0))
        time.sleep(0.1)

    # Gripper je nakon otpustanja jos oko kvake, a put do parkirne poze vodi
    # kroz nju. Ruka se zato prvo povuce po svojoj osi prilaza.
    p_tcp, q_tcp = tcp_pose()
    if p_tcp is not None:
        approach = np.array(quat_rotate_vector(q_tcp, [0.0, 0.0, 1.0]))
        target = p_tcp - RETREAT_M * approach
        node.get_logger().info(f"Odmicem ruku {RETREAT_M*100:.0f} cm od kvake...")
        moveit2.move_to_pose(
            position=list(target), quat_xyzw=list(q_tcp), cartesian=True
        )
        moveit2.wait_until_executed()
    else:
        node.get_logger().warn("Nema TF gripper_tcp - preskacem odmicanje.")

    # --- Faza 2: parkiraj ruku ---
    node.get_logger().info("Parkiram ruku...")
    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES
    pt = JointTrajectoryPoint()
    pt.positions = list(PARK_POSE)
    pt.time_from_start.sec = PARK_TIME_SEC
    traj.points = [pt]
    traj_pub.publish(traj)
    time.sleep(PARK_TIME_SEC + 1.5)
    node.get_logger().info("Ruka parkirana.")

    if back_off_m > 0.0:
        node.get_logger().info(f"Odmicem bazu {back_off_m:.2f} m unatrag...")
        drive_distance(back_off_m, -DRIVE_SPEED_MPS)
        time.sleep(0.5)

    # Baza je nakon otvaranja zakrenuta, pa bi voznja "naprijed" isla
    # dijagonalno u zid. Ispravlja se prema ZIDU, ne prema odometriji: rubovi
    # praznine leze na zidu, njihov spojni vektor daje smjer zida, a robot
    # treba gledati po normali na njega.
    node.get_logger().info("Ispravljam se okomito na zid...")
    t0 = time.monotonic()
    while rclpy.ok() and time.monotonic() - t0 < STRAIGHTEN_TIMEOUT_SEC:
        d = state["doorway"]
        if d is None:
            cmd["wz"] = 0.0
            node.get_logger().warn(
                "Otvor nije vidljiv - stojim.", throttle_duration_sec=2.0
            )
            time.sleep(CONTROL_PERIOD_SEC)
            continue
        _, cx, cy, pa, pb = d
        wall = pb - pa
        nrm = np.array([-wall[1], wall[0]])
        if float(np.dot(nrm, np.array([cx, cy]))) < 0.0:
            nrm = -nrm
        yaw_err = math.atan2(nrm[1], nrm[0])
        if abs(yaw_err) < STRAIGHTEN_TOL_RAD:
            cmd["wz"] = 0.0
            node.get_logger().info(
                f"Ispravljen (greska {math.degrees(yaw_err):+.1f} deg)."
            )
            break
        want = float(np.clip(STRAIGHTEN_GAIN * yaw_err, -0.4, 0.4))
        cmd["wz"] = float(math.copysign((abs(want) + 0.204) / 0.562, want))
        node.get_logger().info(
            f"zakret {math.degrees(yaw_err):+.1f} deg", throttle_duration_sec=2.0
        )
        time.sleep(CONTROL_PERIOD_SEC)
    cmd["wz"] = 0.0
    time.sleep(0.5)

    # --- Faza 4: bocno poravnavanje s otvorom ---
    node.get_logger().info("Poravnavam se s otvorom...")
    t0 = time.monotonic()
    aligned = False
    while rclpy.ok() and time.monotonic() - t0 < ALIGN_TIMEOUT_SEC:
        d = state["doorway"]
        if d is None:
            cmd["vy"] = 0.0
            node.get_logger().warn(
                "Otvor nije vidljiv - stojim.", throttle_duration_sec=2.0
            )
            time.sleep(CONTROL_PERIOD_SEC)
            continue

        _, cx, cy, _, _ = d
        if abs(cy) < ALIGN_TOL_M:
            cmd["vy"] = 0.0
            aligned = True
            node.get_logger().info(f"Poravnat: otvor na ({cx:+.2f}, {cy:+.2f}) m.")
            break
        want_v = min(ALIGN_SPEED_MPS, ALIGN_GAIN * abs(cy) + ALIGN_MIN_MPS)
        cmd["vy"] = math.copysign(want_v, cy)
        node.get_logger().info(
            f"otvor ({cx:+.2f}, {cy:+.2f}) -> vy={cmd['vy']:+.2f}",
            throttle_duration_sec=2.0,
        )
        time.sleep(CONTROL_PERIOD_SEC)

    cmd["vy"] = 0.0
    time.sleep(0.5)

    if not aligned:
        shutdown("Poravnavanje nije uspjelo - ne ulazim.", error=True)
        return

    # --- Faza 4: prolazak ---
    # Kod ZAKRETNIH vrata smjer daje ODBOJNO POLJE svih zapamcenih tocaka (vidi
    # _repulsion). Izlaz prolazi kroz isti filtar i histerezu kao drugdje.
    if steer:
        node.get_logger().info("Prolazim kroz otvor (odbojno polje)...")
        t0 = time.monotonic()
        outcome = "vrijeme isteklo"
        start_xy = None if odom["xy"] is None else odom["xy"].copy()

        while rclpy.ok() and time.monotonic() - t0 < DRIVE_TIMEOUT_SEC:
            if state["too_close"]:
                outcome = (
                    f"prepreka na {state['nearest']*1000:.0f} mm od ruba baze"
                    if state["nearest"] is not None
                    else "prepreka preblizu"
                )
                break

            cmd["vx"] = DRIVE_SPEED_MPS
            clr = clearances()
            d = state["doorway"]
            cx = d[1] if d is not None else 0.0

            vy, wz, fy, yaw_err = 0.0, 0.0, 0.0, 0.0
            if clr["base"] is not None and cx < REPULSE_ACTIVE_CX_M:
                fy = _repulsion(clr["base"])
                sm = gate_smooth((REPULSE_YAW_GAIN * fy, REPULSE_LAT_GAIN * fy))
                if sm is not None:
                    yaw_err, _, yaw_on = sm
                    # Bez bocne komponente: i translacija ima mrtvu zonu (~0.18
                    # m/s), pa je naredba ili puna ili nikakva. Polje daje oko
                    # 0.08, sto se podizalo na prag - bocno gibanje je time bilo
                    # stalno na maksimumu udesno i robot je zanosio u zid.
                    # Skretanje obavlja zakret.
                    if yaw_on:
                        want = float(np.clip(yaw_err, -0.35, 0.35))
                        wz = math.copysign((abs(want) + 0.204) / 0.562, want)
            cmd["vy"], cmd["wz"] = float(vy), float(wz)

            node.get_logger().info(
                f"polje={fy:+.3f}  kut={math.degrees(yaw_err):+.1f}deg  "
                f"vy={vy:+.2f} wz={wz:+.2f}  cx={cx:+.2f}",
                throttle_duration_sec=1.0,
            )

            if start_xy is not None and odom["xy"] is not None:
                if float(np.linalg.norm(odom["xy"] - start_xy)) >= PASS_STRAIGHT_M:
                    outcome = "prosao"
                    break
            time.sleep(CONTROL_PERIOD_SEC)

        cmd["vx"] = cmd["vy"] = cmd["wz"] = 0.0
        node.get_logger().info("=== SAZETAK ===")
        node.get_logger().info(f"  ishod: {outcome}")
        shutdown()
        return

    node.get_logger().info("Prolazim kroz otvor...")
    t0 = time.monotonic()
    lost = 0
    last_cx = None
    outcome = "vrijeme isteklo"
    while rclpy.ok() and time.monotonic() - t0 < DRIVE_TIMEOUT_SEC:
        if state["too_close"]:
            outcome = (
                f"prepreka na {state['nearest']*1000:.0f} mm od ruba baze"
                if state["nearest"] is not None
                else "prepreka preblizu"
            )
            break

        cmd["vx"] = DRIVE_SPEED_MPS

        # Bocna korekcija drzi robota IZMEDU krila i zida. Prolaz je uzak, pa
        # bjezanje samo od krila zavrsi udarcem u suprotni zid. Obje udaljenosti
        # se racunaju iz memorije u odom okviru, pa vrijede i kad ih lidar vise
        # ne vidi.
        vy = 0.0
        d = state["doorway"]
        clr = clearances()
        left, right = clr["left"], clr["right"]

        yaw_err, yaw_on = 0.0, False
        if steer:
            gate = gate_smooth(_gate_target(left, right))
            if gate is not None:
                yaw_err, lat_err, yaw_on = gate
                vy = float(
                    np.clip(GATE_LAT_GAIN * lat_err, -ALIGN_SPEED_MPS, ALIGN_SPEED_MPS)
                )
        # Bez steer nema NIKAKVE korekcije: poravnanje je gotovo prije voznje,
        # a krilo kliznih vrata se povuce u stranu i ne strsi u prolaz. Stalno
        # korigiranje prema sredini otvora je ondje samo titralo.
        node.get_logger().info(
            f"lijevo={'n/a' if left is None else f'{left[0]*1000:.0f}mm'}  "
            f"desno={'n/a' if right is None else f'{right[0]*1000:.0f}mm'}  "
            f"kut={math.degrees(yaw_err):+.1f}deg  vy={vy:+.2f}",
            throttle_duration_sec=1.0,
        )
        # Ispod praga trenja se baza bocno ne mice, pa se mala naredba podize
        # na prag umjesto da se odbaci.
        if 1e-6 < abs(vy) < ALIGN_MIN_MPS:
            vy = math.copysign(ALIGN_MIN_MPS, vy)
        cmd["vy"] = float(np.clip(vy, -ALIGN_SPEED_MPS, ALIGN_SPEED_MPS))

        # Zakret krece tek kad prednji rub baze prijedje ravninu zidova, i ide
        # OD krila. Rotacija ima mrtvu zonu od 0.36 rad/s, pa se naredba skalira
        # inverzom izmjerene relacije.
        # Zakret ovisi samo o blizini krila, ne o vidljivosti otvora - otvor
        # nestane iz skena prije nego prednji rub baze prijedje ravninu zidova.
        # Zakret prema CILJNOJ orijentaciji - okomici na spojnicu najuzeg
        # prolaza. Ranije se zadavala samo kutna brzina bez cilja, pa se zakret
        # gomilao i robot je spiralno skretao.
        wz = 0.0
        if yaw_on:
            want_w = float(np.clip(GATE_YAW_GAIN * yaw_err, -0.3, 0.3))
            wz = math.copysign((abs(want_w) + 0.204) / 0.562, want_w)
        cmd["wz"] = float(wz)

        if d is None:
            lost += 1
            if lost >= LOST_DOORWAY_STEPS:
                # Otvor je nestao iz vidnog polja (rubovi su izasli iz maske
                # samozaklona), ne znaci da smo prosli. Zadnji vidjeni cx je
                # udaljenost do ravnine zida; do nje treba dodati jos pola
                # duljine baze da i straznji rub prodje, plus marza.
                if last_cx is None:
                    outcome = "otvor nikad nije vidjen - ne ulazim"
                    break
                need = last_cx + KMR_LENGTH_M / 2.0 + PASS_MARGIN_M
                node.get_logger().info(
                    f"Otvor izasao iz vidnog polja na cx={last_cx:+.2f} m - "
                    f"vozim jos {need:.2f} m (mjereno odometrijom)."
                )
                done = drive_distance(need, DRIVE_SPEED_MPS, avoid=steer)
                outcome = "prosao" if done is not None else "prepreka preblizu"
                break
        else:
            lost = 0
            _, cx, cy, _, _ = d
            last_cx = cx
            node.get_logger().info(
                f"otvor na cx={cx:+.2f} m", throttle_duration_sec=2.0
            )
            if cx < PASS_CX_M:
                outcome = f"prosao (otvor iza, cx={cx:+.2f} m)"
                break
        time.sleep(CONTROL_PERIOD_SEC)

    cmd["vx"] = 0.0
    node.get_logger().info("=== SAZETAK ===")
    node.get_logger().info(f"  ishod: {outcome}")
    shutdown()

    cmd["vx"] = 0.0
    node.get_logger().info("=== SAZETAK ===")
    node.get_logger().info(f"  ishod: {outcome}")
    shutdown()


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
