"""
door_probe.py - procjena ogranicenja gibanja i krutosti okoline iz mjerenja
sile i momenta, dok gripper vec drzi kvaku.

Iz uhvacene poze rade se mali, silom-nadzirani pomaci u sest smjerova (+-3
osi okvira gripera). Za svaki smjer biljezi se ostvareni pomak i porast sile,
a smjerovi se klasificiraju po OMJERU ostvareno/naredjeno.

Krutost se i dalje racuna i zapisuje, ali NIJE kriterij. Zakretna vrata nemaju
povratnu krutost - slobodno se okrecu, a otpor im dolazi iz trenja u zglobu i
inercije krila. Izmjerena "krutost" je zato dominantno krutost pozicijskog
upravljanja ruke (pogoni na 100000) koja gura u tu inerciju, pa ispada u
stotinama kN/m i ne govori nista o vratima. Uz to se racunala iz jednog uzorka
(sila/pomak), sto pri pomacima ispod desetinke milimetra daje i negativne
vrijednosti.

Omjer razdvaja smjerove: izmjereno 0,47 u slobodnima naspram 0,003-0,05 u
ogranicenima, dakle red velicine razlike.

Kao izlazna velicina biljezi se OTPORNI MOMENT (sila puta udaljenost kvake od
sarke). Za zakretna vrata je izmjereno oko 280 Nm u smjeru otvaranja, sto se
poklapa s 280-310 Nm izracunatim iz stvarnih runova otvaranja - dakle dvije
neovisne metode daju isti otpor.

KLASIFIKACIJA TIPA VRATA IZ SILE
approach je slobodan kod zakretnih (normala na krilo poklapa se s tangentom
luka sarka-kvaka pri zatvorenim vratima) a blokiran kod kliznih (gura u krilo).
closing je obrnuto: blokiran kod zakretnih (vertikala) a slobodan kod kliznih
(smjer klizanja). Koja je od te dvije osi meksa, takva su vrata - neovisno o
vidu, cime se potvrduje klasifikacija iz geometrije tagova.

along_bar se NE koristi za klasifikaciju: slobodan je kod obje vrste, jer
gripper ondje klizi duz kvake. To je ujedno i ogranicenje metode - sila sama ne
razlikuje gibanje vrata od klizanja hvata, oboje daje nisku silu.

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

from kmr_iiwa_task.add_door_collision import COLLISION_OBJECT_ID
from kmr_iiwa_task.geometry import quat_rotate_vector

JOINT_NAMES = [f"iiwa_joint_{i}" for i in range(1, 8)]

STEP_M = 0.003  # velicina jednog koraka
MAX_STEPS = 4  # najvise koraka po smjeru
FORCE_ABORT_N = 700.0  # prekid smjera cim sila predje ovo. Na 100 i 300 N se
# approach smjerovi prekidali vec u prvom koraku (izmjereno 424 i 535 N pri
# pomaku od 3 mm), sto je ispod sile potrebne da se vrata uopce pokrenu.
HANDLE_RADIUS_M = 0.65  # udaljenost kvake od osi sarke, za otporni moment
SETTLE_SEC = 0.4  # da se sila smiri nakon koraka
# Kriterij je NAGIB sile po naredenom pomaku [N/m]. Izmjereno: slobodan smjer
# oko 2 000, vertikala oko 19 000, blokiran oko 120 000 N/m.
FREE_SLOPE_N_PER_M = 8000.0  # ispod = slobodan
BLOCKED_SLOPE_N_PER_M = 40000.0  # iznad = ogranicen

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
    # inace mijesa obicne pretplate na istom nodu.
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
                # Omjer se uzima iz PRVOG koraka, ne zadnjeg. U zadnjem je sila
                # najveca, pa dominira popustanje hvata: izmjereno je da
                # approach- (graniznik zgloba, ne smije se micati) daje 3.14 mm
                # pri 810 N, dok je u prvom koraku 0.42 mm pri 450 N.
                first = samples[0]
                ratio = (
                    first["achieved_m"] / first["commanded_m"]
                    if first["commanded_m"]
                    else 0.0
                )
                # Kriterij: kako sila raste s NAREDENIM pomakom.
                #
                # Slobodan smjer: sila naraste do razine potrebne da se svlada
                # trenje u zglobu i tu STAGNIRA, jer se vrata gibaju
                # (izmjereno 421 -> 449 -> 449 -> 442 N).
                # Blokiran smjer: sila monotono RASTE, jer se nista ne giba pa
                # se samo napinje lanac (450 -> 810 N).
                #
                # Dijeli se naredenim pomakom, ne ostvarenim: ostvareni sadrzi
                # prodiranje prstiju u model kvake i zakretanje poluge u hvatu,
                # pa je izmedu runova varirao i do 3 mm u blokiranim smjerovima.
                slope = None
                if len(samples) >= 2:
                    dcmd = samples[-1]["commanded_m"] - samples[0]["commanded_m"]
                    dfrc = samples[-1]["force_along_N"] - samples[0]["force_along_N"]
                    if abs(dcmd) > 1e-9:
                        slope = dfrc / dcmd
                elif aborted and samples:
                    # Jedan uzorak iznad praga sile znaci da se smjer nije mogao
                    # ni zapoceti - najjasniji moguci znak blokade. Nagib se
                    # procjenjuje iz te jedne tocke.
                    slope = samples[0]["force_along_N"] / samples[0]["commanded_m"]

                # Krutost se zapisuje, ali nije kriterij (vidi docstring).
                # Racuna se kao NAGIB kroz uzorke, ne iz jednog - inace u nju
                # ulazi i konstantni pomak od pocetka.
                stiffness = None
                if len(samples) >= 2:
                    dx = samples[-1]["achieved_m"] - samples[0]["achieved_m"]
                    df = samples[-1]["force_along_N"] - samples[0]["force_along_N"]
                    if abs(dx) > 1e-4:
                        stiffness = df / dx
                elif abs(last["achieved_m"]) > 1e-4:
                    stiffness = last["force_along_N"] / last["achieved_m"]

                # Pomak suprotnog predznaka od naredbe znaci da se gibalo nesto
                # drugo - poluga u prstima, ne vrata. Izmjereno: ostvareni pomak
                # je unutar istog smjera prelazio iz -2.83 u +3.70 mm.
                if slope is None:
                    verdict = "NEPOZNAT"
                elif abs(slope) <= FREE_SLOPE_N_PER_M:
                    verdict = "SLOBODAN"
                elif abs(slope) >= BLOCKED_SLOPE_N_PER_M:
                    verdict = "OGRANICEN"
                else:
                    verdict = "DJELOMICAN"

                # Otporni moment je fizikalno smislena velicina za vrata;
                # "krutost" nije, jer vrata nisu opruga (vidi docstring).
                torque = last["force_along_N"] * HANDLE_RADIUS_M

                results[label] = {
                    "verdict": verdict,
                    "slope_N_per_m": slope,
                    "ratio": ratio,
                    "torque_Nm": torque,
                    "aborted": aborted,
                    "stiffness_N_per_m": stiffness,
                    "samples": samples,
                }

    goto(start_pos, grasp_quat)

    node.get_logger().info("=== SAZETAK ===")
    for label, r in results.items():
        sl = r["slope_N_per_m"]
        node.get_logger().info(
            f"  {label:12s} {r['verdict']:12s} "
            f"nagib={'n/a' if sl is None else f'{sl/1000:+.1f} N/mm'}  "
            f"moment={r['torque_Nm']:.0f} Nm  ({len(r['samples'])} uzoraka"
            f"{', prekinuto silom' if r['aborted'] else ''})"
        )

    # Klasifikacija tipa vrata iz sile, neovisno o vidu.
    #
    # approach je slobodan kod zakretnih (vrata se otvaraju po toj normali) a
    # blokiran kod kliznih (gura u krilo); closing je obrnuto - blokiran kod
    # zakretnih (vertikala) a slobodan kod kliznih (smjer klizanja). Koja je od
    # te dvije osi mekša, takva su vrata.
    #
    # along_bar se NE koristi: slobodan je kod obje vrste, jer gripper ondje
    # klizi duz kvake. Sila sama ne razlikuje gibanje vrata od klizanja hvata -
    # oboje daje nisku silu.
    def softest(prefix):
        vals = [
            abs(r["slope_N_per_m"])
            for lbl, r in results.items()
            if lbl.startswith(prefix) and r["slope_N_per_m"] is not None
        ]
        return min(vals) if vals else None

    s_app, s_clo = softest("approach"), softest("closing")
    if s_app is not None and s_clo is not None:
        door_type = "ZAKRETNA" if s_app < s_clo else "KLIZNA"
        node.get_logger().info(
            f"  tip vrata iz sile: {door_type}  "
            f"(approach {s_app/1000:.1f} N/mm, closing {s_clo/1000:.1f} N/mm)"
        )
        results["_door_type"] = {
            "type": door_type,
            "approach_slope_N_per_m": s_app,
            "closing_slope_N_per_m": s_clo,
        }

    with open(LOG_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    node.get_logger().info(f"Detalji spremljeni u {LOG_PATH}")

    sensor_exec.shutdown()
    sensor_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
