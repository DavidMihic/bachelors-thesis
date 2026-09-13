"""base_speed_test.py - mjeri postize li baza naredjenu brzinu BEZ kontakta s
vratima, i kako izgleda profil brzine kroz voznju.

Svrha: razdvojiti tri moguca uzroka toga sto baza ne prelazi naredjeni put:
  a) kasnjenje starta - naredba stigne, ali robot krene tek nakon nekog
     vremena, pa se gubi dio intervala
  b) baza ne dosegne naredjenu brzinu
  c) dosegne je, ali je ne zadrzi

Zato se uz ukupni pomak biljezi i PROFIL brzine, racunat iz uzastopnih poza s
/ground_truth/base_pose. Prosjek put/vrijeme sam po sebi ne razlikuje ta tri
slucaja - robot koji pola intervala stoji pa vozi punom brzinom daje isti
prosjek kao onaj koji cijelo vrijeme vozi upola sporije.

PREDUVJET
- cmd_vel_bridge radi, s aktivnim door_gt_publisherom
- robot NE drzi kvaku i ima slobodan prostor u smjeru voznje
- ruka u neutralnoj pozi

Zadano vozi UNATRAG (-x), dalje od vrata. Provjeri da iza robota ima barem 2 m
slobodnog prostora prije pokretanja.

Pokretanje:
    ros2 run kmr_iiwa_task base_speed_test
    ros2 run kmr_iiwa_task base_speed_test --ros-args -p axis:=y
    ros2 run kmr_iiwa_task base_speed_test --ros-args -p speeds:="[0.15]" \
        -p duration_sec:=10.0
"""

import json
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node

PUBLISH_PERIOD_SEC = 0.02  # 50 Hz, isti ritam kao open_sliding/open_revolute
SETTLE_SEC = 1.5  # pauza izmedju mjerenja, da se baza smiri
LOG_PATH = "/tmp/base_speed_test.json"


class BaseSpeedTest(Node):
    def __init__(self):
        super().__init__("base_speed_test")

        self.declare_parameter("axis", "x")
        self.declare_parameter("direction", -1.0)  # -1 = unatrag, dalje od vrata
        self.declare_parameter("speeds", [0.05, 0.10, 0.15, 0.22])
        self.declare_parameter("duration_sec", 3.0)

        self.axis = self.get_parameter("axis").get_parameter_value().string_value
        self.direction = (
            self.get_parameter("direction").get_parameter_value().double_value
        )
        self.speeds = list(
            self.get_parameter("speeds").get_parameter_value().double_array_value
        )
        self.duration = (
            self.get_parameter("duration_sec").get_parameter_value().double_value
        )

        self.base_xy = None
        # Uzorci brzine: popunjavaju se samo dok je samples lista (tj. tijekom
        # mjerene dionice), inace se preskacu.
        self.samples = None
        self.t0 = 0.0
        self._last_t = None
        self.create_subscription(
            PoseStamped, "/ground_truth/base_pose", self._on_base, 10
        )
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.cmd_speed = 0.0
        self.stop_flag = False

    def _on_base(self, msg: PoseStamped):
        p = np.array([msg.pose.position.x, msg.pose.position.y])
        t = time.monotonic()
        if self.samples is not None and self.base_xy is not None:
            dt = t - self._last_t
            if dt > 1e-6:
                self.samples.append(
                    (t - self.t0, float(np.linalg.norm(p - self.base_xy)) / dt)
                )
        self.base_xy = p
        self._last_t = t

    def publisher_loop(self):
        """cmd_vel se salje iz zasebne niti u stalnom ritmu - cmd_vel_bridge
        primjenjuje zadnju primljenu poruku svaki fizicki korak i nema failsafe
        timeout, pa neujednacen ritam znaci trzajno gibanje."""
        while rclpy.ok() and not self.stop_flag:
            tw = Twist()
            if self.axis == "y":
                tw.linear.y = self.cmd_speed
            else:
                tw.linear.x = self.cmd_speed
            self.cmd_pub.publish(tw)
            time.sleep(PUBLISH_PERIOD_SEC)


def main():
    rclpy.init()
    node = BaseSpeedTest()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    node.get_logger().info("Cekam /ground_truth/base_pose...")
    t0 = time.monotonic()
    while node.base_xy is None and rclpy.ok():
        time.sleep(0.1)
        if time.monotonic() - t0 > 10.0:
            node.get_logger().error(
                "Nema /ground_truth/base_pose - je li door_gt_publisher aktivan "
                "u cmd_vel_bridgeu? Bez njega ovaj test nema sto mjeriti."
            )
            rclpy.shutdown()
            return

    threading.Thread(target=node.publisher_loop, daemon=True).start()

    node.get_logger().info(
        f"Os {node.axis}, smjer {node.direction:+.0f}, "
        f"{node.duration:.1f}s po brzini. PAZI: robot se giba!"
    )

    results = []
    for speed in node.speeds:
        # smiri se prije mjerenja
        node.cmd_speed = 0.0
        time.sleep(SETTLE_SEC)

        p_start = node.base_xy.copy()
        t_start = time.monotonic()
        node.t0 = t_start
        node._last_t = t_start
        node.samples = []
        node.cmd_speed = node.direction * speed

        while time.monotonic() - t_start < node.duration and rclpy.ok():
            time.sleep(0.02)

        p_end = node.base_xy.copy()  # ocitaj PRIJE zaustavljanja
        node.cmd_speed = 0.0
        t_end = time.monotonic()
        samples = node.samples or []
        node.samples = None
        time.sleep(0.5)

        elapsed = t_end - t_start
        commanded = speed * elapsed
        actual = float(np.linalg.norm(p_end - p_start))
        ratio = actual / commanded if commanded > 1e-9 else float("nan")

        node.get_logger().info(
            f"  naredjeno {speed:.2f} m/s -> presao {actual*1000:6.0f} mm od "
            f"{commanded*1000:6.0f} mm  ({100*ratio:5.1f}%),  "
            f"prosjek put/vrijeme {actual/elapsed:.3f} m/s"
        )

        # Profil razlikuje kasnjenje starta od gubitka brzine - prosjek
        # put/vrijeme to ne moze.
        if samples:
            vmax = max(v for _, v in samples)
            t_reach = next((t for t, v in samples if v >= 0.9 * speed), None)
            v_late = [v for t, v in samples if t > 0.7 * elapsed]
            node.get_logger().info(
                f"    najveca izmjerena brzina {vmax:.3f} m/s"
                + (
                    f", 90% naredjene dosegnuto u {t_reach:.2f} s"
                    if t_reach is not None
                    else ", 90% naredjene NIKAD dosegnuto"
                )
                + (
                    f", prosjek zadnje trecine {sum(v_late)/len(v_late):.3f} m/s"
                    if v_late
                    else ""
                )
            )
            node.get_logger().info("    profil (s -> m/s):")
            for t_rel, v in samples[::5]:
                node.get_logger().info(f"      {t_rel:5.2f}  {v:.3f}")
        results.append(
            {
                "commanded_speed_mps": speed,
                "elapsed_s": elapsed,
                "commanded_m": commanded,
                "actual_m": actual,
                "ratio": ratio,
                "actual_speed_mps": actual / elapsed,
                "profile": [[round(t, 3), round(v, 4)] for t, v in samples],
            }
        )

    node.stop_flag = True
    time.sleep(0.1)
    node.cmd_pub.publish(Twist())
    time.sleep(0.5)

    node.get_logger().info("=== SAZETAK ===")
    ratios = [r["ratio"] for r in results if math.isfinite(r["ratio"])]
    if ratios:
        node.get_logger().info(
            f"  omjer stvarno/naredjeno: min {min(ratios)*100:.0f}%, "
            f"max {max(ratios)*100:.0f}%"
        )
        if max(ratios) > 0.9:
            node.get_logger().info(
                "  Baza slobodno postize naredjeno -> zastajkivanje pri "
                "otvaranju vrata dolazi od OTPORA VRATA, ne od platforme."
            )
        else:
            node.get_logger().warn(
                "  Baza NE postize naredjeno ni bez kontakta -> problem je u "
                "cmd_vel_bridgeu ili fizici platforme, a ne u vratima."
            )

    with open(LOG_PATH, "w") as fh:
        json.dump(
            {"axis": node.axis, "direction": node.direction, "runs": results},
            fh,
            indent=2,
        )
    node.get_logger().info(f"Detalji u {LOG_PATH}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
