"""base_speed_test.py - mjeri postize li baza naredjenu brzinu BEZ kontakta s
vratima.

Svrha: razdvojiti dva moguca uzroka toga sto baza pri otvaranju vrata postize
samo ~23% naredjene brzine (izmjereno protiv ground trutha, i na kliznim i na
zakretnim vratima):
  a) vrata pruzaju otpor -> baza zastaje  (ocekivano, rjesiv problem)
  b) baza ni slobodno ne postize naredjeno -> problem je u cmd_vel_bridgeu ili
     u fizici platforme, a sva dosadasnja mjerenja na vratima su posljedica

Test vozi bazu kroz nekoliko brzina i za svaku usporedi naredjeni pomak sa
stvarnim iz /ground_truth/base_pose (koji objavljuje door_gt_publisher).

PREDUVJET
- cmd_vel_bridge radi, s aktivnim door_gt_publisherom
- robot NE drzi kvaku i ima slobodan prostor u smjeru voznje
- ruka u neutralnoj pozi

Zadano vozi UNATRAG (-x), dalje od vrata, da ne udari u njih. Provjeri da iza
robota ima barem 2 m slobodnog prostora prije pokretanja.

Pokretanje:
    ros2 run kmr_iiwa_task base_speed_test
    ros2 run kmr_iiwa_task base_speed_test --ros-args -p axis:=y
    ros2 run kmr_iiwa_task base_speed_test --ros-args -p speeds:="[0.05,0.1,0.2]"
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
        self.create_subscription(
            PoseStamped, "/ground_truth/base_pose", self._on_base, 10
        )
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.cmd_speed = 0.0
        self.stop_flag = False

    def _on_base(self, msg: PoseStamped):
        self.base_xy = np.array([msg.pose.position.x, msg.pose.position.y])

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
        node.cmd_speed = node.direction * speed

        while time.monotonic() - t_start < node.duration and rclpy.ok():
            time.sleep(0.02)

        p_end = node.base_xy.copy()  # ocitaj PRIJE zaustavljanja
        node.cmd_speed = 0.0
        t_end = time.monotonic()
        time.sleep(0.5)

        elapsed = t_end - t_start
        commanded = speed * elapsed
        actual = float(np.linalg.norm(p_end - p_start))
        ratio = actual / commanded if commanded > 1e-9 else float("nan")

        node.get_logger().info(
            f"  naredjeno {speed:.2f} m/s -> presao {actual*1000:6.0f} mm od "
            f"{commanded*1000:6.0f} mm  ({100*ratio:5.1f}%),  "
            f"stvarna brzina {actual/elapsed:.3f} m/s"
        )
        results.append(
            {
                "commanded_speed_mps": speed,
                "elapsed_s": elapsed,
                "commanded_m": commanded,
                "actual_m": actual,
                "ratio": ratio,
                "actual_speed_mps": actual / elapsed,
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
