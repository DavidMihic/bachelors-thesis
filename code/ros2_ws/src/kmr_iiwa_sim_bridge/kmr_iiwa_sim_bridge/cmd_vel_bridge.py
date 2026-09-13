"""
cmd_vel_bridge.py — pokrece Isaac Sim, ucitava kmr_iiwa.usd, i pretplacuje se
direktno na /cmd_vel (geometry_msgs/Twist). Svaki fizicki korak, zadnja
primljena Twist poruka se konvertira iz robot-lokalnog u world-frame i
postavlja kao linearna/kutna brzina na base_link (kinematicko gibanje cijele
baze, bez fizike po-kotacu - konzistentno s dogovorenom arhitekturom).

Pokretanje (iz IsaacLab root direktorija, NE preko `ros2 run` jer treba
Isaac Simov python environment):

    ./isaaclab.sh -p <putanja>/kmr_iiwa_sim_bridge/cmd_vel_bridge.py \
        --usd_path <putanja>/assets/kmr_iiwa.usd

U drugom terminalu (obican ROS2 environment):

    ros2 run teleop_twist_keyboard teleop_twist_keyboard

Uz pretplatu na /cmd_vel, most objavljuje i ODOMETRIJU na /odom
(nav_msgs/Odometry) te TF odom -> base_link.

Odometrija se racuna iz poze tijela (get_world_poses), NE iz integracije
zadane brzine. Izmjereno: baza ne postize naredjenu brzinu (oko 45% uzduzno),
a prijedjeni put ne odgovara ni izmjerenoj brzini, pa svaki proracun puta iz
naredbe i vremena visestruko promasuje. Zato se put mjeri, a ne racuna.

U modelu baza nema kotace (jedna kolizijska kutija), pa nema enkodera iz kojih
bi se odometrija inace izvela. Pravi KMR iiwa odometriju objavljuje kao dio
isporucenog softvera, pa je rijec o senzoru koji robot stvarno posjeduje - za
razliku od stanja zglobova vrata, koje se cita samo kao ground truth za
validaciju. Ova odometrija je pritom savrsena: bez proklizavanja i bez
akumulirajuceg drifta, sto pravi enkoderi imaju.

Napomena o API pozivima: set_linear_velocities/set_angular_velocities i
get_world_poses su nazivi iz isaacsim.core.prims.Articulation u trenutnim
verzijama Isaac Sima. Ako tvoja verzija ima drugacije nazive metoda, pokreni
`print(dir(robot))` odmah nakon world.reset() da pronadjes tocne nazive.
"""

import argparse
import threading

import numpy as np

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--usd_path", type=str, required=True, help="Putanja do kmr_iiwa.usd"
)
parser.add_argument("--headless", action="store_true", help="Pokreni bez GUI-ja")
parser.add_argument(
    "--skip-ground-plane",
    action="store_false",
    help="Ne dodaji default ground plane - koristi ako ucitavas integracijsku "
    "scenu (npr. build_integration_scene.py output) koja ga vec ima.",
)
parser.add_argument(
    "--articulation-prim-path",
    type=str,
    default="/World/Robot/base_link",
    help="Prim path root artikulacije (provjeri u check.py ispisu ako se razlikuje)",
)
parser.add_argument(
    "--cmd_vel_topic", type=str, default="/cmd_vel", help="ROS2 topic za Twist poruke"
)
parser.add_argument(
    "--odom-topic", type=str, default="/odom", help="ROS2 topic za odometriju"
)
parser.add_argument(
    "--odom-every",
    type=int,
    default=2,
    help="Objavi odometriju svaki n-ti fizicki korak (60 Hz / n).",
)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

# Isaac Sim / omni importi moraju doci nakon SimulationApp starta
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

# Bez ovoga, standalone SimulationApp ne ucitava ROS2 bridge ekstenziju
# automatski (GUI aplikacija je ima ukljucenu po defaultu) - kamera na
# robotu se nece publishati preko ROS2 dok ovo ne prodje.
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# ROS2 se moze importati bilo kad, ali logicki grupiramo ovdje
import rclpy  # noqa: E402
from geometry_msgs.msg import Twist, TransformStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from tf2_ros import TransformBroadcaster  # noqa: E402


class CmdVelBuffer:
    """Thread-safe spremnik za zadnju primljenu Twist poruku."""

    def __init__(self):
        self._lock = threading.Lock()
        self.linear_x = 0.0
        self.linear_y = 0.0
        self.angular_z = 0.0

    def update(self, msg: Twist):
        with self._lock:
            self.linear_x = msg.linear.x
            self.linear_y = msg.linear.y
            self.angular_z = msg.angular.z

    def get(self):
        with self._lock:
            return self.linear_x, self.linear_y, self.angular_z


class CmdVelSubscriber(Node):
    def __init__(self, buffer: CmdVelBuffer, topic: str):
        super().__init__("kmr_cmd_vel_bridge")
        self._buffer = buffer
        self.create_subscription(Twist, topic, self._callback, 10)
        self.get_logger().info(f"Slusam {topic}...")

    def _callback(self, msg: Twist):
        self._buffer.update(msg)


class OdomPublisher(Node):
    """Objavljuje /odom i TF odom -> base_link iz poze baze u simulaciji."""

    def __init__(self, topic: str, publish_every: int):
        super().__init__("kmr_odom_publisher")
        self._pub = self.create_publisher(Odometry, topic, 10)
        self._tf = TransformBroadcaster(self)
        self._every = max(1, publish_every)
        self._i = 0
        # Ishodiste odom okvira je poza baze pri pokretanju, pa odometrija
        # krece od nule bez obzira gdje je robot spawnan u sceni.
        self._origin_pos = None
        self._origin_yaw = 0.0
        self.get_logger().info(f"Objavljujem odometriju na {topic}.")

    def publish(self, position, quat_wxyz, lin_vel, ang_vel) -> None:
        self._i += 1
        if self._i % self._every:
            return

        yaw = quat_to_yaw(quat_wxyz)
        if self._origin_pos is None:
            self._origin_pos = np.array(position, dtype=float).copy()
            self._origin_yaw = yaw

        d = np.array(position, dtype=float) - self._origin_pos
        c, sn = np.cos(-self._origin_yaw), np.sin(-self._origin_yaw)
        x = float(d[0] * c - d[1] * sn)
        y = float(d[0] * sn + d[1] * c)
        rel_yaw = float(
            np.arctan2(np.sin(yaw - self._origin_yaw), np.cos(yaw - self._origin_yaw))
        )
        qz, qw = float(np.sin(rel_yaw / 2.0)), float(np.cos(rel_yaw / 2.0))

        stamp = self.get_clock().now().to_msg()

        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        # Brzina se po ROS konvenciji izrazava u okviru djeteta (base_link).
        vx, vy = float(lin_vel[0]), float(lin_vel[1])
        cy, sy = np.cos(-yaw), np.sin(-yaw)
        msg.twist.twist.linear.x = vx * cy - vy * sy
        msg.twist.twist.linear.y = vx * sy + vy * cy
        msg.twist.twist.angular.z = float(ang_vel[2])
        self._pub.publish(msg)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self._tf.sendTransform(tf)


def quat_to_yaw(quat_wxyz: np.ndarray) -> float:
    """USD/Isaac Sim koristi (w, x, y, z) konvenciju za kvaternione."""
    w, x, y, z = quat_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def main():
    stage_utils.open_stage(args.usd_path)

    world = World(stage_units_in_meters=1.0)
    if not args.skip_ground_plane:
        world.scene.add_default_ground_plane()
    world.reset()

    robot = Articulation(prim_paths_expr=args.articulation_prim_path, name="kmr_iiwa")
    world.scene.add(robot)
    world.reset()

    # --- ROS2 strana: spin u pozadinskom threadu da ne blokira sim petlju ---
    rclpy.init()
    buffer = CmdVelBuffer()
    ros_node = CmdVelSubscriber(buffer, args.cmd_vel_topic)
    odom_node = OdomPublisher(args.odom_topic, args.odom_every)

    # Oba nodea u istom izvrsavacu: rclpy.spin bez izricitog izvrsavaca koristi
    # globalni, pa bi ga dvije niti vrtjele istovremeno.
    ros_executor = MultiThreadedExecutor(2)
    ros_executor.add_node(ros_node)
    ros_executor.add_node(odom_node)
    ros_thread = threading.Thread(target=ros_executor.spin, daemon=True)
    ros_thread.start()

    # from handle_gt_publisher import HandleGroundTruth
    from door_gt_publisher import DoorGroundTruth

    # gt = HandleGroundTruth(
    #     tag_a_path="/World/Door/handle_tag_a",
    #     tag_b_path="/World/Door/handle_tag_b",
    #     base_path="/World/Robot/base_link",
    # )

    gt_door = DoorGroundTruth()

    print(f"[INFO] Artikulacija: {args.articulation_prim_path}")
    print(f"[INFO] cmd_vel topic: {args.cmd_vel_topic}")
    print("[INFO] Pokrecem simulacijsku petlju. Ctrl+C za izlaz.")

    try:
        while simulation_app.is_running():
            lin_x, lin_y, ang_z = buffer.get()

            # Twist dolazi u robot-lokalnom (base_link) frameu; treba ga
            # rotirati u world frame prije nego ga upisemo u simulaciju.
            positions, orientations = robot.get_world_poses()
            yaw = quat_to_yaw(orientations[0])

            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            world_vx = lin_x * cos_yaw - lin_y * sin_yaw
            world_vy = lin_x * sin_yaw + lin_y * cos_yaw

            robot.set_linear_velocities(np.array([[world_vx, world_vy, 0.0]]))
            robot.set_angular_velocities(np.array([[0.0, 0.0, ang_z]]))

            world.step(render=not args.headless)

            # DIJAGNOSTIKA: usporedi zadanu i stvarnu brzinu odmah nakon koraka
            # v_meas = robot.get_linear_velocities()[0]
            # print(
            #     f"[VEL] zadano ({world_vx:+.3f}, {world_vy:+.3f})  "
            #     f"izmjereno ({v_meas[0]:+.3f}, {v_meas[1]:+.3f}, {v_meas[2]:+.3f})"
            # )

            # gt.publish()
            gt_door.publish()
            positions, orientations = robot.get_world_poses()
            odom_node.publish(
                positions[0],
                orientations[0],
                robot.get_linear_velocities()[0],
                robot.get_angular_velocities()[0],
            )

    except KeyboardInterrupt:
        pass
    finally:
        ros_executor.shutdown()
        ros_node.destroy_node()
        odom_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
