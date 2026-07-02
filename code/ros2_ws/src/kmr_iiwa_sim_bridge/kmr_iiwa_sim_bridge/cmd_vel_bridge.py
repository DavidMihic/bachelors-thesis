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
    "--articulation_prim_path",
    type=str,
    default="/kmr_iiwa/base_link",
    help="Prim path root artikulacije (provjeri u check.py ispisu ako se razlikuje)",
)
parser.add_argument(
    "--cmd_vel_topic", type=str, default="/cmd_vel", help="ROS2 topic za Twist poruke"
)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

# Isaac Sim / omni importi moraju doci nakon SimulationApp starta
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402

# ROS2 se moze importati bilo kad, ali logicki grupiramo ovdje
import rclpy  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from rclpy.node import Node  # noqa: E402


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


def quat_to_yaw(quat_wxyz: np.ndarray) -> float:
    """USD/Isaac Sim koristi (w, x, y, z) konvenciju za kvaternione."""
    w, x, y, z = quat_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def main():
    stage_utils.open_stage(args.usd_path)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    world.reset()

    robot = Articulation(prim_paths_expr=args.articulation_prim_path, name="kmr_iiwa")
    world.scene.add(robot)
    world.reset()

    # --- ROS2 strana: spin u pozadinskom threadu da ne blokira sim petlju ---
    rclpy.init()
    buffer = CmdVelBuffer()
    ros_node = CmdVelSubscriber(buffer, args.cmd_vel_topic)
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

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
    except KeyboardInterrupt:
        pass
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
