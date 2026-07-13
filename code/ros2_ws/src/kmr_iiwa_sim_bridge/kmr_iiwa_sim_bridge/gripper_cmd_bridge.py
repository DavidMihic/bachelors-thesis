"""
Konvencija: 0.0 = potpuno otvoren, 1.0 = potpuno zatvoren (isto kao q=0/q=0.025
u URDF-u/gripper.xacro, pa je mapiranje direktno mnozenje, bez inverzije).

Pokretanje:
    ./isaaclab.sh -p <putanja>/kmr_iiwa_sim_bridge/gripper_cmd_bridge.py \
        --usd_path <putanja>/assets/kmr_iiwa_full.usd

"""

import argparse
import threading

import numpy as np

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--usd_path", type=str, required=True, help="Putanja do kmr_iiwa_full.usd"
)
parser.add_argument("--headless", action="store_true", help="Pokreni bez GUI-ja")
parser.add_argument(
    "--articulation_prim_path",
    type=str,
    default="/kmr_iiwa/base_link",
    help="Prim path root artikulacije (gripper zglobovi su dio ISTE artikulacije kao ruka).",
)
parser.add_argument(
    "--gripper_cmd_topic",
    type=str,
    default="/gripper_cmd",
    help="ROS2 topic za Float32 komandu",
)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

# Isaac Sim / omni importi moraju doci nakon SimulationApp starta
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402

# ROS2 se moze importati bilo kad, ali logicki grupiramo ovdje
import rclpy  # noqa: E402
from std_msgs.msg import Float32  # noqa: E402
from rclpy.node import Node  # noqa: E402

FINGER_JOINTS = [f"gripper_finger_{i}_joint" for i in range(1, 5)]
STROKE = 0.025  # m; 0.0=otvoren, STROKE=zatvoren (isto kao URDF joint limit)


class GripperCmdBuffer:
    """Thread-safe spremnik za zadnju primljenu closing-fraction vrijednost."""

    def __init__(self):
        self._lock = threading.Lock()
        self.closing_fraction = 0.0  # pocetno stanje: potpuno otvoren

    def update(self, msg: Float32):
        with self._lock:
            self.closing_fraction = float(np.clip(msg.data, 0.0, 1.0))

    def get(self):
        with self._lock:
            return self.closing_fraction


class GripperCmdSubscriber(Node):
    def __init__(self, buffer: GripperCmdBuffer, topic: str):
        super().__init__("kmr_gripper_cmd_bridge")
        self._buffer = buffer
        self.create_subscription(Float32, topic, self._callback, 10)
        self.get_logger().info(f"Slusam {topic} (0.0=otvoren, 1.0=zatvoren)...")

    def _callback(self, msg: Float32):
        self._buffer.update(msg)


def main():
    stage_utils.open_stage(args.usd_path)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    world.reset()

    robot = Articulation(prim_paths_expr=args.articulation_prim_path, name="kmr_iiwa")
    world.scene.add(robot)
    world.reset()

    dof_names = robot.dof_names
    finger_indices = np.array([dof_names.index(n) for n in FINGER_JOINTS])

    # --- ROS2 strana: spin u pozadinskom threadu da ne blokira sim petlju ---
    rclpy.init()
    buffer = GripperCmdBuffer()
    ros_node = GripperCmdSubscriber(buffer, args.gripper_cmd_topic)
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    print(f"[INFO] Artikulacija: {args.articulation_prim_path}")
    print(f"[INFO] gripper_cmd topic: {args.gripper_cmd_topic}")
    print("[INFO] Pokrecem simulacijsku petlju. Ctrl+C za izlaz.")

    try:
        while simulation_app.is_running():
            closing_fraction = buffer.get()
            target = np.full((1, len(finger_indices)), closing_fraction * STROKE)

            robot.set_joint_position_targets(target, joint_indices=finger_indices)

            world.step(render=not args.headless)
    except KeyboardInterrupt:
        pass
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
