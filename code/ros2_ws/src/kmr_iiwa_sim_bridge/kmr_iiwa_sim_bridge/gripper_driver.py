"""
gripper_driver.py - generički gripper driver za placeholder 4-prsti gripper.
Nasljeđuje gripper_cmd_bridge.py, sad dvosmjeran: prima komandu I javlja stanje,
oboje preko standardnih primitivnih tipova (bez custom .msg definicije).

ARHITEKTURNA NAMJERA: ovaj file je JEDINO mjesto u cijelom stacku koje smije
znati da gripper ima 4 prsta, prismatic zglobove, 25mm hod, itd. Sav ostali
kod (task orkestracija, door-opening logika, bilo koji buduci node) smije
pricati SAMO s ova tri topica - nikad direktno s robot.get_joint_positions()
za gripper zglobove. Kad se jednom (izvan opsega ovog rada) zamijeni za pravi
hardverski gripper, mijenja se SAMO ovaj file (ili se zamijeni pravim vendor
driverom koji publisha/subscriba ista tri topica) - nijedan drugi node se ne dira.

Sučelje:
  SUB  /gripper_cmd      (std_msgs/Float32)  0.0=otvori, 1.0=zatvori
  PUB  /gripper_state    (std_msgs/Float32)  0.0..1.0, stvarna trenutna pozicija
  PUB  /gripper_stalled  (std_msgs/Bool)     True = gura prema targetu ali se
                                              vise ne mice i nije stigao ->
                                              vjerojatno je nesto uhvatio

Isto po smislu kao control_msgs/action/GripperCommand (position/stalled/
reached_goal), samo bez action servera - ako ikad zatreba pravi ROS-standard
interop s gotovim driverom za stvarni hardver, ova tri polja se izravno
premotaju u action goal/result, arhitektura (odvojen driver, generičko
sučelje) se ne mijenja.

Pokretanje (isti obrazac kao cmd_vel_bridge.py):
    ./isaaclab.sh -p <putanja>/kmr_iiwa_sim_bridge/gripper_driver.py \
        --usd_path <putanja>/assets/kmr_iiwa_full.usd

Ručni test u drugom terminalu:
    ros2 topic pub /gripper_cmd std_msgs/msg/Float32 "{data: 1.0}" -1
    ros2 topic echo /gripper_state
    ros2 topic echo /gripper_stalled
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
parser.add_argument("--gripper_cmd_topic", type=str, default="/gripper_cmd")
parser.add_argument("--gripper_state_topic", type=str, default="/gripper_state")
parser.add_argument("--gripper_stalled_topic", type=str, default="/gripper_stalled")
parser.add_argument(
    "--state_publish_every_n_steps",
    type=int,
    default=12,
    help="Throttle za state/stalled publish (izbjegava ROS spam na ~240Hz fizici).",
)
parser.add_argument(
    "--stall_velocity_threshold",
    type=float,
    default=0.0003,
    help="m po publish-intervalu ispod kojega se smatra 'ne mice se'.",
)
parser.add_argument(
    "--stall_position_threshold",
    type=float,
    default=0.0015,
    help="m odstupanja od targeta iznad kojega se, ako se ne mice, smatra 'zapeo'.",
)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

import isaacsim.core.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402

import rclpy  # noqa: E402
from std_msgs.msg import Bool, Float32  # noqa: E402
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


class GripperDriverNode(Node):
    def __init__(
        self,
        buffer: GripperCmdBuffer,
        cmd_topic: str,
        state_topic: str,
        stalled_topic: str,
    ):
        super().__init__("kmr_gripper_driver")
        self._buffer = buffer
        self.create_subscription(Float32, cmd_topic, self._callback, 10)
        self._state_pub = self.create_publisher(Float32, state_topic, 10)
        self._stalled_pub = self.create_publisher(Bool, stalled_topic, 10)

    def _callback(self, msg: Float32):
        self._buffer.update(msg)

    def publish_state(self, closing_fraction: float, stalled: bool):
        self._state_pub.publish(Float32(data=closing_fraction))
        self._stalled_pub.publish(Bool(data=stalled))


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

    rclpy.init()
    buffer = GripperCmdBuffer()
    ros_node = GripperDriverNode(
        buffer,
        args.gripper_cmd_topic,
        args.gripper_state_topic,
        args.gripper_stalled_topic,
    )
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    print(f"[INFO] Artikulacija: {args.articulation_prim_path}")
    print(
        f"[INFO] cmd={args.gripper_cmd_topic}  state={args.gripper_state_topic}  stalled={args.gripper_stalled_topic}"
    )
    print("[INFO] Pokrecem simulacijsku petlju. Ctrl+C za izlaz.")

    step = 0
    prev_mean_pos = None
    try:
        while simulation_app.is_running():
            closing_fraction = buffer.get()
            target_pos = closing_fraction * STROKE
            target = np.full((1, len(finger_indices)), target_pos)

            robot.set_joint_position_targets(target, joint_indices=finger_indices)
            world.step(render=not args.headless)

            if step % args.state_publish_every_n_steps == 0:
                current = robot.get_joint_positions(joint_indices=finger_indices)
                mean_pos = float(np.mean(current[0]))

                stalled = False
                if prev_mean_pos is not None:
                    velocity = abs(mean_pos - prev_mean_pos)
                    pos_error = abs(target_pos - mean_pos)
                    stalled = (
                        velocity < args.stall_velocity_threshold
                        and pos_error > args.stall_position_threshold
                    )
                prev_mean_pos = mean_pos

                ros_node.publish_state(mean_pos / STROKE, stalled)

            step += 1
    except KeyboardInterrupt:
        pass
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
