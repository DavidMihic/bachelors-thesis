"""
gripper_bridge.py - standalone gripper driver, NE pokrece vlastiti Isaac Sim.
Umjesto toga koristi VEC POSTOJECI /isaac_joint_commands i /isaac_joint_states
par (isti kojeg koristi ros2_control preko topic_based_ros2_control, vidi
add_arm_ros_control_graph.py) - radi unutar DIJELJENE simulacije koju vec
hostira sto god trenutno pokrece world.step() (npr. cmd_vel_bridge.py),
umjesto da otvara svoj vlastiti, odvojeni Isaac Sim prozor.

Zamjenjuje stari gripper_driver.py za slucajeve kad vec postoji zajednicka
simulacija - stari file ostaje za samostalno pokretanje/testiranje ako
zatreba, arhitektura i sucelje su namjerno identicni.

Ista stall-detekcijska logika kao stari gripper_driver.py (isti brojevi/
pragovi - referentni izvor). Jedina stvarna razlika: stari throttling
"svaki N fizickih koraka" (jer je taj node sam kontrolirao world.step()
petlju) postaje ovdje "svaki N milisekundi" wall-clock timer, jer ovaj node
vise ne kontrolira fizicku petlju direktno, samo se pretplacuje na tudja
stanja preko topica.

Sucelje (identicno starom gripper_driver.py):
  SUB  /gripper_cmd      (std_msgs/Float32)  0.0=otvori, 1.0=zatvori
  PUB  /gripper_state    (std_msgs/Float32)  0.0..1.0, stvarna trenutna pozicija
  PUB  /gripper_stalled  (std_msgs/Bool)     True = gura prema targetu ali se
                                              vise ne mice i nije stigao ->
                                              vjerojatno je nesto uhvatio
  PUB  /joint_states     (sensor_msgs/JointState) - trenutne pozicije 4 prsta,
                                              DODATNO uz gornje - inace MoveIt-ov
                                              planning_scene_monitor nikad ne
                                              vidi gripper zglobove (samo
                                              joint_state_broadcaster/ruku
                                              publisha na taj topic). Vise
                                              publishera na isti topic je OK -
                                              current_state_monitor akumulira.

Interno (nova stvar naspram starog gripper_driver.py):
  PUB  /isaac_joint_commands (sensor_msgs/JointState) - pozicijski cilj za
       4 prsta, ide u DIJELJENU simulaciju preko postojeceg arm-control grafa
  SUB  /isaac_joint_states (sensor_msgs/JointState) - trenutno stanje CIJELE
       artikulacije (isti graf publisha za sve, ukljucujuci ruku) - filtriramo
       samo 4 prsta po imenu

Pokretanje (Isaac Sim + arm-control graf VEC MORAJU raditi - npr. preko
cmd_vel_bridge.py na USD-u koji ima add_arm_ros_control_graph.py bakiran):
    ros2 run kmr_iiwa_sim_bridge gripper_bridge

Rucni test u drugom terminalu (identicno starom gripper_driver.py):
    ros2 topic pub /gripper_cmd std_msgs/msg/Float32 "{data: 1.0}" -1
    ros2 topic echo /gripper_state
    ros2 topic echo /gripper_stalled
"""

import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32

FINGER_JOINTS = [f"gripper_finger_{i}_joint" for i in range(1, 5)]
STROKE = 0.025  # m; 0.0=otvoren, STROKE=zatvoren (isto kao URDF joint limit)


class GripperBridgeNode(Node):
    def __init__(self):
        super().__init__("kmr_gripper_bridge")

        self.declare_parameter("gripper_cmd_topic", "/gripper_cmd")
        self.declare_parameter("gripper_state_topic", "/gripper_state")
        self.declare_parameter("gripper_stalled_topic", "/gripper_stalled")
        self.declare_parameter("isaac_commands_topic", "/isaac_joint_commands")
        self.declare_parameter("isaac_states_topic", "/isaac_joint_states")
        self.declare_parameter("check_period_sec", 0.1)
        self.declare_parameter("stall_velocity_threshold", 0.0003)
        self.declare_parameter("stall_position_threshold", 0.0015)

        self._lock = threading.Lock()
        self._closing_fraction = 0.0  # pocetno stanje: potpuno otvoren
        self._current_positions = None  # zadnje poznato stanje 4 prsta
        self._prev_mean_pos = None

        self.create_subscription(
            Float32, self.get_parameter("gripper_cmd_topic").value, self._on_cmd, 10
        )
        self.create_subscription(
            JointState,
            self.get_parameter("isaac_states_topic").value,
            self._on_isaac_state,
            10,
        )
        self._isaac_cmd_pub = self.create_publisher(
            JointState, self.get_parameter("isaac_commands_topic").value, 10
        )
        self._state_pub = self.create_publisher(
            Float32, self.get_parameter("gripper_state_topic").value, 10
        )
        self._stalled_pub = self.create_publisher(
            Bool, self.get_parameter("gripper_stalled_topic").value, 10
        )
        # Dodatan publisher na standardni /joint_states - inace MoveIt-ov
        # planning_scene_monitor nikad ne vidi gripper zglobove (samo
        # joint_state_broadcaster/arm ih publisha, gripper prsti idu mimo
        # ros2_control-a preko /isaac_joint_commands). Vise publishera na
        # isti topic je uobicajeno - current_state_monitor akumulira
        # stanje iz odvojenih poruka, ne treba jedna monolitna poruka.
        self._joint_states_pub = self.create_publisher(JointState, "/joint_states", 10)

        self.stall_vel_thresh = self.get_parameter("stall_velocity_threshold").value
        self.stall_pos_thresh = self.get_parameter("stall_position_threshold").value

        period = self.get_parameter("check_period_sec").value
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            "kmr_gripper_bridge pokrenut - koristi dijeljenu simulaciju preko "
            f"{self.get_parameter('isaac_commands_topic').value}/"
            f"{self.get_parameter('isaac_states_topic').value}, ne otvara vlastiti Isaac Sim."
        )

    def _on_cmd(self, msg: Float32):
        with self._lock:
            self._closing_fraction = float(np.clip(msg.data, 0.0, 1.0))

    def _on_isaac_state(self, msg: JointState):
        # Izvuci samo 4 prsta iz stanja CIJELE artikulacije (poruka sadrzi i
        # ruku - vidi napomenu u add_arm_ros_control_graph.py).
        positions = {}
        for name, pos in zip(msg.name, msg.position):
            if name in FINGER_JOINTS:
                positions[name] = pos
        if len(positions) == len(FINGER_JOINTS):
            with self._lock:
                self._current_positions = [positions[n] for n in FINGER_JOINTS]

    def _tick(self):
        with self._lock:
            closing_fraction = self._closing_fraction
            current_positions = self._current_positions

        target_pos = closing_fraction * STROKE

        # Salji komandu u dijeljenu simulaciju svaki tick - jednostavno i
        # dovoljno (ArticulationController drzi zadnju primljenu vrijednost
        # izmedju poruka, isti princip kao cmd_vel_bridge.py).
        cmd_msg = JointState()
        cmd_msg.name = FINGER_JOINTS
        cmd_msg.position = [target_pos] * len(FINGER_JOINTS)
        self._isaac_cmd_pub.publish(cmd_msg)

        if current_positions is None:
            return  # jos nismo culi /isaac_joint_states, nema sto provjeriti

        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = FINGER_JOINTS
        joint_state_msg.position = list(current_positions)
        self._joint_states_pub.publish(joint_state_msg)

        mean_pos = float(np.mean(current_positions))

        stalled = False
        if self._prev_mean_pos is not None:
            velocity = abs(mean_pos - self._prev_mean_pos)
            pos_error = abs(target_pos - mean_pos)
            stalled = (
                velocity < self.stall_vel_thresh and pos_error > self.stall_pos_thresh
            )
        self._prev_mean_pos = mean_pos

        self._state_pub.publish(Float32(data=mean_pos / STROKE))
        self._stalled_pub.publish(Bool(data=stalled))


def main():
    rclpy.init()
    node = GripperBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
