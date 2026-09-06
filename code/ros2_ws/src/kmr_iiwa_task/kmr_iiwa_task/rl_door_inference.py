"""rl_door_inference.py - istrenirana politika kao ROS2 cvor.

Zamjenjuje door_probe + door_open/door_pull_base iz klasicnog pristupa.
Prilaz i hvat ostaju klasicni (§0): ovaj cvor se pokrece TEK kad gripper vec
drzi kvaku.

DVOSLOJNA ARHITEKTURA. Politika NE izlazi momente nego referencu i krutost:

    politika (60 Hz)  ->  delta poza TCP-a + krutost po osi
                          + (vx, vy, omega) za bazu
    OSC / kartezijski
    impedancijski
    regulator (1 kHz) ->  momenti zglobova

Ovaj cvor je gornji sloj. Donji sloj je ili Isaac Labov OSC (kad se vrti nad
simulatorom) ili ros2_control s kartezijskim impedancijskim regulatorom, npr.
ChengTang62/Cartesian-Impedance-Controller-ROS2, odnosno FRI kod stvarnog
iiwa. Bez donjeg sloja ovaj cvor nema komu slati.

REDOSLIJED OPAZANJA MORA BITI IDENTICAN ONOME U door_env_cfg.PolicyCfg.
Trideset sedam brojeva, tim redom:

    tcp_pose      7   pozicija (3) + kvaternion wxyz (4), u okviru base_link
    tcp_velocity  6   linearna (3) + kutna (3), u okviru base_link
    base_velocity 3   (vx, vy, omega) fiktivnih zglobova, u SVIJETU
    wrench        6   sila (3) + moment (3), u LOKALNOM okviru gripper_base
    last_action  15   prethodni izlaz politike, neskaliran

Svaka zamjena mjesta ili okvira daje politiku koja radi u simulaciji a ne na
robotu, i to bez ijedne poruke o gresci.

DVIJE ZAMKE OKO OKVIRA:

1. base_velocity su brzine fiktivnih zglobova base_x/base_y, a te su osi
   SVJETSKE - u treningu se baza giba po svjetskim osima. Odometrija stvarnog
   robota daje twist u okviru baze, pa ga treba rotirati u svijet PRIJE nego
   ude u opazanje. Isto vrijedi obrnuto za izlaz: politika daje svjetski
   (vx, vy), a /cmd_vel ocekuje brzine u okviru baze.

2. wrench je u lokalnom okviru linka gripper_base, ne u svijetu i ne u bazi.
   tcp_wrench_estimator publisha u okviru koji treba provjeriti i po potrebi
   transformirati.

SKALIRANJE. Politika izlazi neskalirane brojeve; Isaac Lab ih mnozi
konstantama iz ActionsCfg. Iste konstante moraju biti ovdje, inace robot radi
istu stvar u krivom mjerilu.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import rclpy
import tf2_ros
from geometry_msgs.msg import Twist, WrenchStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64MultiArray

# --- Konstante iz door_env_cfg.ActionsCfg. Moraju se poklapati. ---
POSITION_SCALE = 0.015
ORIENTATION_SCALE = 0.1
STIFFNESS_SCALE = 300.0
STIFFNESS_LIMITS = (200.0, 15000.0)
BASE_SCALE = np.array([0.1, 0.1, 0.17])

# decimation=2 uz sim.dt=1/120 daje korak politike od 1/60 s.
POLICY_RATE_HZ = 60.0

OBS_DIM = 37
ACTION_DIM = 15


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Konjugat jedinicnog kvaterniona (w, x, y, z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotiraj vektor kvaternionom (w, x, y, z). Ista formula kao u
    door_mdp.quat_apply - ne mijenjati bez provjere."""
    qw, qv = q[0], q[1:]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


class DoorPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("rl_door_inference")

        self.declare_parameter("policy_path", "")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tcp_frame", "gripper_tcp")
        self.declare_parameter("enabled", False)

        path = self.get_parameter("policy_path").value
        if not path:
            raise RuntimeError("Parametar policy_path nije postavljen.")
        self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name

        self.base_frame = self.get_parameter("base_frame").value
        self.tcp_frame = self.get_parameter("tcp_frame").value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.wrench = np.zeros(6, dtype=np.float32)
        self.base_velocity_world = np.zeros(3, dtype=np.float32)
        self.prev_tcp: tuple[float, np.ndarray, np.ndarray] | None = None
        self.tcp_velocity = np.zeros(6, dtype=np.float32)

        self.create_subscription(
            WrenchStamped, "/estimation/tcp_wrench", self._on_wrench, 10
        )
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)

        # Gornji sloj publisha referencu i krutost; donji sloj (kartezijski
        # impedancijski regulator) iz toga racuna momente.
        self.pub_setpoint = self.create_publisher(
            Float64MultiArray, "/cartesian_impedance/command", 10
        )
        self.pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)

        # Estimator ne modelira gravitaciju, pa je apsolutna procjena sile
        # neupotrebljiva - u mirovanju daje ~46 N tezine ruke. Politika je
        # trenirana na wrenchu koji je u mirovanju nula
        # (get_link_incoming_joint_force nema gravitacijski clan), pa se prije
        # prvog slanja naredbi mora poslati tare.
        self.pub_tare = self.create_publisher(Empty, "/estimation/tare", 10)
        self.tared = False

        self.create_timer(1.0 / POLICY_RATE_HZ, self._step)
        self.get_logger().info(
            f"Politika ucitana ({path}). Cekam enabled=true prije slanja naredbi."
        )

    # --- ulazi -----------------------------------------------------------

    def _on_wrench(self, msg: WrenchStamped) -> None:
        """Sila i moment na zapescu, rotirani u LOKALNI okvir gripper_base.

        tcp_wrench_estimator publisha u base_link (header.frame_id), a
        opazanje ocekuje lokalni okvir linka - u simulaciji je to izravno
        rezultat get_link_incoming_joint_force, koji je uvijek lokalan.
        Bez ove rotacije politika dobiva silu okrenutu drugamo, sto se ne
        prijavljuje kao greska nego samo kao cudno ponasanje.
        """
        f, t = msg.wrench.force, msg.wrench.torque
        wrench_base = np.array([f.x, f.y, f.z, t.x, t.y, t.z], dtype=np.float32)

        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, "gripper_base", rclpy.time.Time()
            )
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(
                f"nema transformacije za wrench: {exc}", throttle_duration_sec=2.0
            )
            return
        r = tf.transform.rotation
        q_inv = quat_inverse(np.array([r.w, r.x, r.y, r.z]))

        self.wrench = np.concatenate(
            [
                quat_apply(q_inv, wrench_base[:3]),
                quat_apply(q_inv, wrench_base[3:]),
            ]
        ).astype(np.float32)

    def _on_joint_states(self, msg: JointState) -> None:
        """Brzina baze. U simulaciji su to fiktivni zglobovi base_x/base_y/
        base_theta; na stvarnom robotu se uzima iz odometrije i rotira iz
        okvira baze u svijet (vidi zamku 1 u docstringu)."""
        try:
            ix = msg.name.index("base_x_joint")
            iy = msg.name.index("base_y_joint")
            it = msg.name.index("base_theta_joint")
        except ValueError:
            return
        self.base_velocity_world = np.array(
            [msg.velocity[ix], msg.velocity[iy], msg.velocity[it]],
            dtype=np.float32,
        )

    def _read_tcp(self) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.tcp_frame, rclpy.time.Time()
            )
        except tf2_ros.TransformException:
            return None
        t, r = tf.transform.translation, tf.transform.rotation
        return (
            np.array([t.x, t.y, t.z], dtype=np.float32),
            np.array([r.w, r.x, r.y, r.z], dtype=np.float32),
        )

    # --- petlja ----------------------------------------------------------

    def _step(self) -> None:
        tcp = self._read_tcp()
        if tcp is None:
            return
        position, orientation = tcp

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_tcp is not None:
            dt = now - self.prev_tcp[0]
            if dt > 1e-4:
                linear = (position - self.prev_tcp[1]) / dt
                delta = quat_multiply(orientation, quat_inverse(self.prev_tcp[2]))
                angle = 2.0 * np.arccos(np.clip(delta[0], -1.0, 1.0))
                axis = delta[1:]
                norm = np.linalg.norm(axis)
                angular = (axis / norm * angle / dt) if norm > 1e-6 else np.zeros(3)
                self.tcp_velocity = np.concatenate([linear, angular]).astype(np.float32)
        self.prev_tcp = (now, position, orientation)

        observation = np.concatenate(
            [
                position,
                orientation,
                self.tcp_velocity,
                self.base_velocity_world,
                self.wrench,
                self.last_action,
            ]
        ).astype(np.float32)

        self._tick = getattr(self, "_tick", 0) + 1
        if self._tick % 60 == 0:
            self.get_logger().info(f"obs: {observation.tolist()}")

        if observation.shape[0] != OBS_DIM:
            self.get_logger().error(
                f"Opazanje ima {observation.shape[0]} brojeva umjesto {OBS_DIM}."
            )
            return

        action = self.session.run(None, {self.input_name: observation[None, :]})[0][0]
        self.last_action = action.astype(np.float32)

        if not self.get_parameter("enabled").value:
            return

        self._publish(action, orientation)

    def _publish(self, action: np.ndarray, orientation: np.ndarray) -> None:
        if not self.tared:
            self.pub_tare.publish(Empty())
            self.tared = True
            self.get_logger().info(
                "Tare poslan. Preskacem ovaj ciklus da procjena stigne na nulu."
            )
            return

        # --- ruka: delta poza + krutost, isto skaliranje kao ActionsCfg ---
        delta_position = action[0:3] * POSITION_SCALE
        delta_orientation = action[3:6] * ORIENTATION_SCALE
        stiffness = np.clip(
            action[6:12] * STIFFNESS_SCALE, STIFFNESS_LIMITS[0], STIFFNESS_LIMITS[1]
        )

        message = Float64MultiArray()
        message.data = np.concatenate(
            [delta_position, delta_orientation, stiffness]
        ).tolist()
        self.pub_setpoint.publish(message)

        # --- baza: iz svijeta u okvir baze, jer /cmd_vel je u okviru baze ---
        world_velocity = np.array(
            [action[12] * BASE_SCALE[0], action[13] * BASE_SCALE[1], 0.0]
        )
        yaw = 2.0 * np.arctan2(orientation[3], orientation[0])
        cos_yaw, sin_yaw = np.cos(-yaw), np.sin(-yaw)

        twist = Twist()
        twist.linear.x = float(
            cos_yaw * world_velocity[0] - sin_yaw * world_velocity[1]
        )
        twist.linear.y = float(
            sin_yaw * world_velocity[0] + cos_yaw * world_velocity[1]
        )
        twist.angular.z = float(action[14] * BASE_SCALE[2])
        self.pub_cmd_vel.publish(twist)


def main() -> None:
    rclpy.init()
    node = DoorPolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
