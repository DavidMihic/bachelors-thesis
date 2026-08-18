"""
tcp_wrench_estimator.py - procjenjuje vanjsku silu i moment na gripper_tcp
iz izmjerenih momenata zglobova ruke.

Princip: tau = J^T * F, gdje je J geometrijski Jacobian ruke u tocki TCP-a,
tau vektor momenata 7 zglobova, F traženi wrench [fx,fy,fz,tx,ty,tz] u
base_link okviru. Sustav je preodredjen (7 jednadzbi, 6 nepoznanica) pa se
rjesava metodom najmanjih kvadrata; norma reziduala se objavljuje kao
dijagnostika (koliko momenta se NE moze objasniti silom na TCP-u).

Jacobian se racuna iz TF-a. Sve osi zglobova su (0,0,1) u lokalnom okviru
pripadnog linka, pa je os u base_link okviru treci stupac rotacije
base_link->iiwa_link_N:
    z_i = R(base<-iiwa_link_i) * [0,0,1]
    J_v_i = z_i x (p_tcp - p_i)
    J_w_i = z_i

Gravitacija nije modelirana, a sirovi tau je dominantno gravitacijski, pa je
apsolutna vrijednost neupotrebljiva. Zato node podrzava tare: prazna poruka
na /estimation/tare pamti trenutnu procjenu kao nulu, a sve nakon toga je
razlika u odnosu na taj trenutak - upravo ono sto treba za detekciju kontakta
i procjenu krutosti.

Tare vrijedi samo u blizini poze u kojoj je uzet, jer se gravitacijski moment
mijenja s konfiguracijom ruke. Za pomake od nekoliko milimetara oko iste poze
zanemarivo je; za vece pokrete treba ga ponoviti.

Preduvjet: /isaac_joint_states (arm-control graf u USD-u) i TF za
iiwa_link_1..7 + gripper_tcp.

Pokretanje:
    ros2 run kmr_iiwa_perception tcp_wrench_estimator

Provjera:
    ros2 topic echo /estimation/tcp_wrench
    ros2 topic pub /estimation/tare std_msgs/msg/Empty {} -1
"""

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float32
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

ARM_JOINTS = [f"iiwa_joint_{i}" for i in range(1, 8)]
ARM_LINKS = [f"iiwa_link_{i}" for i in range(1, 8)]
BASE_FRAME = "base_link"
TCP_FRAME = "gripper_tcp"


def quat_to_rotmat(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    X, Y, Z = x * s, y * s, z * s
    xx, xy, xz = x * X, x * Y, x * Z
    yy, yz, zz = y * Y, y * Z, z * Z
    wx, wy, wz = w * X, w * Y, w * Z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


class TcpWrenchEstimator(Node):
    def __init__(self):
        super().__init__("tcp_wrench_estimator")

        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("warn_residual_nm", 15.0)

        self._tau = None
        self._offset = np.zeros(6)
        self._last = np.zeros(6)

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)

        self.create_subscription(JointState, "/isaac_joint_states", self._on_states, 10)
        self.create_subscription(Empty, "/estimation/tare", self._on_tare, 10)

        self._pub = self.create_publisher(WrenchStamped, "/estimation/tcp_wrench", 10)
        self._pub_raw = self.create_publisher(
            WrenchStamped, "/estimation/tcp_wrench_raw", 10
        )
        self._pub_res = self.create_publisher(
            Float32, "/estimation/wrench_residual", 10
        )

        rate = self.get_parameter("publish_rate_hz").value
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            "tcp_wrench_estimator pokrenut. Gravitacija NIJE modelirana - "
            "posalji /estimation/tare u zeljenoj pozi prije mjerenja."
        )

    def _on_states(self, msg: JointState):
        if not msg.effort:
            return
        lookup = dict(zip(msg.name, msg.effort))
        try:
            self._tau = np.array([lookup[j] for j in ARM_JOINTS])
        except KeyError:
            pass

    def _on_tare(self, _msg):
        self._offset = self._last.copy()
        self.get_logger().info(f"Tare postavljen: {np.round(self._offset, 2)}")

    def _jacobian(self):
        """Geometrijski Jacobian 6x7 u base_link okviru, ili None ako TF nije
        dostupan."""
        t0 = rclpy.time.Time()
        try:
            tf_tcp = self.tf_buffer.lookup_transform(BASE_FRAME, TCP_FRAME, t0)
            p_tcp = np.array(
                [
                    tf_tcp.transform.translation.x,
                    tf_tcp.transform.translation.y,
                    tf_tcp.transform.translation.z,
                ]
            )
            J = np.zeros((6, 7))
            for i, link in enumerate(ARM_LINKS):
                tf_i = self.tf_buffer.lookup_transform(BASE_FRAME, link, t0)
                p_i = np.array(
                    [
                        tf_i.transform.translation.x,
                        tf_i.transform.translation.y,
                        tf_i.transform.translation.z,
                    ]
                )
                q = tf_i.transform.rotation
                R = quat_to_rotmat(q.x, q.y, q.z, q.w)
                z_i = R[:, 2]  # sve osi su (0,0,1) lokalno
                J[0:3, i] = np.cross(z_i, p_tcp - p_i)
                J[3:6, i] = z_i
            return J
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            self.get_logger().warn(
                f"TF nedostupan ({exc}) - preskacem.", throttle_duration_sec=5.0
            )
            return None

    def _tick(self):
        if self._tau is None:
            return
        J = self._jacobian()
        if J is None:
            return

        # tau = J^T F  ->  F preko najmanjih kvadrata
        F, *_ = np.linalg.lstsq(J.T, self._tau, rcond=None)
        self._last = F

        residual = float(np.linalg.norm(J.T @ F - self._tau))
        self._pub_res.publish(Float32(data=residual))
        if residual > self.get_parameter("warn_residual_nm").value:
            self.get_logger().warn(
                f"Velik rezidual ({residual:.1f} Nm) - dio momenta se ne moze "
                "objasniti silom na TCP-u (nullspace, model gravitacije, sum).",
                throttle_duration_sec=5.0,
            )

        stamp = self.get_clock().now().to_msg()
        for pub, vec in ((self._pub_raw, F), (self._pub, F - self._offset)):
            msg = WrenchStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = BASE_FRAME
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = map(
                float, vec[0:3]
            )
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = map(
                float, vec[3:6]
            )
            pub.publish(msg)


def main():
    rclpy.init()
    node = TcpWrenchEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
