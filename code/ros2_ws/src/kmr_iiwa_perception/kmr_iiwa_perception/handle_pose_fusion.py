"""
handle_pose_fusion.py - spaja 2 nezavisna AprilTag TF-a (handle_tag_a,
handle_tag_b, publishani preko apriltag_ros) u jedinstvenu 6D pozu hvatišta
kvake, u base_link frameu.

ARHITEKTURNA NAMJERA: ovo je JEDINI izlaz cijelog percepcijskog modula (vidi
originalni handoff dokument, §2) - 6D poza hvatišta, ništa više. Klasifikacija
tipa vrata (zakretna/klizna) namjerno NIJE ovdje - to radi kasniji modul iz
sila/momenata tijekom interakcije. Ovaj node ne zna niti treba znati koji je
tip vrata u pitanju.

ZASTO NE NAIVNI PROSJEK dviju poza: pojedinacni mali tag (25mm) na udaljenosti
1-1.5m ima relativno slabu preciznost orijentacije iz vlastite homografije
(kratka bazna linija unutar samog taga). Umjesto toga:
  - pozicija hvatista = polovište (pos_a + pos_b) / 2
  - os kvake (Y) = normaliziran vektor pos_b -> pos_a - efektivna bazna linija
    je duljina cijele kvake (~10x dulja od pojedinog taga), puno robusnija
    procjena orijentacije nego iz jednog taga
  - "gleda prema" os (Z) = prosjek pojedinacnih Z-osi oba taga (obje bi trebale
    gledati priblizno isti smjer, uz sitan sum), re-ortogonaliziran
    Gram-Schmidtom protiv Y-osi
  - X os = Y x Z (desnokretni ortonormalni sustav)

Sucelje:
  Koristi TF (base_link -> handle_tag_a, base_link -> handle_tag_b) -
  ne treba direktnu pretplatu na apriltag_ros topice, tf2 stablo vec
  sadrzi kompletan lanac (kamera ekstrinsika iz URDF-a + apriltag_ros TF).
  PUB  /perception/handle_pose  (geometry_msgs/PoseStamped, frame_id=base_link)

Objavljuje SAMO kad su oba taga trenutno vidljiva u TF-u i njihovi stampovi
medjusobno bliski (max_stamp_skew_sec) -
namjerno konzervativno, bolje nista nego nepouzdana poza. Downstream (RL
policy / MoveIt) treba zadrzati zadnju dobru pozu ako ovaj topic privremeno
prestane objavljivati.

Pokretanje:
    ros2 run kmr_iiwa_perception handle_pose_fusion
    ros2 run kmr_iiwa_perception handle_pose_fusion --ros-args \
        -p tag_a_frame:=handle_tag_a -p tag_b_frame:=handle_tag_b \
        -p base_frame:=base_link -p publish_rate_hz:=10.0
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


def quat_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Standardna kvaternion -> 3x3 rotacijska matrica konverzija."""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    X, Y, Z, W = x * s, y * s, z * s, w * s
    xx, xy, xz = x * X, x * Y, x * Z
    yy, yz, zz = y * Y, y * Z, z * Z
    wx, wy, wz = w * X, w * Y, w * Z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def rotmat_to_quat(R: np.ndarray):
    """3x3 rotacijska matrica -> kvaternion (x,y,z,w). Trace-based metoda."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def transform_to_pos_rotmat(t: TransformStamped):
    tr = t.transform.translation
    q = t.transform.rotation
    pos = np.array([tr.x, tr.y, tr.z])
    R = quat_to_rotmat(q.x, q.y, q.z, q.w)
    return pos, R


class HandlePoseFusionNode(Node):
    def __init__(self):
        super().__init__("handle_pose_fusion")

        self.declare_parameter("tag_a_frame", "handle_tag_a")
        self.declare_parameter("tag_b_frame", "handle_tag_b")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("output_topic", "/perception/handle_pose")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("max_stamp_skew_sec", 1.0)  # koliko A i B smiju biti
                                                            # vremenski razmaknuti MEDJUSOBNO
                                                            # (ne naspram node-ovog sata - vidi
                                                            # napomenu kod _tick)
        self.declare_parameter("min_tag_distance_m", 0.05)  # sanity granice - vidi napomenu
        self.declare_parameter("max_tag_distance_m", 0.40)  # stvarne vrijednosti: 0.13m (revolute),
                                                              # 0.22m (sliding) - potvrdjeno direktno
                                                              # iz geometrije oba URDF-a. 0.40 daje
                                                              # ~0.18m margine iznad veceg od ta dva,
                                                              # dovoljno za AprilTag procjensku gresku
                                                              # (opazeno do ~38% na 25mm tagu), a i dalje
                                                              # hvata stvarno krive parove (npr. tagovi
                                                              # s razlicitih vrata).

        self.tag_a_frame = self.get_parameter("tag_a_frame").value
        self.tag_b_frame = self.get_parameter("tag_b_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.max_stamp_skew = self.get_parameter("max_stamp_skew_sec").value
        self.min_dist = self.get_parameter("min_tag_distance_m").value
        self.max_dist = self.get_parameter("max_tag_distance_m").value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pub = self.create_publisher(
            PoseStamped, self.get_parameter("output_topic").value, 10
        )

        rate = self.get_parameter("publish_rate_hz").value
        self.timer = self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f"handle_pose_fusion: {self.tag_a_frame} + {self.tag_b_frame} -> "
            f"{self.base_frame}, izlaz na {self.get_parameter('output_topic').value}"
        )

    def _lookup_latest(self, frame: str):
        """Vrati TransformStamped (najnoviji dostupan) ili None ako uopce ne postoji
        u bufferu - NE provjeravamo "starost" naspram node-ovog vlastitog sata (nema
        /clock topica u Isaac Simu, pa self.get_clock().now() pada na pravi zidni sat,
        potpuno druga domena od TF stampova - svaka takva usporedba je besmislena).
        Umjesto toga: tf2_ros.Buffer sam interno izbacuje prestare unose (cache_time),
        a dodatnu provjeru "jesu li A i B iz istog trenutka" radimo u _tick usporedbom
        njihovih stampova MEDJUSOBNO, ne naspram vanjskog sata."""
        try:
            return self.tf_buffer.lookup_transform(self.base_frame, frame, Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _tick(self):
        t_a = self._lookup_latest(self.tag_a_frame)
        t_b = self._lookup_latest(self.tag_b_frame)
        if t_a is None or t_b is None:
            missing = []
            if t_a is None:
                missing.append(self.tag_a_frame)
            if t_b is None:
                missing.append(self.tag_b_frame)
            self.get_logger().warn(
                f"Nema TF za: {missing} (base_frame={self.base_frame}) - preskacem ciklus.",
                throttle_duration_sec=2.0,
            )
            return

        # Usporedba stampova A i B MEDJUSOBNO (ne naspram node-ovog sata - vidi
        # napomenu u _lookup_latest) - hvata slucaj kad je jedan tag "zaglavljen"
        # na staroj detekciji dok je drugi svjez.
        stamp_a = Time.from_msg(t_a.header.stamp)
        stamp_b = Time.from_msg(t_b.header.stamp)
        skew = abs((stamp_a - stamp_b).nanoseconds) / 1e9
        if skew > self.max_stamp_skew:
            self.get_logger().warn(
                f"Stampovi {self.tag_a_frame}/{self.tag_b_frame} razmaknuti {skew:.2f}s "
                f"(> {self.max_stamp_skew}s) - preskacem ciklus.",
                throttle_duration_sec=2.0,
            )
            return

        pos_a, R_a = transform_to_pos_rotmat(t_a)
        pos_b, R_b = transform_to_pos_rotmat(t_b)

        axis_vec = pos_b - pos_a
        dist = float(np.linalg.norm(axis_vec))
        if not (self.min_dist <= dist <= self.max_dist):
            self.get_logger().warn(
                f"Razmak tagova ({dist:.3f}m) izvan ocekivanog raspona "
                f"[{self.min_dist},{self.max_dist}]m - preskacem ovaj ciklus."
            )
            return

        midpoint = (pos_a + pos_b) / 2.0
        y_axis = axis_vec / dist

        z_a = R_a[:, 2]
        z_b = R_b[:, 2]
        z_avg = z_a + z_b
        z_avg_norm = np.linalg.norm(z_avg)
        if z_avg_norm < 1e-6:
            self.get_logger().warn("Z-osi oba taga se ponistavaju (suprotne) - preskacem ciklus.")
            return
        z_raw = z_avg / z_avg_norm

        # Gram-Schmidt: makni komponentu z_raw duz y_axis, pa normaliziraj
        z_axis = z_raw - np.dot(z_raw, y_axis) * y_axis
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-6:
            self.get_logger().warn("Degenerirana geometrija (Z paralelan s Y) - preskacem ciklus.")
            return
        z_axis /= z_norm

        x_axis = np.cross(y_axis, z_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])
        qx, qy, qz, qw = rotmat_to_quat(R)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.pose.position.x = float(midpoint[0])
        msg.pose.position.y = float(midpoint[1])
        msg.pose.position.z = float(midpoint[2])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = HandlePoseFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
