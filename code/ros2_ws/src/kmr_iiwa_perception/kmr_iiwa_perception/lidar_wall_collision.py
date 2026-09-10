"""
lidar_wall_collision.py - odrzava MoveIt kolizijske objekte za zidove UZIVO
iz /scan (sensor_msgs/LaserScan), NEOVISNO o AprilTagu na vratima.

Zasto ovako, a ne kao door_panel (tag + poznati offset): door_panel radi jer
tocno znamo geometriju krila iz URDF-a. Zidovi bi TAKODJER mogli ici preko
poznatog offseta od door_tag_center, ali to znaci da robot "zna" gdje su
zidovi SAMO ako vidi bas taj tag i SAMO dok su vrata zatvorena - suprotno
svrsi lidara, koji treba dati percepciju prepreka neovisnu o tagovima.

Umjesto toga:
  1. cita /scan iz lidar_link,
  2. baca van vlastite pogotke (fiksna maska ±94°, vidi
     SELF_OCCLUSION_HALF_ANGLE_DEG),
  3. klasterira preostale tocke u kontinuirane segmente (prekid = rupa,
     npr. sam otvor vrata),
  4. za svaki dovoljno dug/gust klaster fita pravac (PCA) i doda tanku
     OKOMITU kutiju (ispod/iznad ravnine skena) u MoveIt scenu kao "zid".

VAZNA PRETPOSTAVKA: lidar je JEDAN vodoravan presjek na 8 cm visine. Skena
ne zna nista o stvarnoj visini prepreke - "zid" koji ovdje dodajemo je NASA
pretpostavka da se sve sto se detektira u toj ravnini prostire od poda do
WALL_HEIGHT_M. Ispravno za prave zidove (nas slucaj), NIJE opcenito
ispravno za proizvoljne prepreke (npr. nizak stol bi ispao kao "zid do
stropa").

TAKODJER: maska od ±94° pretpostavlja da su zidovi generalno ISPRED robota
(vrijedi za prilaz vratima). Ne vrijedi opcenito za hodnik s paralelnim
zidovima sa strane (ti bi upali blizu ±90°, na rubu maske) - ako se ovo
prosiruje na opcenitu navigaciju, masku treba revidirati.

Pokretanje:
    ros2 run kmr_iiwa_perception lidar_wall_collision
"""

import math
import threading

import numpy as np
import rclpy
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

JOINT_NAMES = [
    "iiwa_joint_1",
    "iiwa_joint_2",
    "iiwa_joint_3",
    "iiwa_joint_4",
    "iiwa_joint_5",
    "iiwa_joint_6",
    "iiwa_joint_7",
]

# Geometrijski izvod (vidi raniju raspravu o samozaklonu): sa senzora na
# (0.57, 0, 0.08) u base_link, prednji uglovi kucista (kmr_length=1.08,
# kmr_width=0.63) su tangentne tocke sjene pod ±95.44°. Ovdje 94° - malo
# UZE nego geometrijski nuzno, da se pojas oko ruba (gdje snop ima konacnu
# sirinu pa moze okrznuti kuciste) sigurno odbaci.
SELF_OCCLUSION_HALF_ANGLE_DEG = 94.0

MIN_CLUSTER_POINTS = 4
MIN_SEGMENT_LENGTH_M = 0.3
CLUSTER_GAP_M = 0.15  # prekid klastera ako je razmak do sljedece tocke veci

WALL_HEIGHT_M = 2.2  # pretpostavka pod/strop, vidi docstring
WALL_THICKNESS_M = 0.10
UPDATE_PERIOD_S = 0.5  # ne azuriraj scenu brze od ovoga (zidovi ne biju)

LIDAR_FRAME = "lidar_link"
BASE_FRAME = "base_link"
COLLISION_ID_PREFIX = "lidar_wall_"


def quat_rotate_vector(q, v):
    """Rotiraj vektor v kvaternionom q=(x,y,z,w)."""
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return [rx, ry, rz]


def _wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _cluster_points(points, gap_m):
    if not points:
        return []
    clusters = [[points[0]]]
    for p in points[1:]:
        prev = clusters[-1][-1]
        if math.hypot(p[0] - prev[0], p[1] - prev[1]) > gap_m:
            clusters.append([p])
        else:
            clusters[-1].append(p)
    return clusters


def _fit_segment(cluster):
    """PCA pravac kroz klaster. Vraca ((cx,cy), yaw, duljina)."""
    pts = np.array(cluster)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]
    yaw = math.atan2(direction[1], direction[0])

    projections = centered @ direction
    length = float(projections.max() - projections.min())
    mid_offset = direction * ((projections.max() + projections.min()) / 2.0)
    center = centroid + mid_offset
    return (center[0], center[1]), yaw, length


class LidarWallCollision(Node):
    def __init__(self):
        super().__init__("lidar_wall_collision")

        callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=JOINT_NAMES,
            base_link_name="base_link",
            end_effector_name="gripper_tcp",
            group_name="iiwa_arm",
            callback_group=callback_group,
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        # (translacija[3], kvaternion_xyzw[4]) - fiksan lanac, cache-iramo jednom.
        self._lidar_to_base = None

        self._active_ids = set()
        self._last_update = self.get_clock().now()

        self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
        self.get_logger().info("Cekam TF lidar_link -> base_link...")

    def _ensure_static_tf(self):
        if self._lidar_to_base is not None:
            return True
        try:
            tf = self._tf_buffer.lookup_transform(
                BASE_FRAME, LIDAR_FRAME, rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return False
        t = tf.transform.translation
        q = tf.transform.rotation
        self._lidar_to_base = ([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
        self.get_logger().info(
            f"lidar_link -> base_link uhvacen: t=({t.x:.3f},{t.y:.3f},{t.z:.3f})"
        )
        return True

    def _on_scan(self, msg: LaserScan):
        if not self._ensure_static_tf():
            return

        now = self.get_clock().now()
        if (now - self._last_update).nanoseconds < UPDATE_PERIOD_S * 1e9:
            return
        self._last_update = now

        translation, quat = self._lidar_to_base
        mask_rad = math.radians(SELF_OCCLUSION_HALF_ANGLE_DEG)

        points_base = []
        angle = msg.angle_min
        for r in msg.ranges:
            a = angle
            angle += msg.angle_increment
            if not math.isfinite(r) or r < msg.range_min or r > msg.range_max:
                continue
            if abs(_wrap_pi(a)) > mask_rad:
                continue
            x_l = r * math.cos(a)
            y_l = r * math.sin(a)
            x_b, y_b, _ = quat_rotate_vector(quat, [x_l, y_l, 0.0])
            points_base.append((translation[0] + x_b, translation[1] + y_b))

        clusters = _cluster_points(points_base, CLUSTER_GAP_M)

        new_ids = set()
        for i, cluster in enumerate(clusters):
            if len(cluster) < MIN_CLUSTER_POINTS:
                continue
            (cx, cy), yaw, length = _fit_segment(cluster)
            if length < MIN_SEGMENT_LENGTH_M:
                continue

            obj_id = f"{COLLISION_ID_PREFIX}{i}"
            new_ids.add(obj_id)
            quat_z = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
            self.moveit2.add_collision_primitive(
                id=obj_id,
                primitive_type=SolidPrimitive.BOX,
                dimensions=[length + 0.05, WALL_THICKNESS_M, WALL_HEIGHT_M],
                position=[cx, cy, WALL_HEIGHT_M / 2.0],
                quat_xyzw=quat_z,
            )

        stale = self._active_ids - new_ids
        for obj_id in stale:
            self.moveit2.remove_collision_object(obj_id)
        self._active_ids = new_ids


def main():
    rclpy.init()
    node = LidarWallCollision()
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
