"""
add_door_collision.py - dodaje panel vrata kao kolizijski objekt u MoveIt
planning scenu, tako da planer izbjegava vrata umjesto da prolazi kroz njih.

Geometrija (iz sliding_door.urdf):
  - door_leaf collision box: 0.04 x 0.85 x 2.0, centriran na lokalni
    (0, 0.425, 1.0) relativno na door_leaf
  - door_tag_center: lokalni (0.021, 0.425, 1.0)
  - dakle centar panela = door_tag_center + lokalni offset (-0.021, 0, 0)

Orijentacija kutije gradi se oko poznate vertikale, a iz tag orijentacije se
uzima samo smjer prema vratima. Tag ima oko 20 stupnjeva odstupanja u nagibu,
sto bi inace polozilo kutiju na bok.

Dimenzije se podstavljaju da pokriju sum procjene, i to razlicito po osi:
tanko u dubini, gdje ruka mora precizno prici kvaki, a velikodusno u sirini i
visini, gdje margina ne smeta a najvise koristi.

Preduvjet: move_group radi, apriltag vidi door_tag_center.

Pokretanje:
    ros2 run kmr_iiwa_task add_door_collision
"""

import math
import threading

import numpy as np
import rclpy
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
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

# Stvarna geometrija panela (provjereno u sliding_door.urdf).
DOOR_PANEL_SIZE = [0.04, 0.85, 2.0]  # X, Y, Z u LOKALNOM frameu vrata

# Offset od door_tag_center do centra panela - vidi docstring.
TAG_TO_PANEL_CENTER_OFFSET = [-0.021, 0.0, 0.0]

# Sigurnosna margina, razlicita po osi. Tanka u dubini jer grasp cilj lezi
# svega par centimetara od povrsine vrata i inace bi upao u vlastitu
# sigurnosnu zonu; velikodusna u sirini i visini gdje ne smeta.
COLLISION_PADDING_DEPTH_M = 0.02
COLLISION_PADDING_WIDTH_HEIGHT_M = 0.15

COLLISION_OBJECT_ID = "door_panel"


def quat_rotate_vector(q, v):
    """Rotiraj vektor v kvaternionom q=(x,y,z,w). Standardna formula."""
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return [rx, ry, rz]


def rotmat_to_quat(r):
    """3x3 rotacijska matrica (lista 3 stupca, svaki [x,y,z]) -> kvaternion
    (x,y,z,w). Standardna Shepperd/trace metoda."""
    m = np.array(r).T  # stupci u retke za standardnu formulu
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return [x, y, z, w]


def build_vertical_panel_orientation(tag_quat):
    """Orijentacija kutije panela koja uvijek stoji uspravno. Iz tag
    orijentacije se uzima samo Z-os (van iz povrsine) za vodoravni dio; puna
    tag orijentacija bi zbog roll/pitch suma polozila kutiju na bok."""
    z_world_up = np.array([0.0, 0.0, 1.0])

    # Tag Z-os projicirana na vodoravnu ravninu - cisti nagib taga.
    tag_z = np.array(quat_rotate_vector(tag_quat, [0.0, 0.0, 1.0]))
    tag_z_horizontal = tag_z - np.dot(tag_z, z_world_up) * z_world_up
    norm = np.linalg.norm(tag_z_horizontal)
    if norm < 1e-6:
        # Degenerirano (tag gleda tocno gore/dolje?) - koristi X kao fallback.
        tag_z_horizontal = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    box_x_axis = tag_z_horizontal / norm  # "dubina" panela (najmanja dimenzija)

    box_z_axis = z_world_up  # "visina" panela - UVIJEK gore
    box_y_axis = np.cross(box_z_axis, box_x_axis)  # "sirina" panela

    return rotmat_to_quat([box_x_axis, box_y_axis, box_z_axis])


def main():
    rclpy.init()
    node = Node("add_door_collision")
    callback_group = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name="base_link",
        end_effector_name="gripper_tcp",
        group_name="iiwa_arm",
        callback_group=callback_group,
    )

    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    node.get_logger().info("Cekam TF base_link -> door_tag_center...")
    transform = None
    while transform is None and rclpy.ok():
        try:
            transform = tf_buffer.lookup_transform(
                "base_link", "door_tag_center", rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            pass

    t = transform.transform.translation
    q = transform.transform.rotation
    tag_pos = [t.x, t.y, t.z]
    tag_quat = [q.x, q.y, q.z, q.w]

    node.get_logger().info(
        f"door_tag_center pozicija=({t.x:.3f}, {t.y:.3f}, {t.z:.3f})"
    )

    # Offset se primjenjuje duz X-osi izgradjene orijentacije (dubina panela),
    # ne duz sirove tag orijentacije.
    panel_quat = build_vertical_panel_orientation(tag_quat)
    depth_axis = quat_rotate_vector(panel_quat, [1.0, 0.0, 0.0])
    panel_center = [
        tag_pos[i] + TAG_TO_PANEL_CENTER_OFFSET[0] * depth_axis[i] for i in range(3)
    ]

    padded_dims = [
        DOOR_PANEL_SIZE[0] + COLLISION_PADDING_DEPTH_M,
        DOOR_PANEL_SIZE[1] + COLLISION_PADDING_WIDTH_HEIGHT_M,
        DOOR_PANEL_SIZE[2] + COLLISION_PADDING_WIDTH_HEIGHT_M,
    ]

    node.get_logger().info(
        f"Panel centar={panel_center}, dimenzije (s paddingom)={padded_dims}"
    )

    moveit2.add_collision_primitive(
        id=COLLISION_OBJECT_ID,
        primitive_type=SolidPrimitive.BOX,
        dimensions=padded_dims,
        position=panel_center,
        quat_xyzw=panel_quat,
    )

    node.get_logger().info(
        f"Kolizijski objekt '{COLLISION_OBJECT_ID}' dodan. Provjeri u RViz "
        "Scene Objects panelu da se pojavio."
    )

    rclpy.shutdown()
    executor_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
