"""
add_door_collision.py - dodaje PANEL VRATA kao kolizijski objekt u MoveIt
planning scenu. Ovo popravlja temeljni problem otkriven analizom - Scene
Objects panel u RViz je bio prazan, sto znaci da RRT/MoveIt CIJELO VRIJEME
nije znao da vrata postoje, pa nije imao poticaj ih izbjegavati. Sva
prijasnja "divlja putanja" ponasanja (STANDOFF, brzina, path constraints,
hill-climbing) rjesavala su simptome dok je pravi uzrok bio ovaj.

Geometrija (provjereno u izvoru, sliding_door.urdf):
  - door_leaf collision box: size 0.04 x 0.85 x 2.0, centriran na lokalni
    (0, 0.425, 1.0) relativno na door_leaf
  - door_tag_center: lokalni (0.021, 0.425, 1.0) relativno na door_leaf
    (iz ranijeg rada u ovom projektu)
  - Znaci: centar panela = door_tag_center + lokalni offset (-0.021, 0, 0)

Koristimo door_tag_center-ovu TF ORIJENTACIJU za kutiju - to je AprilTag
procjena (ima poznati sum), ALI za sigurnosnu zonu to je prihvatljivo:
generozno podstavljamo dimenzije (COLLISION_PADDING_DEPTH_M,
COLLISION_PADDING_WIDTH_HEIGHT_M - razliciti po osi, vidi te konstante) da
pokrijemo i
poziciju i orijentacijsku netocnost. Ovo NIJE precizan grasp cilj (gdje bi
sum bio problem), nego namjerno konzervativna "ne prilazi ovome" zona.

NAPOMENA - ako ikad zatreba PRECIZNIJA verzija: Isaac Sim moguce publisha
TF i za door_leaf direktno (ground truth iz simulacije, ne AprilTag
procjena) - provjeri `ros2 topic echo /tf --once` postoji li takav frame.
To bi bilo bolje za precizne stvari, ali za sigurnosnu zonu ovo je dovoljno.

Preduvjet: move_group.launch.py mora raditi, apriltag_detection.launch.py
mora vidjeti door_tag_center (robot mora gledati prema vratima).

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

# Lokalni offset od door_tag_center do centra panela (izveden usporedbom
# dvije poznate lokalne pozicije, vidi docstring).
TAG_TO_PANEL_CENTER_OFFSET = [-0.021, 0.0, 0.0]

# Koliko dodati na svaku dimenziju kutije - konzervativna sigurnosna margina
# koja pokriva orijentacijski sum AprilTag procjene (izmjereno ~20 stupnjeva
# odstupanja, vidi razgovor) I pozicijski sum. Namjerno velikodusno - svrha
# ovog objekta je sprijeciti divlje putanje blizu vrata, ne precizno opisati
# geometriju (za to bi trebao live vizualni feedback koji trenutno nemamo
# pouzdano na ovoj udaljenosti).
# Padding je RAZLICIT po osi - jedinstven padding je uzrokovao da sam
# GRASP CILJ upadne unutar vlastite sigurnosne zone (grasp trazi ~1.5cm od
# kvake, a jedinstveni padding od 0.15m protezao je kutiju ~7.5cm izvan
# povrsine vrata u SVIM smjerovima, ukljucujuci dubinu gdje moramo precizno
# prici). Sad: tanak padding u dubini (X - okomito na vrata, gdje treba
# preciznost za sam hvat), velikodusan u sirini/visini (Y,Z - gdje ne
# smeta, i tamo najvise pomaze protiv divljih rotacija sto zadire u strane).
COLLISION_PADDING_DEPTH_M = 0.02
COLLISION_PADDING_WIDTH_HEIGHT_M = 0.15

COLLISION_OBJECT_ID = "door_panel"


def quat_to_rpy_deg(q):
    """Kvaternion (x,y,z,w) -> (roll, pitch, yaw) u stupnjevima, standardna
    ZYX Euler konvencija. Samo za citljiv ispis/debug, ne za racunanje."""
    x, y, z, w = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


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
    """Izgradi orijentaciju za kutiju panela vrata koja garantirano stoji
    USPRAVNO (Z uvijek gore, base_link/world konvencija - vrata se fizicki
    ne naginju), koristeci SAMO tag Z-os (van iz povrsine) za vodoravni dio.
    NE koristi punu tag orijentaciju (roll/pitch bi mogli uzrokovati da
    kutija legne na bok, kao sto smo upravo vidjeli)."""
    z_world_up = np.array([0.0, 0.0, 1.0])

    # Tag Z-os (van iz povrsine vrata) - projeciraj na vodoravnu ravninu i
    # normaliziraj (ako je tag blago nakrivljen, ovo cisti tu gresku).
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
    node.get_logger().info(f"door_tag_center SIROVI kvaternion (x,y,z,w)={tag_quat}")
    node.get_logger().info(
        f"door_tag_center RPY (stupnjevi)={quat_to_rpy_deg(tag_quat)}"
    )

    # Izgradi USPRAVNU orijentaciju (Z uvijek gore) umjesto da koristimo
    # sirovu tag orijentaciju direktno - vidi build_vertical_panel_orientation
    # docstring. Offset primijeni duz NJENE X-osi (dubina panela), ne duz
    # sirove tag orijentacije, jer su sad usklade (obje predstavljaju "van
    # iz povrsine, vodoravno").
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
        f"Panel centar={panel_center}, orijentacija={panel_quat}, "
        f"RPY (stupnjevi)={quat_to_rpy_deg(panel_quat)}, "
        f"dimenzije (s paddingom)={padded_dims}"
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
