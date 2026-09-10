"""door_gt_publisher.py - objavljuje STVARNO stanje vrata i baze iz simulatora.

Mora se izvoditi UNUTAR Isaac Sim procesa (isaaclab.sh -p), jer cita poze prima
izravno sa scene. Ne pokrece se preko `ros2 run`.

Tip vrata se prepoznaje SAM, po tome koji prim postoji na stageu:
    <door_root>/slide_frame  -> klizna   (prismatic, slide_joint, metri)
    <door_root>/door_frame   -> zakretna (revolute, hinge_joint, radijani)
Time ista linija u cmd_vel_bridgeu radi za obje scene, bez rucnog mijenjanja.

Uporaba: importaj u cmd_vel_bridge.py i pozovi u petlji.

    from door_gt_publisher import DoorGroundTruth
    gt_door = DoorGroundTruth()          # sve zadano, prepoznaje samo
    ...
    while simulation_app.is_running():
        world.step(render=True)
        gt_door.publish()

Izlazi:
  /ground_truth/door_joint  (sensor_msgs/JointState)
      Otvorenost vrata: radijani za zakretna, metri za klizna. Racuna se iz
      poza prima, ne iz articulation API-ja, pa radi jednako za oba tipa bez
      poznavanja strukture artikulacije.

  /ground_truth/hinge_pose  (geometry_msgs/PoseStamped, okvir base_link)
      Poza ishodista krila (kod zakretnih vrata os sarke).

  /ground_truth/base_pose   (geometry_msgs/PoseStamped, okvir world)
      Stvarna poza platforme u svijetu.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

import omni.usd

# Isaac Sim 5.x je preimenovao namespace. Provjeri koji od ova dva radi kod tebe.
try:
    from isaacsim.core.prims import XFormPrim
except ImportError:  # starije inacice
    from omni.isaac.core.prims import XFormPrim


# (naziv okvira, tip zgloba, naziv zgloba, lokalna os klizanja)
DOOR_VARIANTS = [
    ("slide_frame", "prismatic", "slide_joint", 1),
    ("door_frame", "revolute", "hinge_joint", None),
]


def _quat_to_R(q_wxyz: np.ndarray) -> np.ndarray:
    """Kvaternion (w, x, y, z) -> matrica rotacije 3x3."""
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    """Matrica rotacije 3x3 -> kvaternion (w, x, y, z), Shepperdova metoda."""
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array(
            [
                0.25 * s,
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
            ]
        )
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2.0
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def _yaw_from_R(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class DoorGroundTruth:
    """Cita poze krila, okvira vrata i platforme sa scene i objavljuje ih."""

    def __init__(
        self,
        door_root: str = "/World/Door",
        leaf_name: str = "door_leaf",
        base_path: str = "/World/Robot/base_link",
        joint_topic: str = "/ground_truth/door_joint",
        hinge_topic: str = "/ground_truth/hinge_pose",
        base_topic: str = "/ground_truth/base_pose",
        base_frame: str = "base_link",
        world_frame: str = "world",
        publish_every: int = 4,
    ) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(
                "door_gt_publisher: nema aktivnog USD stagea. Konstruiraj "
                "DoorGroundTruth tek nakon sto je scena ucitana."
            )

        # --- Prepoznavanje tipa vrata po tome koji okvir postoji na stageu ---
        self._frame_path = None
        for frame_name, joint_type, joint_name, slide_axis in DOOR_VARIANTS:
            candidate = f"{door_root}/{frame_name}"
            if stage.GetPrimAtPath(candidate).IsValid():
                self._frame_path = candidate
                self._joint_type = joint_type
                self._joint_name = joint_name
                self._slide_axis = slide_axis
                break
        if self._frame_path is None:
            trazeno = ", ".join(f"{door_root}/{v[0]}" for v in DOOR_VARIANTS)
            raise ValueError(
                f"door_gt_publisher: ne nalazim okvir vrata. Trazeno: {trazeno}. "
                "Provjeri putanje u Stage panelu Isaac Sima."
            )

        leaf_path = f"{door_root}/{leaf_name}"
        for label, path in (("krilo", leaf_path), ("platforma", base_path)):
            if not stage.GetPrimAtPath(path).IsValid():
                raise ValueError(
                    f"door_gt_publisher: prim za '{label}' ne postoji na '{path}'."
                )

        if not rclpy.ok():
            rclpy.init()
        self._node = Node("door_gt_publisher")
        self._joint_pub = self._node.create_publisher(JointState, joint_topic, 10)
        self._hinge_pub = self._node.create_publisher(PoseStamped, hinge_topic, 10)
        self._base_pub = self._node.create_publisher(PoseStamped, base_topic, 10)
        self._leaf = XFormPrim(leaf_path)
        self._frame = XFormPrim(self._frame_path)
        self._base = XFormPrim(base_path)
        self._base_frame = base_frame
        self._world_frame = world_frame
        self._every = max(1, publish_every)
        self._i = 0
        # Referentna vrijednost pri prvom pozivu: vrata se spawnaju zatvorena,
        # pa je sve nakon toga mjereno u odnosu na to.
        self._ref: float | None = None

        jedinica = "m" if self._joint_type == "prismatic" else "rad"
        tip = "klizna" if self._joint_type == "prismatic" else "zakretna"
        print(
            f"[door_gt_publisher] prepoznata {tip} vrata: {self._frame_path}, "
            f"zglob '{self._joint_name}' [{jedinica}]"
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _world_pose(prim: XFormPrim) -> tuple[np.ndarray, np.ndarray]:
        p, q = prim.get_world_poses()
        return np.asarray(p[0], dtype=float), np.asarray(q[0], dtype=float)

    def _joint_value(self, p_leaf, q_leaf, p_frame, q_frame) -> float:
        """Otvorenost: kut (rad) za zakretna, pomak (m) za klizna."""
        if self._joint_type == "revolute":
            return _wrap_pi(
                _yaw_from_R(_quat_to_R(q_leaf)) - _yaw_from_R(_quat_to_R(q_frame))
            )
        # prismatic: komponenta pomaka krila duz osi klizanja, izrazene u
        # okviru dovratnika - vrijedi i ako je cijela scena zakrenuta.
        R_frame = _quat_to_R(q_frame)
        return float((R_frame.T @ (p_leaf - p_frame))[self._slide_axis])

    # ------------------------------------------------------------------ #
    def publish(self) -> None:
        self._i += 1
        if self._i % self._every:
            return

        p_leaf, q_leaf = self._world_pose(self._leaf)
        p_frame, q_frame = self._world_pose(self._frame)
        p_base, q_base = self._world_pose(self._base)

        raw = self._joint_value(p_leaf, q_leaf, p_frame, q_frame)
        if self._ref is None:
            self._ref = raw
        value = raw - self._ref
        if self._joint_type == "revolute":
            value = _wrap_pi(value)

        js = JointState()
        js.header.stamp = self._node.get_clock().now().to_msg()
        js.name = [self._joint_name]
        js.position = [float(value)]
        self._joint_pub.publish(js)

        R_base = _quat_to_R(q_base)
        p_b = R_base.T @ (p_leaf - p_base)
        q_b = _R_to_quat(R_base.T @ _quat_to_R(q_leaf))

        msg = PoseStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = p_b
        msg.pose.orientation.w = float(q_b[0])
        msg.pose.orientation.x = float(q_b[1])
        msg.pose.orientation.y = float(q_b[2])
        msg.pose.orientation.z = float(q_b[3])
        self._hinge_pub.publish(msg)

        bmsg = PoseStamped()
        bmsg.header.stamp = self._node.get_clock().now().to_msg()
        bmsg.header.frame_id = self._world_frame
        bmsg.pose.position.x, bmsg.pose.position.y, bmsg.pose.position.z = p_base
        bmsg.pose.orientation.w = float(q_base[0])
        bmsg.pose.orientation.x = float(q_base[1])
        bmsg.pose.orientation.y = float(q_base[2])
        bmsg.pose.orientation.z = float(q_base[3])
        self._base_pub.publish(bmsg)

        # NAPOMENA: ovdje se NE poziva rclpy.spin_once. cmd_vel_bridge vec vrti
        # vlastiti executor u pozadinskoj niti; drugi spin iz glavne niti nad
        # istim kontekstom rusi proces s "IndexError: wait set index too big".
