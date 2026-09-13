"""handle_gt_publisher.py — objavljuje STVARNU pozu hvatišta iz simulatora.

Mora se izvoditi UNUTAR Isaac Sim procesa (isaaclab.sh -p), jer čita poze prima
izravno sa scene. Ne pokreće se preko `ros2 run`.

Uporaba: importaj u cmd_vel_bridge.py i pozovi u petlji, npr.

    from handle_gt_publisher import HandleGroundTruth
    gt = HandleGroundTruth(
        tag_a_path="/World/Door/handle_tag_a",
        tag_b_path="/World/Door/handle_tag_b",
        base_path="/World/Robot/base_link",
    )
    ...
    while simulation_app.is_running():
        world.step(render=True)
        gt.publish()          # <-- jedini dodatak u postojeću petlju

Izlaz: /ground_truth/handle_pose (geometry_msgs/PoseStamped, okvir base_link),
konstruirana ISTIM postupkom kao handle_pose_fusion, da usporedba mjeri
percepciju, a ne razliku u definiciji hvatišta.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

# Isaac Sim 5.x je preimenovao namespace. Provjeri koji od ova dva radi kod tebe.
try:
    from isaacsim.core.prims import XFormPrim
except ImportError:  # starije inačice
    from omni.isaac.core.prims import XFormPrim


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


class HandleGroundTruth:
    """Čita poze oznaka i platforme sa scene i objavljuje pozu hvatišta."""

    def __init__(
        self,
        tag_a_path: str,
        tag_b_path: str,
        base_path: str,
        topic: str = "/ground_truth/handle_pose",
        base_frame: str = "base_link",
        normal_axis: int = 2,  # 2 = lokalna os z oznake je njezina normala
        publish_every: int = 4,  # objavi svaki n-ti korak simulacije
    ) -> None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        for label, path in (
            ("tag_a", tag_a_path),
            ("tag_b", tag_b_path),
            ("platforma", base_path),
        ):
            if stage is None or not stage.GetPrimAtPath(path).IsValid():
                raise ValueError(
                    f"handle_gt_publisher: prim za '{label}' ne postoji na "
                    f"'{path}'. Provjeri putanje u Stage panelu Isaac Sima."
                )

        if not rclpy.ok():
            rclpy.init()
        self._node = Node("handle_gt_publisher")
        self._pub = self._node.create_publisher(PoseStamped, topic, 10)
        self._tag_a = XFormPrim(tag_a_path)
        self._tag_b = XFormPrim(tag_b_path)
        self._base = XFormPrim(base_path)
        self._base_frame = base_frame
        self._axis = normal_axis
        self._every = max(1, publish_every)
        self._i = 0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _world_pose(prim: XFormPrim) -> tuple[np.ndarray, np.ndarray]:
        p, q = prim.get_world_poses()
        return np.asarray(p[0], dtype=float), np.asarray(q[0], dtype=float)

    def _handle_frame_world(self) -> tuple[np.ndarray, np.ndarray]:
        """Konstrukcija identična onoj u handle_pose_fusion."""
        p_a, q_a = self._world_pose(self._tag_a)
        p_b, q_b = self._world_pose(self._tag_b)

        z_a = _quat_to_R(q_a)[:, self._axis]
        z_b = _quat_to_R(q_b)[:, self._axis]

        p = 0.5 * (p_a + p_b)

        e_y = p_b - p_a
        e_y /= np.linalg.norm(e_y)

        z_bar = 0.5 * (z_a + z_b)
        e_z = z_bar - np.dot(z_bar, e_y) * e_y
        e_z /= np.linalg.norm(e_z)

        e_x = np.cross(e_y, e_z)
        return p, np.column_stack((e_x, e_y, e_z))

    # ------------------------------------------------------------------ #
    def publish(self) -> None:
        self._i += 1
        if self._i % self._every:
            return

        p_w, R_w = self._handle_frame_world()
        p_base, q_base = self._world_pose(self._base)
        R_base = _quat_to_R(q_base)

        # svijet -> platforma
        p_b = R_base.T @ (p_w - p_base)
        R_b = R_base.T @ R_w
        q_b = _R_to_quat(R_b)

        msg = PoseStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = p_b
        msg.pose.orientation.w = float(q_b[0])
        msg.pose.orientation.x = float(q_b[1])
        msg.pose.orientation.y = float(q_b[2])
        msg.pose.orientation.z = float(q_b[3])
        self._pub.publish(msg)

        # NAPOMENA: ovdje se NE poziva rclpy.spin_once. cmd_vel_bridge vec vrti
        # vlastiti executor u pozadinskoj niti; drugi spin iz glavne niti nad
        # istim kontekstom rusi proces s "IndexError: wait set index too big".
