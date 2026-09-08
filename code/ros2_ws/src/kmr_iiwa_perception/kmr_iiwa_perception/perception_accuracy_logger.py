"""perception_accuracy_logger.py — snima jednu mjernu točku u CSV.

Pokreće se sa strane ROS-a 2 (Python 3.10), dok Isaac Sim i cijeli stog rade.
Robot mora MIROVATI tijekom snimanja, jer se procjena i stvarna vrijednost ne
sinkroniziraju po vremenu, nego se usredni po intervalu.

Postupak jednog mjerenja:
    1. dovezi platformu u novi položaj (door_task_node ili ručno /cmd_vel)
    2. pričekaj da se zaustavi
    3. ros2 run kmr_iiwa_perception perception_accuracy_logger \
           --ros-args -p label:=p01 -p door:=revolute

Ponovi za 20-ak različitih položaja, po oba tipa vrata. Rezultat se dopisuje u
perception_accuracy.csv.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


def _quat_to_R(q_wxyz) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


class AccuracyLogger(Node):

    FIELDS = [
        "label", "door", "n_est", "n_gt", "distance_m",
        "err_norm_mm", "err_along_mm", "err_perp_mm",
        "err_x_mm", "err_y_mm", "err_z_mm",
        "axis_err_deg", "rot_err_deg",
        "std_est_pos_mm", "std_est_axis_deg",
    ]

    def __init__(self) -> None:
        super().__init__("perception_accuracy_logger")
        self.declare_parameter("label", "p00")
        self.declare_parameter("door", "revolute")
        self.declare_parameter("duration", 5.0)
        self.declare_parameter("csv", "perception_accuracy.csv")
        self.declare_parameter("est_topic", "/perception/handle_pose")
        self.declare_parameter("gt_topic", "/ground_truth/handle_pose")

        self._est: list[tuple[np.ndarray, np.ndarray]] = []
        self._gt: list[tuple[np.ndarray, np.ndarray]] = []

        self.create_subscription(
            PoseStamped, self.get_parameter("est_topic").value,
            lambda m: self._est.append(self._unpack(m)), 20)
        self.create_subscription(
            PoseStamped, self.get_parameter("gt_topic").value,
            lambda m: self._gt.append(self._unpack(m)), 20)

        dur = float(self.get_parameter("duration").value)
        self.get_logger().info(f"snimam {dur:.1f} s — robot mora mirovati")
        self.create_timer(dur, self._finish)

    @staticmethod
    def _unpack(m: PoseStamped) -> tuple[np.ndarray, np.ndarray]:
        p = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])
        q = np.array([m.pose.orientation.w, m.pose.orientation.x,
                      m.pose.orientation.y, m.pose.orientation.z])
        return p, q

    # ------------------------------------------------------------------ #
    @staticmethod
    def _mean_pose(samples) -> tuple[np.ndarray, np.ndarray]:
        """Srednji položaj i srednja rotacija (glavni svojstveni vektor kvaterniona)."""
        P = np.array([s[0] for s in samples])
        Q = np.array([s[1] for s in samples])
        Q[np.einsum("ij,j->i", Q, Q[0]) < 0] *= -1.0     # ukloni dvoznačnost predznaka
        _, vecs = np.linalg.eigh(Q.T @ Q)
        return P.mean(axis=0), vecs[:, -1]

    def _finish(self) -> None:
        if len(self._est) < 5 or len(self._gt) < 5:
            self.get_logger().error(
                f"premalo uzoraka (procjena {len(self._est)}, stvarno {len(self._gt)}) "
                "— rade li oba čvora?")
            raise SystemExit(1)

        p_e, q_e = self._mean_pose(self._est)
        p_g, q_g = self._mean_pose(self._gt)
        R_e, R_g = _quat_to_R(q_e), _quat_to_R(q_g)

        err = p_e - p_g
        axis = R_g[:, 1]                                  # stvarni smjer duž kvake
        along = float(np.dot(err, axis))
        perp = float(np.linalg.norm(err - along * axis))

        axis_err = float(np.degrees(np.arccos(
            np.clip(abs(np.dot(R_e[:, 1], axis)), -1.0, 1.0))))
        rot_err = float(np.degrees(np.arccos(
            np.clip((np.trace(R_g.T @ R_e) - 1.0) / 2.0, -1.0, 1.0))))

        P = np.array([s[0] for s in self._est])
        A = np.array([_quat_to_R(s[1])[:, 1] for s in self._est])
        std_pos = float(np.linalg.norm(P.std(axis=0)) * 1000.0)
        std_axis = float(np.degrees(np.arccos(
            np.clip(A @ A.mean(axis=0) / np.linalg.norm(A.mean(axis=0)),
                    -1.0, 1.0))).std())

        row = {
            "label": self.get_parameter("label").value,
            "door": self.get_parameter("door").value,
            "n_est": len(self._est), "n_gt": len(self._gt),
            "distance_m": round(float(np.linalg.norm(p_g)), 4),
            "err_norm_mm": round(float(np.linalg.norm(err)) * 1000, 2),
            "err_along_mm": round(along * 1000, 2),
            "err_perp_mm": round(perp * 1000, 2),
            "err_x_mm": round(err[0] * 1000, 2),
            "err_y_mm": round(err[1] * 1000, 2),
            "err_z_mm": round(err[2] * 1000, 2),
            "axis_err_deg": round(axis_err, 3),
            "rot_err_deg": round(rot_err, 3),
            "std_est_pos_mm": round(std_pos, 3),
            "std_est_axis_deg": round(std_axis, 3),
        }

        path = self.get_parameter("csv").value
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)

        self.get_logger().info(
            f"{row['label']}: udaljenost {row['distance_m']:.2f} m, "
            f"pogreška {row['err_norm_mm']:.1f} mm "
            f"(okomito {row['err_perp_mm']:.1f} mm), "
            f"os {row['axis_err_deg']:.2f}°")
        raise SystemExit(0)


def main() -> None:
    rclpy.init()
    node = AccuracyLogger()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
