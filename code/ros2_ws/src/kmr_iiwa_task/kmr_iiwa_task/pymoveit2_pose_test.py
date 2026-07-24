"""
pymoveit2_pose_test.py - potvrdi da move_to_pose (kartezijski cilj s
orijentacijom) radi, prije nego se osloni na stvarnu percepciju
(/perception/handle_pose). Za razliku od ranijeg hardkodiranog pokusaja
(propao - vjerojatno kombinacija zadanog cartesian=True i neispitane
identity orijentacije), ovaj test:

  1. cita TRENUTNU pozu gripper_tcp preko TF-a (isti obrazac kao
     door_task_node.py),
  2. trazi mali pomak POZICIJE uz ISTU orijentaciju (zajamceno dostizno -
     mala perturbacija iz stanja koje vec znamo da je validno),
  3. eksplicitno postavlja cartesian=False.

Isaac Sim (Terminal A), ros2_control_test.launch.py (Terminal B) i
move_group.launch.py (Terminal C) moraju raditi.

Pokretanje:
    ros2 run kmr_iiwa_task pymoveit2_pose_test
"""

import threading

import rclpy
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
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


def main():
    rclpy.init()
    node = Node("pymoveit2_pose_test")
    callback_group = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name="base_link",
        end_effector_name="gripper_tcp",
        group_name="iiwa_arm",
        callback_group=callback_group,
    )

    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    node.get_logger().info("Cekam TF base_link -> gripper_tcp...")
    transform = None
    while transform is None and rclpy.ok():
        try:
            transform = tf_buffer.lookup_transform("base_link", "gripper_tcp", rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            pass

    t = transform.transform.translation
    q = transform.transform.rotation
    node.get_logger().info(f"Trenutna pozicija gripper_tcp: ({t.x:.3f}, {t.y:.3f}, {t.z:.3f})")

    # Mali pomak (5cm naprijed po x) uz identicnu orijentaciju.
    target_position = [t.x + 0.05, t.y, t.z]
    target_orientation = [q.x, q.y, q.z, q.w]

    node.get_logger().info(f"Saljem pose goal: pozicija={target_position}, orijentacija={target_orientation}")
    moveit2.move_to_pose(
        position=target_position,
        quat_xyzw=target_orientation,
        cartesian=False,
    )
    moveit2.wait_until_executed()
    node.get_logger().info("Gotovo.")

    rclpy.shutdown()
    executor_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
