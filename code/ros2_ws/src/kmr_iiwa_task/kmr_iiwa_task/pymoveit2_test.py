"""
pymoveit2_test.py - samostalna test skripta: potvrdi da pymoveit2 radi u
nasoj konfiguraciji prije nego se ugradi u pravi state-machine cvor. Salje
FIKSAN, hardkodiran cilj (jos ne cita /perception/handle_pose) grupi
iiwa_arm.

Za razliku od moveit_py (nije uspio instalirati - nema binarnog paketa za
ovu Humble konfiguraciju), pymoveit2 je cist Python bez kompajliranih
bindinga i razgovara s VEC POKRENUTIM move_group-om preko ROS2 akcija/
servisa - NE hostira vlastiti planning stack. Zato move_group.launch.py
MOZE (i treba) ostati pokrenut dok ovo radi, za razliku od moveit_py
pristupa.

NAPOMENA - provjeri protiv izvora: API pymoveit2 biblioteke se mijenja
izmedju verzija. Prije pokretanja, usporedi ovo s
code/ros2_ws/src/pymoveit2/examples/ex_pose_goal.py (dio kloniranog repoa) -
ako se konstruktor/nazivi metoda razlikuju od onoga dolje, prilagodi po
stvarnom izvoru, ne po ovom komentaru.

Isaac Sim (Terminal A) i ros2_control_test.launch.py (Terminal B) i dalje
moraju raditi, kao i move_group.launch.py (Terminal C) - pymoveit2 samo
salje zahtjeve tom vec pokrenutom move_group-u.

Pokretanje:
    ros2 run kmr_iiwa_task pymoveit2_test
"""

import threading

import rclpy
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

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
    node = Node("pymoveit2_test")
    callback_group = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name="base_link",
        end_effector_name="gripper_tcp",
        group_name="iiwa_arm",
        callback_group=callback_group,
    )

    # MoveIt2 klasa ceka na akcijske/servisne odgovore preko futures-a - treba
    # zaseban executor thread da spin() ne blokira glavnu nit dok cekamo.
    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # Joint-space cilj, ne pose - izbjegava IK/orijentacijska pitanja u
    # potpunosti, isti tip vrijednosti koje smo vec rucno slali preko
    # arm_controllera pa znamo da su fizicki dostizne. Cilj ovog testa je
    # samo potvrditi da pymoveit2 -> move_group -> arm_controller lanac radi;
    # pose-goal (kartezijski cilj s orijentacijom) je zaseban test, dodaj ga
    # tek kad ovo prodje.
    JOINT_GOAL = [0.3, -0.3, 0.0, -0.5, 0.0, 0.5, 0.0]
    node.get_logger().info(f"Saljem joint-space cilj: {JOINT_GOAL}")
    moveit2.move_to_configuration(JOINT_GOAL)
    moveit2.wait_until_executed()
    node.get_logger().info("Gotovo.")

    rclpy.shutdown()
    executor_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
