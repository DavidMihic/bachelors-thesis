"""
ros2_control_test.launch.py - pokrece controller_manager (ros2_control_node) s
kmr_iiwa_full.urdf kao robot_description i ros2_controllers.yaml, pa odmah
spawna joint_state_broadcaster i arm_controller. Namjerno odvojeno od
MoveIt-a (move_group) - prvo provjeravamo da ros2_control lanac sam radi
(controller_manager <-> topic_based_ros2_control <-> Isaac Sim preko
/isaac_joint_commands, /isaac_joint_states), prije nego dodamo MoveIt sloj
iznad.

Preduvjet: Isaac Sim vec mora raditi (npr. cmd_vel_bridge.py) na USD-u koji
ima add_arm_ros_control_graph.py bakiran graf, inace topic_based_ros2_control
nema s kim razgovarati.

Pokretanje:
    ros2 launch kmr_iiwa_description ros2_control_test.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("kmr_iiwa_description")
    urdf_path = os.path.join(pkg_share, "urdf", "kmr_iiwa_full.urdf")
    controllers_yaml = os.path.join(pkg_share, "config", "ros2_controllers.yaml")

    with open(urdf_path, "r") as f:
        robot_description_content = f.read()

    # robot_state_publisher publisha TF (treba nam za RViz/MoveIt kasnije) i
    # robot_description kao topic. ros2_control_node NAMJERNO ne cita s tog
    # topica nego dobiva robot_description direktno kao parametar ispod -
    # "pravi" nacin (topic) ima race condition ako robot_state_publisher i
    # ros2_control_node startaju istovremeno (cesto se dogodi u launch fileu),
    # jer poruka moze biti objavljena prije nego se pretplata uspostavi, a
    # nikad se ne salje ponovno. Direktan parametar je pouzdaniji, samo
    # generira (bezopasan) deprecation warning.
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description_content}],
    )

    controller_manager_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": robot_description_content},
            controllers_yaml,
        ],
        output="screen",
    )

    spawn_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    spawn_arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller"],
        output="screen",
    )

    return LaunchDescription(
        [
            robot_state_publisher_node,
            controller_manager_node,
            spawn_joint_state_broadcaster,
            spawn_arm_controller,
        ]
    )
