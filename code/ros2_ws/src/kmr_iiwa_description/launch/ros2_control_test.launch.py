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

    # NAPOMENA: robot_state_publisher NAMJERNO nije ovdje. Isaac Sim
    # (add_camera_ros_graph.py, bakirano u USD) vec publisha TF za CIJELO
    # kinematicko stablo od base_link nanize, ukljucujuci iiwa_link_1..7,
    # direktno iz stvarnog stanja fizike, sa sim-time pecatima. Da smo ovdje
    # takodjer pokrenuli robot_state_publisher, on bi NEOVISNO racunao i
    # publishao TF za ISTE frameove iz /joint_states, sa svojim (wall-clock)
    # satom - dva izvora, dvije vremenske domene, isti frameovi -> TF2
    # "ignoring data from the past" konflikt (proslo iskustvo, vidi handoff
    # razgovor). move_group dobiva URDF/SRDF direktno preko
    # MoveItConfigsBuilder-a, ne treba robot_state_publisher-ov
    # /robot_description topic.
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
            controller_manager_node,
            spawn_joint_state_broadcaster,
            spawn_arm_controller,
        ]
    )
