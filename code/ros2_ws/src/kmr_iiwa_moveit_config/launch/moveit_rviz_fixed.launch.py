"""
moveit_rviz_fixed.launch.py - rucno sastavljen RViz launch, zamjena za
generirani moveit_rviz.launch.py.

Zasto ovo treba: generate_moveit_rviz_launch(moveit_config) iz
moveit_configs_utils ne postavlja robot_description/robot_description_semantic
kao parametre na rviz node u ovoj verziji (potvrdjeno: "ros2 param get /rviz
robot_description" vraca "not set" iako move_group, koji koristi ISTI
MoveItConfigsBuilder poziv, ucitava model bez problema). Ovaj launch radi
isto sto MoveItConfigsBuilder inace automatski odradi - eksplicitno
prosljedjuje svih pet standardnih MoveIt parametara direktno rviz2 nodu.

Pokretanje (umjesto ros2 launch kmr_iiwa_moveit_config moveit_rviz.launch.py):
    ros2 launch kmr_iiwa_moveit_config moveit_rviz_fixed.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("kmr_iiwa", package_name="kmr_iiwa_moveit_config")
        .to_moveit_configs()
    )

    rviz_config_path = os.path.join(
        get_package_share_directory("kmr_iiwa_moveit_config"),
        "config",
        "moveit.rviz",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_path],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    return LaunchDescription([rviz_node])
