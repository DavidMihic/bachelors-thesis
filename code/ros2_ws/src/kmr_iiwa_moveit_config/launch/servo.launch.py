"""
servo.launch.py - pokrece MoveIt Servo (servo_node_main) za real-time
kartezijsko vodjenje end-effectora, kao alternativa move_to_pose planiranju
za kratke, precizne pokrete (npr. grasp-dive).

PRVI POKUSAJ - vidi napomenu u config/servo_config.yaml o mogucim
nedostajucim parametrima.

Preduvjet: move_group.launch.py MORA vec raditi (is_primary_planning_scene_monitor
je false u konfiguraciji, servo se oslanja na move_groupov planning scene monitor).

Pokretanje:
    ros2 launch kmr_iiwa_moveit_config servo.launch.py
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("kmr_iiwa", package_name="kmr_iiwa_moveit_config")
        .to_moveit_configs()
    )

    servo_yaml_path = os.path.join(
        get_package_share_directory("kmr_iiwa_moveit_config"),
        "config",
        "servo_config.yaml",
    )
    with open(servo_yaml_path, "r") as f:
        servo_yaml = yaml.safe_load(f)
    servo_params = {"moveit_servo": servo_yaml}

    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        parameters=[
            servo_params,
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
        ],
        output="screen",
    )

    return LaunchDescription([servo_node])
