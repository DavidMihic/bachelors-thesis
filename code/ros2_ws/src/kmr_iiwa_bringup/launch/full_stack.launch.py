"""
full_stack.launch.py - jedan launch file za sve sto NE treba isaaclab.sh
(Isaac Simov Python) - cmd_vel_bridge.py i dalje pokreces RUCNO, zasebno,
PRIJE ovog launch filea:
    ./isaaclab.sh -p code/ros2_ws/src/kmr_iiwa_sim_bridge/kmr_iiwa_sim_bridge/cmd_vel_bridge.py \
        --usd_path <putanja do USD-a>

Ukljuceno:
  - ros2_control_test.launch.py (controller_manager + joint_state_broadcaster
    + arm_controller, iz kmr_iiwa_description)
  - apriltag_detection.launch.py (door_tag_center + handle_tag_a/b, iz
    kmr_iiwa_perception)
  - handle_pose_fusion node (iz kmr_iiwa_perception)
  - gripper_bridge node (iz kmr_iiwa_sim_bridge - standalone, dijeljena
    simulacija, NE otvara vlastiti Isaac Sim)
  - move_group.launch.py (MoveIt, iz kmr_iiwa_moveit_config) - odgodjen 5s
    da arm_controller stigne postati aktivan prije MoveIt-a
  - moveit_rviz_fixed.launch.py (opcionalno, launch_rviz:=true) - odgodjen
    8s da move_group vec bude gore
  - tcp_wrench_estimator

NAMJERNO izostavljeno (pokreni rucno, po potrebi):
  - cmd_vel_bridge.py - vidi gore, treba isaaclab.sh
  - door_task_node, handle_approach_test i slicni jednokratni test/task
    skriptovi - ovo su radnje koje pokrecemo RUCNO kad testiramo konkretnu
    fazu, ne stalni servisi koji trebaju uvijek raditi

PREDUVJET: cmd_vel_bridge.py (Isaac Sim) mora VEC raditi i biti spreman
(scena ucitana, Play pritisnut) prije nego pokrenes ovaj launch file - isti
redoslijed kao i dosad kroz sve nase testove, samo sad umjesto pet-sest
zasebnih terminala treba samo taj jedan plus ovaj launch.

Odgode (5s/8s) su POCETNA PROCJENA - ako controller_manager/move_group jos
nisu spremni kad im dodje red, produlji ih. Ne postoji čist nacin da launch
file ceka "Isaac Sim je gotov" jer je Isaac Sim izvan njegove kontrole
(rucno pokrenut proces).

Pokretanje:
    ros2 launch kmr_iiwa_bringup full_stack.launch.py
    ros2 launch kmr_iiwa_bringup full_stack.launch.py launch_rviz:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    launch_rviz_arg = DeclareLaunchArgument(
        "launch_rviz",
        default_value="false",
        description="Pokreni i RViz (moveit_rviz_fixed.launch.py) uz ostalo.",
    )
    launch_rviz = LaunchConfiguration("launch_rviz")

    ros2_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("kmr_iiwa_description"),
                "launch",
                "ros2_control_test.launch.py",
            )
        )
    )

    apriltag = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("kmr_iiwa_perception"),
                "launch",
                "apriltag_detection.launch.py",
            )
        )
    )

    handle_pose_fusion = Node(
        package="kmr_iiwa_perception",
        executable="handle_pose_fusion",
        output="screen",
    )

    tcp_wrench_estimator = Node(
        package="kmr_iiwa_perception",
        executable="tcp_wrench_estimator",
        output="screen",
    )

    gripper_bridge = Node(
        package="kmr_iiwa_sim_bridge",
        executable="gripper_bridge",
        output="screen",
    )

    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("kmr_iiwa_moveit_config"),
                "launch",
                "move_group.launch.py",
            )
        )
    )
    move_group_delayed = TimerAction(period=5.0, actions=[move_group])

    moveit_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("kmr_iiwa_moveit_config"),
                "launch",
                "moveit_rviz_fixed.launch.py",
            )
        ),
        condition=IfCondition(launch_rviz),
    )
    moveit_rviz_delayed = TimerAction(period=8.0, actions=[moveit_rviz])

    return LaunchDescription(
        [
            launch_rviz_arg,
            ros2_control,
            apriltag,
            handle_pose_fusion,
            tcp_wrench_estimator,
            gripper_bridge,
            move_group_delayed,
            moveit_rviz_delayed,
        ]
    )
