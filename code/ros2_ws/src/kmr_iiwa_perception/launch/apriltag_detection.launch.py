"""Pokreće dva apriltag_ros node-a (36h11 za vrata, 16h5 za kvaku) na istom
kamera streamu. Zasebni node-ovi jer apriltag_ros podržava samo jednu tag
familiju po node instanci.

NAPOMENA - image_rect vs image_raw: apriltag_ros se pretplaćuje na REKTIFICIRANU
sliku (image_rect), ne raw. U simulaciji (Isaac Sim pinhole kamera, bez
modelirane leće) raw i rect su brojevno identični, pa je remap image_raw -> image_rect
opravdan ovdje. NA PRAVOM ROBOTU s pravim D435 ovo NIJE dovoljno - treba pravi
image_proc/rectify node između, jer stvarna leća ima distorziju. Ne zaboraviti
dodati kad se prebacuje na hardver (isti razlog zašto smo odabrali D435 - isti
downstream kod, ali OVAJ launch file treba dopuniti tim korakom).

Pokretanje:
  ros2 launch kmr_iiwa_perception apriltag_detection.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("kmr_iiwa_perception")
    door_cfg = os.path.join(pkg_share, "config", "tags_door.yaml")
    handle_cfg = os.path.join(pkg_share, "config", "tags_handle.yaml")

    common_remap = [
        ("image_rect", "/camera/color/image_raw"),   # vidi napomenu gore - vrijedi samo za sim
        ("camera_info", "/camera/color/camera_info"),
    ]

    door_node = Node(
        package="apriltag_ros",
        executable="apriltag_node",
        name="apriltag_door",
        namespace="apriltag_door",
        parameters=[door_cfg],
        remappings=common_remap,
        output="screen",
    )

    handle_node = Node(
        package="apriltag_ros",
        executable="apriltag_node",
        name="apriltag_handle",
        namespace="apriltag_handle",
        parameters=[handle_cfg],
        remappings=common_remap,
        output="screen",
    )

    return LaunchDescription([door_node, handle_node])
