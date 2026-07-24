import os
from glob import glob

from setuptools import find_packages, setup

package_name = "kmr_iiwa_bringup"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="David Mihić",
    maintainer_email="david.mihic@gmail.com",
    description="Jedan launch file koji pokrece cijelu ROS2 stranu infrastrukture (osim cmd_vel_bridge.py)",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [],
    },
)
