from setuptools import find_packages, setup

package_name = "kmr_iiwa_task"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="David Mihić",
    maintainer_email="david.mihic@gmail.com",
    description="State-machine orkestracija zadatka otvaranja vrata (model-based pristup)",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "door_task_node = kmr_iiwa_task.door_task_node:main",
            "handle_approach = kmr_iiwa_task.handle_approach:main",
            "door_probe = kmr_iiwa_task.door_probe:main",
            "door_open = kmr_iiwa_task.door_open:main",
            "open_sliding = kmr_iiwa_task.open_sliding:main",
            "open_revolute = kmr_iiwa_task.open_revolute:main",
            "rl_door_inference = kmr_iiwa_task.rl_door_inference:main",
            "base_speed_test = kmr_iiwa_task.base_speed_test:main",
            "pass_through_door = kmr_iiwa_task.pass_through_door:main",
        ],
    },
)
