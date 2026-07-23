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
        ],
    },
)
