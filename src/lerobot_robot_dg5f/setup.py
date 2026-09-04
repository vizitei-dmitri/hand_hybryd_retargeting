from glob import glob

from setuptools import find_packages, setup


package_name = "lerobot_robot_dg5f"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "lerobot==0.4.4"],
    zip_safe=True,
    maintainer="DG5F workspace maintainer",
    maintainer_email="maintainer@example.com",
    description="LeRobot plugin and ROS 2 bridge for the Tesollo DG5F hand",
    license="MIT",
    entry_points={
        "console_scripts": [
            "ros_bridge = lerobot_robot_dg5f.ros_bridge_node:main",
            "sdk_check = lerobot_robot_dg5f.sdk_check:main",
            "debug_recorder = lerobot_robot_dg5f.debug_recorder:main",
        ],
    },
)
