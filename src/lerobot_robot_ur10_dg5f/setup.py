from setuptools import find_packages, setup


package_name = "lerobot_robot_ur10_dg5f"


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
    ],
    install_requires=["setuptools", "lerobot==0.4.4"],
    zip_safe=True,
    maintainer="DG5F workspace maintainer",
    maintainer_email="maintainer@example.com",
    description="LeRobot plugin and gym environment for a UR10e arm with a DG5F hand",
    license="MIT",
)
