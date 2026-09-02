from glob import glob

from setuptools import find_packages, setup


package_name = "dg5f_teleop"

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
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="DG5F workspace maintainer",
    maintainer_email="maintainer@example.com",
    description="Quest hand-landmark retargeting and MuJoCo bridge for DG5F",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mujoco_bridge = dg5f_teleop.mujoco_bridge_node:main",
            "retarget = dg5f_teleop.retarget_node:main",
        ],
    },
)
