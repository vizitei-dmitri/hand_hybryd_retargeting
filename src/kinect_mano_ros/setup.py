from glob import glob

from setuptools import find_packages, setup


setup(
    name="kinect_mano_ros",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/kinect_mano_ros"]),
        ("share/kinect_mano_ros", ["package.xml", "README.md"]),
        ("share/kinect_mano_ros/launch", glob("launch/*.launch.py")),
        ("share/kinect_mano_ros/config", glob("config/*.yaml")),
        ("share/kinect_mano_ros/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools", "numpy", "scipy"],
    zip_safe=True,
    maintainer="hand workspace maintainer",
    maintainer_email="maintainer@example.com",
    description="Optional ROS 2 transport for native Kinect MANO results",
    license="Apache-2.0",
    entry_points={
        "console_scripts": ["tracker = kinect_mano_ros.node:main"],
    },
)
