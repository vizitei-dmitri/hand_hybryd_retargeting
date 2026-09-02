from glob import glob
import os

from setuptools import find_packages, setup


package_name = "dg5f_unity_teleop"


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
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hand workspace maintainer",
    maintainer_email="maintainer@example.com",
    description="Unity Quest Mano landmarks adapter and DG5F V2 launch",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mano_adapter = dg5f_unity_teleop.mano_adapter_node:main",
            "fake_mano = dg5f_unity_teleop.fake_mano_node:main",
        ],
    },
)
