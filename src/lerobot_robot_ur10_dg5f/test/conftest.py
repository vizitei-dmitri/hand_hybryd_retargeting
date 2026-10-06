import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def mjcf_path():
    pytest.importorskip("mujoco")
    path = Path(os.environ.get("UR10E_DG5F_MJCF", REPO_ROOT / "models" / "ur10e_dg5f" / "scene.xml"))
    if not path.is_file():
        pytest.skip(f"MJCF not found: {path}")
    return path


@pytest.fixture
def make_sim_robot(mjcf_path, tmp_path):
    from lerobot_robot_ur10_dg5f import Ur10Dg5f, Ur10Dg5fConfig

    robots = []

    def make(**overrides):
        overrides.setdefault("sim_cameras", ())
        config = Ur10Dg5fConfig(id="test", calibration_dir=tmp_path, mjcf_path=str(mjcf_path), **overrides)
        robot = Ur10Dg5f(config)
        robot.connect()
        robots.append(robot)
        return robot

    yield make
    for robot in robots:
        robot.disconnect()
