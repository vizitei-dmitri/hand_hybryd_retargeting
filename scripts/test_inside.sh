#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
set -u

run_package_tests() {
  local package_dir=$1
  echo "Testing ${package_dir#/workspace/src/}"
  (
    cd "${package_dir}"
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q test
  )
}

run_package_tests /workspace/src/dg5f_teleop
run_package_tests /workspace/src/dg5f_unity_teleop
run_package_tests /workspace/src/lerobot_robot_dg5f
run_package_tests /workspace/src/ros_tcp_endpoint
