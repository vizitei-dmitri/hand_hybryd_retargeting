#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
test_dir=$(mktemp -d /tmp/dg5f-fake-sdk.XXXXXX)
g++ -std=c++20 -pthread -I/usr/include/eigen3 \
  -I"${project_dir}/vendor/tesollo_control/include" \
  "${project_dir}/vendor/tesollo_control/dg_control/control.cpp" \
  "${project_dir}/vendor/tesollo_control/tests/fake_sdk_test.cpp" \
  -o "${test_dir}/fake_sdk_test"
timeout 10s "${test_dir}/fake_sdk_test"
