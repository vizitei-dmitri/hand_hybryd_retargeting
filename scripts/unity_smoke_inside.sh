#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
set -u

smoke_port="${SMOKE_TCP_PORT:-10001}"
log_file=$(mktemp /tmp/dg5f-hybrid-smoke.XXXXXX.log)
launch_pid=""
client_pid=""

cleanup() {
  if [[ -n "${client_pid}" ]] && kill -0 "${client_pid}" 2>/dev/null; then
    kill -TERM "${client_pid}" 2>/dev/null || true
    wait "${client_pid}" 2>/dev/null || true
  fi
  if [[ -n "${launch_pid}" ]] && kill -0 "${launch_pid}" 2>/dev/null; then
    kill -TERM -- "-${launch_pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${launch_pid}" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "${launch_pid}" 2>/dev/null; then
      kill -KILL -- "-${launch_pid}" 2>/dev/null || true
    fi
    wait "${launch_pid}" 2>/dev/null || true
  fi
  echo "----- Unity V2 smoke launch log -----"
  sed -n '1,240p' "${log_file}"
  find "${log_file}" -delete
}
trap cleanup EXIT

setsid ros2 launch dg5f_unity_teleop unity_dg5f.launch.py \
  mujoco_viewer:=false \
  tcp_port:="${smoke_port}" \
  >"${log_file}" 2>&1 &
launch_pid=$!

for _ in $(seq 1 30); do
  if ! kill -0 "${launch_pid}" 2>/dev/null; then
    echo "Unity V2 smoke launch exited before becoming ready."
    exit 1
  fi

  nodes=$(ros2 node list 2>/dev/null || true)
  if grep -qx '/unity_endpoint' <<<"${nodes}" \
      && grep -qx '/unity_mano_adapter' <<<"${nodes}" \
      && grep -qx '/dg5f_retarget' <<<"${nodes}" \
      && grep -qx '/dg5f_mujoco' <<<"${nodes}"; then
    break
  fi
  sleep 0.5
done

nodes=$(ros2 node list)
grep -qx '/unity_endpoint' <<<"${nodes}"
grep -qx '/unity_mano_adapter' <<<"${nodes}"
grep -qx '/dg5f_retarget' <<<"${nodes}"
grep -qx '/dg5f_mujoco' <<<"${nodes}"

python3 /workspace/scripts/fake_unity_tcp_client.py \
  --port "${smoke_port}" \
  --frames 180 &
client_pid=$!

timeout 5s ros2 topic echo /quest/hand_pose --once >/dev/null
timeout 5s ros2 topic echo /hands/right/landmarks --once >/dev/null
timeout 5s ros2 topic echo /dg5f/joint_command --once >/dev/null
timeout 5s ros2 topic echo /dg5f/joint_states --once >/dev/null
timeout 5s ros2 topic echo /dg5f/tracking_ok --once | grep -q 'data: true'
wait "${client_pid}"
client_pid=""

echo "Hybrid V2 smoke passed: TCP/CDR -> Mano[21] -> Hybrid -> DG5F[20] -> MuJoCo."
