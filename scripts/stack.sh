#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "${script_dir}/.." && pwd)
script_path="${script_dir}/stack.sh"
compose_file="${project_dir}/compose.yaml"

export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"
video_gid=$(getent group video | cut -d: -f3 || true)
render_gid=$(getent group render | cut -d: -f3 || true)
export VIDEO_GID="${VIDEO_GID:-${video_gid:-${HOST_GID}}}"
export RENDER_GID="${RENDER_GID:-${render_gid:-${HOST_GID}}}"
export DISPLAY="${DISPLAY:-:0}"

compose=(docker compose -f "${compose_file}")
service=lerobot_hand
source_workspace='source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash'

show_help() {
  cat <<'EOF'
Usage: bash scripts/stack.sh COMMAND

Commands:
  setup          Build image, start container, compile, test and smoke-test
  build          Build/rebuild the standalone Docker image
  up             Start the background container
  compile        Build all five ROS 2 packages
  test           Run package tests
  smoke          Offline TCP -> Hybrid -> MuJoCo + LeRobot mock test
  gui-on         Allow the container to open the MuJoCo window
  gui-off        Revoke that X11 permission
  usb            Configure adb reverse localhost:10000 for Quest
  host-ip        Print host IPv4 addresses for a Wi-Fi connection
  launch [port] [hybrid|vector|dexpilot]
                 Start simulation plus the LeRobot mock backend
  headless [port] [hybrid|vector|dexpilot]
                 Start simulation/mock without the MuJoCo window
  hardware [port] [mode] [hand_ip]
                 Start MuJoCo plus the REAL Tesollo LeRobot backend (disarmed)
  hardware-headless [port] [mode] [hand_ip]
                 Start the real backend without the MuJoCo window (disarmed)
  arm            Enable real motion after passive health/tracking checks
  disarm         Stop forwarding new commands to the physical hand
  recover        Explicitly recover only the Tesollo SDK session; stays disarmed
  sdk-check [ip] Connect and read 20 positions without sending a command
  lerobot-check  Verify LeRobot plugin discovery and its action schema
  endpoint [port]
                 Start only ROS-TCP-Endpoint for connection diagnostics
  quest-view [fps]
                 Mirror the authorized Quest display to this PC
  topics         List ROS topics
  record [name]  Record Quest, normalized landmarks and DG5F state
  debug-record [--ping] [--tcpdump]
                 Passively record DG5F/ROS/network diagnostics (no control)
  debug-mark LABEL
                 Add an event marker to the active passive recording
  debug-record-network
                 ROS recorder + host ping, tcpdump and link counters
  dg-status      Read current ROS diagnostics; no SDK connection
  shell          Open a shell in the container
  logs           Show container logs
  stop           Stop only this project's container
EOF
}

require_container() {
  if ! "${compose[@]}" ps --status running --quiet "${service}" | grep -q .; then
    echo "Container is not running. Run: bash scripts/stack.sh up"
    exit 1
  fi
}

mode_config() {
  local mode=$1
  case "${mode}" in
    hybrid|vector|dexpilot) ;;
    *)
      echo "Retargeting mode must be 'hybrid', 'vector', or 'dexpilot'." >&2
      return 2
      ;;
  esac
  echo "/workspace/src/dg5f_teleop/config/dg5f_${mode}.yaml"
}

ensure_pipeline_slot_free() {
  local tcp_port=$1
  local running
  running=$("${compose[@]}" exec -T "${service}" bash -lc \
    "pgrep -af '[r]os2 launch dg5f_unity_teleop unity_dg5f.launch.py' || true")
  if [[ -n "${running}" ]]; then
    echo "Another DG5F pipeline is already running:" >&2
    echo "${running}" >&2
    echo "Stop its terminal with Ctrl+C before starting another mode." >&2
    return 1
  fi
  if ss -H -ltn "sport = :${tcp_port}" | grep -q .; then
    echo "TCP port ${tcp_port} is already in use." >&2
    echo "Stop the old RSL endpoint or choose another port." >&2
    return 1
  fi
}

case "${1:-help}" in
  setup)
    "${script_path}" build
    "${script_path}" up
    "${script_path}" compile
    "${script_path}" test
    "${script_path}" smoke
    ;;
  build)
    "${compose[@]}" build
    ;;
  up)
    "${compose[@]}" up -d "${service}"
    ;;
  compile)
    require_container
    "${compose[@]}" exec "${service}" bash -lc \
      'source /opt/ros/humble/setup.bash; cd /workspace; colcon build --symlink-install --base-paths /workspace/src'
    ;;
  test)
    require_container
    "${compose[@]}" exec "${service}" bash /workspace/scripts/test_inside.sh
    ;;
  smoke)
    require_container
    smoke_domain="${SMOKE_ROS_DOMAIN_ID:-91}"
    smoke_port="${SMOKE_TCP_PORT:-10001}"
    "${compose[@]}" exec \
      -e ROS_DOMAIN_ID="${smoke_domain}" \
      -e SMOKE_TCP_PORT="${smoke_port}" \
      "${service}" bash /workspace/scripts/unity_smoke_inside.sh
    ;;
  gui-on)
    xhost "+SI:localuser:$(id -un)"
    ;;
  gui-off)
    xhost "-SI:localuser:$(id -un)"
    ;;
  usb)
    adb start-server
    adb devices -l
    adb reverse tcp:10000 tcp:10000
    adb reverse --list
    ;;
  host-ip)
    hostname -I | tr ' ' '\n' | sed '/^$/d'
    ;;
  launch|headless)
    require_container
    tcp_port="${2:-${RSL_TCP_PORT:-10000}}"
    ensure_pipeline_slot_free "${tcp_port}"
    retarget_mode="${3:-${DG5F_RETARGET_MODE:-hybrid}}"
    retarget_config=$(mode_config "${retarget_mode}")
    viewer=true
    if [[ "${1}" == "headless" ]]; then
      viewer=false
    fi
    "${compose[@]}" exec "${service}" bash -lc \
      "${source_workspace}; ros2 launch dg5f_unity_teleop unity_dg5f.launch.py mujoco_viewer:=${viewer} tcp_port:=${tcp_port} retarget_config:=${retarget_config} lerobot_control_smoothing:=${DG5F_CONTROL_SMOOTHING:-false} lerobot_max_direct_step_deg:=${DG5F_MAX_DIRECT_STEP_DEG:-5.0} lerobot_startup_blend_s:=${DG5F_STARTUP_BLEND_S:-0.70} max_joint_velocity:=${DG5F_MAX_JOINT_VELOCITY:-0.0} mujoco_actuator_kp:=${DG5F_MUJOCO_KP:-40.0} mujoco_actuator_kd:=${DG5F_MUJOCO_KD:-0.5} mujoco_self_collision:=${DG5F_MUJOCO_SELF_COLLISION:-tip_only}"
    ;;
  hardware|hardware-headless)
    require_container
    tcp_port="${2:-${RSL_TCP_PORT:-10000}}"
    ensure_pipeline_slot_free "${tcp_port}"
    retarget_mode="${3:-${DG5F_RETARGET_MODE:-hybrid}}"
    hand_ip="${4:-${DG5F_HAND_IP:-169.254.186.72}}"
    retarget_config=$(mode_config "${retarget_mode}")
    viewer=true
    if [[ "${1}" == "hardware-headless" ]]; then
      viewer=false
    fi
    echo "Starting REAL Tesollo backend at ${hand_ip}:502 in DISARMED state."
    echo "After checking tracking and MuJoCo, use: bash scripts/stack.sh arm"
    "${compose[@]}" exec "${service}" bash -lc \
      "${source_workspace}; ros2 launch dg5f_unity_teleop unity_dg5f.launch.py mujoco_viewer:=${viewer} tcp_port:=${tcp_port} retarget_config:=${retarget_config} lerobot_backend:=tesollo lerobot_auto_enable:=false lerobot_ip:=${hand_ip} lerobot_control_smoothing:=${DG5F_CONTROL_SMOOTHING:-false} lerobot_max_direct_step_deg:=${DG5F_MAX_DIRECT_STEP_DEG:-5.0} lerobot_startup_blend_s:=${DG5F_STARTUP_BLEND_S:-0.70} max_joint_velocity:=${DG5F_MAX_JOINT_VELOCITY:-0.0} mujoco_actuator_kp:=${DG5F_MUJOCO_KP:-40.0} mujoco_actuator_kd:=${DG5F_MUJOCO_KD:-0.5} mujoco_self_collision:=${DG5F_MUJOCO_SELF_COLLISION:-tip_only}"
    ;;
  arm|disarm)
    require_container
    enable=true
    if [[ "${1}" == "disarm" ]]; then
      enable=false
    fi
    "${compose[@]}" exec "${service}" bash -lc \
      "${source_workspace}; ros2 service call /dg5f/lerobot/enable std_srvs/srv/SetBool '{data: ${enable}}'"
    ;;
  recover)
    require_container
    echo "Explicit DG5F SDK recovery requested. High-level output will remain DISARMED."
    "${compose[@]}" exec "${service}" bash -lc \
      "${source_workspace}; ros2 service call /dg5f/lerobot/recover std_srvs/srv/Trigger '{}'"
    ;;
  sdk-check)
    require_container
    hand_ip="${2:-${DG5F_HAND_IP:-169.254.186.72}}"
    echo "Read-only DG5F SDK check at ${hand_ip}:502 (no target is sent)."
    "${compose[@]}" exec "${service}" bash -lc \
      "${source_workspace}; ros2 run lerobot_robot_dg5f sdk_check --ip ${hand_ip} --port 502 --slave-id 1"
    ;;
  lerobot-check)
    require_container
    "${compose[@]}" exec "${service}" bash -lc \
      "${source_workspace}; python3 -c 'from lerobot.utils.import_utils import register_third_party_plugins; register_third_party_plugins(); from lerobot.robots import RobotConfig; print(\"dg5f registered:\", \"dg5f\" in RobotConfig.get_known_choices()); from lerobot_robot_dg5f import Dg5f, Dg5fConfig; robot=Dg5f(Dg5fConfig(id=\"check\", backend=\"mock\")); print(\"actions:\", len(robot.action_features), \"observations:\", len(robot.observation_features))'"
    ;;
  endpoint)
    require_container
    tcp_port="${2:-${RSL_TCP_PORT:-10000}}"
    ensure_pipeline_slot_free "${tcp_port}"
    "${compose[@]}" exec "${service}" bash -lc \
      "${source_workspace}; ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0 -p ROS_TCP_PORT:=${tcp_port}"
    ;;
  quest-view)
    view_fps="${2:-30}"
    if ! [[ "${view_fps}" =~ ^[0-9]+$ ]] || (( view_fps < 1 || view_fps > 90 )); then
      echo "Quest mirror FPS must be an integer from 1 to 90."
      exit 2
    fi

    adb start-server
    device_state=$(adb devices | awk 'NR > 1 && NF >= 2 {print $2; exit}')
    if [[ "${device_state:-}" != "device" ]]; then
      echo "Quest ADB state is '${device_state:-not-found}'."
      echo "Put on the headset, accept USB debugging, then retry."
      exit 1
    fi

    if command -v scrcpy >/dev/null 2>&1; then
      scrcpy_bin=$(command -v scrcpy)
    else
      tools_dir="${project_dir}/.tools"
      scrcpy_dir="${tools_dir}/scrcpy-linux-x86_64-v4.1"
      archive_path="${tools_dir}/scrcpy-linux-x86_64-v4.1.tar.gz"
      install -d "${tools_dir}"
      if [[ ! -x "${scrcpy_dir}/scrcpy" ]]; then
        curl --fail --location --output "${archive_path}" \
          https://github.com/Genymobile/scrcpy/releases/download/v4.1/scrcpy-linux-x86_64-v4.1.tar.gz
        echo "ad56ae8bfeedf41e824945c11dbf55fcb092b3e615b9b486f48a50e30d389635  ${archive_path}" \
          | sha256sum --check --status
        tar -xzf "${archive_path}" -C "${tools_dir}"
      fi
      scrcpy_bin="${scrcpy_dir}/scrcpy"
    fi

    restore_quest_power() {
      adb shell am broadcast \
        -a com.oculus.vrpowermanager.automation_disable >/dev/null 2>&1 || true
    }
    trap restore_quest_power EXIT INT TERM
    adb shell am broadcast -a com.oculus.vrpowermanager.prox_close >/dev/null
    "${scrcpy_bin}" \
      --no-control \
      --no-audio \
      --max-fps="${view_fps}" \
      --video-bit-rate=12M \
      --window-title="Quest 3 - RSL LeRobot retargeting"
    ;;
  topics)
    require_container
    "${compose[@]}" exec "${service}" bash -lc \
      "${source_workspace}; ros2 topic list | sort"
    ;;
  record)
    require_container
    bag_name="${2:-lerobot_hand_$(date +%Y%m%d_%H%M%S)}"
    install -d "${project_dir}/bags"
    "${compose[@]}" exec "${service}" bash -lc \
      "${source_workspace}; ros2 bag record -o /workspace/bags/${bag_name} --storage mcap /quest/hand_pose /quest/hand_points /quest/hand_gesture /hands/right/landmarks /dg5f/joint_command /dg5f/target_joint_states /dg5f/joint_states /dg5f/tracking_ok /dg5f/lerobot/joint_states /dg5f/lerobot/commanded_joint_states /dg5f/lerobot/temperatures /dg5f/lerobot/connected /dg5f/lerobot/armed /dg5f/lerobot/diagnostics /dg5f/debug_marker /tf"
    ;;
  debug-record|debug-record-network)
    require_container
    hand_ip="${DG5F_HAND_IP:-169.254.186.72}"
    network_interface="${DG5F_NETWORK_INTERFACE:-enp49s0}"
    extra_args=()
    if [[ "$1" == debug-record-network ]]; then
      extra_args=(--ping --tcpdump)
    fi
    for option in "${@:2}"; do
      case "${option}" in
        --ping|--tcpdump) extra_args+=("${option}") ;;
        *)
          echo "debug-record accepts only --ping and --tcpdump." >&2
          exit 2
          ;;
      esac
    done
    if (( ${#extra_args[@]} )); then
      exec bash "${script_dir}/dg5f_network_capture.sh" \
        --interface "${network_interface}" --hand-ip "${hand_ip}" "${extra_args[@]}"
    fi
    echo "Starting passive recorder. It does not connect to or control DG-5F."
    "${compose[@]}" exec "${service}" bash -lc \
      "${source_workspace}; ros2 run lerobot_robot_dg5f debug_recorder --output-root /workspace/debug_runs --project-dir /workspace --interface ${network_interface} --hand-ip ${hand_ip} ${extra_args[*]}"
    ;;
  dg-status)
    require_container
    "${compose[@]}" exec "${service}" bash -lc \
      "${source_workspace}; timeout 8s ros2 topic echo /dg5f/lerobot/diagnostics --once"
    ;;
  debug-mark)
    require_container
    marker="${2:-}"
    if [[ -z "${marker}" || ! "${marker}" =~ ^[[:alnum:]_.:-]+$ ]]; then
      echo "Marker must contain only letters, digits, underscore, dot, colon or dash." >&2
      exit 2
    fi
    "${compose[@]}" exec "${service}" bash -lc \
      "${source_workspace}; ros2 topic pub --once /dg5f/debug_marker std_msgs/msg/String \"{data: '${marker}'}\""
    ;;
  shell)
    require_container
    "${compose[@]}" exec "${service}" bash
    ;;
  logs)
    "${compose[@]}" logs --tail 200 "${service}"
    ;;
  stop)
    "${compose[@]}" down
    ;;
  help|-h|--help)
    show_help
    ;;
  *)
    echo "Unknown command: $1"
    show_help
    exit 2
    ;;
esac
