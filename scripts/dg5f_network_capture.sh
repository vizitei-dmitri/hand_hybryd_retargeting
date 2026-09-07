#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# sudo credentials are optional; absence of privilege never blocks ROS logs.
if command -v tcpdump >/dev/null 2>&1 && [[ " $* " == *" --tcpdump "* ]]; then
  sudo -v || echo 'TCP capture unavailable: sudo not authorized; recording continues.' >&2
fi
exec python3 "${script_dir}/dg5f_network_capture.py" "$@"
