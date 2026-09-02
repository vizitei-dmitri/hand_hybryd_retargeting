#!/usr/bin/env bash
set -euo pipefail

BLEND="${1:-0.80}"
START="${2:-0.055}"
FULL="${3:-0.025}"
MAX_CORR="${4:-0.45}"
LATERAL="${5:-0.12}"
CORRECTION_ALPHA="${6:-0.35}"
RELEASE_ALPHA="${7:-0.18}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "${script_dir}/.." && pwd)
FILE="${project_dir}/src/dg5f_teleop/config/retarget.params.yaml"

python3 - "$FILE" "$BLEND" "$START" "$FULL" "$MAX_CORR" \
  "$LATERAL" "$CORRECTION_ALPHA" "$RELEASE_ALPHA" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
values = {
    "hybrid_max_blend": float(sys.argv[2]),
    "hybrid_contact_start": float(sys.argv[3]),
    "hybrid_contact_full": float(sys.argv[4]),
    "hybrid_max_joint_correction": float(sys.argv[5]),
    "hybrid_lateral_max_correction": float(sys.argv[6]),
    "hybrid_correction_alpha": float(sys.argv[7]),
    "hybrid_release_alpha": float(sys.argv[8]),
}

if not 0.0 <= values["hybrid_max_blend"] <= 1.0:
    raise SystemExit("hybrid_max_blend must be in [0,1]")
if values["hybrid_contact_start"] <= values["hybrid_contact_full"]:
    raise SystemExit("contact_start must be > contact_full")
if values["hybrid_max_joint_correction"] <= 0:
    raise SystemExit("max correction must be > 0")
if values["hybrid_lateral_max_correction"] < 0:
    raise SystemExit("lateral correction must be >= 0")
if not 0.0 <= values["hybrid_correction_alpha"] <= 1.0:
    raise SystemExit("correction alpha must be in [0,1]")
if not 0.0 <= values["hybrid_release_alpha"] <= 1.0:
    raise SystemExit("release alpha must be in [0,1]")

text = path.read_text()
for key, value in values.items():
    text, n = re.subn(
        rf"(?m)^(\s*{re.escape(key)}:\s*).*$",
        rf"\g<1>{value}",
        text,
        count=1,
    )
    if n != 1:
        raise SystemExit(f"Missing parameter: {key}")
path.write_text(text)

for key, value in values.items():
    print(f"{key}: {value}")
PY

echo "Restart the launch to apply the new values."
