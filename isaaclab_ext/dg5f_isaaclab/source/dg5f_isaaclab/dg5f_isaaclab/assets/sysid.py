"""Validated, name-indexed DG5F SysID data. No Isaac Sim or pickle dependency."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from collections.abc import Sequence


DEFAULT_SYSID_PATH = Path(__file__).parent / "data/hand_sysid_params.json"


@dataclass(frozen=True)
class HandSysID:
    source: str
    sha256: str
    joint_order: tuple[str, ...]
    parameters: dict[str, dict[str, float]]
    delay_control_steps_by_finger: dict[str, int]
    disabled_joint: str
    fit_info_by_finger: dict

    def delays_for(self, joint_names: Sequence[str]) -> list[int]:
        return [self.delay_control_steps_by_finger[re.fullmatch(r"rj_dg_(\d+)_(\d+)", name)[1]]
                for name in joint_names]


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate SysID JSON key: {name}")
        result[name] = value
    return result


def load_sysid(path: str | Path, expected_joint_order: Sequence[str]) -> HandSysID:
    """Validate version, exact URDF order, complete dictionaries and finite values."""
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=_unique_object)
    if type(data.get("format_version")) is not int or data["format_version"] != 1:
        raise ValueError("Unsupported SysID format_version; expected 1")
    expected = tuple(expected_joint_order)
    order = data.get("joint_order")
    if not isinstance(order, list) or tuple(order) != expected or len(set(order)) != len(expected):
        raise ValueError("SysID joint_order must exactly match the URDF joint order")
    parameters = {}
    for parameter in ("stiffness", "damping", "armature", "friction"):
        values = data.get("parameters", {}).get(parameter, {})
        if not isinstance(values, dict) or set(values) != set(expected):
            raise ValueError(f"SysID {parameter} must contain every URDF joint exactly once")
        parameters[parameter] = {}
        for name in expected:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid SysID {parameter} for {name}: {value!r}")
            parameters[parameter][name] = float(value)
    fingers = set()
    for name in expected:
        match = re.fullmatch(r"rj_dg_(\d+)_(\d+)", name)
        if match is None:
            raise ValueError(f"Cannot resolve finger from joint name: {name}")
        fingers.add(match[1])
    delays = data.get("delay_control_steps_by_finger")
    if not isinstance(delays, dict) or set(delays) != fingers:
        raise ValueError("SysID delays must cover every finger exactly once")
    if any(type(delay) is not int or delay < 0 for delay in delays.values()):
        raise ValueError("SysID delay must be a nonnegative integer number of control steps")
    disabled = data.get("disabled_joint")
    if disabled not in expected:
        raise ValueError(f"SysID disabled_joint is absent from the URDF: {disabled!r}")
    return HandSysID(
        str(path), hashlib.sha256(raw).hexdigest(), expected, parameters, dict(delays),
        disabled, data.get("fit_info_by_finger", {}),
    )
