"""Shared diagnostic reason mapping; no current/error collision thresholds."""

import math


def age_ms(timestamp, now):
    return math.nan if timestamp is None else max(0.0, (now - timestamp) * 1000.0)


def disarm_reason(status):
    if status.get("diagnostics_error"):
        return "INTERNAL_ERROR"
    if not status.get("transport_connected", False):
        return "SDK_DISCONNECTED"
    if not status.get("telemetry_valid", False):
        return "TELEMETRY_STALE"
    if not status.get("temperature_safe", False):
        return "TEMPERATURE_LIMIT"
    if not status.get("system_started", False):
        return "SYSTEM_NOT_STARTED"
    if status.get("last_motion_result", 0) != 0:
        return "SDK_COMMAND_REJECTED"
    if not status.get("motion_ready", False):
        return "SDK_NOT_MOTION_READY"
    return "NONE"
