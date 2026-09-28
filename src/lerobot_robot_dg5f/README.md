# `lerobot_robot_dg5f`

LeRobot 0.4.4 plugin and ROS 2 bridge for the Tesollo DG5F right hand.

The plugin implements LeRobot's `Robot` contract with 20 degree-valued
`<joint>.pos` actions and position, velocity, current and temperature
observations. The ROS bridge converts the existing radian-valued
`/dg5f/joint_command` trajectory into that contract.

The default `control_mode=legacy` preserves the existing path. The optional
`control_mode=servo` adds a 60 Hz persistent position command with a 120 deg/s
limit before the existing current guard. See [launch instructions, telemetry
and the legacy audit](../../docs/DG5F_SERVO_CONTROL.md).
The [hardware regression report](../../docs/DG5F_SERVO_HARDWARE_FIX.md) explains
the contact-offset fix, stage diagnostics and speed comparison commands.
The current [servo runtime policy and hardware trace audit](../../docs/DG5F_SERVO_RUNTIME_POLICY.md)
supersedes the earlier servo shutdown/resume policy: indefinite tracking hold,
automatic smooth resume, local load handling and a separate physical lead bound.
Manual ARM/disarm and explicit SDK recovery remain available; legacy is unchanged.
Servo ARM requires healthy hardware and fresh measured feedback, independently of
Quest/tracking. It seeds both controller and backend hold command from measured
pose once, then waits in TRACKING_HOLD until fresh tracking and a new valid target
arrive. No second ARM is needed; see the [ARM regression fix](../../docs/DG5F_SERVO_ARM_WITHOUT_TRACKING.md).
After contact release, servo uses its existing rate limiter without the legacy
15 deg/s resume ramp. See the [index slowdown audit and scale diagnostics](../../docs/DG5F_INDEX_FREE_MOTION.md).

In legacy, `command_shaper.py` owns the authoritative command trajectory. It applies
input filtering, deadband, velocity and acceleration limits, overshoot
protection and a minimum useful SDK output step. Measured feedback is read with
a bounded latest-sample drain for observations, but never reseeds that command
trajectory. `send_action()` returns the effective setpoint accepted by the
backend, which is suitable for LeRobot dataset actions.

The real backend is a direct adapter over
`dg5f_python.DGApi.set_target_position()`. ROS is not used as a motor driver and
there is no `ros2_control`, trajectory-controller, torque or current-control
path.

`rj_dg_5_1` is fixed at 0 degrees in both the plugin and the upstream ROS
command because the physical prototype's first pinky joint is unavailable.

Use the repository-level README for Docker, RSL and hardware instructions.
