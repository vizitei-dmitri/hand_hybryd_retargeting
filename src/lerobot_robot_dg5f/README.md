# `lerobot_robot_dg5f`

LeRobot 0.4.4 plugin and ROS 2 bridge for the Tesollo DG5F right hand.

The plugin implements LeRobot's `Robot` contract with 20 degree-valued
`<joint>.pos` actions and position, velocity, current and temperature
observations. The ROS bridge converts the existing radian-valued
`/dg5f/joint_command` trajectory into that contract.

`command_shaper.py` owns the authoritative command trajectory. It applies
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
