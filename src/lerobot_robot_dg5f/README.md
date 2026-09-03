# `lerobot_robot_dg5f`

LeRobot 0.4.4 plugin and ROS 2 bridge for the Tesollo DG5F right hand.

The plugin implements LeRobot's `Robot` contract with 20 degree-valued
`<joint>.pos` actions and position, velocity, current and temperature
observations. The ROS bridge converts the existing radian-valued
`/dg5f/joint_command` trajectory into that contract.

`rj_dg_5_1` is fixed at 0 degrees in both the plugin and the upstream ROS
command because the physical prototype's first pinky joint is unavailable.

Use the repository-level README for Docker, RSL and hardware instructions.
