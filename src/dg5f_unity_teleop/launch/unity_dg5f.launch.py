"""Start the Unity, retargeting, MuJoCo and LeRobot DG5F pipeline."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    unity_share = FindPackageShare("dg5f_unity_teleop")
    core_share = FindPackageShare("dg5f_teleop")
    lerobot_share = FindPackageShare("lerobot_robot_dg5f")

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_endpoint", default_value="true"),
            DeclareLaunchArgument("start_adapter", default_value="true"),
            DeclareLaunchArgument("start_retarget", default_value="true"),
            DeclareLaunchArgument("start_mujoco", default_value="true"),
            DeclareLaunchArgument("start_lerobot", default_value="true"),
            DeclareLaunchArgument("fake_mano", default_value="false"),
            DeclareLaunchArgument("mujoco_viewer", default_value="true"),
            DeclareLaunchArgument("tcp_port", default_value="10000"),
            DeclareLaunchArgument("lerobot_backend", default_value="mock"),
            DeclareLaunchArgument("lerobot_auto_enable", default_value="true"),
            DeclareLaunchArgument("lerobot_ip", default_value="169.254.186.72"),
            DeclareLaunchArgument("lerobot_port", default_value="502"),
            DeclareLaunchArgument("lerobot_slave_id", default_value="1"),
            DeclareLaunchArgument(
                "lerobot_max_relative_target_deg", default_value="7.0"
            ),
            DeclareLaunchArgument("max_joint_velocity", default_value="0.0"),
            DeclareLaunchArgument("lerobot_control_smoothing", default_value="false"),
            DeclareLaunchArgument("mujoco_actuator_kp", default_value="40.0"),
            DeclareLaunchArgument("mujoco_actuator_kd", default_value="0.5"),
            DeclareLaunchArgument(
                "mujoco_self_collision", default_value="tip_only"
            ),
            DeclareLaunchArgument(
                "retarget_config",
                default_value=PathJoinSubstitution(
                    [core_share, "config", "dg5f_hybrid.yaml"]
                ),
            ),
            Node(
                package="ros_tcp_endpoint",
                executable="default_server_endpoint",
                name="unity_endpoint",
                output="screen",
                parameters=[
                    {
                        "ROS_IP": "0.0.0.0",
                        "ROS_TCP_PORT": LaunchConfiguration("tcp_port"),
                    }
                ],
                condition=IfCondition(LaunchConfiguration("start_endpoint")),
            ),
            Node(
                package="dg5f_unity_teleop",
                executable="mano_adapter",
                name="unity_mano_adapter",
                output="screen",
                parameters=[
                    PathJoinSubstitution(
                        [unity_share, "config", "mano_adapter.params.yaml"]
                    )
                ],
                condition=IfCondition(LaunchConfiguration("start_adapter")),
            ),
            Node(
                package="dg5f_unity_teleop",
                executable="fake_mano",
                name="fake_unity_mano",
                output="screen",
                condition=IfCondition(LaunchConfiguration("fake_mano")),
            ),
            Node(
                package="dg5f_teleop",
                executable="retarget",
                name="dg5f_retarget",
                output="screen",
                parameters=[
                    PathJoinSubstitution(
                        [core_share, "config", "retarget.params.yaml"]
                    ),
                    {
                        "urdf_path": (
                            "/workspace/models/dg5f/urdf/dg5f_right.urdf"
                        ),
                        "retargeting_config": LaunchConfiguration("retarget_config"),
                        "hybrid_dexpilot_config": PathJoinSubstitution(
                            [core_share, "config", "dg5f_dexpilot.yaml"]
                        ),
                        "max_joint_velocity": ParameterValue(
                            LaunchConfiguration("max_joint_velocity"),
                            value_type=float,
                        ),
                    },
                ],
                condition=IfCondition(LaunchConfiguration("start_retarget")),
            ),
            Node(
                package="dg5f_teleop",
                executable="mujoco_bridge",
                name="dg5f_mujoco",
                output="screen",
                parameters=[
                    PathJoinSubstitution(
                        [core_share, "config", "mujoco.params.yaml"]
                    ),
                    {
                        "model_path": (
                            "/workspace/models/dg5f/scene_dg.xml"
                        ),
                        "render": LaunchConfiguration("mujoco_viewer"),
                        "actuator_kp": ParameterValue(
                            LaunchConfiguration("mujoco_actuator_kp"),
                            value_type=float,
                        ),
                        "actuator_kd": ParameterValue(
                            LaunchConfiguration("mujoco_actuator_kd"),
                            value_type=float,
                        ),
                        "self_collision_mode": ParameterValue(
                            LaunchConfiguration("mujoco_self_collision"),
                            value_type=str,
                        ),
                    },
                ],
                condition=IfCondition(LaunchConfiguration("start_mujoco")),
            ),
            Node(
                package="lerobot_robot_dg5f",
                executable="ros_bridge",
                name="dg5f_lerobot_bridge",
                output="screen",
                parameters=[
                    PathJoinSubstitution(
                        [lerobot_share, "config", "bridge.params.yaml"]
                    ),
                    {
                        "backend": ParameterValue(
                            LaunchConfiguration("lerobot_backend"),
                            value_type=str,
                        ),
                        "control_smoothing": ParameterValue(
                            LaunchConfiguration("lerobot_control_smoothing"), value_type=bool,
                        ),
                        "auto_enable": ParameterValue(
                            LaunchConfiguration("lerobot_auto_enable"),
                            value_type=bool,
                        ),
                        "ip": ParameterValue(
                            LaunchConfiguration("lerobot_ip"),
                            value_type=str,
                        ),
                        "port": ParameterValue(
                            LaunchConfiguration("lerobot_port"),
                            value_type=int,
                        ),
                        "slave_id": ParameterValue(
                            LaunchConfiguration("lerobot_slave_id"),
                            value_type=int,
                        ),
                        "max_relative_target_deg": ParameterValue(
                            LaunchConfiguration("lerobot_max_relative_target_deg"),
                            value_type=float,
                        ),
                    },
                ],
                condition=IfCondition(LaunchConfiguration("start_lerobot")),
            ),
        ]
    )
