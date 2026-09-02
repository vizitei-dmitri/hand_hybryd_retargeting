"""Start Unity ROS-TCP Endpoint, Mano adapter, retargeting, and MuJoCo."""

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

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_endpoint", default_value="true"),
            DeclareLaunchArgument("start_adapter", default_value="true"),
            DeclareLaunchArgument("start_retarget", default_value="true"),
            DeclareLaunchArgument("start_mujoco", default_value="true"),
            DeclareLaunchArgument("fake_mano", default_value="false"),
            DeclareLaunchArgument("mujoco_viewer", default_value="true"),
            DeclareLaunchArgument("tcp_port", default_value="10000"),
            DeclareLaunchArgument("max_joint_velocity", default_value="3.0"),
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
        ]
    )
