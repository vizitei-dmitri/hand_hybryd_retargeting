"""Isolated visualization/replay; retargeting is an explicit separate launch."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("kinect_mano_ros")
    arguments = [
        DeclareLaunchArgument("namespace", default_value="kinect_mano"),
        DeclareLaunchArgument("mode", default_value="replay", choices=["replay", "live"]),
        DeclareLaunchArgument("reconstruction", default_value=""),
        DeclareLaunchArgument("quality_json", default_value=""),
        DeclareLaunchArgument("source_axes", default_value="optical", choices=["optical", "ros"]),
        DeclareLaunchArgument("frame_id", default_value="kinect_mano_world"),
        DeclareLaunchArgument("speed", default_value="1.0"),
        DeclareLaunchArgument("loop", default_value="false"),
        DeclareLaunchArgument("publish_cloud", default_value="false"),
        DeclareLaunchArgument("start_rviz", default_value="false"),
        DeclareLaunchArgument("landmarks_topic", default_value="landmarks"),
        DeclareLaunchArgument("live_result_topic", default_value="input/result"),
        DeclareLaunchArgument("live_jsonl", default_value=""),
        DeclareLaunchArgument("max_live_age_s", default_value="0.35"),
        DeclareLaunchArgument(
            "rviz_config", default_value=PathJoinSubstitution([share, "rviz", "kinect_mano.rviz"]),
        ),
    ]
    parameters = {
        name: ParameterValue(LaunchConfiguration(name), value_type=value_type)
        for name, value_type in (
            ("mode", str), ("reconstruction", str), ("quality_json", str),
            ("source_axes", str), ("frame_id", str), ("speed", float),
            ("loop", bool), ("publish_cloud", bool),
            ("live_jsonl", str),
            ("max_live_age_s", float),
        )
    }
    return LaunchDescription(arguments + [
        Node(
            package="kinect_mano_ros", executable="tracker", name="tracker",
            namespace=LaunchConfiguration("namespace"), output="screen",
            parameters=[PathJoinSubstitution([share, "config", "replay.params.yaml"]), parameters],
            remappings=[
                ("landmarks", LaunchConfiguration("landmarks_topic")),
                ("input/result", LaunchConfiguration("live_result_topic")),
            ],
        ),
        Node(
            package="rviz2", executable="rviz2", name="rviz",
            namespace=LaunchConfiguration("namespace"),
            arguments=["-d", LaunchConfiguration("rviz_config")],
            condition=IfCondition(LaunchConfiguration("start_rviz")),
        ),
    ])
