"""Select Quest or Kinect and reuse the existing DG5F simulation pipeline."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _selected_pipeline(context):
    source = LaunchConfiguration("source").perform(context)
    if source not in ("quest", "kinect"):
        raise ValueError("source must be quest or kinect")
    # Guard explicit overrides too: this selector only starts visualization/simulation.
    if LaunchConfiguration("start_lerobot").perform(context) != "false":
        raise ValueError("hand_source.launch.py requires start_lerobot:=false")
    if LaunchConfiguration("lerobot_backend").perform(context) != "mock":
        raise ValueError("hand_source.launch.py requires lerobot_backend:=mock")
    actions = []
    if source == "kinect":
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare("kinect_mano_ros"), "launch", "kinect_mano.launch.py",
            ])),
            launch_arguments=({
                name: LaunchConfiguration(name) for name in (
                    "reconstruction", "quality_json", "mode", "source_axes", "frame_id",
                    "speed", "loop", "publish_cloud", "start_rviz", "live_result_topic",
                    "live_jsonl",
                    "max_live_age_s",
                )
            } | {"landmarks_topic": "/hands/right/landmarks"}).items(),
        ))
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("dg5f_unity_teleop"), "launch", "unity_dg5f.launch.py",
        ])),
        launch_arguments={
            "start_endpoint": "true" if source == "quest" else "false",
            "start_adapter": "true" if source == "quest" else "false",
            "fake_mano": "false", "start_lerobot": "false",
            "lerobot_backend": "mock", "lerobot_auto_enable": "false",
            "start_retarget": LaunchConfiguration("start_retarget"),
            "start_mujoco": LaunchConfiguration("start_mujoco"),
            "mujoco_viewer": LaunchConfiguration("mujoco_viewer"),
        }.items(),
    ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="kinect", choices=["quest", "kinect"]),
        DeclareLaunchArgument("reconstruction", default_value=""),
        DeclareLaunchArgument("quality_json", default_value=""),
        DeclareLaunchArgument("mode", default_value="replay", choices=["replay", "live"]),
        DeclareLaunchArgument("source_axes", default_value="optical", choices=["optical", "ros"]),
        DeclareLaunchArgument("frame_id", default_value="kinect_mano_world"),
        DeclareLaunchArgument("speed", default_value="1.0"),
        DeclareLaunchArgument("loop", default_value="false"),
        DeclareLaunchArgument("publish_cloud", default_value="false"),
        DeclareLaunchArgument("start_rviz", default_value="false"),
        DeclareLaunchArgument("live_result_topic", default_value="input/result"),
        DeclareLaunchArgument("live_jsonl", default_value=""),
        DeclareLaunchArgument("max_live_age_s", default_value="0.35"),
        DeclareLaunchArgument("start_retarget", default_value="true"),
        DeclareLaunchArgument("start_mujoco", default_value="true"),
        DeclareLaunchArgument("mujoco_viewer", default_value="false"),
        DeclareLaunchArgument("start_lerobot", default_value="false", choices=["false"]),
        DeclareLaunchArgument("lerobot_backend", default_value="mock", choices=["mock"]),
        OpaqueFunction(function=_selected_pipeline),
    ])
