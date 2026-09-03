# Third-party components

This snapshot contains the third-party source, model and runtime components
required to run the V2 LeRobot pipeline.

## ROS-TCP-Endpoint

- Upstream: <https://github.com/leggedrobotics/ROS-TCP-Endpoint/tree/main-ros2>
- Snapshot revision: `05a3b25a11d9639549ae128e471bd696430b91f7`
- License: Apache-2.0 (`src/ros_tcp_endpoint/LICENSE`)
- Local fixes: deserialize ROS 2 CDR before publishing, drop malformed
  transition frames without disconnecting, and guard `rclpy.shutdown()`.

## vr_haptic_msgs

- Upstream: <https://github.com/leggedrobotics/vr_haptic_msgs>
- Snapshot revision: `f5a5f56d42a8c379a563d6df5ff13f16dc4cd747`
- License: MIT (`src/vr_haptic_msgs/LICENSE`)
- Local fix: explicit `ament_cmake` build type for colcon/rosbag discovery.

## Tesollo DG5F MuJoCo model

- Upstream snapshot: <https://github.com/VAlikV/tesollo_dg5f_mujoco>
- Snapshot revision: `783b992745cd800b339cfdde7a2ae968cb76c59d`
- Included subset: DG5F scene, model, URDF and DG5F meshes only.
- The retained upstream notice is at `models/dg5f/UPSTREAM.md`.
- Local fix: URDF mesh paths resolve against the included assets directory.

## Tesollo DG5F control wrapper and DGSDK

- Source snapshot: local `tesollo_control` checkout based on revision
  `1ce62601d90dc8dbea5010ba9748b67f71517a16`.
- Included at: `vendor/tesollo_control`.
- Python distribution: `dg5f-python==1.0`, built inside the Docker image.
- The upstream license and notices are retained in the vendor directory.
- `libDGSDK.so` is a prebuilt vendor library and remains subject to Tesollo's
  distribution terms.
- Local safety change: command element `16` (`rj_dg_5_1`) is held at zero to
  mirror the known hardware fault.

## LeRobot

- Upstream: <https://github.com/huggingface/lerobot>
- Installed version: `0.4.4` from PyPI.
- License: Apache-2.0.
- Local integration: `src/lerobot_robot_dg5f` implements a third-party
  `Robot` plugin and a ROS 2 command/state bridge; the upstream source is not
  vendored.

## Runtime Python packages

Pinned versions are in `docker/python-requirements.txt`, including
`lerobot==0.4.4`, `dex-retargeting==0.5.0`, `mujoco==3.3.7`, `pin==3.4.0`
and `numpy==2.2.6`. They are downloaded during the Docker build and their own
licenses apply.
