These examples describe the native PyTorch MANO tracker and the optional ROS
transport. The numerical core uses no ROS and no `smplx` dependency.

`fit.example.json` contains `kinect_mano.fitting.FitConfig` fields. It selects
temporal fitting, uses metres/seconds, and breaks history across gaps longer
than 0.25 s. The file contains fitting settings, not camera calibration. The
CLI exposes fitting choices as flags (`--mode temporal --iterations 80
--points-per-camera 2048`), rather than a `--config` file option. Python callers
can use `FitConfig(**json.loads(path.read_text()))` to load this complete example.

The official converted MANO `.npz` must be supplied locally. No licensed model
is bundled or downloaded. The demo fixture is synthetic and cannot establish
anatomical reconstruction accuracy.

Camera calibration must contain actual intrinsics, distortion, resolution and
`T_world_from_camera`, acting on column vectors in metres. Deliberately no
identity extrinsic example is supplied: obtain and validate the transform for
the recording. Core camera coordinates are right/down/forward. For a world
frame already using ROS forward/left/up, pass `source_axes:=ros` to ROS.

ROS parameter defaults and RViz configuration are installed by `kinect_mano_ros`
from `src/kinect_mano_ros/config` and `src/kinect_mano_ros/rviz`. Its README
documents replay, live JSONL and simulation-only source selection.
