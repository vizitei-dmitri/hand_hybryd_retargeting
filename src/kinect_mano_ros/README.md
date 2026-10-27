# Kinect MANO ROS 2 adapter

This optional ament package consumes the standalone tracker's public arrays.
Replay and transport only need NumPy/SciPy plus ROS; they never load a MANO
model or import Torch. Importing the package and `node` module without ROS is
supported. Constructing the node requires a sourced ROS 2 Humble environment.

Build from the workspace with `colcon build --symlink-install`, then source
`install/setup.bash`. The standalone core is separately installed with
`python3 -m pip install -e kinect_mano` in its own environment.

## Isolated replay and visualization

```bash
ros2 launch kinect_mano_ros kinect_mano.launch.py \
  reconstruction:=/absolute/path/reconstruction.npz start_rviz:=true
```

The default namespace is `/kinect_mano`; this launch starts no retargeter or
robot bridge. Topics are `landmarks` (`PoseArray`, wrist/thumb/index/middle/ring/
little order, 21 points), `markers` (`MarkerArray`, triangle mesh, skeleton and
joints), `mano` and `quality` (`String` JSON), plus optional `cloud`
(`PointCloud2` XYZ) with `publish_cloud:=true`. Separate `cloud_A` (orange) and
`cloud_B` (blue) expose the two actual camera clouds. RViz uses fixed frame
`kinect_mano_world`. With a custom namespace/frame, update the RViz topics/fixed
frame or pass `rviz_config:=...`. RViz's optional PoseArray display needs
`/hands/right/landmarks` when using the integration launch.

Mandatory NPZ arrays are `timestamps_s[F]`, `joints_m[F,21,3]`,
`vertices_m[F,778,3]`, integer `faces[K,3]`, `theta[F,45]`, `beta[10]`,
`global_orient[F,3]` (axis-angle radians), `translation_m[F,3]`, and `valid[F]`.
An optional numeric `cloud_m[F,N,3]` contains combined observed points. For
separate cameras supply `cloud_camera_ids[C]` (strings) together with
`clouds_per_camera_m[F,C,N,3]` (NaN padding). Camera IDs `camera_A`/`camera_B`
or `synthetic_0`/`synthetic_1` map to `cloud_A`/`cloud_B`; unknown IDs remain
available in the Python data contract and are not guessed into an A/B slot.
Loading uses `allow_pickle=False`. Missing/invalid arrays fail before
replay begins; invalid fits may contain NaNs and publish diagnostics only.

Supply quality with `quality_json:=...`; otherwise the adapter checks
`reconstruction.quality.json`, then sibling `quality.json`, then an embedded
scalar JSON string `quality_json`. A list or a dict's `frames`/`per_frame` list
is indexed by frame; a summary object is retained as sequence-level quality.

Replay defaults to acquisition timestamp spacing, with `speed:=1.0` and
`loop:=false`. Absolute acquisition epochs are ignored for pacing. Current ROS
time stamps the messages; original time is preserved as `source_timestamp_s` in
JSON. Offline reconstruction may have used future frames and is for analysis
and simulation. Invalid frames never publish landmarks; marker deletion and an
empty optional cloud clear stale geometry. Markers also expire after 0.5 s.

`source_axes:=optical` is the default and maps `(x,y,z)` to `(z,-x,-y)` before
publishing: right/down/forward becomes forward/left/up. Choose
`source_axes:=ros` only when the reconstruction's common world frame already
uses ROS axes. This proper rotation affects joints, mesh, cloud, translation
and model-to-world global orientation. Articulation `theta` and shape `beta`
are unchanged. This axis convention does not estimate camera extrinsics or
publish a TF tree; all geometry is expressed directly in `frame_id`.

## Live results

The standalone `online` producer can append and flush one UTF-8 JSON object per
line to a shared local file, then ROS follows that file:

```bash
ros2 launch kinect_mano_ros kinect_mano.launch.py \
  mode:=live live_jsonl:=/absolute/path/online.jsonl
```

The core CLI's `online --output DIR` writes `DIR/live.jsonl`; point `live_jsonl`
at that file. Additional producer fields such as per-camera `clouds_m` are
allowed. The publisher uses the concatenated `cloud_m` for PointCloud2 and
`clouds_m` for the separate A/B topics using the same camera aliases as replay.
Model metadata is retained in the `mano` JSON, including a numerical fixture
label when present. Declared left-hand models are rejected by this right-hand
interface.

Each line is one frame with `timestamp_s` (scalar), `joints_m[21,3]`,
`vertices_m[778,3]`, `faces[K,3]`, `theta[45]`, `beta[10]`,
`global_orient[3]`, `translation_m[3]`, `valid`, `quality`, and optional
`cloud_m[N,3]` and integer `frame_index`. Axes/units match replay. The reader
waits for a missing file, buffers incomplete final lines, detects truncation
and replacement, rejects malformed lines, and publishes the newest complete
validated record in each read chunk. Use a new stream file for each run to
avoid old records being replayed at startup. A quality-only invalid record
still needs the fixed-size arrays (NaNs allowed when `valid` is false).

Without `live_jsonl`, live mode receives the same object as `String` JSON on
`/kinect_mano/input/result` (`live_result_topic:=...` changes it). An in-process
producer may instead call `KinectManoNode.publish_result(mapping_or_ResultFrame)`
on the ROS executor thread. The ROS adapter does not start cameras or fitting;
run the standalone online producer separately. Neither path calls a cloud API.

All live inputs (JSONL, ROS topic, and direct callbacks) must include finite
`quality.host_received_perf_counter_s` from the producer's `time.perf_counter()`
on the **same host/monotonic clock** as ROS. This measures age from camera
receipt, including preprocessing and fitting time. `fit_completed_perf_counter_s`
is preserved as a diagnostic; it does not reset the acquisition age. The default
`max_live_age_s:=0.35` rejects older results. Missing provenance, future or
incompatible clocks, and non-finite values also suppress landmarks and clear
all visual clouds/markers. They still publish current diagnostics with
`valid:false`, original `source_valid`, and `quality.live_freshness` explaining
the rejection. An external ROS producer must provide this same-host provenance;
device timestamps alone do not establish live freshness. Replay does not use
this gate and always labels itself as offline reconstruction. Compose exposes
the threshold as `KINECT_MAX_LIVE_AGE_S`.

## Select Quest or Kinect for the existing simulation

```bash
ros2 launch kinect_mano_ros hand_source.launch.py source:=kinect \
  reconstruction:=/absolute/path/reconstruction.npz
ros2 launch kinect_mano_ros hand_source.launch.py source:=quest
```

Both selections include the original `dg5f_unity_teleop/unity_dg5f.launch.py`
and reuse its retargeting and MuJoCo algorithm/configuration. Kinect explicitly
remaps its `landmarks` publisher to `/hands/right/landmarks` and sets
`start_endpoint:=false`, `start_adapter:=false`, `fake_mano:=false`. Quest
starts the original endpoint and adapter. The selector always passes
`start_lerobot:=false`, `lerobot_backend:=mock`, and `lerobot_auto_enable:=false`;
attempts to request a physical backend or start the bridge are rejected.
MuJoCo runs without a viewer by default; use `mujoco_viewer:=true` with a display.

## Docker

From the repository root:

```bash
docker compose -f compose.kinect.yaml --profile core build kinect_core
docker compose -f compose.kinect.yaml --profile core run --rm kinect_core --help
docker compose -f compose.kinect.yaml --profile ros build kinect_ros
docker compose -f compose.kinect.yaml --profile ros up kinect_ros
docker compose -f compose.kinect.yaml --profile sim up kinect_sim
```

`KINECT_MANO_DATA_DIR` selects the host directory mounted at `/data`; ROS mounts
it read-only. `KINECT_RECONSTRUCTION` defaults to `reconstruction.npz` relative
to `/data`. `MANO_MODEL_DIR` mounts licensed converted model files read-only in
the CPU core at `/models/mano`. Obtain the model separately. Core image
entrypoint is `python -m kinect_mano`, so append any supported CLI arguments.
Replay and live examples can use Compose `run --rm kinect_ros ros2 launch ...`.
Set `KINECT_ROS_MODE=live KINECT_LIVE_JSONL=/data/online.jsonl` to follow a shared
stream through Compose. `HAND_SOURCE=quest` selects Quest in the `sim` profile.

The `core` target is Python 3.10 with a modern CPU Torch stack. The `ros` target
extends local `dg5f-lerobot-retargeting:humble` (override `KINECT_ROS_BASE_IMAGE`)
and builds the ament workspace into the image. It installs the core editable
and visualization extras with `--no-deps` in a separate environment inheriting
the base stack. Build checks assert Torch/NumPy/SciPy versions remain unchanged.
It reuses the base image's actual `/ros_entrypoint.sh`; no runtime source bind
mount or startup build is required. No cameras, GPU, or physical hand are
attached by default. For host CUDA use the standalone core's isolated host
environment with the already installed compatible Torch. Optional MediaPipe
and pyk4a/SDK are separate dependencies for their respective capture paths.

The CPU and ROS images support offline fitting/replay without camera packages.
Real `capture`/`online` needs a native Azure Kinect Sensor SDK runtime and
development headers (`libk4a`, `k4a.pc`), USB permissions/udev rules, and `pyk4a`;
`online` also needs MediaPipe and a locally supplied hand-landmarker model.
Use the [Azure Kinect SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK)
packages/build instructions for the host distribution and
[pyk4a setup](https://github.com/etiennedub/pyk4a),
verify both devices with `k4aviewer`, and check `pkg-config --modversion k4a`
before installing pyk4a. Keep this setup outside the ROS/LeRobot environment:

```bash
python3 -m venv --system-site-packages /path/to/kinect-mano-native
/path/to/kinect-mano-native/bin/python -m pip install 'setuptools==79.0.1'
/path/to/kinect-mano-native/bin/python -m pip install --no-deps --no-build-isolation -e kinect_mano
# Install optional packages in this environment, retaining host Torch/NumPy.
/path/to/kinect-mano-native/bin/python -m pip install --no-deps \
  'pyk4a>=1.5,<2' 'mediapipe>=0.10.20,<0.11'
/path/to/kinect-mano-native/bin/python -c \
  'import torch, cv2, pyk4a, mediapipe; print(torch.__version__, cv2.__version__)'
```

`--no-deps` intentionally requires the optional packages' own compatible
transitive dependencies to be prepared in that isolated environment. Use
`python -m pip check` to identify them, without replacing host CUDA Torch or
the base ROS numerical stack. The CLI reports missing pyk4a/SDK rather than
starting a real camera from an unprepared image. Device passthrough and camera
permission setup are explicit deployment steps; the default Compose profiles
do not include them.

An optional test target adds the standalone tests and pytest without changing
the production core dependencies:

```bash
docker build -f docker/kinect-mano.Dockerfile --target test -t kinect-mano-test .
docker run --rm --network none kinect-mano-test
```

`KINECT_START_RVIZ=true` enables RViz with an already authorized local X display.
The profiles use localhost ROS discovery (`ROS_DOMAIN_ID` configurable), and
keep all physical robot startup disabled. No hardware verification is implied.

## Checks

```bash
PYTHONPATH=src/kinect_mano_ros python3 -m unittest discover \
  -s src/kinect_mano_ros/test -p 'test_*.py'
```

Data/scheduling/JSONL tests work without ROS. ROS transport/launch tests skip
when ROS is unavailable and run inside a built/sourced ROS environment.
