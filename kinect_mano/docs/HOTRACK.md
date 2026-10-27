# HOTrack audit and external joint initializer

## Implementation status

Implemented in [`hotrack.py`](../kinect_mano/hotrack.py): a generic, local-only
NPZ/process adapter with `HOTrackInitializer.estimate_joints(observation,
previous_state=None)` and `load_joint_estimate(path, frame_index=...)`. It returns
the shared `data.JointEstimate(joints_m, confidence, valid_mask, valid)`.

**Official HOTrack inference integration is not implemented.** The module logs
this distinction when configured and exposes `OFFICIAL_INFERENCE_IMPLEMENTED =
False`. Supplying a checkpoint does not create an inference implementation.
The user must supply an executable that actually loads the desired model,
prepares its inputs and converts its results into the protocol below.
No neural model was trained or evaluated here. No dataset or checkpoint was
downloaded, CUDA extension modified, or fitting algorithm added to this adapter.
MANO fitting and acceptance/reinitialization decisions belong to the main pipeline.

## Official sources audited

Audit: **2026-10-27 Europe/Moscow** (2026-10-26 UTC). Upstream `main` resolved
through `git ls-remote` to **`476b726bb6be681739429fc42f3b1fb27351996e`**.
These links pin the inspected code, rather than assuming a released Python API:

- [Official README](https://github.com/PKU-EPIC/HOTrack/blob/476b726bb6be681739429fc42f3b1fb27351996e/README.md):
  upstream reports testing Ubuntu 20.04, CUDA 11.7, Python 3.8 and PyTorch 1.9.1;
  explicitly requires **PyTorch <1.11** for its PointNet extension. Its example
  wheels use `torch==1.9.1+cu111`, `torchvision==0.10.1+cu111`, and
  `torchaudio==0.9.1`; that cu111 example and the reported CUDA 11.7 environment
  are distinct upstream statements, not a compatibility test performed here.
  Upstream documents MANO assets, dataset preparation, a pretrained archive, and
  dataset-specific evaluation. Its stated license for the work/dataset is
  **CC BY-NC 4.0**; check upstream and the separate MANO asset terms before use.
- [Exact requirements](https://github.com/PKU-EPIC/HOTrack/blob/476b726bb6be681739429fc42f3b1fb27351996e/requirements.txt):
  `tqdm==4.64.1`, `matplotlib==3.6.2`, `chumpy==0.70`,
  `opencv-python==4.6.0.66`, `pyyaml==6.0`, `transforms3d==0.4.1`,
  `trimesh==3.17.1`, `open3d==0.16.0`, `numpy==1.20.3`,
  `plyfile==0.7.4`, `scikit-image==0.19.3`, `tensorboardx==2.5.1`.
  Torch is specified by the README, not pinned in this requirements file.
- [PointNet build](https://github.com/PKU-EPIC/HOTrack/blob/476b726bb6be681739429fc42f3b1fb27351996e/network/models/pointnet_lib/setup.py)
  compiles `pointnet2_cuda` with PyTorch's CUDAExtension. Keep the upstream
  environment separate from the modern fitting environment. Installing its
  pinned NumPy/torch into the fitter would conflict with this package's versions.
- [`network/test.py`](https://github.com/PKU-EPIC/HOTrack/blob/476b726bb6be681739429fc42f3b1fb27351996e/network/test.py)
  constructs a configured dataset/dataloader, creates `Trainer`, resumes weights
  and tests trajectories. It does not implement this adapter's NPZ protocol.
  [`trainer.py`](https://github.com/PKU-EPIC/HOTrack/blob/476b726bb6be681739429fc42f3b1fb27351996e/network/trainer.py)
  loads experiment checkpoints containing a `model` state dictionary.
- [HO3D evaluation config](https://github.com/PKU-EPIC/HOTrack/blob/476b726bb6be681739429fc42f3b1fb27351996e/configs/all_config/handtracknet_test_HO3D.yml)
  references `pretrained_handtracknet_SimGrasp`, uses 512 points, batch size 1,
  tracking mode and a joint-derived hand frame. These settings are evidence of
  upstream expectations, not an automatically executed Kinect recipe.

## Mirror availability and archive size

The [official mirror](https://mirrors.pku.edu.cn/dl-release/HOTrack_AAAI2023/)
was **actually probed using HTTP HEAD requests**, including each archive URL.
The first parallel Python probe began at **2026-10-26T22:19:18 UTC**.

| Resource | Observed result | Size verified from headers |
| --- | --- | --- |
| Directory index | HTTPS TLS handshake timeout | Not available |
| `pretrained_models.zip` | HTTPS TLS handshake timeout; repeated IPv4 curl probe also timed out | **Unknown: no Content-Length or Content-Range received** |
| `SimGrasp.zip` | HTTPS TLS handshake timeout | Unknown |
| `SimGrasp_rawimg.zip` | HTTPS TLS handshake timeout | Unknown |

The resolved IPv4 was `162.105.120.100`. Curl exited 28 and printed HTTP code
`000` for HTTPS; **000 is curl's lack-of-response marker, not a server status**.
An HTTP/port-80 HEAD request to the pretrained archive also timed out without
headers. An IPv6 attempt exited 7 (could not connect). Browser retrieval failed
too. These observations do not establish a 404, a healthy 200, or global mirror
unavailability. The README links the archive, but **its current availability,
byte size, contents and checkpoint integrity remain unverified from this host**.
Do not invent an archive size or treat dataset sizes as the weights size.

Recheck from a host that can reach the mirror, without downloading bodies:

```bash
curl --head --location --connect-timeout 10 --max-time 30 \
  --write-out '\nHTTP=%{http_code} downloaded=%{size_download}\n' \
  'https://mirrors.pku.edu.cn/dl-release/HOTrack_AAAI2023/pretrained_models.zip'
```

Record the final HTTP status, `Content-Length`, date and any redirects. A length
on an error page is not the archive length. If a successful response omits the
length, size remains unknown; do not start an unrestricted dataset download.
The adapter never contacts this mirror and never automatically retrieves weights.

## Units, initialization and domain limits

[`HandTrackNet.forward`](https://github.com/PKU-EPIC/HOTrack/blob/476b726bb6be681739429fc42f3b1fb27351996e/network/models/hand_network.py)
consumes `hand_points` and `jittered_hand_kp`; the tracking path also supplies a
palm template. Its joint-derived frame uses a 0.2 m canonical scale.
[`canonicalize` / `decanonicalize`](https://github.com/PKU-EPIC/HOTrack/blob/476b726bb6be681739429fc42f3b1fb27351996e/network/models/hand_utils.py)
apply rotation/translation and divide/multiply by this scale. This is internal
normalization, not permission to hand the worker millimetres or multiply its
camera-space joint output by five.

[`track_network.py`](https://github.com/PKU-EPIC/HOTrack/blob/476b726bb6be681739429fc42f3b1fb27351996e/network/models/track_network.py)
uses the previous predicted joints relative to the previous cloud centroid,
then adds the current cloud centroid. At the first frame the dataset supplies
the joint guess. An independent first-frame initialization and reset policy
are therefore still required for a Kinect wrapper. The generic adapter merely
passes previous state; it does not reproduce upstream's temporal transport.

The [HO3D loader](https://github.com/PKU-EPIC/HOTrack/blob/476b726bb6be681739429fc42f3b1fb27351996e/datasets/HO3D_dataset.py)
uses annotated joints to crop the hand within 0.15 m and constructs jittered
initial joints, then samples points. The
[SimGrasp loader](https://github.com/PKU-EPIC/HOTrack/blob/476b726bb6be681739429fc42f3b1fb27351996e/datasets/SimGrasp_dataset.py)
uses labels, MANO annotations and jittered initialization. The
[DexYCB loader](https://github.com/PKU-EPIC/HOTrack/blob/476b726bb6be681739429fc42f3b1fb27351996e/datasets/DexYCB_dataset.py)
divides depth and Manopth output by 1000. Upstream data preprocessing cannot
simply be run on an unsegmented Kinect scene without replacing these inputs.

The [upstream MANO layer](https://github.com/PKU-EPIC/HOTrack/blob/476b726bb6be681739429fc42f3b1fb27351996e/third_party/mano/our_mano.py)
reorders 21 landmarks into wrist followed by thumb/index/middle/ring/pinky,
each from base to tip. The adapter requires that same output order in **world
metres**. A wrapper must explicitly handle sensor depth scaling, camera/world
transforms, handedness and any dataset axis conversions. The adapter does not
infer units, flip axes, recenter, sample 512 points or rescale the hand.

Inference from this preprocessing: synthetic-to-Kinect noise, incomplete masks,
occlusion, fused multi-view sampling and left hands are domain changes requiring
validation. No Kinect accuracy or confidence calibration is claimed. Default
confidence of 1 means unspecified weight, not a calibrated probability.

## Adapter protocol (implemented)

All archives are opened with `allow_pickle=False`. Protocol fields containing
Python object arrays, complex/string coordinates, malformed shapes or invalid
metadata are rejected. Unrecognized archive keys are ignored.
Raw upstream `.pkl`, `.npy` dictionaries and checkpoint files are not joint NPZs.

### Precomputed joints

```python
import numpy as np
from kinect_mano.hotrack import HOTrackInitializer

# joints_world_m must already be ordered and transformed into world metres.
np.savez("hand_joints.npz", joints_m=joints_world_m, confidence=joint_weights)
initializer = HOTrackInitializer(joints_npz="hand_joints.npz")
estimate = initializer.estimate_joints(None)
```

Required: `joints_m[21,3]`. Optional: `confidence[21]` in [0,1] (default 1),
`valid_mask[21]` (default `confidence > 0`), scalar `valid` (default any valid
joint), and scalar integer `schema_version=1`. Boolean fields must be booleans
or integer 0/1. Valid joints require finite coordinates and positive confidence.
Missing joints may contain NaN only where their mask is false; returned missing
coordinates are NaN and their confidence is zero, per `data.JointEstimate`.
Infinity is rejected everywhere. `valid=True` requires at least one valid joint.
An explicit `valid=False` is preserved even if some individual joints are usable.

For sequences use `joints_m[T,21,3]` with optional `confidence[T,21]`,
`valid_mask[T,21]`, `valid[T]`. Every frame is checked when configuring the
initializer. No broadcast or hidden frame counter is used. Select an exact row
with `estimate_joints({"frame_index": i})`; an object with `.frame_index` also
works. A sequence, including T=1, requires a nonnegative integer index; an
out-of-range index raises. A single-frame archive is reusable and does not use
the index to select a sequence. Cache results are copied before returning them.
There is no timestamp matching; the caller must align sequence rows to frames.

### User-supplied external command

```python
initializer = HOTrackInitializer(
    command=["/opt/hotrack-env/bin/python", "/local/my_worker.py",
             "--input", "{input}", "--output", "{output}",
             "--checkpoint", "{weights}"],
    weights_path="/local/checkpoints/model.pt",
    cwd="/local/HOTrack", timeout_s=120,
)
estimate = initializer.estimate_joints(
    {"points_m": segmented_world_cloud_m, "frame_index": i},
    previous_state=previous_estimate,  # or a finite joints ndarray [21,3]
)
```

`my_worker.py` is **user code, not a provided official command**. A direct cloud
ndarray [N,3], or an object with `.points_m`, is also accepted. The cloud must be
finite and nonempty. This standalone contract intentionally leaves construction
of initializer observations to the caller; fitting observations are separate.
Previous state may be a joint ndarray or mapping/object with `joints_m` and
optional `confidence`, `valid_mask`, `valid`; a fitting state with `.joints_m`
therefore works. The caller manages resets across time gaps or failed tracks.

Input NPZ always contains scalar integer `schema_version=1` and `points_m[N,3]`.
If supplied, it also contains scalar integer `frame_index`, and the four
`previous_joints_m[21,3]`, `previous_confidence[21]`,
`previous_valid_mask[21]`, `previous_valid` fields. Missing previous joints retain
their NaN/zero/false representation. Omitted previous state adds no such fields.
The worker must validate its inputs and model requirements before inference.

Output NPZ must contain **one frame** using the precomputed single-frame schema
above. The worker must write exactly the provided output path, finish writing
it, and exit zero. Returning a sequence is an error. A normal missed detection
can be encoded with all-NaN joints, zero confidences and scalar `valid=False`.

Commands are argv lists executed with `shell=False`, no stdin, a fresh temporary
input/output directory per call, and a finite timeout. `{input}`, `{output}` and
optional `{weights}` are literal substitutions; Python/JSON braces in other
arguments are preserved. Use an absolute interpreter/executable path for a
separate environment. Wrappers should execute their inference process directly;
daemon/background workers are outside this per-call lifecycle.
Checkpoint paths are resolved before changing cwd and checked as nonempty files
at configuration and before each call. Their format is the worker's responsibility.
Precomputed mode needs no weights. A generic command may omit weights if it does
not use them; requesting `{weights}` without `weights_path` is a configuration error.

Bad local input/configuration raises `ValueError`, missing resources raise
`FileNotFoundError`, and sequence bounds raise `IndexError`. Process startup,
timeout, nonzero exit, absent output and invalid output raise
`HOTrackProcessError`. A nonzero exit includes at most the last 4096 stderr
bytes. Logs are directed to temporary files to avoid unbounded memory capture;
temporary exchange files are cleaned on success and failure. There is no silent
previous-state fallback or fabricated detection after an error. The caller
decides how to record a failed initializer and whether fitting should continue.

## Validation

The **21 contract tests passed** with **Python 3.10.12 and NumPy 2.2.6**.
They use NumPy and the Python standard library only; no PyTorch/CUDA is imported
by this adapter. Both Python files also passed `py_compile`.

```bash
PYTHONPATH=kinect_mano python3 -m unittest discover \
  -s kinect_mano/tests -p test_hotrack.py -v
```

Tests cover precomputed selection/copying, complete-sequence validation,
NaN/mask/confidence semantics, schema errors, pickle/NPY rejection, missing/empty
weights, real subprocess exchange of cloud/prior data, cwd and paths with spaces,
bounded error text, missing executables, timeout, nonzero exits and absent or
malformed outputs. This validates the generic adapter; it does not validate the
official GPU model or its pretrained checkpoints.
