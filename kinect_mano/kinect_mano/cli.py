"""Command-line entry points. Hardware is opened only by capture/online."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from .fitting import FitConfig, ManoFitter, calibrate_shape
from .io import json_safe, load_observations, save_observations, save_reconstruction, sha256, state_mapping, write_json
from .model import ManoModel, convert_official_pickle


def _model(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available; choose --device cpu")
    return ManoModel(args.model, device=device, allow_test_model=args.allow_test_model)


def _config(args):
    return FitConfig(mode=args.mode, iterations=args.iterations, max_points_per_camera=args.points_per_camera,
                     surface_weight=args.surface_weight, temporal_weight=args.temporal_weight,
                     acceleration_weight=args.acceleration_weight)


def _shape(path, model):
    if not path:
        return np.zeros(10)
    profile = json.loads(Path(path).read_text())
    if profile.get("model_sha256") != model.sha256 or profile.get("handedness") != model.metadata["handedness"]:
        raise ValueError("Shape profile model hash/handedness does not match this model")
    beta = np.asarray(profile["beta"])
    if beta.shape != (10,) or not np.isfinite(beta).all():
        raise ValueError("Invalid shape profile beta")
    return beta


def _check_observation_model(metadata,model):
    if metadata.get("model_type") == "numerical_fixture" and model.model_type != "numerical_fixture":
        raise ValueError("Numerical fixture data cannot be labelled as real MANO input")
    if "handedness" in metadata and metadata["handedness"] != model.metadata["handedness"]:
        raise ValueError("Observation and MANO model handedness differ")


def _run_fit(args):
    from .pipeline import fit_sequence
    from .visualization import render_frame
    model, config = _model(args), _config(args)
    observations, _, metadata = load_observations(args.input)
    _check_observation_model(metadata,model)
    beta = _shape(args.shape, model)
    fitter = ManoFitter(model, config, beta)
    fingerprint = dict(input_sha256=sha256(args.input), model_sha256=model.sha256, config=asdict(config), beta=beta.tolist())
    states = fit_sequence(fitter, observations, args.output, fingerprint=fingerprint, resume=args.resume)
    # Regenerate from checked checkpoints on resume, repairing interrupted exports.
    path = save_reconstruction(args.output, states, model, config, observations=observations, manifest=fingerprint,overwrite=args.resume)
    state = next((s for s in states if s.valid), states[0])
    frame = next(i for i, s in enumerate(states) if s is state)
    render_frame(state.vertices_m, model.faces.cpu().numpy(), state.joints_m, Path(args.output)/"preview.png", views=observations[frame].views, title=f"{model.model_type} | {config.mode} | {state.status}")
    return dict(reconstruction=str(path), frames=len(states), valid_frames=sum(s.valid for s in states), model_type=model.model_type)


def _prepare(args):
    from .calibration import load_calibrations
    from .data import load_recording
    from .initializers import MediaPipeInitializer, PrecomputedJointsInitializer
    from .pipeline import RGBDProcessor
    rig = load_calibrations(args.rig)
    a = load_recording(args.camera_a,calibration=rig["camera_A"])
    b = load_recording(args.camera_b,calibration=rig["camera_B"])
    detector = MediaPipeInitializer(args.mediapipe_model, handedness=args.hand) if args.mediapipe_model else None
    precomputed = PrecomputedJointsInitializer(args.joints, tolerance_s=args.tolerance) if args.joints else None
    try:
        processor = RGBDProcessor(rig, detector=detector, precomputed=precomputed, allow_polygon_roi=args.allow_polygon_roi, points_per_camera=args.points_per_camera)
        observations, synchronization = processor.process_recordings(a, b, tolerance_s=args.tolerance, offset_b_s=args.offset_b)
        save_observations(args.output, observations, metadata=dict(source="recorded_rgbd", rig_sha256=sha256(args.rig), polygon_roi=args.allow_polygon_roi,handedness=args.hand))
        write_json(str(args.output)+".sync.json", synchronization)
    finally:
        if detector:
            detector.close()
    return dict(observations=args.output, matched_pairs=len(observations), unmatched_a=len(a.frames)-len(observations), unmatched_b=len(b.frames)-len(observations))


def _calibrate(args):
    from .calibration import ChessboardSpec, calibrate_chessboard, save_calibrations
    from .data import load_recording
    if Path(args.output).exists():
        raise FileExistsError(f"Refusing to overwrite rig: {args.output}")
    a, b = load_recording(args.camera_a, allow_uncalibrated=True), load_recording(args.camera_b, allow_uncalibrated=True)
    result = calibrate_chessboard(a.frames, b.frames, ChessboardSpec(args.columns, args.rows, args.square_size),
        intrinsics_a=a.calibration.intrinsics, distortion_a=a.calibration.distortion,
        intrinsics_b=b.calibration.intrinsics, distortion_b=b.calibration.distortion,
        T_world_from_camera_a=np.eye(4), tolerance_s=args.tolerance, offset_b_s=args.offset_b,
        min_pairs=args.min_pairs, max_rms_px=args.max_rms)
    metadata = result.metadata() | dict(world_frame="camera_A_color", camera_a_sha256=sha256(args.camera_a), camera_b_sha256=sha256(args.camera_b), columns=args.columns, rows=args.rows, square_size_m=args.square_size, offset_b_s=args.offset_b, tolerance_s=args.tolerance, serials={"camera_A":a.metadata.get("serial"),"camera_B":b.metadata.get("serial")})
    save_calibrations(args.output, [result.calibration_a, result.calibration_b], metadata=metadata)
    return dict(rig=args.output, **metadata)


def _check_alignment(args):
    from scipy.spatial import cKDTree
    from .calibration import load_calibrations
    from .data import load_recording, match_frames_by_timestamp
    from .geometry import frame_point_cloud
    from .pipeline import filtered_cloud
    from .visualization import render_frame
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rig = load_calibrations(args.rig)
    a = load_recording(args.camera_a,calibration=rig["camera_A"])
    b = load_recording(args.camera_b,calibration=rig["camera_B"])
    pairs = match_frames_by_timestamp(a.frames, b.frames, tolerance_s=args.tolerance, offset_b_s=args.offset_b)
    if not 0 <= args.frame < len(pairs):
        raise ValueError("Alignment frame out of range; no suitable synchronized pair")
    clouds = [filtered_cloud(frame_point_cloud(f, rig[f.camera_id], require_mask=False, min_depth_m=args.min_depth, max_depth_m=args.max_depth), args.points_per_camera) for f in pairs[args.frame]]
    if any(len(c) < 3 for c in clouds):
        raise ValueError("Too few depth points in static alignment scene")
    distances = [cKDTree(clouds[1-i]).query(clouds[i])[0] for i in (0, 1)]
    report = dict(frame=args.frame, points=[len(c) for c in clouds], nearest_neighbor_median_mm=[float(np.median(d))*1000 for d in distances], nearest_neighbor_p90_mm=[float(np.percentile(d,90))*1000 for d in distances], note="Static-scene consistency residual; occlusion and non-overlap affect this metric, not independent calibration accuracy")
    directory = Path(args.output)
    write_json(directory/"alignment.json", report)
    figure = plt.figure(figsize=(9,7))
    ax = figure.add_subplot(111, projection="3d")
    for cloud, label in zip(clouds, ("A", "B")):
        ax.scatter(*cloud.T, s=1, label=label)
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="Static two-camera alignment")
    ax.legend()
    figure.savefig(directory/"alignment.png", dpi=140)
    plt.close(figure)
    return report


def _capture_or_online(args):
    from .calibration import load_calibrations
    from .capture import AzureKinectCamera, DualKinectCapture, record_dual_cameras
    rig = load_calibrations(args.rig) if args.rig else None
    serials = json.loads(Path(args.rig).read_text()).get("metadata",{}).get("serials",{}) if args.rig else {}
    calibration_only = bool(getattr(args, "calibration_only", False))
    if rig is None and not calibration_only:
        raise ValueError("Provide --rig, or explicitly --calibration-only when recording the calibration board")
    if args.frames < 0 or (args.command == "capture" and args.frames == 0):
        raise ValueError("capture frames must be positive; online frames may be 0 for continuous capture")
    if calibration_only and rig is not None:
        raise ValueError("Choose --calibration-only or --rig, not both")
    cameras = []
    for index, camera_id in enumerate(("camera_A", "camera_B")):
        calibration = rig[camera_id] if rig else None
        cameras.append(AzureKinectCamera(camera_id, args.device_a if index == 0 else args.device_b,
            T_world_from_camera=None if calibration is None else calibration.T_world_from_camera,
            calibration_only=calibration_only, sync_mode=("master" if index == 0 else "subordinate") if args.sync == "wired" else "standalone",
            fps=args.fps, exposure_usec=args.exposure_usec, depth_delay_off_color_usec=0 if index == 0 else args.depth_delay_b,
            expected_serial=(args.serial_a if index == 0 else args.serial_b) or serials.get(camera_id)))
    if args.command == "capture":
        paths = record_dual_cameras(*cameras, args.output, frame_count=args.frames, tolerance_s=args.tolerance, offset_b_s=args.offset_b)
        return dict(recordings=[str(p) for p in paths], frames=args.frames, calibration_only=calibration_only)
    # Check ALL local inference resources before opening any camera.
    from .initializers import MediaPipeInitializer
    from .pipeline import RGBDProcessor, LatestPairSource
    model, config = _model(args), _config(args)
    if model.metadata["handedness"] != args.hand:
        raise ValueError("Requested hand and MANO model handedness differ")
    detector = MediaPipeInitializer(args.mediapipe_model, handedness=args.hand)
    fitter = ManoFitter(model, config, _shape(args.shape, model))
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    stream_path = directory/"live.jsonl"
    if stream_path.exists():
        detector.close()
        raise FileExistsError(f"Choose a new online output directory: {stream_path}")
    count, valid_count, sequence = 0, 0, 0
    try:
        with DualKinectCapture(*cameras) as capture:
            # Use factory color intrinsics of the active mode, and only take the
            # verified world extrinsics from the rig. Never reuse stale K blindly.
            active = {c.camera_id: c.calibration for c in cameras}
            for camera_id in active:
                reference = rig[camera_id]
                actual = active[camera_id]
                if (actual.width, actual.height) != (reference.width, reference.height) or not np.allclose(actual.intrinsics, reference.intrinsics, rtol=1e-4, atol=1e-3):
                    raise ValueError(f"{camera_id} factory intrinsics/mode do not match the calibrated rig; recalibrate in this capture mode")
            processor = RGBDProcessor(active, detector=detector, allow_polygon_roi=args.allow_polygon_roi, points_per_camera=args.points_per_camera)
            write_json(directory/"manifest.json", dict(model=model.metadata, model_sha256=model.sha256, config=asdict(config), rig_sha256=sha256(args.rig), captures=[camera.metadata for camera in cameras]))
            source = LatestPairSource(capture, tolerance_s=args.tolerance, offset_b_s=args.offset_b).start()
            try:
                with stream_path.open("x", encoding="utf-8") as stream:
                    while not args.frames or count < args.frames:
                        next_sequence, (a,b), captured_at = source.next(sequence)
                        skipped = max(0, next_sequence-sequence-1)
                        sequence = next_sequence
                        observation = processor.process_pair(a,b, fitter.history[-1] if fitter.history else None)
                        state = fitter.fit(observation)
                        state.quality.update(preprocessing_ms=observation.preprocessing_ms, segmentation=observation.segmentation,
                                             host_receive_to_fit_ms=(perf_counter()-captured_at)*1000, dropped_pairs_since_fit=skipped,
                                             host_received_perf_counter_s=captured_at, fit_completed_perf_counter_s=perf_counter(),
                                             sdk_system_timestamp_a_nsec=a.metadata.get("depth_system_timestamp_nsec"),
                                             sdk_system_timestamp_b_nsec=b.metadata.get("depth_system_timestamp_nsec"),
                                             timestamp_a_s=a.timestamp_s,timestamp_b_s=b.timestamp_s)
                        stream.write(json.dumps(state_mapping(state,model.faces.cpu().numpy(),frame_index=count,observation=observation,metadata=model.metadata),allow_nan=False)+"\n")
                        stream.flush()
                        valid_count += int(state.valid)
                        count += 1
            except KeyboardInterrupt:
                pass
            finally:
                source.close()
    finally:
        detector.close()
    return dict(stream=str(stream_path),frames=count,valid_frames=valid_count)


def build_parser():
    parser = argparse.ArgumentParser(description="Two calibrated Azure Kinect RGB-D views -> MANO parameters; physical DG5F is never opened")
    parser.add_argument("--threads", type=int, default=1, help="Torch CPU threads (small meshes benefit from 1)")
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert-model", help="Convert a trusted official MANO .pkl to metric NPZ")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--hand", choices=["right", "left"], default="right")
    convert.add_argument("--unit-scale", type=float, default=1., help="Official MANO uses metres; use 0.001 only for known mm exports")
    demo = commands.add_parser("demo", help="Generate explicitly labelled numerical fixture, not licensed MANO")
    demo.add_argument("--output", required=True)
    demo.add_argument("--frames", type=int, default=8)
    demo.add_argument("--points-per-camera", type=int, default=128)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--model", help="Use supplied official MANO instead of the numerical fixture")
    demo.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    demo.add_argument("--allow-test-model", action="store_true")
    demo.add_argument("--run", action="store_true", help="Also benchmark all three modes")
    demo.add_argument("--iterations", type=int, default=60)

    def optimizer_arguments(command):
        command.add_argument("--model", required=True)
        command.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
        command.add_argument("--allow-test-model", action="store_true")
        command.add_argument("--shape")
        command.add_argument("--mode", choices=["joint", "hybrid", "temporal"], default="hybrid")
        command.add_argument("--iterations", type=int, default=80)
        command.add_argument("--points-per-camera", type=int, default=2048)
        command.add_argument("--surface-weight", type=float, default=1.)
        command.add_argument("--temporal-weight", type=float, default=.02)
        command.add_argument("--acceleration-weight", type=float, default=.002)

    for name in ("fit", "benchmark", "calibrate-shape"):
        command = commands.add_parser(name)
        optimizer_arguments(command)
        command.add_argument("--input", required=True)
        command.add_argument("--output", required=True)
        if name == "fit":
            command.add_argument("--resume", action="store_true")
        if name == "calibrate-shape":
            command.add_argument("--max-frames", type=int, default=12)

    def paired_files(command):
        command.add_argument("--camera-a", required=True)
        command.add_argument("--camera-b", required=True)
        command.add_argument("--tolerance", type=float, default=.005)
        command.add_argument("--offset-b", type=float, default=0., help="Compare t_A with t_B + offset seconds")
        command.add_argument("--output", required=True)

    prepare = commands.add_parser("prepare", help="Saved RGB-D + masks/local detector -> fitting observations")
    paired_files(prepare)
    prepare.add_argument("--rig", required=True)
    prepare.add_argument("--mediapipe-model")
    prepare.add_argument("--joints", help="Precomputed metric joints with timestamps")
    prepare.add_argument("--hand", choices=["right", "left"], default="right")
    prepare.add_argument("--allow-polygon-roi", action="store_true")
    prepare.add_argument("--points-per-camera", type=int, default=2048)
    calibrate = commands.add_parser("calibrate", help="Paired chessboard recording -> verified external rig")
    paired_files(calibrate)
    calibrate.add_argument("--columns", type=int, required=True, help="Inner corner columns")
    calibrate.add_argument("--rows", type=int, required=True, help="Inner corner rows")
    calibrate.add_argument("--square-size", type=float, required=True, help="Measured square edge, metres")
    calibrate.add_argument("--min-pairs", type=int, default=8)
    calibrate.add_argument("--max-rms", type=float, default=2.)
    check = commands.add_parser("check-calibration", help="Static two-view cloud consistency + PNG")
    paired_files(check)
    check.add_argument("--rig", required=True)
    check.add_argument("--frame", type=int, default=0)
    check.add_argument("--min-depth", type=float, default=.2)
    check.add_argument("--max-depth", type=float, default=1.5)
    check.add_argument("--points-per-camera", type=int, default=4096)

    for name in ("capture", "online"):
        command = commands.add_parser(name, help="Open two real Azure Kinect devices" if name == "capture" else "Live RGB-D -> MANO parameters and optional ROS JSONL stream")
        command.add_argument("--output", required=True)
        command.add_argument("--rig", required=name == "online")
        command.add_argument("--device-a", type=int, default=0)
        command.add_argument("--device-b", type=int, default=1)
        command.add_argument("--sync", choices=["standalone", "wired"], default="standalone")
        command.add_argument("--frames", type=int, default=120 if name == "capture" else 0, help="0 = until Ctrl+C for online")
        command.add_argument("--tolerance", type=float, default=.005)
        command.add_argument("--offset-b", type=float, default=0.)
        command.add_argument("--fps", type=int, choices=[5,15,30], default=30)
        command.add_argument("--exposure-usec", type=int, default=8000)
        command.add_argument("--depth-delay-b", type=int, default=160, help="Active depth staggering in microseconds")
        command.add_argument("--serial-a")
        command.add_argument("--serial-b")
        if name == "capture":
            command.add_argument("--calibration-only", action="store_true")
        else:
            optimizer_arguments(command)
            command.set_defaults(iterations=10,mode="temporal")
            command.add_argument("--mediapipe-model", required=True)
            command.add_argument("--allow-polygon-roi", action="store_true")
            command.add_argument("--hand", choices=["right", "left"], default="right")
    view = commands.add_parser("view", help="Export reconstruction/clouds/joints PNG without display")
    view.add_argument("--input", required=True)
    view.add_argument("--output", required=True)
    view.add_argument("--frame", type=int, default=0)
    inspect = commands.add_parser("inspect", help="Inspect a numeric NPZ safely")
    inspect.add_argument("--input", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        if args.command == "convert-model":
            convert_official_pickle(args.input, args.output, args.hand, args.unit_scale)
            result = dict(model=args.output)
        elif args.command == "demo":
            from .synthetic import make_fixture_model, generate_sequence
            directory = Path(args.output)
            directory.mkdir(parents=True, exist_ok=True)
            if args.model:
                model = _model(args)
            else:
                model_path = directory/"numerical_fixture.npz"
                if model_path.exists():
                    raise FileExistsError(model_path)
                arrays = make_fixture_model(seed=args.seed)
                with model_path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                args.model, args.allow_test_model = str(model_path), True
                model = _model(args)
            observations, truth = generate_sequence(model, frames=args.frames, points_per_camera=args.points_per_camera, seed=args.seed)
            input_path = directory/"observations.npz"
            save_observations(input_path, observations, truth=truth, metadata=dict(model_type=model.model_type, seed=args.seed,handedness=model.metadata["handedness"]))
            from .visualization import render_frame
            render_frame(truth["vertices_m"][0], model.faces.cpu().numpy(), truth["joints_m"][0], directory/"input.png", views=observations[0].views, title=f"Synthetic data | {model.model_type}")
            result = dict(model=args.model, observations=str(input_path), model_type=model.model_type)
            if args.run:
                from .benchmark import run_benchmark
                result["benchmark"] = run_benchmark(model, observations, truth, directory/"benchmark", config=FitConfig(iterations=args.iterations, max_points_per_camera=args.points_per_camera))
        elif args.command == "fit":
            result = _run_fit(args)
        elif args.command == "benchmark":
            from .benchmark import run_benchmark
            model = _model(args)
            observations, truth, metadata = load_observations(args.input)
            _check_observation_model(metadata,model)
            result = run_benchmark(model, observations, truth, args.output, config=_config(args), beta=_shape(args.shape, model), manifest=metadata)
        elif args.command == "calibrate-shape":
            model = _model(args)
            observations, _, metadata = load_observations(args.input)
            _check_observation_model(metadata,model)
            if args.max_frames < 1:
                raise ValueError("max-frames must be positive")
            selected = [observations[i] for i in np.linspace(0,len(observations)-1,min(args.max_frames,len(observations))).astype(int)]
            beta, diagnostics = calibrate_shape(model, selected, iterations=args.iterations, mode=args.mode)
            result = dict(beta=beta, model_sha256=model.sha256, handedness=model.metadata["handedness"], diagnostics=diagnostics)
            if Path(args.output).exists():
                raise FileExistsError(args.output)
            write_json(args.output, result)
        elif args.command == "prepare":
            result = _prepare(args)
        elif args.command == "calibrate":
            result = _calibrate(args)
        elif args.command == "check-calibration":
            result = _check_alignment(args)
        elif args.command in ("capture", "online"):
            result = _capture_or_online(args)
        elif args.command == "view":
            from .visualization import render_reconstruction
            result = dict(image=str(render_reconstruction(args.input,args.output,args.frame)))
        elif args.command == "inspect":
            with np.load(args.input, allow_pickle=False) as archive:
                result = {key: dict(shape=list(archive[key].shape), dtype=str(archive[key].dtype)) for key in archive.files}
        else:
            raise ValueError("Unknown command")
        print(json.dumps(json_safe(result), indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        parser.exit(2, f"kinect-mano: {error}\n")
