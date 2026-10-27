"""Reproducible A/B/C benchmark, never labelled as real Kinect accuracy."""
from dataclasses import replace
from pathlib import Path
import platform
import resource
from time import perf_counter

import numpy as np

from .fitting import FitConfig, ManoFitter
from .io import save_reconstruction, write_json
from .surface import point_triangle_squared_distance, raycast_front_faces


def metrics(states, truth, observations, model):
    import torch
    joints = np.stack([state.joints_m for state in states])
    times = np.array([state.timestamp_s for state in states])
    result = dict(frames=len(states), valid_fraction=float(np.mean([state.valid for state in states])), fit_ms_mean=float(np.mean([state.quality["fit_ms"] for state in states])))
    result["first_frame_fit_ms"] = states[0].quality["fit_ms"]
    result["steady_fit_ms_mean"] = float(np.mean([s.quality["fit_ms"] for s in states[1:]])) if len(states)>1 else None
    if "joints_m" in truth:
        distance = np.linalg.norm(joints - truth["joints_m"], axis=-1) * 1000
        result.update(joint_error_mm=float(distance.mean()), fingertip_error_mm=float(distance[:, [4, 8, 12, 16, 20]].mean()), per_frame_joint_error_mm=distance.mean(1).tolist())
    if len(states) > 2:
        dt = np.diff(times)
        velocity = np.diff(joints, axis=0) / dt[:, None, None]
        acceleration = np.diff(velocity, axis=0) / ((dt[1:] + dt[:-1]) * .5)[:, None, None]
        result["acceleration_rms_m_s2"] = float(np.sqrt(np.mean(acceleration ** 2)))
        if "joints_m" in truth:
            residual = joints - truth["joints_m"]
            result["jitter_error_mm"] = float(np.std(residual, axis=0).mean() * 1000)
    surface_distances, coverage = [], []
    with torch.no_grad():
        for state, observation in zip(states, observations):
            vertices = torch.as_tensor(state.vertices_m, device=model.device, dtype=model.dtype)
            for view in observation.views:
                points = torch.as_tensor(view.points_m, device=model.device, dtype=model.dtype)
                center = torch.as_tensor(view.camera_center_m, device=model.device, dtype=model.dtype)
                indices = raycast_front_faces(points, vertices, model.faces, center)
                valid = indices >= 0
                if len(valid):
                    coverage.append(float(valid.float().mean()))
                if valid.any():
                    distances = point_triangle_squared_distance(points[valid], vertices[model.faces[indices[valid]]]).sqrt()
                    surface_distances.append(float(distances.mean()) * 1000)
    result["surface_residual_mm"] = float(np.mean(surface_distances)) if surface_distances else None
    result["surface_ray_coverage"] = float(np.mean(coverage)) if coverage else 0.
    return result


def run_benchmark(model, observations, truth, output, *, config=None, beta=None, manifest=None):
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode in ("joint", "hybrid", "temporal"):
        actual = replace(config or FitConfig(), mode=mode)
        # Equal warm-up for every mode, outside measured sequence; includes lazy
        # optimizer imports and kernel initialization, with no state leakage.
        warm_start = perf_counter()
        ManoFitter(model,actual,beta=beta).fit(observations[0])
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
            torch.cuda.reset_peak_memory_stats(model.device)
        warmup_ms = (perf_counter()-warm_start)*1000
        fitter = ManoFitter(model, actual, beta=beta)
        start = perf_counter()
        states = [fitter.fit(observation) for observation in observations]
        seconds = perf_counter() - start
        results[mode] = metrics(states, truth, observations, model)
        results[mode].update(total_s=seconds, throughput_fps=len(states)/seconds, config=actual.__dict__,warmup_ms=warmup_ms,
                             process_rss_highwater_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                             peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(model.device)/1024**2 if model.device.type=="cuda" else None,
                             peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(model.device)/1024**2 if model.device.type=="cuda" else None)
        save_reconstruction(output / mode, states, model, actual, observations=observations, manifest=manifest)
    report = dict(model_type=model.model_type, model_sha256=model.sha256, has_independent_real_ground_truth=False,
                  environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=str(model.device),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(model.device) if model.device.type=="cuda" else None,threads=torch.get_num_threads()),
                  note="Numerical fixture is not MANO; synthetic MANO tests are not real Kinect trials. Surface residual is fitting error, not independent accuracy. Throughput measures fitting only after per-mode warmup; excludes loading/detection/ROS. RSS is cumulative process high-water, not incremental per-mode memory.", results=results)
    write_json(output / "benchmark.json", report)
    _plot(results, output / "benchmark.png")
    return report


def _plot(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode, result in results.items():
        if "per_frame_joint_error_mm" in result:
            axes[0].plot(result["per_frame_joint_error_mm"], label=mode)
    axes[0].set(xlabel="Frame", ylabel="Synthetic joint error [mm]")
    if axes[0].lines:
        axes[0].legend()
    axes[1].bar(list(results), [r["throughput_fps"] for r in results.values()])
    axes[1].set(ylabel="Fit throughput [frames/s]")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
