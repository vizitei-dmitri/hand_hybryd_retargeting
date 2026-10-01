"""Verify E2 A/B initial policies, then 100 continuation updates each, sequentially."""
import fcntl
import json
import os
from pathlib import Path
import sys
import traceback

import torch

import overnight_process as process
sys.path.insert(0, str(Path(__file__).resolve().parent / "rsl_rl"))
from resume_verification import assert_same_state

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "logs/rsl_rl/dg5f_cube_direct"
OUT = ROOT / "logs/e2_std_ab_smoke"
TASK = "DG5F-Cube-Stream-Direct-v0"
VARIANTS = {
    "A": (RUNS / "2026-10-01_e2_std05_preserved/model_3599.pt", .001, .5),
    "B": (RUNS / "2026-10-01_14-39-27_e2_continue_3600/model_3599.pt", .005, 5.121748924255371),
}


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    lease = (ROOT / "logs/overnight_ab/driver.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    # The pre-existing desktop Telegram process uses CUDA but is not a simulation.
    # Exempt only this exact executable/PID while its GPU allocation stays <=128 MiB.
    gpu_processes = process.gpu_processes
    desktop = {}
    for pid, memory in gpu_processes().items():
        executable = Path(f"/proc/{pid}/exe").resolve()
        if executable.name == "Telegram" and memory.isdigit() and int(memory) <= 128:
            desktop[pid] = executable
    def checked_gpu_processes():
        return {pid: memory for pid, memory in gpu_processes().items()
                if not (pid in desktop and Path(f"/proc/{pid}/exe").resolve() == desktop[pid]
                        and memory.isdigit() and int(memory) <= 128)}
    process.gpu_processes = checked_gpu_processes
    process.assert_gpu_free()
    state_path = OUT / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "status": "initializing", "iterations_per_variant": 100, "evaluations": {}, "runs": {},
        "desktop_gpu_exceptions": {str(pid): str(exe) for pid, exe in desktop.items()}}
    def persist(status):
        state["status"] = status
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(state_path)
        lines = ["# Проверка старта и короткое продолжение E2 A/B", "", f"Статус: {status}", "",
                 "A: std=0.5, entropy=0.001. B: исходный std≈5.12, entropy=0.005. "
                 "По 100 итераций, 1024 среды, rollout32, seed42, gamma0.99, A/20°. "
                 "Оценки: mean action, 128 эпизодов, seed1234. "
                 "Это сравнение пакета std+entropy, не изолированный эффект std.", "",
                 "Перед первым обновлением сверяются все тензоры модели, Adam, LR, номер "
                 "итерации и вся конфигурация среды (кроме пути логов), включая награду. "
                 "Состояние симулятора/RNG исходный checkpoint не сохранял.", "",
                 "| Вариант | held | drop | целей/эп | reward | std | near limit |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for key, m in state["evaluations"].items():
            lines.append(f'| {key} | {m["held_success_rate"]:.4f} | {m["drop_rate"]:.4f} | '
                         f'{m["goals_completed_per_episode"]:.4f} | {m["episode_reward"]:.3f} | '
                         f'{m["action_std"]:.4f} | {m["action_near_limit_fraction"]:.4f} |')
        for key, run in state["runs"].items():
            lines += ["", f"{key}: `{run['checkpoint']}`", "",
                      "```json", json.dumps(run["verification"], indent=2), "```"]
        if state.get("error"):
            lines += ["", "```", state["error"], "```"]
        (ROOT / "reports/e2_std_ab_smoke.md").write_text("\n".join(lines) + "\n")
        print(f"[SMOKE] {status}", flush=True)
    def evaluate(label, checkpoint):
        path = OUT / f"{label}.json"
        persist(f"evaluating_{label}")
        if not path.exists():
            command = [sys.executable, "scripts/eval_checkpoints.py", "--task", TASK,
                       "--headless", "--num_envs", "128", "--seed", "1234", "--baselines", "",
                       "--stage", "A", "--goal_angle_deg", "20", "--run",
                       f"{label}={checkpoint.parent}:{checkpoint.stem[6:]}", "--output", str(path)]
            process.run_checked(command, path.with_suffix(".log"), ROOT, timeout=1200)
        metrics = next(iter(json.loads(path.read_text()).values()))
        if metrics["episodes"] != 128:
            raise RuntimeError("Incomplete evaluation")
        state["evaluations"][label] = metrics
        persist(f"evaluated_{label}")
        return metrics
    try:
        a, b = [torch.load(value[0], map_location="cpu", weights_only=False) for value in VARIANTS.values()]
        a["model_state_dict"].pop("std")
        b["model_state_dict"].pop("std")
        assert_same_state(a, b, "A_vs_B_except_std")
        for label, (checkpoint, entropy, initial_std) in VARIANTS.items():
            metrics = evaluate(label + "_before", checkpoint)
            if abs(metrics["action_std"] - initial_std) > 1e-5:
                raise RuntimeError(f"Wrong initial std for {label}")
        state["before_metric_differences_except_std"] = {
            key: [value, state["evaluations"]["B_before"][key]]
            for key, value in state["evaluations"]["A_before"].items()
            if key != "action_std" and value != state["evaluations"]["B_before"][key]}
        for label, (checkpoint, entropy, initial_std) in VARIANTS.items():
            if label not in state["runs"]:
                persist(f"training_{label}")
                tag = f"e2_std_ab_{label}_100"
                command = [sys.executable, "scripts/rsl_rl/train.py", "--task", TASK, "--headless",
                           "--num_envs", "1024", "--seed", "42", "--max_iterations", "100",
                           "--resume", "--load_run", checkpoint.parent.name, "--checkpoint", checkpoint.name,
                           "--restore_continuation_state", "--verify_resume_state", "env.goal_stream_stage=A",
                           f"agent.run_name={tag}", f"agent.algorithm.entropy_coef={entropy}"]
                process.run_checked(command, OUT / f"{label}_train.log", ROOT, expected=checkpoint, timeout=3600)
                run = max(RUNS.glob(f"*_{tag}"))
                final = run / "model_3699.pt"
                saved = torch.load(final, map_location="cpu", weights_only=False)
                if saved["iter"] != 3699:
                    raise RuntimeError("Wrong continuation budget")
                verification = json.loads((run / "resume_verification.json").read_text())
                if abs(verification["std_mean"] - initial_std) > 1e-5:
                    raise RuntimeError("Runtime initial std mismatch")
                if Path(verification["checkpoint"]) != checkpoint:
                    raise RuntimeError("Runtime checkpoint path mismatch")
                state["runs"][label] = {"checkpoint": str(final), "verification": verification}
                persist(f"trained_{label}")
            evaluate(label + "_after100", Path(state["runs"][label]["checkpoint"]))
        persist("completed")
    except BaseException:
        state["error"] = traceback.format_exc()
        persist("failed")
        raise


if __name__ == "__main__":
    main()
