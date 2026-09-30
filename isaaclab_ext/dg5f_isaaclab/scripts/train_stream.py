"""Drive the reward-v4 goal-stream experiment: one policy through the directional stages A -> B -> C.

Why an orchestrator instead of a single train.py call. The stage gate has to be decided on
DETERMINISTIC evaluation (the reward-v3 night run reported held_success 0.22 in training rollouts
and 0.000 deterministically -- those were noise-driven passes through the tolerance), and RSL-RL
has no hook for "stop, evaluate elsewhere, maybe change the env, continue". So training runs in
segments as subprocesses, each resuming the previous segment's checkpoint, with an evaluation
between them. The actor is never reinitialized: it is the same policy throughout.

The state file makes the curriculum resumable and auditable: which stage, how many iterations that
stage has had, every evaluation, and why it stopped.

This launches long GPU work. It does not choose to advance or to stop on anything except the
thresholds in StreamCurriculumCfg, and it never lowers them.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/dg5f_isaaclab/dg5f_isaaclab"


def load_module(name, path):
    """Load a module by path: importing the package would pull in pxr, which needs Kit running."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STREAM_CURRICULUM = load_module(
    "dg5f_stream_curriculum", PACKAGE / "tasks/direct/dg5f_cube/curriculum.py").STREAM_CURRICULUM
# The stage names, from the module that defines the samplers, so the two cannot drift apart.
STREAM_STAGES = load_module(
    "dg5f_stream_goals", PACKAGE / "tasks/direct/dg5f_cube/goals.py").STREAM_STAGES

LOG_ROOT = ROOT / "logs/rsl_rl/dg5f_cube_direct"


def run(command, log_path, description):
    """Run a subprocess, tee-ing nothing: Isaac is verbose, so the log goes to a file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[STREAM] {description}\n[STREAM]   {' '.join(str(c) for c in command)}\n"
          f"[STREAM]   log: {log_path}", flush=True)
    started = time.time()
    with log_path.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    print(f"[STREAM]   exit={result.returncode} after {time.time() - started:.0f}s", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed; see {log_path}")


def latest_checkpoint(run_dir: Path) -> tuple[str, int]:
    """The highest-numbered checkpoint in a run directory."""
    checkpoints = sorted(((int(p.stem[6:]), p.name) for p in run_dir.glob("model_*.pt")))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    iteration, name = checkpoints[-1]
    return name, iteration


def newest_run(prefix: str) -> Path:
    runs = sorted(p for p in LOG_ROOT.iterdir() if p.is_dir() and p.name.endswith(prefix))
    if not runs:
        raise FileNotFoundError(f"No run directory ending in {prefix!r} under {LOG_ROOT}")
    return runs[-1]


def gate(metrics: dict) -> tuple[bool, str]:
    """The stage gate. Returns (passed, human-readable reason)."""
    cfg = STREAM_CURRICULUM
    checks = [
        ("first_goal_held", metrics["first_goal_held_success_rate"], ">=", cfg.first_goal_held_success_rate),
        ("drop_rate", metrics["drop_rate"], "<=", cfg.max_drop_rate),
        ("goals_per_episode", metrics["goals_completed_per_episode"], ">=", cfg.goals_completed_per_episode),
    ]
    failures = [f"{name}={value:.3f} not {operator} {threshold}"
                for name, value, operator, threshold in checks
                if not (value >= threshold if operator == ">=" else value <= threshold)]
    summary = "  ".join(f"{name}={value:.3f}" for name, value, _, _ in checks)
    return (not failures), summary + ("" if not failures else "   FAIL: " + "; ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="DG5F-Cube-Stream-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=6500, help="Total across all stages")
    parser.add_argument("--warm_start", type=Path, default=None,
                        help="Run directory holding the actor-only warm-start checkpoint")
    parser.add_argument("--warm_checkpoint", default="model_0.pt")
    parser.add_argument("--freeze_actor_iterations", type=int, default=0,
                        help="Critic-only iterations at the very start, after the actor-only warm "
                             "start. 0 disables; use it only if the first evaluation shows the "
                             "warm-start actor degrading.")
    parser.add_argument("--state", type=Path, default=ROOT / "logs/stream/state.json")
    parser.add_argument("--log_dir", type=Path, default=ROOT / "logs/stream")
    parser.add_argument("--eval_num_envs", type=int, default=None,
                        help="Defaults to the curriculum's eval_episodes (one episode per env)")
    parser.add_argument("--start_stage", default="A", choices=STREAM_STAGES)
    parser.add_argument("--target_angle_deg", type=float, default=20.0,
                        help="The prescribed goal magnitude the run must finish at")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Run the predecessor phase at StreamCurriculumCfg.bootstrap_angle_deg "
                             "first, then return to --target_angle_deg in the same stage with the "
                             "same policy. The gate is the same at both angles.")
    parser.add_argument("--resume_state", action="store_true", help="Continue from an existing state file")
    args = parser.parse_args()

    cfg = STREAM_CURRICULUM
    eval_envs = args.eval_num_envs or cfg.eval_episodes
    if args.state.exists() and args.resume_state:
        state = json.loads(args.state.read_text())
        print(f"[STREAM] resuming state: stage {state['stage']} at {state['total_iterations']} iterations")
    else:
        if args.warm_start is None:
            parser.error("--warm_start is required unless --resume_state continues an existing state")
        state = {
            "stage": args.start_stage, "total_iterations": 0, "stage_iterations": 0,
            "consecutive_passes": 0, "load_run": args.warm_start.name,
            "load_checkpoint": args.warm_checkpoint, "evaluations": [], "stopped": None,
            "stage_passes": {},
            # The bootstrap phase is a property of the RUN, not of the stage: it happens once, and
            # after it the same policy is re-gated at the full angle before any stage advances.
            "phase": "bootstrap" if args.bootstrap else "target",
            "angle_deg": cfg.bootstrap_angle_deg if args.bootstrap else args.target_angle_deg,
            "phase_iterations": 0,
        }

    args.log_dir.mkdir(parents=True, exist_ok=True)

    def save_state():
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(json.dumps(state, indent=1))

    def stop(reason):
        state["stopped"] = reason
        save_state()
        print(f"[STREAM] STOP: {reason}", flush=True)

    save_state()
    while state["total_iterations"] < args.max_iterations and state["stopped"] is None:
        stage = state["stage"]
        segment = min(cfg.eval_every_iterations, args.max_iterations - state["total_iterations"])
        angle = state["angle_deg"]
        suffix = "" if state["phase"] == "target" else f"_{round(angle)}deg"
        tag = f"stream_{stage}{suffix}_{state['total_iterations']:05d}"
        # --resume/--load_run/--checkpoint as CLI FLAGS, never as hydra overrides: cli_args.py does
        # `if args_cli.resume is not None: agent_cfg.resume = args_cli.resume`, and --resume is a
        # store_true with default False, so it is never None and unconditionally overwrites the
        # hydra value with False. `agent.resume=true` is accepted silently and then discarded, while
        # agent.load_run/load_checkpoint DO survive -- so the dumped params/agent.yaml looks correct
        # and the run trains from scratch. Four smoke runs were wasted on exactly that.
        command = [sys.executable, "scripts/rsl_rl/train.py", "--task", args.task, "--headless",
                   "--num_envs", str(args.num_envs), "--max_iterations", str(segment),
                   "--resume", "--load_run", state["load_run"],
                   "--checkpoint", state["load_checkpoint"],
                   f"env.goal_stream_stage={stage}",
                   # Both, always: the env asserts the sampled initial error is at least
                   # min_initial_goal_error_deg, so setting the angle alone trips that assert.
                   f"env.goal_stream_angle_deg={angle}",
                   f"env.min_initial_goal_error_deg={angle}",
                   f"agent.run_name={tag}"]
        if state["total_iterations"] == 0 and args.freeze_actor_iterations:
            command += ["--freeze_actor_iterations", str(args.freeze_actor_iterations)]
        run(command, args.log_dir / f"{tag}_train.log", f"train stage {stage}: {segment} iterations")

        run_dir = newest_run(tag)
        checkpoint, iteration = latest_checkpoint(run_dir)
        state["load_run"], state["load_checkpoint"] = run_dir.name, checkpoint
        state["total_iterations"] += segment
        state["stage_iterations"] += segment
        state["phase_iterations"] += segment

        # Deterministic evaluation of exactly this checkpoint, at exactly this stage.
        eval_path = args.log_dir / f"{tag}_eval.json"
        run([sys.executable, "scripts/eval_checkpoints.py", "--task", args.task, "--headless",
             "--num_envs", str(eval_envs), "--baselines", "", "--stage", stage,
             "--goal_angle_deg", str(angle),
             "--run", f"cur={run_dir}:{iteration}", "--output", str(eval_path)],
            args.log_dir / f"{tag}_eval.log", f"deterministic eval of {run_dir.name}/{checkpoint}")
        metrics = json.loads(eval_path.read_text())[f"cur/model_{iteration}"]
        passed, summary = gate(metrics)
        state["consecutive_passes"] = state["consecutive_passes"] + 1 if passed else 0
        record = {"stage": stage, "total_iterations": state["total_iterations"],
                  "checkpoint": f"{run_dir.name}/{checkpoint}", "passed": passed,
                  "consecutive_passes": state["consecutive_passes"], "metrics": metrics,
                  "phase": state["phase"], "angle_deg": angle}
        state["evaluations"].append(record)
        print(f"[STREAM] eval @{state['total_iterations']} stage {stage} at {angle:g} deg"
              f" ({state['phase']}): {summary}"
              f"  consecutive={state['consecutive_passes']}", flush=True)
        save_state()

        # Exploration watchdog: a high std is only a problem when nothing is improving with it.
        window = [e for e in state["evaluations"]
                  if e["stage"] == stage and e.get("angle_deg", angle) == angle][-cfg.watchdog_window:]
        std = metrics.get("action_std")
        if std is not None and std > cfg.action_std_watchdog and len(window) == cfg.watchdog_window:
            best = max(e["metrics"]["goals_completed_per_episode"] for e in window[:-1])
            if metrics["goals_completed_per_episode"] <= best:
                stop(f"action_std {std:.3f} above {cfg.action_std_watchdog} with no deterministic "
                     f"improvement over the last {cfg.watchdog_window} evaluations")
                break

        if state["consecutive_passes"] >= cfg.consecutive_passes and state["phase"] == "bootstrap":
            # The predecessor phase is done. Same policy, same stage, full angle, gate applied again.
            passed_name = args.log_dir / f"bootstrap_{round(angle)}deg_pass.pt"
            passed_name.write_bytes((run_dir / checkpoint).read_bytes())
            print(f"[STREAM] bootstrap at {angle:g} deg PASSED at {state['total_iterations']} "
                  f"iterations -> {passed_name}; returning to {args.target_angle_deg:g} deg",
                  flush=True)
            state["phase"] = "target"
            state["angle_deg"] = args.target_angle_deg
            state["phase_iterations"] = 0
            state["stage_iterations"] = 0
            state["consecutive_passes"] = 0
            save_state()
            continue
        if state["phase"] == "bootstrap" and state["phase_iterations"] >= cfg.bootstrap_max_iterations:
            stop(f"bootstrap at {angle:g} deg STALLED: {state['phase_iterations']} iterations without "
                 f"{cfg.consecutive_passes} consecutive passing evaluations")
            break
        if state["consecutive_passes"] >= cfg.consecutive_passes:
            passed_name = args.log_dir / f"stage_{stage}_pass.pt"
            passed_name.write_bytes((run_dir / checkpoint).read_bytes())
            state["stage_passes"][stage] = {"iterations": state["total_iterations"],
                                            "checkpoint": f"{run_dir.name}/{checkpoint}"}
            print(f"[STREAM] stage {stage} PASSED at {state['total_iterations']} iterations "
                  f"-> {passed_name}", flush=True)
            index = STREAM_STAGES.index(stage)
            if index + 1 == len(STREAM_STAGES):
                best = args.log_dir / "stage_C_best.pt"
                best.write_bytes((run_dir / checkpoint).read_bytes())
                stop("all stages passed")
                break
            state["stage"] = STREAM_STAGES[index + 1]
            state["stage_iterations"] = 0
            state["consecutive_passes"] = 0
            save_state()
        elif state["stage_iterations"] >= cfg.stage_max_iterations:
            stop(f"stage {stage} STALLED: {state['stage_iterations']} iterations without "
                 f"{cfg.consecutive_passes} consecutive passing evaluations")
            break

    if state["stopped"] is None:
        stop(f"iteration budget reached ({state['total_iterations']})")
    print(f"[STREAM] state: {args.state}", flush=True)


if __name__ == "__main__":
    main()
