"""Durable sequential campaign. Launch detached under systemd-inhibit; never commits."""
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import torch

from night_smoothing_rules import chatter, choose_smoke, competent, early_stop, near, size_action_penalty, working
from overnight_process import check_resume, descendant_groups, gpu_processes, stop_group

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/night_smoothing"
RUNS = ROOT / "logs/rsl_rl/dg5f_cube_direct"
TASK = "DG5F-Cube-Stream-Direct-v0"
H = RUNS / "2026-10-01_14-39-27_e2_continue_3600/model_3599.pt"
L = RUNS / "2026-10-01_e2_std05_preserved/model_3599.pt"
TRANSFER = RUNS / "2026-10-01_18-11-11_goal10_exploration_A_10deg_00000/model_599.pt"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_metrics(path):
    return next(iter(json.loads(Path(path).read_text()).values()))


def compact(m):
    t = m["smoothing"]
    return {**{k: m[k] for k in ("held_success_rate", "target_completion_rate", "goals_completed_per_episode",
            "drop_rate", "episode_reward", "episode_length_s", "action_std", "max_consecutive_goals")},
            "near_limit_fraction": near(m), "historical_near_limit_fraction": m["action_near_limit_fraction"],
            "mean_abs_mu": t["raw_mu"]["mean"], "p95_abs_mu": t["raw_mu"]["p95"],
            "mean_abs_command": t["command"]["mean"], "mean_abs_delta_mu": t["delta_mu"].get("mean", 0),
            "mean_abs_delta_command": t["delta_command"].get("mean", 0),
            "sign_flip_fraction": t["sign_flip_fraction"], "qdot_fd_p99": t["qdot_fd_after_1s"].get("p99"),
            "qdot_physx_p99": t["qdot_physx"].get("p99"), "action_penalty_to_task": t["action_penalty_to_task"],
            "reward_terms": {k: {a: v[a] for a in ("mean_signed", "mean_abs", "sum_per_episode")}
                             for k, v in t["reward_terms"].items()}}


class Campaign:
    def __init__(self):
        self.lease = (ROOT / "logs/overnight_ab/driver.lock").open("a")
        fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state = json.loads((OUT / "night_state.json").read_text())
        self.deadline = datetime.fromisoformat(self.state["hard_deadline"]).timestamp()
        self.cleanup_deadline = datetime.fromisoformat(self.state["cleanup_deadline"]).timestamp()
        self.state.setdefault("jobs", {})
        self.state.setdefault("evaluations", {})
        self.state.setdefault("primary", [])
        self.state["controller_pid"] = os.getpid()
        self.state["detached_mechanism"] = "start_new_session=True + systemd-inhibit --what=sleep; tmux unavailable"
        self.lr = json.loads((OUT / "checkpoint_inspection.json").read_text())["chosen_fixed_lr"]
        self.scale = self.state.get("selected_action_penalty_scale", .002)
        self.entropy = .001  # Identical in H and L; only initial std differs.
        self.desktop = {}
        for pid, memory in gpu_processes().items():
            exe = Path(f"/proc/{pid}/exe").resolve()
            if exe.name == "Telegram" and memory.isdigit() and int(memory) <= 128:
                self.desktop[pid] = str(exe)
        self.state["desktop_gpu_exceptions"] = self.desktop
        self.save("PREPARING", next_step="Baseline + zero preflight; 200-update H/L smokes; select one primary.")

    def save(self, status=None, **fields):
        if status:
            self.state["status"] = status
        self.state.update(fields)
        self.state.update(current_timestamp=now(), elapsed_seconds=time.time() - datetime.fromisoformat(self.state["started_at"]).timestamp(),
                          remaining_to_hard_deadline_seconds=self.deadline - time.time(),
                          free_disk_bytes=shutil.disk_usage(ROOT).free,
                          git_HEAD=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                          git_diff_summary=subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True),
                          git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True))
        self.state["resume_instruction"] = (f"First inspect controller PID {os.getpid()} and GPU; do not duplicate it. "
            "If it exited before COMPLETE, run the same scripts/night_smoothing.py under systemd-inhibit in a new detached session. "
            "Completed jobs/evaluations are reused. Deadlines never reset.")
        temp = OUT / "night_state.tmp"
        temp.write_text(json.dumps(self.state, indent=2))
        temp.replace(OUT / "night_state.json")
        md = OUT / "NIGHT_STATE.md"
        text = "STATUS: " + self.state["status"] + "\n\n" + self.state["resume_instruction"] + "\n\n```json\n" + json.dumps(self.state, indent=2) + "\n```\n"
        md.with_suffix(".tmp").write_text(text)
        md.with_suffix(".tmp").replace(md)

    def foreign_gpu(self):
        return {pid: mem for pid, mem in gpu_processes().items()
                if not (str(Path(f"/proc/{pid}/exe").resolve()) == self.desktop.get(pid)
                        and mem.isdigit() and int(mem) <= 128)}

    def run(self, label, command, kind, source=None, count=None, tag=None):
        prior = self.state["jobs"].get(label)
        if prior and prior.get("status") == "completed":
            return prior
        if self.foreign_gpu():
            raise RuntimeError(f"Foreign GPU process; refusing overlap: {self.foreign_gpu()}")
        if shutil.disk_usage(ROOT).free < (550 if kind == "train" else 180) * 1024**2:
            raise RuntimeError("Disk reserve reached; no new job")
        cutoff = self.deadline if kind == "train" else self.cleanup_deadline
        if time.time() >= cutoff:
            raise TimeoutError("Experiment deadline reached")
        path = OUT / f"{label}.log"
        started = time.time()
        log = path.open("w")
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        owned = {child.pid}
        verified = source is None
        got_gpu = False
        failure_since = None
        position = 0
        initial = ""
        rates = []
        info = {"status": "running", "command": command, "pid": child.pid, "log": str(path), "started": now(),
                "source": str(source) if source else None}
        self.state["jobs"][label] = info
        self.save("TRAINING_DETACHED" if kind == "train" else "EVALUATING_DETACHED", active_experiment=label,
                  exact_command=command, pid=child.pid, source_checkpoint=info["source"], output_run_directory=None,
                  current_iteration=None, next_step=f"Finish {label}; verify saved output; continue predeclared campaign.")
        try:
            while True:
                owned.update(descendant_groups(child.pid))
                with path.open(errors="replace") as f:
                    f.seek(position)
                    fresh = f.read()
                    position = f.tell()
                initial = (initial + fresh)[:300000]
                if source is not None:
                    verified = check_resume(initial, source)
                for line in fresh.splitlines():
                    if "[RESUME_VERIFIED] " in line:
                        info["resume_verification"] = json.loads(line.split("[RESUME_VERIFIED] ", 1)[1])
                    match = re.search(r"\[FIXED_LR\] iteration=(\d+) lr=(\S+) std=(\S+)", line)
                    if match:
                        it, lr, std = int(match[1]), float(match[2]), float(match[3])
                        if lr != self.lr:
                            raise RuntimeError("Effective LR changed under fixed schedule")
                        rates.append({"iteration": it, "lr": lr, "std": std})
                        self.state["current_iteration"] = it
                if re.search(r"Traceback \(most recent call last\)|CUDA out of memory|Mean reward:\s*(?:nan|inf)|Patch buffer overflow|contact buffer overflow", fresh, re.I):
                    failure_since = failure_since or time.time()
                if failure_since and time.time() - failure_since > 15:
                    raise RuntimeError(f"Failure/physics overflow detected in {path}")
                for pid in gpu_processes():
                    try:
                        if os.getpgid(pid) in owned:
                            got_gpu = True
                    except ProcessLookupError:
                        pass
                if tag:
                    dirs = sorted(RUNS.glob("*_" + tag))
                    if dirs:
                        self.state["output_run_directory"] = str(dirs[-1])
                info.update(gpu_process_verified=got_gpu, resume_path_verified=verified, last_update=now())
                if child.poll() is not None:
                    break
                if time.time() > cutoff or time.time() - started > (max(1800, count * 12) if count else 1200):
                    raise TimeoutError("Job exceeded deadline or bounded runtime")
                if time.time() - started > 600 and (not verified or not got_gpu):
                    raise RuntimeError("No verified checkpoint load or GPU process within 600 seconds")
                if shutil.disk_usage(ROOT).free < 180 * 1024**2:
                    raise RuntimeError("Critical disk reserve reached")
                self.save()
                time.sleep(5)
            if child.returncode != 0 or not verified or not got_gpu:
                raise RuntimeError(f"Job failed: return={child.returncode}, resume={verified}, GPU={got_gpu}")
            if kind == "train":
                if len(rates) != count or "resume_verification" not in info:
                    raise RuntimeError(f"Incomplete fixed-LR verification: {len(rates)}/{count}")
                iteration = int(torch.load(source, map_location="cpu", weights_only=False)["iter"]) + count
                if [r["iteration"] for r in rates] != list(range(iteration - count + 1, iteration + 1)):
                    raise RuntimeError("Unexpected iteration sequence")
                final = Path(self.state["output_run_directory"]) / f"model_{iteration}.pt"
                saved = torch.load(final, map_location="cpu", weights_only=False)
                if saved["iter"] != iteration or any(g["lr"] != self.lr for g in saved["optimizer_state_dict"]["param_groups"]):
                    raise RuntimeError("Final checkpoint iteration/LR mismatch")
                info.update(checkpoint=str(final), fixed_lr_samples=rates)
            info.update(status="completed", seconds=time.time() - started)
        except BaseException:
            info.update(status="failed", error=traceback.format_exc())
            raise
        finally:
            if child.poll() is None:
                for group in sorted(owned, reverse=True):
                    stop_group(group)
            child.wait()
            log.close()
            self.save(pid=None)
        # Compress only this campaign's finished logs, never old or unrelated files.
        if path.stat().st_size > 2_000_000:
            with path.open("rb") as f, gzip.open(str(path) + ".gz", "wb") as g:
                shutil.copyfileobj(f, g)
            path.unlink()
            info["log"] = str(path) + ".gz"
            self.save()
        return info

    def evaluate(self, label, checkpoint=None, scale=None, rate=.001, seed=1234):
        path = OUT / f"{label}.json"
        if not path.exists():
            cmd = [sys.executable, "scripts/eval_checkpoints.py", "--task", TASK, "--headless", "--num_envs", "128",
                   "--seed", str(seed), "--baselines", "zero" if checkpoint is None else "", "--stage", "A",
                   "--goal_angle_deg", "20", "--smoothing_telemetry", "--output", str(path),
                   "--action_penalty_scale", str(self.scale if scale is None else scale), "--action_rate_penalty_scale", str(rate)]
            if checkpoint is not None:
                cmd += ["--run", f"{label}={checkpoint.parent}:{checkpoint.stem[6:]}"]
            self.run(label, cmd, "eval")
        m = read_metrics(path)
        if m["episodes"] != 128 or abs(sum(v["sum_per_episode"] for v in m["smoothing"]["reward_terms"].values()) - m["episode_reward"]) > .01:
            raise RuntimeError("Incomplete evaluation or inconsistent reward telemetry")
        self.state["evaluations"][label] = {"checkpoint": str(checkpoint) if checkpoint else "zero", "seed": seed, "json": str(path), **compact(m)}
        self.save(last_deterministic_metrics=self.state["evaluations"][label])
        return m

    def train(self, label, source, count, rate=.001):
        tag = "night_smoothing_" + label
        cmd = [sys.executable, "scripts/rsl_rl/train.py", "--task", TASK, "--headless", "--num_envs", "1024", "--seed", "42",
               "--max_iterations", str(count), "--resume", "--load_run", source.parent.name, "--checkpoint", source.name,
               "--restore_continuation_state", "--verify_resume_state", "--verify_fixed_lr", repr(self.lr),
               "--verify_action_penalty_scale", str(self.scale), "--verify_action_rate_penalty_scale", str(rate),
               f"agent.run_name={tag}", "agent.algorithm.schedule=fixed", f"agent.algorithm.learning_rate={self.lr!r}",
               f"agent.algorithm.entropy_coef={self.entropy}", "env.goal_stream_stage=A",
               f"env.action_penalty_scale={self.scale}", f"env.action_rate_penalty_scale={rate}"]
        return Path(self.run(label, cmd, "train", source, count, tag)["checkpoint"])

    def enough_time(self, count):
        durations = [v["seconds"] / len(v["fixed_lr_samples"]) for v in self.state["jobs"].values()
                     if v.get("status") == "completed" and v.get("fixed_lr_samples")]
        rate = max(5.0, (sum(durations) / len(durations) * 1.25) if durations else 7.0)
        return time.time() + count * rate + 360 < self.deadline and shutil.disk_usage(ROOT).free > 650 * 1024**2

    def execute(self):
        baseline = self.evaluate("BASELINE_3599", H, scale=.002)
        zero = self.evaluate("ZERO", None, scale=.002)
        sizing = size_action_penalty(baseline, zero)
        (OUT / "reward_preflight.json").write_text(json.dumps(sizing, indent=2))
        if sizing["selected"] is None:
            self.state["stop_reason"] = "No affordable action penalty passed zero-policy sanity"
            return
        self.scale = sizing["selected"]["scale"]
        self.save("PREFLIGHT_COMPLETE", selected_action_penalty_scale=self.scale, preflight=sizing,
                  fixed_lr=self.lr, entropy_coef_both_branches=self.entropy,
                  hypothesis="A larger clipped-action magnitude penalty reduces deterministic saturation without sacrificing manipulation.",
                  next_step="10-to-20 transfer eval; identical-reward H/L before,100,200 evaluations.")
        self.evaluate("TRANSFER_10_TO_20", TRANSFER, scale=.002)
        smokes, endpoints = {}, {}
        # Finish both BEFORE evaluations before changing either actor.
        for label, source in (("H", H), ("L", L)):
            self.evaluate(f"SMOKE_{label}_before", source)
        before_h, before_l = [read_metrics(OUT / f"SMOKE_{b}_before.json") for b in ("H", "L")]
        differences = {k: [before_h[k], before_l[k]] for k in before_h
                       if k not in ("action_std", "smoothing") and before_h[k] != before_l[k]}
        if differences:
            raise RuntimeError(f"Before-policy deterministic metrics differ: {differences}")
        if before_h["episode_reward"] <= zero["episode_reward"]:
            raise RuntimeError("Measured zero-policy reward dominates; no training")
        for branch, source in (("H", H), ("L", L)):
            smokes[branch] = []
            for completed in (100, 200):
                source = self.train(f"smoke_{branch}_{completed}", source, 100)
                smokes[branch].append(self.evaluate(f"SMOKE_{branch}_{completed}", source))
            endpoints[branch] = source
        choice, reason = choose_smoke(smokes, baseline)
        self.save("SMOKES_COMPLETE", selected_branch=choice, smoke_decision=reason, next_step="Run selected branch in 500-update segments, or stop primary if neither is competent.")
        history = []
        if choice:
            source = endpoints[choice]
            for completed in range(500, 20001, 500):
                if not self.enough_time(500):
                    self.state["stop_reason"] = "Time/disk reserve: no further 500-update segment fits"
                    break
                source = self.train(f"primary_{completed:05d}", source, 500)
                label = f"PRIMARY_{completed:05d}"
                m = self.evaluate(label, source)
                history.append(m)
                if label not in self.state["primary"]:
                    self.state["primary"].append(label)
                self.save(primary_additional_iterations=completed, next_step="Apply predeclared early-stop rules to deterministic evaluations.")
                reason = early_stop(history, baseline, completed)
                if m["episode_reward"] <= zero["episode_reward"] and len(history) >= 2 and history[-2]["episode_reward"] <= zero["episode_reward"]:
                    reason = "Zero policy dominates for two evaluations"
                if reason or (completed >= 4000 and not working(m, baseline)):
                    self.state["stop_reason"] = reason or "4000 updates without material useful smoothing"
                    break
        else:
            self.state["stop_reason"] = reason
        primary_labels = self.state["primary"]
        if primary_labels:
            def rank(label):
                m = read_metrics(OUT / f"{label}.json")
                return (competent(m, baseline), -near(m), -m["drop_rate"], m["goals_completed_per_episode"])
            best = max(primary_labels, key=rank)
            final = primary_labels[-1]
            self.state.update(best_primary_label=best, final_primary_label=final)
        # Exactly one optional rate experiment; never mix it into the primary hypothesis.
        self.state["action_rate_justified"] = chatter(baseline)
        if chatter(baseline) and self.enough_time(400) and self.state.get("stop_reason"):
            if primary_labels:
                parent = self.state["best_primary_label"]
                source = Path(self.state["evaluations"][parent]["checkpoint"])
                parent_metrics = read_metrics(OUT / f"{parent}.json")
            else:
                branch = max(smokes, key=lambda k: smokes[k][-1]["goals_completed_per_episode"])
                source, parent_metrics = endpoints[branch], smokes[branch][-1]
            t = parent_metrics["smoothing"]
            rate = next((r for r in (.002, .003, .005, .0075, .010) if .05 <= r * t["mean_action_rate_square"] / max(t["task_abs"], 1e-12) <= .15), None)
            projected = parent_metrics["episode_reward"] - ((rate or .001) - .001) * t["action_rate_square_sum_per_episode"]
            self.save(action_rate_preflight={"scale": rate, "projected_return": projected,
                       "fraction_of_task_abs": (rate or .001) * t["mean_action_rate_square"] / max(t["task_abs"], 1e-12),
                       "source": str(source)}, next_step="Optional isolated action-rate smoke only if affordable and parent remains competent.")
            if rate and projected > zero["episode_reward"] and competent(parent_metrics, baseline):
                self.evaluate("RATE_before", source, rate=rate)
                for completed in (100, 200):
                    source = self.train(f"rate_{completed}", source, 100, rate=rate)
                    self.evaluate(f"RATE_{completed}", source, rate=rate)
                self.state["action_rate_ran"] = True
        self.final_comparison()

    def final_comparison(self):
        self.save("FINAL_EVALUATIONS", next_step="Fresh same-seed final comparisons, report and cleanup; no further training.")
        self.evaluate("FINAL_BASELINE", H, scale=.002)
        for role in ("best", "final"):
            key = self.state.get(f"{role}_primary_label")
            if key:
                self.evaluate("FINAL_" + role.upper(), Path(self.state["evaluations"][key]["checkpoint"]))
        self.evaluate("FINAL_TRANSFER", TRANSFER, scale=.002)
        # A second held-out evaluation seed is used only for the final comparison.
        for role, checkpoint, scale in [("BASELINE", H, .002), ("TRANSFER", TRANSFER, .002)]:
            self.evaluate(f"FINAL_{role}_seed4321", checkpoint, scale=scale, seed=4321)
        for role in ("best", "final"):
            key = self.state.get(f"{role}_primary_label")
            if key:
                self.evaluate(f"FINAL_{role.upper()}_seed4321", Path(self.state["evaluations"][key]["checkpoint"]), seed=4321)


def main():
    os.chdir(ROOT)
    if json.loads((OUT / "night_state.json").read_text()).get("status") == "COMPLETE":
        print("Campaign already complete; no new jobs.")
        return
    campaign = Campaign()
    try:
        campaign.execute()
    except BaseException:
        campaign.state["error"] = traceback.format_exc()
        campaign.state["stop_reason"] = "Campaign stopped on explicit failure; see error and individual job logs"
        print(campaign.state["error"], flush=True)
    finally:
        from night_smoothing_report import write_report
        write_report(campaign.state, OUT)
        campaign.save("COMPLETE", pid=None, active_experiment="finished", next_step="Read FINAL_REPORT.md. No further jobs are queued.")


if __name__ == "__main__":
    main()
