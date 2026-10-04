"""Measure how noisy the deterministic gait metrics are, and whether B - A survives that noise.

Why this exists. The 400-iteration H1 smoke gave contradictory verdicts on two evaluation seeds:
the SAME B checkpoint reported meaningful_events_per_episode 0.914 (seed 1234) and 1.484
(seed 4321), while A reported 1.070 and 1.047. The effect under test was therefore comparable to
the measurement noise, which makes "H1 not supported" unsupportable as a claim in either
direction. This script estimates the noise before anyone spends GPU-hours on a long run.

Design notes that matter for the statistics:

- Every checkpoint is evaluated on the SAME seed set, and the seed is the unit of observation, so
  B - A is computed PAIRED within a seed. A and B then face the same goal sequence and reset draws,
  which removes most of the between-seed variance from the comparison. The pairing is approximate
  rather than exact: GPU PhysX is not bit-deterministic, as this project's journal already records.
- Evaluation always uses the STABLE reset distribution, for A and B alike. Evaluating B from its own
  gait-transition resets would count a scripted mid-transition starting state as a learned skill.
- All 8 seeds are re-run here rather than reusing the two existing numbers, so that every
  observation comes from one code path with one set of flags. Uniformity is the point of a noise
  estimate; the 10 minutes saved by reuse are not worth a subtle flag difference.
- Bootstrap over 8 seeds is crude. It is reported as an interval, not as a p-value, and the decision
  rule below asks whether the direction is consistent, not whether a threshold was crossed.
"""

import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/gait_research"
MS = OUT / "multiseed"
PY_EXE = sys.executable
TASK = "DG5F-Cube-Stream-Direct-v0"
SEEDS = [1234, 4321, 2026, 31415, 27182, 9876, 5555, 7777]

CHECKPOINTS = {
    "BASELINE": OUT.parent / "rsl_rl/dg5f_cube_direct/2026-10-01_14-39-27_e2_continue_3600/model_3599.pt",
    "A_STABLE": OUT.parent / "rsl_rl/dg5f_cube_direct/2026-10-02_17-38-56_gait_H1_A/model_3999.pt",
    "B_MIXED":  OUT.parent / "rsl_rl/dg5f_cube_direct/2026-10-02_18-15-45_gait_H1_B/model_3999.pt",
}

# The metrics the H1 decision rests on, plus the ones that say whether a "gait event" was real
# motion or 1 mm of contact flicker.
PRIMARY = ["gait/meaningful_events_per_episode", "gait/successful_gait_cycles_per_episode",
           "goals_completed_per_episode", "drop_rate"]
# Verified against a real eval JSON: there is no contact_switch_distance_p99, only mean/p50/p90,
# and the mask metric is distinct_masks_per_episode. Guessing these names would have produced an
# empty report table rather than an error.
SECONDARY = ["target_completion_rate", "mean_tip_contacts", "support_qualified_goals_per_episode",
             "gait/events_per_episode", "gait/contact_switch_distance_mean",
             "gait/contact_switch_distance_p50", "gait/contact_switch_distance_p90",
             "gait/recovery_to_3plus_rate", "gait/distinct_fingers_switched_per_episode",
             "gait/distinct_masks_per_episode", "gait/successful_gait_cycle_rate",
             "gait/goals_with_meaningful_switch_fraction", "gait/two_tip_support_duration",
             "gait/release_duration_mean", "palm_contact_fraction",
             "raw_mu", "saturation", "action_std"]


def evaluate(name, checkpoint, seed):
    """One deterministic evaluation. Skips work that is already on disk, so this is resumable."""
    label = f"{name}_s{seed}"
    path = MS / f"{label}.json"
    if path.exists():
        return label, path
    MS.mkdir(parents=True, exist_ok=True)
    command = [PY_EXE, "-u", "scripts/eval_checkpoints.py", "--task", TASK, "--headless",
               "--num_envs", "128", "--seed", str(seed), "--baselines", "", "--stage", "A",
               "--goal_angle_deg", "20", "--gait_telemetry", "--smoothing_telemetry",
               "--gait_thresholds", str(OUT / "gait_thresholds.json"),
               "--output", str(path),
               "--run", f"{label}={checkpoint.parent}:{checkpoint.stem[6:]}"]
    log = MS / f"{label}.log"
    print(f"[MS] {label}", flush=True)
    started = time.time()
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not path.exists():
        raise RuntimeError(f"{label} failed; see {log}")
    print(f"[MS]   {time.time() - started:.0f}s", flush=True)
    return label, path


def flatten(path):
    """eval_checkpoints writes one entry per run label; pull it out and flatten the gait block."""
    record = next(iter(json.loads(path.read_text()).values()))
    row = {k: record.get(k) for k in ("goals_completed_per_episode", "target_completion_rate",
                                      "drop_rate", "mean_tip_contacts", "action_std",
                                      "episodes", "actor_output_fingerprint")}
    gait = record.get("gait") or {}
    row.update({k: v for k, v in gait.items() if isinstance(v, (int, float))})
    row["support_qualified_goals_per_episode"] = gait.get("support_qualified_goals_per_episode")
    row["palm_contact_fraction"] = gait.get("palm_contact_fraction")
    row["tip_count_fractions"] = gait.get("tip_count_fractions")
    smoothing = record.get("smoothing") or {}
    row["raw_mu"] = (smoothing.get("raw_mu") or {}).get("mean")
    row["saturation"] = (smoothing.get("command") or {}).get("near_limit_fraction")
    return row


def summarize(values):
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    mean = sum(values) / n
    median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return {"n": n, "mean": mean, "median": median, "std": variance ** .5,
            "min": values[0], "max": values[-1]}


def paired_bootstrap(differences, draws=20000, seed=12345):
    """Percentile bootstrap over SEEDS of the paired mean difference. Crude at n=8; reported as an
    interval, never as a significance test."""
    import random
    differences = [d for d in differences if isinstance(d, (int, float))]
    if len(differences) < 2:
        return None
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(rng.choice(differences) for _ in range(n)) / n for _ in range(draws))
    lo = means[int(.025 * draws)]
    hi = means[int(.975 * draws)]
    observed = sum(differences) / n
    positive = sum(1 for d in differences if d > 0)
    return {"mean_difference": observed, "ci95": [lo, hi], "seeds_positive": positive,
            "seeds_total": n, "ci_excludes_zero": lo > 0 or hi < 0}


def main():
    rows = {name: {} for name in CHECKPOINTS}
    for seed in SEEDS:
        for name, checkpoint in CHECKPOINTS.items():
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            _, path = evaluate(name, checkpoint, seed)
            rows[name][seed] = flatten(path)
            (MS / "progress.json").write_text(json.dumps(
                {n: sorted(r) for n, r in rows.items()}, indent=1))

    metrics = PRIMARY + SECONDARY
    report = {"seeds": SEEDS, "checkpoints": {k: str(v) for k, v in CHECKPOINTS.items()},
              "evaluation_reset_distribution": "stable for every checkpoint",
              "episodes_per_evaluation": 128, "per_seed": {}, "spread": {}, "paired_B_minus_A": {}}
    for name in CHECKPOINTS:
        report["per_seed"][name] = {str(s): rows[name][s] for s in SEEDS}
        report["spread"][name] = {m: summarize([rows[name][s].get(m) for s in SEEDS]) for m in metrics}
    for m in metrics:
        differences = []
        for s in SEEDS:
            a, b = rows["A_STABLE"][s].get(m), rows["B_MIXED"][s].get(m)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                differences.append(b - a)
        report["paired_B_minus_A"][m] = {"per_seed": differences, **(paired_bootstrap(differences) or {})}

    (OUT / "h1_multiseed.json").write_text(json.dumps(report, indent=2))
    print(f"[MS] wrote {OUT / 'h1_multiseed.json'}", flush=True)

    def fmt(x, nd=3):
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"

    lines = ["# H1: многосидовая оценка шума и парное сравнение B − A", "",
             f"Сиды: {SEEDS}. По 128 эпизодов на оценку, детерминированные средние действия.",
             "Оценка ВСЕХ чекпойнтов идёт из стабильных сбросов, включая B: иначе старт из готовой",
             "середины перехода засчитался бы как выученный навык.", "",
             "Чекпойнты:", ""]
    for k, v in CHECKPOINTS.items():
        lines.append(f"- `{k}` — `{v}`")
    lines += ["", "## Разброс по сидам (один и тот же чекпойнт)", "",
              "| чекпойнт | метрика | mean | median | std | min | max |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name in CHECKPOINTS:
        for m in PRIMARY:
            d = report["spread"][name][m]
            if d:
                lines.append(f"| {name} | `{m.replace('gait/','')}` | {fmt(d['mean'])} | "
                             f"{fmt(d['median'])} | {fmt(d['std'])} | {fmt(d['min'])} | {fmt(d['max'])} |")
    lines += ["", "## Парная разность B − A по сидам", "",
              "| метрика | B−A среднее | 95% ДИ | сидов B>A | ДИ исключает 0 |",
              "|---|---:|---|---:|---|"]
    for m in PRIMARY + ["gait/contact_switch_distance_p90", "gait/contact_switch_distance_mean",
                        "gait/recovery_to_3plus_rate", "gait/events_per_episode",
                        "gait/two_tip_support_duration", "gait/distinct_fingers_switched_per_episode"]:
        d = report["paired_B_minus_A"].get(m) or {}
        if d.get("ci95"):
            lines.append(f"| `{m.replace('gait/','')}` | {fmt(d['mean_difference'])} | "
                         f"[{fmt(d['ci95'][0])}, {fmt(d['ci95'][1])}] | "
                         f"{d['seeds_positive']}/{d['seeds_total']} | "
                         f"{'да' if d['ci_excludes_zero'] else 'нет'} |")
    (OUT / "H1_MULTI_SEED_REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[MS] wrote {OUT / 'H1_MULTI_SEED_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
