"""Predeclared, conservative decisions for the bounded overnight experiment."""
import math


def size_action_penalty(baseline, zero):
    t = baseline["smoothing"]
    candidates = [.002, .005, .010, .015, .020, .030, .050]
    if not any(.15 <= s * t["mean_action_square"] / t["task_abs"] <= .35 for s in candidates):
        candidates.append(.15 * t["task_abs"] / t["mean_action_square"])
    rows = []
    for scale in sorted(set(candidates)):
        penalty = scale * t["mean_action_square"]
        projected = baseline["episode_reward"] - (scale - t["action_penalty_scale"]) * t["action_square_sum_per_episode"]
        rows.append({"scale": scale, "mean_abs_action_penalty": penalty,
                     "fraction_of_task_abs": penalty / t["task_abs"], "projected_return": projected,
                     "zero_return": zero["episode_reward"], "beats_zero": projected > zero["episode_reward"]})
    valid = [r for r in rows if .15 - 1e-9 <= r["fraction_of_task_abs"] <= .35 and r["beats_zero"]]
    if not valid:
        # Prefer a smaller affordable candidate to starting a movement-averse long run.
        valid = [r for r in rows if .05 <= r["fraction_of_task_abs"] < .15 and r["beats_zero"]]
    return {"candidates": rows, "selected": min(valid, key=lambda r: r["scale"]) if valid else None,
            "method": "Offline counterfactual on exactly the same first-episode clipped-command trajectory; task_abs includes normalized success bonus."}


def competent(m, baseline):
    return (m["held_success_rate"] >= .70 and
            m["goals_completed_per_episode"] >= .70 * baseline["goals_completed_per_episode"] and
            m["drop_rate"] <= baseline["drop_rate"] + .10)


def near(m):
    return m["smoothing"]["command"]["near_limit_fraction"]


def choose_smoke(smokes, baseline):
    eligible = {k: v for k, v in smokes.items() if competent(v[-1], baseline)}
    if not eligible:
        return None, "Neither smoke preserved the predeclared competence floor; no long run."
    if "L" in eligible:
        low = eligible["L"][-1]
        high = eligible.get("H", [None])[-1]
        if high is None or not (near(high) < near(low) - .03 and
                                high["goals_completed_per_episode"] >= .9 * low["goals_completed_per_episode"]):
            return "L", "Low std preserved competence at fixed LR; high std did not show a clear saturation advantage (>3 percentage points with comparable goals)."
    return "H", "High std preserved competence and low std failed, or H had a clear saturation advantage."


def early_stop(history, baseline, additional_iterations):
    if not history:
        return None
    last = history[-1]
    if not all(math.isfinite(last[k]) for k in ("episode_reward", "action_std", "drop_rate")):
        return "nonfinite_metrics"
    if not .02 <= last["action_std"] <= 10:
        return "pathological_std"
    if len(history) < 2:
        return None
    pair = history[-2:]
    if all(m["goals_completed_per_episode"] < .7 * baseline["goals_completed_per_episode"] for m in pair):
        return "goals_below_70_percent_for_two_evaluations"
    if all(m["held_success_rate"] < .6 * baseline["held_success_rate"] for m in pair):
        return "competence_collapse"
    if all(m["drop_rate"] > baseline["drop_rate"] + .10 for m in pair):
        return "drop_materially_worse_for_two_evaluations"
    if all(m["smoothing"]["action_penalty_to_task"] > .50 for m in pair):
        return "regularization_exceeds_half_task_signal"
    if additional_iterations >= 1500 and all(near(baseline) - near(m) < .01 for m in pair):
        return "saturation_did_not_move_after_1500"
    if additional_iterations >= 1500 and all(near(baseline) - near(m) >= .05 and
                                              m["drop_rate"] >= baseline["drop_rate"] - .03 for m in pair):
        return "action_smoothing_succeeded_grasp_survival_did_not"
    return None


def working(m, baseline):
    return competent(m, baseline) and near(baseline) - near(m) >= .05


def chatter(m):
    t = m["smoothing"]
    return t["sign_flip_fraction"] >= .20 and t["delta_command"]["p90"] >= 1.0
