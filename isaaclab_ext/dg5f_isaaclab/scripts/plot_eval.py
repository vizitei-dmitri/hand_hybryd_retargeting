"""Plot the deterministic checkpoint evaluation written by eval_checkpoints.py.

The training curves in <run>/analysis are per-iteration and come from exploration rollouts, which
overstate success (the night run logged held_success 0.22 while deterministic eval gives 0.000).
This compares whole policies against each other and against the zero-action baseline, which is the
only comparison worth judging by. No simulator needed: it reads the eval JSON.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One panel per question, grouping only series that share a unit.
PANELS = [
    ("what fraction of episodes", None, [
        ("drop_rate", "dropped the cube"),
        ("entered_tolerance_rate", "entered 5 deg"),
        ("held_success_rate", "held 5 deg for 0.3 s"),
    ]),
    ("orientation error [deg]", None, [
        ("initial_error_deg", "initial"),
        ("min_error_deg", "closest reached"),
        ("final_error_deg_not_dropped", "final (not dropped)"),
    ]),
    ("episode length [s]", 24.0, [("episode_length_s", "survived")]),
    ("fraction of steps", None, [
        ("action_near_limit_fraction", "action at +-1"),
        ("torque_saturation_fraction", "torque saturated"),
    ]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Defaults to eval_comparison.png beside the JSON")
    args = parser.parse_args()

    data = json.loads(args.json_path.read_text())
    # "zero" first: every other bar is only meaningful relative to doing nothing.
    names = (["zero"] if "zero" in data else []) + [key for key in data if key != "zero"]
    labels = [name.split("/")[-1] for name in names]
    positions = range(len(names))

    figure, axes = plt.subplots(len(PANELS), 1, figsize=(1.6 * len(names) + 5, 3.1 * len(PANELS)), sharex=True)
    for axis, (ylabel, reference, series) in zip(axes, PANELS):
        width = 0.8 / len(series)
        for index, (key, label) in enumerate(series):
            values = [data[name].get(key, math.nan) for name in names]
            offset = (index - (len(series) - 1) / 2) * width
            bars = axis.bar([p + offset for p in positions], values, width=width, label=label)
            if len(series) == 1 or len(names) <= 8:
                axis.bar_label(bars, fmt="%.3g", fontsize=7, padding=1)
        if reference is not None:
            axis.axhline(reference, color="0.4", linestyle=":", linewidth=1)
        # The zero-action baseline as a line across the panel, so "better than nothing" is readable.
        if "zero" in data:
            for index, (key, _) in enumerate(series):
                baseline = data["zero"].get(key)
                if baseline is not None and math.isfinite(baseline):
                    axis.axhline(baseline, color=f"C{index}", linestyle="--", linewidth=0.8, alpha=0.5)
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8, ncols=len(series))
        axis.grid(axis="y", alpha=0.3)
    axes[-1].set_xticks(list(positions), labels, rotation=20, ha="right")
    figure.suptitle(f"deterministic eval, {data[names[0]]['episodes']} episodes per policy"
                    f"  (dashed = zero-action baseline)")
    figure.tight_layout()
    out = args.out or args.json_path.with_name("eval_comparison.png")
    figure.savefig(out, dpi=140)
    print(f"[PLOT] wrote {out}")


if __name__ == "__main__":
    main()
