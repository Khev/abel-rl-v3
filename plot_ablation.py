#!/usr/bin/env python3
"""Plot multiple labeled groups of seeds on one figure (4 metrics).

Each group = (label, [seed_dir, seed_dir, ...]). Plots mean ± min/max
across seeds within each group, with a distinct color per group.

Auto-detects train_cov format (accuracies.csv) vs train_abel format
(learning_curves.csv).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS_COV = ["train_greedy_acc", "train_beam_acc", "test_greedy_acc", "test_beam_acc"]
METRICS_ABEL = ["coverage", "test_greedy", "test_beam", "test_at10"]


def load_group(dirs):
    """Load all seeds in a group, return list of (df, time_col, metrics)."""
    runs = []
    for d in dirs:
        d = Path(d)
        if (d / "accuracies.csv").exists():
            df = pd.read_csv(d / "accuracies.csv")
            runs.append((df, "timesteps", METRICS_COV))
        elif (d / "learning_curves.csv").exists():
            df = pd.read_csv(d / "learning_curves.csv")
            time_col = "step" if "step" in df.columns else "timesteps"
            runs.append((df, time_col, METRICS_ABEL))
        else:
            print(f"[warn] no CSV in {d}", file=sys.stderr)
    return runs


def aggregate(group_runs, metric):
    """Return common_t, mean, lo, hi across the group's seeds for a metric.
    Returns (None, None, None, None) if no data."""
    if not group_runs:
        return None, None, None, None
    # use first run's time axis as canonical
    common_t = group_runs[0][0][group_runs[0][1]].values
    stack = []
    for df, time_col, _ in group_runs:
        if metric not in df.columns:
            continue
        t = df[time_col].values
        y = df[metric].values
        if len(t) == 0:
            continue
        if not np.array_equal(t, common_t):
            y = np.interp(common_t, t, y)
        stack.append(y)
    if not stack:
        return None, None, None, None
    stack = np.stack(stack, axis=0)
    return common_t, stack.mean(0), stack.min(0), stack.max(0)


def plot(groups, out_path, title):
    """groups = dict {label: [dir, dir, ...]}."""
    # decide metrics by inspecting first valid group
    first_runs = next((load_group(v) for v in groups.values() if v), [])
    if not first_runs:
        print("no data loaded")
        return
    metrics = first_runs[0][2]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes = axes.flatten()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, metric in zip(axes, metrics):
        for ci, (label, dirs) in enumerate(groups.items()):
            runs = load_group(dirs)
            t, mean, lo, hi = aggregate(runs, metric)
            if t is None:
                continue
            color = colors[ci % len(colors)]
            ax.plot(t, mean, color=color, linewidth=2, label=f"{label} (n={len(runs)})")
            ax.fill_between(t, lo, hi, color=color, alpha=0.18)
        ax.set_title(metric)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[2].set_xlabel("step")
    axes[3].set_xlabel("step")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures/ablation.png")
    p.add_argument("--title", default="")
    p.add_argument("--group", action="append", required=True,
                   help='format: "label:dir1,dir2,dir3"; repeatable')
    args = p.parse_args()

    groups = {}
    for g in args.group:
        if ":" not in g:
            print(f"[warn] skipping bad group spec: {g}", file=sys.stderr)
            continue
        label, dirs_str = g.split(":", 1)
        dirs = [d.strip() for d in dirs_str.split(",") if d.strip()]
        groups[label] = dirs

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    title = args.title or f"ablation across {len(groups)} configs"
    plot(groups, args.out, title)


if __name__ == "__main__":
    main()
