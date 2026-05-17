#!/usr/bin/env python3
"""Plot learning curves across seeds from train_cov.py runs.

Usage:
    python plot_seeds.py <run_dir1> <run_dir2> ...    # explicit dirs
    python plot_seeds.py --glob 'gemini/cov/ppo_*'    # glob pattern
    python plot_seeds.py --latest 5                   # most recent N dirs

Each run dir is expected to contain `accuracies.csv` with columns:
    timesteps, train_greedy_acc, train_beam_acc, test_greedy_acc, test_beam_acc
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS = ["train_greedy_acc", "train_beam_acc", "test_greedy_acc", "test_beam_acc"]


def load_runs(dirs):
    runs = []
    for d in dirs:
        p = Path(d) / "accuracies.csv"
        if not p.exists():
            print(f"[warn] no accuracies.csv in {d}", file=sys.stderr)
            continue
        df = pd.read_csv(p)
        runs.append((str(d), df))
    return runs


def plot(runs, out_path, title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes = axes.flatten()
    for ax, metric in zip(axes, METRICS):
        # individual seeds
        for label, df in runs:
            ax.plot(df["timesteps"], df[metric], alpha=0.35, linewidth=1)
        # mean ± min/max envelope across seeds
        if len(runs) >= 2:
            common_t = runs[0][1]["timesteps"].values
            stack = []
            for _, df in runs:
                if not np.array_equal(df["timesteps"].values, common_t):
                    # interpolate if timesteps don't align
                    y = np.interp(common_t, df["timesteps"].values, df[metric].values)
                else:
                    y = df[metric].values
                stack.append(y)
            stack = np.stack(stack, axis=0)
            mean = stack.mean(axis=0)
            lo, hi = stack.min(axis=0), stack.max(axis=0)
            ax.plot(common_t, mean, color="black", linewidth=2, label=f"mean (n={len(runs)})")
            ax.fill_between(common_t, lo, hi, color="black", alpha=0.12, label="min/max")
            ax.legend(loc="lower right", fontsize=8)
        ax.set_title(metric)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[2].set_xlabel("timesteps")
    axes[3].set_xlabel("timesteps")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="*", help="explicit run dirs containing accuracies.csv")
    p.add_argument("--glob", help="glob pattern for run dirs")
    p.add_argument("--latest", type=int, help="take N most-recently-modified dirs matching --glob or default gemini/cov/*")
    p.add_argument("--out", default="figures/learning_curves.png")
    p.add_argument("--title", default="")
    args = p.parse_args()

    dirs = list(args.dirs)
    if args.glob:
        dirs += glob.glob(args.glob)
    if args.latest:
        candidates = dirs or glob.glob("gemini/cov/*")
        candidates = [c for c in candidates if os.path.isdir(c)]
        candidates.sort(key=lambda d: os.path.getmtime(d), reverse=True)
        dirs = candidates[: args.latest]

    if not dirs:
        print("no run dirs supplied", file=sys.stderr)
        sys.exit(2)

    runs = load_runs(dirs)
    if not runs:
        print("no readable runs", file=sys.stderr)
        sys.exit(2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    title = args.title or f"learning curves: {len(runs)} run(s)"
    plot(runs, args.out, title)


if __name__ == "__main__":
    main()
