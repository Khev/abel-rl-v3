#!/usr/bin/env python3
"""Regenerate the curiosity-ablation figure (appendix Fig. 7).

Curiosity bonuses (ICM, RND, NGU) vs. a no-curiosity baseline on
abel_level3. The baseline is a mean +/- min/max band over the seeds
that reach the full 3M-step horizon; each curiosity variant is a single
seed. The previous version of this figure was clipped to 5e5 steps, at
which the baseline had not yet risen -- this regenerates it at 3e6.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("data/dynamic_actions/abel_level3_hidden_dim256_nenvs1")
HORIZON = 3_000_000

# Baseline seeds on the 500k grid that reach the 3M horizon.
BASELINE_SEEDS = ["seed7001", "seed7006", "seed7011", "seed8001"]

# (label, run dir, color)
VARIANTS = [
    ("ppo-tree + RND",  BASE / "ppo-tree-RND" / "seed4000", "#ff7f0e"),
    ("ppo-tree + ICM",  BASE / "ppo-tree-ICM" / "seed1000", "#2ca02c"),
    ("ppo-tree + NGU",  BASE / "ppo-tree-NGU" / "seed6000", "#d62728"),
]

METRICS = [
    ("coverage",    "coverage"),
    ("test_greedy", "test\\_greedy"),
    ("test_beam",   "test\\_beam"),
    ("test_at10",   "test\\_at10"),
]


def load(csv):
    df = pd.read_csv(csv)
    return df[df["step"] <= HORIZON]


def main():
    base = [load(BASE / "ppo-tree" / s / "learning_curves.csv")
            for s in BASELINE_SEEDS]
    # Common step grid for the baseline band.
    steps = sorted(set(base[0]["step"]).intersection(*(set(d["step"]) for d in base[1:])))
    steps = np.array([s for s in steps if s <= HORIZON])

    variants = [(label, load(d / "learning_curves.csv"), color)
                for label, d, color in VARIANTS]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, title) in zip(axes.flat, METRICS):
        stacked = np.array([[d.set_index("step").loc[s, col] for s in steps]
                            for d in base])
        ax.plot(steps / 1e6, stacked.mean(0), color="#1f77b4",
                linewidth=2, label=f"ppo-tree (baseline, n={len(base)})")
        ax.fill_between(steps / 1e6, stacked.min(0), stacked.max(0),
                        color="#1f77b4", alpha=0.18)
        for label, df, color in variants:
            ax.plot(df["step"] / 1e6, df[col], color=color,
                    linewidth=2, label=f"{label} (n=1)")
        ax.set_title(title)
        ax.set_xlabel("training steps (millions)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("fraction solved")
    axes[1, 0].set_ylabel("fraction solved")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Curiosity bonuses do not improve TreeMLP learning "
                 "on abel\\_level3", fontsize=12)
    fig.tight_layout()

    out = Path("papers/curiosity-tree-hurts.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")

    print("\nFinal values at 3M steps:")
    base_final = {col: np.mean([d.set_index("step").loc[steps[-1], col]
                                for d in base]) for col, _ in METRICS}
    print(f"  baseline (n={len(base)}):  " +
          "  ".join(f"{c}={base_final[c]:.2f}" for c, _ in METRICS))
    for label, df, _ in variants:
        last = df.iloc[-1]
        print(f"  {label:18s}  " +
              "  ".join(f"{c}={last[c]:.2f}" for c, _ in METRICS))


if __name__ == "__main__":
    main()
