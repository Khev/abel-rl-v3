#!/usr/bin/env python3
"""Generate the paper's headline closed-equations.png learning-curve figure.

Layout: 3 datasets (rows) x 3 metrics (coverage / test_greedy / test_beam) =
9 panels. Each panel overlays the 4 algorithms: ppo, ppo-tree, ppo-tree-rc,
ppo-tree-rc-buf. Mean + min/max envelope across seeds.

Coverage is omitted for poesia (training set is unbounded).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Mapping: (path_prefix_under_data/, agent_subdir) per algo + dataset
DATASETS = ["abel_level3", "abel_level4", "poesia-full"]
LABELS_DATASET = {"abel_level3": "small", "abel_level4": "large", "poesia-full": "poesia"}
METRICS = ["coverage", "test_greedy", "test_beam"]
METRIC_LABELS = {"coverage": "coverage", "test_greedy": "test greedy", "test_beam": "test beam (w=5)"}

# Toggle directories that mean specific algorithms
ALGOS = [
    ("ppo",             "dynamic_actions",                                                  "ppo"),
    ("ppo-tree",        "dynamic_actions",                                                  "ppo-tree"),
    ("ppo-tree-rc",     "dynamic_actions/use_relabel_constants",                            "ppo-tree"),
    ("ppo-tree-rc-buf", "dynamic_actions/use_relabel_constants/use_buffer",                 "ppo-tree"),
]


def find_seed_csvs(dataset, toggle_path, agent_subdir):
    """Locate all learning_curves.csv files for a (dataset, toggle, agent)."""
    base = Path("data") / toggle_path / f"{dataset}_hidden_dim256_nenvs1" / agent_subdir
    if not base.exists():
        return []
    return sorted(base.glob("seed*/learning_curves.csv"))


def aggregate(csvs, metric):
    if not csvs:
        return None, None, None, None
    dfs = []
    for c in csvs:
        try:
            df = pd.read_csv(c)
            if metric in df.columns and len(df):
                dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None, None, None, None
    # use first run's time axis as canonical (column "step")
    common_t = dfs[0]["step"].values
    stack = []
    for df in dfs:
        t = df["step"].values
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


def main():
    fig, axes = plt.subplots(len(DATASETS), len(METRICS),
                             figsize=(11, 8), sharex="row")
    if len(DATASETS) == 1:
        axes = axes[None, :]
    colors = ["#888888", "#1f77b4", "#2ca02c", "#d62728"]

    for i, ds in enumerate(DATASETS):
        for j, m in enumerate(METRICS):
            ax = axes[i, j]
            if ds == "poesia-full" and m == "coverage":
                ax.axis("off")
                continue
            for k, (algo_label, toggle, agent_subdir) in enumerate(ALGOS):
                csvs = find_seed_csvs(ds, toggle, agent_subdir)
                t, mean, lo, hi = aggregate(csvs, m)
                if t is None:
                    continue
                ax.plot(t, mean, color=colors[k], linewidth=1.6,
                        label=f"{algo_label} (n={len(csvs)})" if (i==0 and j==1) else None)
                ax.fill_between(t, lo, hi, color=colors[k], alpha=0.15)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(METRIC_LABELS[m])
            if j == 0:
                ax.set_ylabel(LABELS_DATASET[ds])
            if i == len(DATASETS) - 1:
                ax.set_xlabel("steps")
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # one legend for the figure (in top-middle panel)
    axes[0, 1].legend(loc="lower right", fontsize=8, frameon=True)
    fig.suptitle("Learning curves for closed equations (mean ± min/max across seeds)",
                 fontsize=11, y=0.995)
    fig.tight_layout()
    out = "figures/closed_equations_headline.png"
    os.makedirs("figures", exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
