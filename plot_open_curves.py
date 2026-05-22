#!/usr/bin/env python3
"""Open-equation learning curves -- the Fig-2 equivalent for the open
(mixed) dataset.

Styled to match the closed-equation headline figure (closed-equations.png):
matplotlib categorical palette, panel titles on top, sci-notation step
axis, single legend in the first panel.

Three panels: Coverage, Test Greedy, Test Beam vs training steps.
Two curves: baseline (seed8001) and the full-stack fresh-buffer method.
The intermediate ablation rungs live in the ablation table, not here -- the
figure is the before/after money shot.

The full-stack curve aggregates over every seed listed for it that exists
on disk: as the extra background seeds (16000, 17000) finish, rerunning
this script automatically turns the single line into a mean +/- min/max
band, matching Figure 2.

The headline runs used --eval_lite, so test_beam is not in their
learning_curves.csv; it is reconstructed by eval_checkpoint_curve.py
into test_beam_curve.csv (column test_beam_value).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Match the closed-equation figure (Fig. 2): seaborn whitegrid + despine.
sns.set_theme(style="whitegrid")

BASE = Path("data/dynamic_actions/use_relabel_constants/use_buffer/"
            "mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree")

# (label, [seed dirs], color, test_beam_is_in_csv)
# Colors mirror closed-equations.png: blue = baseline, red = best/full stack.
RUNS = [
    ("baseline (ppo-tree-rc-buf-cov)", ["seed8001"], "#1f77b4", True),
    ("full method stack",
     ["seed14000", "seed16000", "seed17000"], "#d62728", False),
]

PANELS = [
    ("Coverage",    "coverage",    "step"),
    ("Test Greedy", "test_greedy", "step"),
    ("Test Beam",   "test_beam",   "beam_step"),
]


def load_seed(seed_dir, has_beam):
    """Load one seed's curves; return None if the run does not exist yet."""
    d = BASE / seed_dir
    lc_path = d / "learning_curves.csv"
    if not lc_path.exists():
        return None
    lc = pd.read_csv(lc_path)
    out = {"step": lc["step"].values,
           "coverage": lc["coverage"].values,
           "test_greedy": lc["test_greedy"].values}
    if has_beam and "test_beam" in lc and lc["test_beam"].notna().any():
        out["beam_step"] = lc["step"].values
        out["test_beam"] = lc["test_beam"].values
    else:
        cf = d / "test_beam_curve.csv"
        if cf.exists():
            cc = pd.read_csv(cf)
            out["beam_step"] = cc["step"].values
            out["test_beam"] = cc["test_beam_value"].values
        else:
            out["beam_step"] = None
            out["test_beam"] = None
    return out


def aggregate(seeds, key, xkey):
    """Mean / min / max across seeds on the first seed's x grid."""
    series = [(s[xkey], s[key]) for s in seeds
              if s.get(xkey) is not None and s.get(key) is not None]
    if not series:
        return None, None, None, None
    grid = np.asarray(series[0][0], dtype=float)
    stack = []
    for x, y in series:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        stack.append(y if np.array_equal(x, grid) else np.interp(grid, x, y))
    stack = np.stack(stack, axis=0)
    return grid, stack.mean(0), stack.min(0), stack.max(0)


def main():
    runs = []
    for label, seed_dirs, color, hb in RUNS:
        seeds = [s for s in (load_seed(sd, hb) for sd in seed_dirs)
                 if s is not None]
        if seeds:
            runs.append((label, seeds, color))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharex=True)
    for ax, (title, key, xkey) in zip(axes, PANELS):
        for label, seeds, color in runs:
            x, mean, lo, hi = aggregate(seeds, key, xkey)
            if x is None:
                continue
            n = sum(1 for s in seeds if s.get(key) is not None)
            ax.plot(x, mean, color=color, linewidth=1.6,
                    label=(f"{label} (n={n})" if title == "Coverage" else None))
            if n > 1:
                ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylim(-0.02, 1.02)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    axes[0].set_ylabel("fraction solved")
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    sns.despine(fig=fig)
    fig.tight_layout()

    out = Path("figures/open_eqn_curves.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")

    print("\nFinal values:")
    for label, seeds, _ in runs:
        cov = seeds[0]["coverage"][-1]
        tg = seeds[0]["test_greedy"][-1]
        tb = (seeds[0]["test_beam"][-1]
              if seeds[0]["test_beam"] is not None else float("nan"))
        print(f"  {label:35s}  n={len(seeds)}  "
              f"cov={cov:.2f}  greedy={tg:.2f}  beam={tb:.2f}")


if __name__ == "__main__":
    main()
