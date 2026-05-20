#!/usr/bin/env python3
"""Open-equation learning curves -- the Fig-2 equivalent for the open
(mixed) dataset.

Three panels: coverage, test_greedy, test_beam vs training steps.
Three curves: baseline (seed8001), anti-loop (seed9100), and the
full-stack fresh-buffer headline run (seed14000).

The headline runs used --eval_lite, so test_beam is not in their
learning_curves.csv; it is reconstructed by eval_checkpoint_curve.py
into test_beam_curve.csv (column test_beam_value).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("data/dynamic_actions/use_relabel_constants/use_buffer/"
            "mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree")

# (label, seed dir, color, has_beam_in_csv)
RUNS = [
    ("baseline (ppo-tree-rc-buf-cov)", "seed8001", "#888888", True),
    ("+ anti-loop penalty",            "seed9100", "#1f77b4", True),
    ("+ fresh-buffer (full stack)",    "seed14000", "#d62728", False),
]


def load_run(seed_dir, has_beam):
    d = BASE / seed_dir
    lc = pd.read_csv(d / "learning_curves.csv")
    out = {"step": lc["step"].values,
           "coverage": lc["coverage"].values,
           "test_greedy": lc["test_greedy"].values}
    if has_beam and "test_beam" in lc and lc["test_beam"].notna().any():
        out["beam_step"] = lc["step"].values
        out["test_beam"] = lc["test_beam"].values
    else:
        # reconstructed curve from checkpoint re-eval
        cf = d / "test_beam_curve.csv"
        if cf.exists():
            cc = pd.read_csv(cf)
            out["beam_step"] = cc["step"].values
            out["test_beam"] = cc["test_beam_value"].values
        else:
            out["beam_step"] = None
            out["test_beam"] = None
    return out


def main():
    runs = [(label, load_run(sd, hb), color) for label, sd, color, hb in RUNS]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [
        ("coverage", "coverage", "step"),
        ("test\\_greedy", "test_greedy", "step"),
        ("test\\_beam", "test_beam", "beam_step"),
    ]
    for ax, (title, key, xkey) in zip(axes, panels):
        for label, data, color in runs:
            x = data.get(xkey)
            y = data.get(key)
            if x is None or y is None:
                continue
            ax.plot(np.asarray(x) / 1e6, y, label=label, color=color, linewidth=2)
        ax.set_xlabel("training steps (millions)")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction solved")
    axes[2].legend(fontsize=8, loc="lower right")
    fig.suptitle("Open-equation learning curves (mixed\\_v2\\_easy, 91 test eqns)",
                 fontsize=12)
    fig.tight_layout()

    out = Path("figures/open_eqn_curves.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")

    # Also print the final numbers for the caption
    print("\nFinal values:")
    for label, data, _ in runs:
        cov = data["coverage"][-1]
        tg = data["test_greedy"][-1]
        tb = data["test_beam"][-1] if data["test_beam"] is not None else float("nan")
        print(f"  {label:35s}  cov={cov:.2f}  greedy={tg:.2f}  beam={tb:.2f}")


if __name__ == "__main__":
    main()
