#!/usr/bin/env python3
"""Aggregate all summary.csv files under data/ into a master table.

Each summary.csv comes from a single train_abel.py run and looks like:
    agent, coverage_mean, coverage_std,
    final_test_greedy_mean, final_test_greedy_std,
    final_test_beam_mean,   final_test_beam_std,
    final_test_at10_mean,   final_test_at10_std

The path tells us the dataset (`abel_level3`, `cov_level5`, `poesia-full`, ...)
and the inductive-bias toggles (use_relabel_constants/use_buffer/etc.).

Output: one row per (dataset, inductive-bias combo, agent), with all metrics.
"""
import os
import re
import sys
from pathlib import Path
import pandas as pd


ROOT = Path("data")


# Inductive-bias toggle directory names we recognize
TOGGLES = {
    "dynamic_actions":         "dyn",
    "use_relabel_constants":   "rc",
    "use_buffer":              "buf",
    "no_curriculum":           "no-curr",
    "sparse_rewards":          "sparse",
}


def parse_path(p: Path):
    """Return (dataset, toggles_set) from a path like
    data/dynamic_actions/use_relabel_constants/use_buffer/abel_level3_hidden_dim256_nenvs1/summary.csv
    """
    # everything from after `data/` up to (not including) the leaf dir
    parts = list(p.relative_to(ROOT).parts[:-1])
    # the leaf-1 part is the run-config dir, e.g. abel_level3_hidden_dim256_nenvs1
    cfg = parts[-1]
    # match dataset prefix: greedy match until _hidden_dim or end
    m = re.match(r"(.+?)(?:_hidden_dim\d+)?(?:_nenvs\d+)?$", cfg)
    dataset = m.group(1) if m else cfg
    toggles = set()
    for d in parts[:-1]:
        if d in TOGGLES:
            toggles.add(TOGGLES[d])
    return dataset, toggles


def main():
    rows = []
    for p in sorted(ROOT.rglob("summary.csv")):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[warn] could not read {p}: {e}", file=sys.stderr)
            continue
        dataset, toggles = parse_path(p)
        toggle_str = "+".join(sorted(toggles)) if toggles else "(base)"
        for _, r in df.iterrows():
            rows.append({
                "dataset":  dataset,
                "toggles":  toggle_str,
                "agent":    r.get("agent"),
                "cov":      r.get("coverage_mean"),
                "cov_std":  r.get("coverage_std"),
                "greedy":   r.get("final_test_greedy_mean"),
                "greedy_std": r.get("final_test_greedy_std"),
                "beam":     r.get("final_test_beam_mean"),
                "beam_std": r.get("final_test_beam_std"),
                "at10":     r.get("final_test_at10_mean"),
                "at10_std": r.get("final_test_at10_std"),
                "path":     str(p),
            })

    if not rows:
        print("No summary.csv files found.")
        return

    out = pd.DataFrame(rows)
    out.to_csv("results_all.csv", index=False)
    print(f"Wrote results_all.csv with {len(out)} rows.\n")

    # Compact view focused on closed-eq paper Table 1 inputs
    print("=" * 88)
    print("Compact view (closed-eq runs only, sorted by dataset, toggles, agent):")
    print("=" * 88)
    closed_mask = out["dataset"].str.startswith("abel_level") | out["dataset"].str.startswith("poesia")
    closed = out[closed_mask].copy()
    closed = closed.sort_values(["dataset", "toggles", "agent"])
    # Round for readability
    for col in ["cov", "greedy", "beam", "at10"]:
        closed[col] = closed[col].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (int, float)) else "—"
        )
    print(closed[["dataset", "toggles", "agent", "cov", "greedy", "beam", "at10"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
