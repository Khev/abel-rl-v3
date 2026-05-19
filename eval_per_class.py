#!/usr/bin/env python3
"""Per-equation-class breakdown of test accuracy.

For each checkpoint, run greedy + plain beam + value beam on the
mixed_v2_easy test set, then group results by source class
(quadratic/cubic/quartic/exponential/abel_level{1,2,3}).
"""
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path

try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass

import sympy as sp
from sb3_contrib import MaskablePPO

from envs.env_multi_eqn import multiEqn
from train_abel import greedy_solve_one, beam_solve_one


# Classes in mixed_v2_easy (matches mixed_v2_easy/build.py)
SOURCES = [
    ("cov_v2_small/quadratic",   "quadratic"),
    ("cov_v2_small/cubic",       "cubic"),
    ("cov_v2_small/quartic",     "quartic"),
    ("cov_v2_small/exponential", "exponential"),
    ("abel_level1",              "abel_level1"),
    ("abel_level2",              "abel_level2"),
    ("abel_level3",              "abel_level3"),
]

CKPTS = OrderedDict([
    ("baseline_seed8001",  "data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/seed8001/checkpoints/latest.zip"),
    ("antiloop_seed9100",  "data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/seed9100/checkpoints/latest.zip"),
    ("full_seed10000",     "data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/seed10000/checkpoints/latest.zip"),
])

TEST_FILE = "equation_templates/mixed_v2_easy/test_eqns.txt"


def load_eqns_with_class():
    """Return list of (eqn, class_name) for each test eqn, by lookup against
    the source files."""
    # Read all source test eqns
    source_eqns = {}
    for src_path, label in SOURCES:
        path = f"equation_templates/{src_path}/test_eqns.txt"
        try:
            with open(path) as f:
                source_eqns[label] = set(
                    ln.strip() for ln in f
                    if ln.strip() and not ln.startswith("#")
                )
        except FileNotFoundError:
            source_eqns[label] = set()

    # Read mixed test eqns and tag
    out = []
    with open(TEST_FILE) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            tag = None
            for src_path, label in SOURCES:
                if ln in source_eqns.get(label, set()):
                    tag = label
                    break
            out.append((sp.sympify(ln), tag if tag else "unknown"))
    return out


def main():
    eqns_tagged = load_eqns_with_class()
    print(f"Loaded {len(eqns_tagged)} test eqns. Class counts:")
    by_class = defaultdict(list)
    for e, c in eqns_tagged:
        by_class[c].append(e)
    for c, lst in by_class.items():
        print(f"  {c:20s} {len(lst):3d}")
    print()

    env = multiEqn(
        gen="mixed_v2_easy",
        state_rep="graph_integer_1d",
        use_cov=True,
        use_relabel_constants=True,
        use_success_replay=True,
    )

    rows = []  # (ckpt_name, class, method, n_solved, n_total)
    for ckpt_name, ckpt_path in CKPTS.items():
        print(f"=== {ckpt_name} ===")
        model = MaskablePPO.load(ckpt_path, env=env, device="cpu")
        for c, eqs in by_class.items():
            n = len(eqs)
            if n == 0:
                continue
            # Greedy
            g = sum(greedy_solve_one(model, env, e, max_steps=10,
                                     per_eqn_seconds=1.0) for e in eqs)
            # Plain beam
            pb = sum(beam_solve_one(model, env, e, beam_width=5, topk_per_node=5,
                                    max_steps=10, per_eqn_seconds=1.0,
                                    beam_lambda=0.0) for e in eqs)
            # Value beam
            vb = sum(beam_solve_one(model, env, e, beam_width=5, topk_per_node=5,
                                    max_steps=10, per_eqn_seconds=1.0,
                                    beam_lambda=1.0) for e in eqs)
            rows.append((ckpt_name, c, g, pb, vb, n))
            print(f"  {c:15s}  greedy={g}/{n}  plain={pb}/{n}  value={vb}/{n}")
        print()

    # Print summary table
    print("\n=== Summary (greedy / plain / value, fractions) ===")
    header = f"{'class':15s} | " + " | ".join(f"{ck:18s}" for ck in CKPTS) + " |"
    print(header)
    print("-" * len(header))
    classes = list(by_class.keys())
    for c in classes:
        cells = []
        for ck in CKPTS:
            row = next((r for r in rows if r[0] == ck and r[1] == c), None)
            if row:
                _, _, g, pb, vb, n = row
                cells.append(f"{g/n:.2f}/{pb/n:.2f}/{vb/n:.2f}")
            else:
                cells.append("n/a")
        print(f"{c:15s} | " + " | ".join(f"{x:18s}" for x in cells) + " |")


if __name__ == "__main__":
    main()
