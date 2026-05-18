#!/usr/bin/env python3
"""Diagnostic: distribution of first greedy actions on test equations.

Hypothesis: agent fails on test eqns because it doesn't reliably pick
`relabel_const` as the FIRST action, so test eqns never get canonicalized.

Usage:
    python diagnose_first_action.py
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import sympy as sp
from sb3_contrib import MaskablePPO

from envs.env_multi_eqn import multiEqn
from utils.utils_env import make_actions


CKPT = "data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/seed8001/checkpoints/latest.zip"
TEST_FILE = "equation_templates/mixed_v2_easy/test_eqns.txt"
TRAIN_FILE = "equation_templates/mixed_v2_easy/train_eqns.txt"


def load_eqns(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def first_action(env, model, eqn):
    """Reset env, pin eqn, return (op_name, term) of greedy first action + solved-in-1?."""
    env.reset()
    env.main_eqn = sp.sympify(eqn)
    env.lhs = env.main_eqn
    env.rhs = 0
    env.setup()
    env.state, _ = env.to_vec(env.lhs, env.rhs)
    obs = env.state

    mask = env.get_valid_action_mask()
    action, _ = model.predict(obs, deterministic=True, action_masks=mask)
    action = int(action)

    action_list, _ = make_actions(env.lhs, env.rhs, env.actions_fixed, env.action_dim)
    op, term = action_list[action]
    op_name = op.__name__ if hasattr(op, "__name__") else str(op)
    return op_name, term


def main():
    print(f"Loading {CKPT}")
    test = load_eqns(TEST_FILE)
    train = load_eqns(TRAIN_FILE)
    print(f"Test set: {len(test)} eqns, Train set: {len(train)} eqns\n")

    env = multiEqn(
        gen="mixed_v2_easy",
        state_rep="graph_integer_1d",
        use_cov=True,
        use_relabel_constants=True,
        use_success_replay=True,
    )
    model = MaskablePPO.load(CKPT, env=env, device="cpu")

    for label, eqns in [("TEST", test), ("TRAIN", train[:200])]:
        print(f"=== {label} first-action distribution ({len(eqns)} eqns) ===")
        counter = Counter()
        for e in eqns:
            try:
                op_name, _ = first_action(env, model, e)
                counter[op_name] += 1
            except Exception as ex:
                counter[f"_err:{type(ex).__name__}"] += 1
        total = sum(counter.values())
        for op_name, n in counter.most_common():
            print(f"  {op_name:30s} {n:4d} ({100*n/total:.1f}%)")
        print()


if __name__ == "__main__":
    main()
