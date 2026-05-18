#!/usr/bin/env python3
"""Trace full greedy action sequences on test eqns; categorize solved vs failed.

Goal: see where the policy diverges between solved and failed test eqns.
"""
import sys
from collections import Counter
from pathlib import Path

import sympy as sp
from sb3_contrib import MaskablePPO

from envs.env_multi_eqn import multiEqn
from utils.utils_env import make_actions


CKPT = "data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/seed8001/checkpoints/latest.zip"
TEST_FILE = "equation_templates/mixed_v2_easy/test_eqns.txt"


def load_eqns(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def trace_one(env, model, eqn, max_steps=10):
    """Return (solved, ops_sequence, final_lhs, final_rhs)."""
    env.reset()
    env.main_eqn = sp.sympify(eqn)
    env.lhs = env.main_eqn
    env.rhs = 0
    env.setup()
    env.state, _ = env.to_vec(env.lhs, env.rhs)
    obs = env.state

    ops = []
    info = {}
    for _ in range(max_steps):
        mask = env.get_valid_action_mask()
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        action = int(action)
        action_list, _ = make_actions(env.lhs, env.rhs, env.actions_fixed, env.action_dim)
        op, term = action_list[action]
        op_name = op.__name__ if hasattr(op, "__name__") else str(op)
        ops.append(op_name)
        obs, _, term_done, _, info = env.step(action)
        if term_done:
            break
    return bool(info.get("is_solved")), ops, env.lhs, env.rhs


def main():
    test = load_eqns(TEST_FILE)
    env = multiEqn(
        gen="mixed_v2_easy",
        state_rep="graph_integer_1d",
        use_cov=True,
        use_relabel_constants=True,
        use_success_replay=True,
    )
    model = MaskablePPO.load(CKPT, env=env, device="cpu")

    solved, failed = [], []
    for e in test:
        try:
            ok, ops, lhs, rhs = trace_one(env, model, e)
        except Exception as ex:
            failed.append((e, f"<err:{type(ex).__name__}>", [], None, None))
            continue
        (solved if ok else failed).append((e, ops, lhs, rhs))

    print(f"Solved: {len(solved)}/{len(test)} ({100*len(solved)/len(test):.1f}%)")
    print(f"Failed: {len(failed)}/{len(test)}\n")

    def short(o):
        return {
            "cov_action_placeholder": "COV",
            "relabel_const_custom": "REL",
            "custom_expand": "EXP",
            "custom_collect": "CLT",
            "custom_square": "SQR",
            "custom_sqrt": "SRT",
            "custom_log": "LOG",
            "custom_exp": "EXP_",
            "custom_sin": "SIN",
            "inverse_sin": "iSIN",
            "custom_cos": "COS",
            "inverse_cos": "iCOS",
            "mul": "*",
            "truediv": "/",
            "add": "+",
            "sub": "-",
            "pow": "^",
        }.get(o, o[:4].upper())

    print("=== Length distribution ===")
    sl = Counter(len(t[1]) for t in solved)
    fl = Counter(len(t[1]) for t in failed if isinstance(t[1], list))
    print(f"Solved   lengths: {dict(sorted(sl.items()))}")
    print(f"Failed   lengths: {dict(sorted(fl.items()))}")

    print("\n=== First-action distribution by outcome ===")
    cs = Counter(t[1][0] if t[1] else "<empty>" for t in solved)
    cf = Counter(t[1][0] if t[1] else "<empty>" for t in failed if isinstance(t[1], list))
    print("Solved first action:")
    for k, v in cs.most_common():
        print(f"  {k:30s} {v:3d}")
    print("Failed first action:")
    for k, v in cf.most_common():
        print(f"  {k:30s} {v:3d}")

    print("\n=== Sample solved traces (first 8) ===")
    for eqn, ops, lhs, rhs in solved[:8]:
        print(f"  {' '.join(short(o) for o in ops):30s}  eqn={eqn}  ⇒  x={rhs}")

    print("\n=== Sample failed traces (first 12) ===")
    for eqn, ops, lhs, rhs in failed[:12]:
        ops_repr = ' '.join(short(o) for o in ops) if isinstance(ops, list) else ops
        print(f"  {ops_repr:30s}  eqn={eqn}  ⇒  {lhs} = {rhs}")


if __name__ == "__main__":
    main()
