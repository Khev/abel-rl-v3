#!/usr/bin/env python3
"""Compare greedy action traces: baseline (seed8001 final) vs anti-loop (seed9100 best).

Per the action-trace diagnostic on baseline seed8001, the dominant failure mode
was the agent falling into REL-REL-REL... loops on test equations. Anti-loop
penalty (α=0.1) was launched in response. seed9100 trained with anti-loop hit
test_beam=0.341 at t=1.65M (vs baseline 0.308 at t=3M).

This script:
  1. Loads both models.
  2. Runs greedy on all 91 test eqns for each.
  3. Categorizes solved/failed AND identifies which actions each policy preferred.
  4. Shows the per-eqn "did anti-loop unlock this eqn?" / "did it break a baseline win?"
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import sympy as sp
from sb3_contrib import MaskablePPO

from envs.env_multi_eqn import multiEqn
from utils.utils_env import make_actions


BASELINE_CKPT = "data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/seed8001/checkpoints/latest.zip"
ANTILOOP_CKPT = "data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/seed9100/checkpoints/model_step1650000.zip"
TEST_FILE     = "equation_templates/mixed_v2_easy/test_eqns.txt"


def load_eqns(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def trace_one(env, model, eqn, max_steps=10):
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
    return bool(info.get("is_solved")), ops


SHORT = {
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
    "mul": "*", "truediv": "/", "add": "+", "sub": "-", "pow": "^",
}


def short(o):
    return SHORT.get(o, o[:4].upper())


def fraction_repeating(ops):
    """Heuristic: count fraction of steps that match the previous one."""
    if len(ops) < 2:
        return 0.0
    rep = sum(1 for i in range(1, len(ops)) if ops[i] == ops[i-1])
    return rep / (len(ops) - 1)


def main():
    test = load_eqns(TEST_FILE)
    env = multiEqn(
        gen="mixed_v2_easy",
        state_rep="graph_integer_1d",
        use_cov=True,
        use_relabel_constants=True,
        use_success_replay=True,
    )

    print(f"Loading baseline: {BASELINE_CKPT}")
    baseline = MaskablePPO.load(BASELINE_CKPT, env=env, device="cpu")
    print(f"Loading anti-loop: {ANTILOOP_CKPT}")
    antiloop = MaskablePPO.load(ANTILOOP_CKPT, env=env, device="cpu")

    # Per-eqn results
    rows = []  # (eqn, baseline_solved, baseline_ops, antiloop_solved, antiloop_ops)
    for e in test:
        try:
            bs, bo = trace_one(env, baseline, e)
        except Exception:
            bs, bo = False, []
        try:
            asv, ao = trace_one(env, antiloop, e)
        except Exception:
            asv, ao = False, []
        rows.append((e, bs, bo, asv, ao))

    n = len(rows)
    base_solved = sum(1 for r in rows if r[1])
    ant_solved  = sum(1 for r in rows if r[3])
    both        = sum(1 for r in rows if r[1] and r[3])
    only_base   = sum(1 for r in rows if r[1] and not r[3])
    only_ant    = sum(1 for r in rows if not r[1] and r[3])
    neither     = sum(1 for r in rows if not r[1] and not r[3])

    print(f"\n=== Greedy solve count ===")
    print(f"  Baseline (seed8001 final) : {base_solved}/{n} ({100*base_solved/n:.1f}%)")
    print(f"  Anti-loop (seed9100 peak) : {ant_solved}/{n} ({100*ant_solved/n:.1f}%)")
    print(f"\n=== Overlap ===")
    print(f"  Both solved   : {both}")
    print(f"  Only baseline : {only_base}")
    print(f"  Only anti-loop: {only_ant}")
    print(f"  Neither       : {neither}")

    # Loop fraction on failed traces
    base_fail_rep = [fraction_repeating(r[2]) for r in rows if not r[1] and r[2]]
    ant_fail_rep  = [fraction_repeating(r[4]) for r in rows if not r[3] and r[4]]
    if base_fail_rep:
        print(f"\n=== Repeat-action fraction on FAILED traces ===")
        print(f"  Baseline failed traces: mean={sum(base_fail_rep)/len(base_fail_rep):.2f}, n={len(base_fail_rep)}")
        print(f"  Anti-loop failed traces: mean={sum(ant_fail_rep)/len(ant_fail_rep):.2f}, n={len(ant_fail_rep)}")

    # First-action distribution
    print(f"\n=== First-action distribution (test set) ===")
    bf = Counter(r[2][0] if r[2] else "<empty>" for r in rows)
    af = Counter(r[4][0] if r[4] else "<empty>" for r in rows)
    all_keys = sorted(set(bf) | set(af), key=lambda k: -max(bf.get(k, 0), af.get(k, 0)))
    print(f"  {'op':30s}  base  ant")
    for k in all_keys:
        print(f"  {k:30s}  {bf.get(k, 0):4d}  {af.get(k, 0):4d}")

    # Show eqns ANTILOOP unlocked
    print(f"\n=== Equations anti-loop unlocked (failed in baseline, solved here) ===")
    for e, bs, bo, asv, ao in rows:
        if not bs and asv:
            print(f"  eqn = {e}")
            print(f"    base : {' '.join(short(x) for x in bo)}")
            print(f"    anti : {' '.join(short(x) for x in ao)}")

    # Show eqns baseline solved but anti-loop didn't (regressions)
    print(f"\n=== Regressions (baseline solved, anti-loop failed) ===")
    for e, bs, bo, asv, ao in rows:
        if bs and not asv:
            print(f"  eqn = {e}")
            print(f"    base : {' '.join(short(x) for x in bo)}")
            print(f"    anti : {' '.join(short(x) for x in ao)}")


if __name__ == "__main__":
    main()
