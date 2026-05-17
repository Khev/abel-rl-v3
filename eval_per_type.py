#!/usr/bin/env python3
"""Evaluate a trained closed-equation model and report accuracy per equation type.

Buckets test equations by structural type (polynomial / rational / trig / log /
exp / radical / mixed) using SymPy's atom inspection. Then runs greedy eval
on each bucket and reports per-type accuracy.

Usage:
    python eval_per_type.py --run_dir data/dynamic_actions/.../seed7001 \
                            --gen abel_level3
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import sympy as sp
import numpy as np

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.env_multi_eqn import multiEqn as multiEqnDynamic
from envs.env_multi_eqn_fixed import multiEqn as multiEqnFixed


def _get_action_mask(env):
    return env.get_valid_action_mask()


def classify(eqn):
    """Return a coarse structural label for a SymPy equation.
    Order of checks matters: more specific first."""
    x = sp.symbols("x")
    expr = sp.sympify(eqn)
    atoms_funcs = expr.atoms(sp.Function)
    has_trig = any(isinstance(a.func, type(sp.sin)) or
                   a.func in (sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan)
                   for a in atoms_funcs)
    has_log = any(a.func is sp.log for a in atoms_funcs)
    has_exp = any(a.func is sp.exp for a in atoms_funcs)
    has_sqrt = expr.has(sp.sqrt) or any(
        isinstance(a, sp.Pow) and a.exp == sp.Rational(1, 2)
        for a in sp.preorder_traversal(expr)
    )

    # rational? look for x in denominator
    num, den = sp.together(expr).as_numer_denom()
    has_rational = den != 1 and den.has(x)

    # polynomial-only check
    is_polynomial = expr.as_poly(x) is not None
    if is_polynomial and not has_trig and not has_log and not has_exp and not has_sqrt:
        deg = expr.as_poly(x).degree()
        return f"poly-deg{deg}"

    if has_trig and not (has_log or has_exp):
        return "trig"
    if has_log and not has_exp:
        return "log"
    if has_exp and not has_log:
        return "exp"
    if has_log and has_exp:
        return "log+exp"
    if has_sqrt and not (has_log or has_exp or has_trig):
        return "radical"
    if has_rational and not (has_log or has_exp or has_trig or has_sqrt):
        return "rational"
    return "mixed"


def greedy_solve(env, model, eqn, max_steps=10):
    """Reset env on eqn, run greedy, return True if solved."""
    obs, _ = env.reset()
    setter = getattr(env, "set_equation", None) or getattr(getattr(env, "unwrapped", env), "set_equation", None)
    if setter:
        setter(sp.sympify(eqn))
        underlying = env.unwrapped if hasattr(env, "unwrapped") else env
        obs = getattr(underlying, "state", obs)
    for _ in range(max_steps):
        try:
            mask = env.get_valid_action_mask()
        except Exception:
            mask = None
        if isinstance(model, MaskablePPO) and mask is not None:
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        if info.get("is_solved"):
            return True
        if terminated or truncated:
            return False
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--gen", required=True)
    p.add_argument("--action_space", default="dynamic", choices=["dynamic", "fixed"])
    p.add_argument("--use_cov", action="store_true")
    p.add_argument("--use_relabel_constants", action="store_true")
    p.add_argument("--state_rep", default=None)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--split", choices=["train", "test", "both"], default="test")
    args = p.parse_args()

    # state rep
    if args.state_rep is None:
        run_name = os.path.basename(os.path.dirname(args.run_dir.rstrip("/")))
        args.state_rep = "graph_integer_1d" if any(t in run_name for t in ("tree","gnn","sage")) else "integer_1d"

    EnvCls = multiEqnDynamic if args.action_space == "dynamic" else multiEqnFixed
    env = EnvCls(
        gen=args.gen,
        state_rep=args.state_rep,
        use_cov=args.use_cov,
        use_relabel_constants=args.use_relabel_constants,
        sparse_rewards=False,
        use_curriculum=False,
    )
    if args.action_space == "dynamic":
        env = ActionMasker(env, _get_action_mask)

    rd = Path(args.run_dir)
    model_path = None
    for cand in ["final_model.zip", "model.zip"]:
        if (rd / cand).exists():
            model_path = rd / cand
            break
    if model_path is None:
        zips = list(rd.glob("*.zip"))
        model_path = zips[0] if zips else None
    if model_path is None or not model_path.exists():
        print(f"no model.zip in {rd}", file=sys.stderr)
        sys.exit(2)

    try:
        model = MaskablePPO.load(str(model_path), env=env, device="cpu")
    except Exception:
        model = PPO.load(str(model_path), env=env, device="cpu")

    underlying = env.unwrapped if hasattr(env, "unwrapped") else env
    train_eqns = list(getattr(underlying, "train_eqns", []))
    test_eqns = list(getattr(underlying, "test_eqns", []))

    splits = {}
    if args.split in ("train", "both"):
        splits["train"] = train_eqns
    if args.split in ("test", "both"):
        splits["test"] = test_eqns

    for split_name, eqns in splits.items():
        print(f"\n=== {split_name} ({len(eqns)} eqns) ===")
        by_type = defaultdict(list)
        for e in eqns:
            try:
                tag = classify(e)
            except Exception:
                tag = "unknown"
            by_type[tag].append(e)

        # Sort buckets by size
        rows = []
        for tag, group in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
            solved = sum(greedy_solve(env, model, e, max_steps=args.max_steps) for e in group)
            rate = solved / len(group)
            rows.append((tag, len(group), solved, rate))
            print(f"  {tag:<14s} n={len(group):4d}  solved={solved:4d}  acc={rate:.3f}")

        # Aggregate
        total_n = sum(r[1] for r in rows)
        total_solved = sum(r[2] for r in rows)
        print(f"  {'OVERALL':<14s} n={total_n:4d}  solved={total_solved:4d}  "
              f"acc={total_solved/total_n:.3f}")


if __name__ == "__main__":
    main()
