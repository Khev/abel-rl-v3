#!/usr/bin/env python3
"""Load a trained CoV model and print its solution trace on a target equation.

Usage:
    python trace_cov.py --run_dir gemini/cov/ppo_2026-05-17_10-06-51 \
                        --gen 1 --main_eqn 'a*x + b'

    # use train set from a class:
    python trace_cov.py --run_dir <dir> --gen 5
"""
import argparse
import os
import sys
from pathlib import Path

import sympy as sp
import numpy as np

# Imports must match training-side
from stable_baselines3 import PPO
from envs.env_cov import covEnv


def parse_term_bank(s: str):
    return [sp.sympify(t) for t in s.split(",")]


def trace_one(env, model, equation, deterministic=True, max_steps=20):
    """Run one episode greedily and print the trace."""
    obs, _ = env.reset()
    if equation is not None:
        env.main_eqn = sp.sympify(equation)
        # re-encode obs after pinning equation
        env.base_cmplx = env.__class__.__mro__[0].__init__.__defaults__  # placeholder
        try:
            from envs.env_cov import C
            env.base_cmplx = C(env.main_eqn)
        except Exception:
            pass
        env.obs = env.to_vec(env.main_eqn, 0)[0]
        obs = env._augment_obs(env.obs)

    print(f"\nEquation: {env.main_eqn} = 0")
    print(f"Initial complexity: {env.base_cmplx}")
    print("-" * 60)

    actions_taken = []
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)
        op, term = env.actions[action]
        actions_taken.append((op, term))
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step {step+1}: action=({op}, {term}), reward={reward:.3f}, "
              f"depth={env.depth}, cov_so_far={env.cov}")
        if terminated or truncated:
            break

    print("-" * 60)
    print(f"Final substitution f(x) = {info.get('cov', env.cov)}")
    eqn_after = info.get("equation_after")
    if eqn_after is not None:
        print(f"Equation after CoV: {eqn_after} = 0")
        print(f"Post-CoV complexity: {info.get('after_complexity')}")
        print(f"Δ complexity: {info.get('delta_complexity')}")
    print(f"Success: {info.get('delta_complexity', 0) > 0}")
    return actions_taken, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="dir containing model.zip")
    p.add_argument("--gen", type=int, default=5)
    p.add_argument("--dataset_path", type=str, default=None,
                   help="Override gen-based lookup; load train/test_eqns from this dir.")
    p.add_argument("--main_eqn", type=str, default="a*x**2 + b*x + c")
    p.add_argument("--term_bank", type=str, default="a,b,c,d,e,2,3,4")
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--state_rep", type=str, default=None,
                   help="Override state rep. Auto-detected from agent name if not given.")
    p.add_argument("--n_train_samples", type=int, default=5,
                   help="How many train equations to trace (set 0 to skip).")
    p.add_argument("--n_test_samples", type=int, default=5,
                   help="How many test equations to trace.")
    args = p.parse_args()

    # detect state_rep from run dir name if not given
    if args.state_rep is None:
        run_name = os.path.basename(args.run_dir.rstrip("/"))
        args.state_rep = "graph_integer_1d" if "tree" in run_name else "integer_1d"

    main_eqn = sp.sympify(args.main_eqn)
    term_bank = parse_term_bank(args.term_bank)

    env = covEnv(
        main_eqn=main_eqn,
        term_bank=term_bank,
        max_depth=args.max_depth,
        step_penalty=0.1,
        state_rep=args.state_rep,
        hist_len=10,
        multi_eqn=True,
        use_curriculum=False,
        gen=args.gen,
        dataset_path=args.dataset_path,
    )

    model_path = Path(args.run_dir) / "model.zip"
    if not model_path.exists():
        print(f"no model at {model_path}", file=sys.stderr)
        sys.exit(2)
    model = PPO.load(str(model_path), env=env, device="cpu")

    print(f"\n=== Run: {args.run_dir} ===")
    print(f"Train pool: {len(env.train_eqns)} eqns")
    print(f"Test pool:  {len(env.test_eqns)} eqns")

    successes_tr = successes_te = 0
    for i, eqn in enumerate(env.train_eqns[: args.n_train_samples]):
        print(f"\n>>> TRAIN [{i+1}/{args.n_train_samples}]")
        _, info = trace_one(env, model, str(eqn))
        if info.get("delta_complexity", 0) > 0:
            successes_tr += 1

    for i, eqn in enumerate(env.test_eqns[: args.n_test_samples]):
        print(f"\n>>> TEST [{i+1}/{args.n_test_samples}]")
        _, info = trace_one(env, model, str(eqn))
        if info.get("delta_complexity", 0) > 0:
            successes_te += 1

    print("\n" + "=" * 60)
    if args.n_train_samples > 0:
        print(f"train successes: {successes_tr}/{args.n_train_samples}")
    if args.n_test_samples > 0:
        print(f"test successes:  {successes_te}/{args.n_test_samples}")


if __name__ == "__main__":
    main()
