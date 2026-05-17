#!/usr/bin/env python3
"""Load a trained closed-equation model and print solution traces.

Mirrors trace_cov.py but for train_abel.py / multiEqn / multiEqnDynamic.

Usage:
    python trace_abel.py --run_dir data/dynamic_actions/abel_level3.../ppo-tree/seed7001 \
                         --gen abel_level3 [--n_train_samples 5 --n_test_samples 5]
"""
import argparse
import os
import sys
from pathlib import Path

import sympy as sp
import numpy as np
import torch

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.env_multi_eqn import multiEqn as multiEqnDynamic
from envs.env_multi_eqn_fixed import multiEqn as multiEqnFixed
from utils.utils_env import make_actions


def _get_action_mask(env):
    return env.get_valid_action_mask()


def trace_one(env, model, equation, *, max_steps=10, deterministic=True):
    """Run one greedy episode on a pinned equation; print the trace."""
    env.reset()
    if callable(getattr(env, "set_equation", None)):
        env.set_equation(sp.sympify(equation))
    else:
        env.main_eqn = sp.sympify(equation)
        env.lhs = env.main_eqn
        env.rhs = 0
        env.setup()
        env.state, _ = env.to_vec(env.lhs, env.rhs)

    print(f"\nEquation: {env.main_eqn} = 0")
    print(f"Initial complexity (proxy): cnt={env.action_dim} actions; lhs nodes ≈ count_ops")
    print("-" * 70)

    obs = env.state
    for step in range(max_steps):
        # If env is wrapped with ActionMasker, mask is auto; for raw env we pass it ourselves
        try:
            mask = env.get_valid_action_mask()
        except Exception:
            mask = None
        if isinstance(model, MaskablePPO) and mask is not None:
            action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)
        # action -> (op, term) lookup against fresh action list
        action_list, _ = make_actions(env.lhs, env.rhs, env.actions_fixed, env.action_dim)
        op, term = action_list[action]
        op_name = op.__name__ if hasattr(op, "__name__") else str(op)

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step {step+1}: action=({op_name}, {term}), reward={reward:.3f}, "
              f"lhs={env.lhs}, rhs={env.rhs}, "
              f"solved={info.get('is_solved')}, valid={info.get('is_valid_eqn')}")
        if terminated or truncated:
            break

    print("-" * 70)
    print(f"Final: lhs={env.lhs}, rhs={env.rhs}")
    print(f"  is_solved={info.get('is_solved')}, action_taken={info.get('action_taken')}")
    print(f"  cov_depth={info.get('cov_depth', 0)}")
    return bool(info.get("is_solved"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="dir containing final_model.zip or seed<N>.zip")
    p.add_argument("--gen", type=str, required=True, help="dataset (e.g. abel_level3)")
    p.add_argument("--action_space", type=str, default="dynamic", choices=["dynamic", "fixed"])
    p.add_argument("--use_cov", action="store_true")
    p.add_argument("--use_relabel_constants", action="store_true")
    p.add_argument("--state_rep", type=str, default=None,
                   help="default: inferred from run_dir name (tree/gnn/sage -> graph_integer_1d).")
    p.add_argument("--n_train_samples", type=int, default=5)
    p.add_argument("--n_test_samples", type=int, default=5)
    args = p.parse_args()

    # state rep
    if args.state_rep is None:
        run_name = os.path.basename(os.path.dirname(args.run_dir.rstrip("/")))
        args.state_rep = "graph_integer_1d" if any(t in run_name for t in ("tree","gnn","sage")) else "integer_1d"

    # env (dynamic by default, matches train_abel.py default)
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

    # locate model
    rd = Path(args.run_dir)
    model_path = None
    for cand in ["final_model.zip", *[f"seed{rd.name.replace('seed','')}.zip"], "model.zip"]:
        if (rd / cand).exists():
            model_path = rd / cand
            break
    if model_path is None:
        zips = list(rd.glob("*.zip"))
        if zips:
            model_path = zips[0]
    if model_path is None or not model_path.exists():
        print(f"no model.zip in {rd}", file=sys.stderr)
        sys.exit(2)

    # load (try Maskable first since dynamic action space uses it)
    print(f"loading {model_path}")
    try:
        model = MaskablePPO.load(str(model_path), env=env, device="cpu")
        print("  (MaskablePPO)")
    except Exception:
        model = PPO.load(str(model_path), env=env, device="cpu")
        print("  (PPO)")

    underlying = env.unwrapped if hasattr(env, "unwrapped") else env
    train_eqns = getattr(underlying, "train_eqns", [])
    test_eqns = getattr(underlying, "test_eqns", [])
    print(f"\nTrain pool: {len(train_eqns)}, Test pool: {len(test_eqns)}")

    succ_tr = succ_te = 0
    for i, eqn in enumerate(train_eqns[: args.n_train_samples]):
        print(f"\n>>> TRAIN [{i+1}/{args.n_train_samples}]")
        if trace_one(env, model, str(eqn)):
            succ_tr += 1
    for i, eqn in enumerate(test_eqns[: args.n_test_samples]):
        print(f"\n>>> TEST [{i+1}/{args.n_test_samples}]")
        if trace_one(env, model, str(eqn)):
            succ_te += 1

    print("\n" + "=" * 70)
    if args.n_train_samples > 0:
        print(f"train: {succ_tr}/{args.n_train_samples}")
    if args.n_test_samples > 0:
        print(f"test:  {succ_te}/{args.n_test_samples}")


if __name__ == "__main__":
    main()
