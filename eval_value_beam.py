#!/usr/bin/env python3
"""Compare plain beam vs value-guided beam on an existing checkpoint.

Loads a trained MaskablePPO model, runs beam search on the test set under
multiple (lambda, alpha, beta) settings, and reports test_beam for each.

The model is unchanged — we just re-use its policy + value heads at decode time.
"""
import argparse
import sys
from pathlib import Path

# Match train_abel.py: disable Python 3.11 int-string limit (sympy can hit it).
try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass

import sympy as sp
from sb3_contrib import MaskablePPO

from envs.env_multi_eqn import multiEqn
from train_abel import beam_accuracy, greedy_accuracy


def load_eqns(path):
    """Return sympified expressions (matches what `env.test_eqns` looks like)."""
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            out.append(sp.sympify(ln))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to .zip checkpoint")
    p.add_argument("--gen", default="mixed_v2_easy")
    p.add_argument("--test_file", default=None,
                   help="Path to test_eqns.txt. Default: equation_templates/<gen>/test_eqns.txt")
    p.add_argument("--beam_width", type=int, default=5)
    p.add_argument("--topk_per_node", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--per_eqn_seconds", type=float, default=0.75)
    # Sweep grid for (lambda, alpha, beta). Each is a comma-separated list.
    p.add_argument("--lambdas", default="0.0,0.1,0.3,1.0",
                   help="Comma-separated lambda values (value-head bonus weight)")
    p.add_argument("--alphas", default="0.0",
                   help="Comma-separated alpha values (per-step length penalty)")
    p.add_argument("--betas", default="0.0",
                   help="Comma-separated beta values (complexity penalty)")
    p.add_argument("--include_greedy", action="store_true",
                   help="Also report greedy accuracy for reference")
    args = p.parse_args()

    test_file = args.test_file or f"equation_templates/{args.gen}/test_eqns.txt"
    test = load_eqns(test_file)
    print(f"Test set: {test_file} ({len(test)} eqns)")
    print(f"Checkpoint: {args.ckpt}")

    env = multiEqn(
        gen=args.gen,
        state_rep="graph_integer_1d",
        use_cov=True,
        use_relabel_constants=True,
        use_success_replay=True,
    )
    model = MaskablePPO.load(args.ckpt, env=env, device="cpu")

    if args.include_greedy:
        gacc = greedy_accuracy(model, env, test,
                               max_steps=args.max_steps,
                               per_eqn_seconds=args.per_eqn_seconds)
        print(f"\ngreedy: test_acc = {gacc:.4f}")

    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    alphas  = [float(x) for x in args.alphas.split(",") if x.strip()]
    betas   = [float(x) for x in args.betas.split(",") if x.strip()]

    print(f"\n{'lambda':>8s}  {'alpha':>6s}  {'beta':>6s}  test_beam  delta_vs_base")
    base = None
    for lam in lambdas:
        for a in alphas:
            for b in betas:
                acc = beam_accuracy(
                    model, env, test,
                    beam_width=args.beam_width,
                    topk_per_node=args.topk_per_node,
                    max_steps=args.max_steps,
                    per_eqn_seconds=args.per_eqn_seconds,
                    beam_lambda=lam, beam_alpha=a, beam_beta=b,
                )
                if base is None and lam == 0.0 and a == 0.0 and b == 0.0:
                    base = acc
                delta = (acc - base) if base is not None else 0.0
                print(f"  {lam:6.2f}  {a:6.2f}  {b:6.2f}  {acc:9.4f}  {delta:+.4f}")


if __name__ == "__main__":
    main()
