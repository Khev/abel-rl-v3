#!/usr/bin/env python3
"""End-to-end evaluation of a trained pi_cov model on open equations.

For each test equation:
  1. Apply pi_cov(eqn) -> substitution f(x).
  2. Substitute x := f(x) into eqn, simplify.
  3. Check whether the post-CoV equation is "pure" (<=2 polynomial terms).
  4. Optionally, solve the pure equation with SymPy as the final closed-eq step.

Reports per-class accuracy: fraction of test equations the pi_cov successfully
depresses. This is the natural ICLR Table 3 / Figure 7 number for the open-
equations section.

Usage:
    python eval_pi_cov.py --model_dir gemini/cov/ppo-tree-mem_2026-... \\
                          --dataset_path equation_templates/cov_v2_small/quadratic
"""
import argparse
import os
import sys
from pathlib import Path

import sympy as sp
import numpy as np


def is_pure_post_cov(after, max_terms=2):
    """Same convention as eqn_gen/base.py: post-CoV must be <= max_terms in x."""
    x = sp.symbols("x")
    try:
        # Try polynomial form first (covers quadratic/cubic/quartic depression)
        poly = after.as_poly(x)
        if poly is not None:
            return len([c for c in poly.all_coeffs() if c != 0]) <= max_terms
        # Rational form (exp depression yields a/x + bx + c after substitution)
        simplified = sp.together(after)
        num, _ = simplified.as_numer_denom()
        poly = sp.expand(num).as_poly(x)
        if poly is not None:
            return len([c for c in poly.all_coeffs() if c != 0]) <= max_terms + 1
        return False
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True,
                   help="dir containing model.zip (the trained pi_cov)")
    p.add_argument("--dataset_path", required=True,
                   help="cov_v2_small/<class> directory")
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--state_rep", default="graph_integer_1d")
    p.add_argument("--term_bank", default="a,b,c,d,e,2,3,4")
    p.add_argument("--max_n", type=int, default=200,
                   help="cap test set size")
    args = p.parse_args()

    from utils.pi_cov_learned import make_pi_cov_learned

    model_dir = Path(args.model_dir)
    model_path = None
    for cand in ["model.zip", "final_model.zip"]:
        if (model_dir / cand).exists():
            model_path = model_dir / cand; break
    if model_path is None:
        print(f"no model.zip in {model_dir}", file=sys.stderr)
        sys.exit(2)

    pi_cov = make_pi_cov_learned(
        model_path=model_path,
        dataset_path=args.dataset_path,
        term_bank=args.term_bank,
        max_depth=args.max_depth,
        state_rep=args.state_rep,
    )

    # Load test set
    test_file = Path(args.dataset_path) / "test_eqns.txt"
    with open(test_file) as f:
        eqns = [sp.sympify(line.strip()) for line in f if line.strip() and not line.startswith("#")]
    if args.max_n and len(eqns) > args.max_n:
        eqns = eqns[: args.max_n]
    print(f"loaded {len(eqns)} test equations from {test_file}")

    x = sp.symbols("x")
    n_depressed = 0
    n_nontrivial = 0
    n_none = 0
    examples_pass = []
    examples_fail = []

    for i, eqn in enumerate(eqns):
        f = pi_cov(eqn)
        if f is None:
            n_none += 1
            continue
        try:
            after = sp.simplify(eqn.subs(x, f))
        except Exception:
            n_none += 1
            continue
        if is_pure_post_cov(after):
            n_depressed += 1
            if f != x:  # x→x is trivial; doesn't count as a real cov
                n_nontrivial += 1
                if len(examples_pass) < 3:
                    examples_pass.append((eqn, f, after))
        else:
            if len(examples_fail) < 3:
                examples_fail.append((eqn, f, after))

    n = len(eqns)
    print(f"\n=== Results on {Path(args.dataset_path).name} ===")
    print(f"  depressed (post-CoV is pure):     {n_depressed}/{n} = {n_depressed/n:.3f}")
    print(f"  nontrivial depression (f != x):   {n_nontrivial}/{n} = {n_nontrivial/n:.3f}")
    print(f"  no substitution returned:         {n_none}/{n}")

    print("\nFirst few passes:")
    for eqn, f, after in examples_pass[:3]:
        print(f"  {eqn}\n     -> f={f}\n     -> after={after}")
    print("\nFirst few failures:")
    for eqn, f, after in examples_fail[:3]:
        print(f"  {eqn}\n     -> f={f}\n     -> after={after}")


if __name__ == "__main__":
    main()
