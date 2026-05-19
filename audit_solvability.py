#!/usr/bin/env python3
"""Audit: for each eqn in our datasets, check whether the closed-form
solution requires an action our env doesn't have.

Tracks the operations that appear in sp.solve()'s output. If any solution
uses something we can't apply (e.g., cube-root, fourth-root with rational
denominator, etc.), flag it.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass

import sympy as sp

# Operations our env can produce on the lhs/rhs
SUPPORTED_FUNCS = {"Pow_1_2", "log", "exp", "sin", "asin", "cos", "acos", "Mul_neg1"}
# We also have add/sub/mul/truediv with arbitrary terms, expand, collect.

# Functions that appear in solutions but indicate UNSUPPORTED inverse ops:
# - Pow with exponent 1/3 or 1/k for k > 4 = cube root / nth root
# - any "RootOf" expression = irrational degree-3+ algebraic
# - acot, asinh, etc. = exotic inverses


def required_ops(sol):
    """Return a set of op-name strings used in `sol` that we'd need."""
    used = set()
    for node in sp.preorder_traversal(sol):
        if isinstance(node, sp.Pow):
            base, exp = node.args
            if exp.is_Rational:
                num, den = exp.as_numer_denom()
                if den == 1:
                    continue  # integer power
                if abs(int(num)) == 1 and int(den) == 2:
                    used.add("sqrt")
                elif abs(int(num)) == 1 and int(den) == 4:
                    used.add("sqrt+sqrt")
                elif abs(int(num)) == 1 and int(den) == 3:
                    used.add("cbrt")
                else:
                    used.add(f"Pow_{num}/{den}")
            else:
                used.add(f"Pow_{exp}")
        elif isinstance(node, sp.Function):
            used.add(type(node).__name__)
        elif type(node).__name__ in ("RootOf", "CRootOf"):
            used.add("RootOf")
    return used


def audit_file(path: Path, max_n: int = None):
    x = sp.symbols("x")
    eqns = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            eqns.append(ln)
    if max_n:
        eqns = eqns[:max_n]

    ops_counter = Counter()
    unsolvable = []
    no_solution = []
    examples = defaultdict(list)
    for e_str in eqns:
        try:
            e = sp.sympify(e_str)
            sols = sp.solve(e, x, check=False)
        except Exception as ex:
            no_solution.append((e_str, f"err: {type(ex).__name__}"))
            continue
        if not sols:
            no_solution.append((e_str, "empty"))
            continue
        # Pick the FIRST solution (simplest assumption)
        sol = sols[0]
        ops = required_ops(sol)
        for op in ops:
            ops_counter[op] += 1
            examples[op].append(e_str)
        # Flag if any unsupported op
        unsupported = {o for o in ops if o == "cbrt" or o.startswith("Pow_") and o not in ("Pow_1/2", "Pow_1/4")}
        if unsupported:
            unsolvable.append((e_str, unsupported, str(sol)[:80]))
    return ops_counter, unsolvable, no_solution, examples, len(eqns)


def main():
    classes = [
        "cov_v2_small/quadratic",
        "cov_v2_small/cubic",
        "cov_v2_small/quartic",
        "cov_v2_small/exponential",
        "abel_level1",
        "abel_level2",
        "abel_level3",
    ]
    print(f"{'class':30s}  n  | ops used (count) | n_unsupported")
    print("-" * 100)
    for c in classes:
        path = Path(f"equation_templates/{c}/test_eqns.txt")
        if not path.exists():
            print(f"{c:30s}  -- not found --")
            continue
        ops, unsupported, no_sol, examples, n = audit_file(path, max_n=30)
        ops_str = ", ".join(f"{k}({v})" for k, v in ops.most_common(6))
        n_unsupp = len(unsupported)
        print(f"{c:30s}  {n:3d} | {ops_str[:50]:50s} | {n_unsupp:3d}")
        if unsupported and len(unsupported) <= 3:
            for e_str, missing, sol in unsupported[:3]:
                print(f"    needs={missing}: {e_str[:70]}  →  {sol}")
        if no_sol:
            print(f"    no-solve (n={len(no_sol)}): first = {no_sol[0]}")


if __name__ == "__main__":
    main()
