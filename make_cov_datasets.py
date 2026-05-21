#!/usr/bin/env python3
"""Build cov_tiny / cov_small / cov_large -- pure change-of-variables datasets
for training pi_cov independently (train_cov.py / covEnv).

Design (decided 2026-05-20):
  * 7 coefficient symbols a..g  (matches the open_small / mixed_v2_easy pool).
  * 4 families x {general, monic} forms; exponential has a single form.
    Equations are symbol PERMUTATIONS of fixed templates -- no integer
    coefficients, so pi_cov must learn the *structural* substitution rule.
  * Every template depresses to a pure x^n + const by construction; this is
    re-verified here with sympy before anything is written.
  * One shared held-out test set (~20%, stratified by family AND form).
  * Nested train sets: cov_tiny  subset  cov_small  subset  cov_large.

Outputs: equation_templates/cov_{tiny,small,large}/{train,test}_eqns.txt
Each equation string means `<expr> = 0`.
"""
import itertools
import random
from pathlib import Path

import sympy as sp

SEED = 0
SYMBOLS = list("abcdefg")            # 7-symbol pool, matches open_small
x = sp.Symbol("x")
OUT = Path(__file__).resolve().parent / "equation_templates"
TEST_FRAC = 0.20

# --- templates: general builders take 3 distinct symbols; monic take 2 ------
def quad_general(A, B, C):    return f"{A}*x**2 + {B}*x + {C}"
def quad_monic(b, c):         return f"x**2 + 2*{b}*x + {c}"
def cubic_general(A, B, D):   return f"{A}*x**3 + {B}*x**2 + {B}**2*x/(3*{A}) + {D}"
def cubic_monic(b, d):        return f"x**3 + 3*{b}*x**2 + 3*{b}**2*x + {d}"
def quartic_general(A, B, E): return f"{A}*x**4 + {B}*x**3 + 3*{B}**2*x**2/(8*{A}) + {B}**3*x/(16*{A}**2) + {E}"
def quartic_monic(b, e):      return f"x**4 + 4*{b}*x**3 + 6*{b}**2*x**2 + 4*{b}**3*x + {e}"
def exp_form(A, B, C):        return f"{A}*exp(x) + {B}*exp(-x) + {C}"

# (name, n_slots, builder, kind)
BUCKETS = [
    ("quadratic_general", 3, quad_general,    "poly"),
    ("quadratic_monic",   2, quad_monic,      "poly"),
    ("cubic_general",     3, cubic_general,   "poly"),
    ("cubic_monic",       2, cubic_monic,     "poly"),
    ("quartic_general",   3, quartic_general, "poly"),
    ("quartic_monic",     2, quartic_monic,   "poly"),
    ("exponential",       3, exp_form,        "exp"),
]
MONIC = {"quadratic_monic", "cubic_monic", "quartic_monic"}

# per-dataset train counts: (per general-type bucket, per monic bucket).
# None = take the whole remaining pool.
SIZE_SPEC = {
    "cov_tiny":  (7,    3),
    "cov_small": (50,   10),
    "cov_large": (None, None),
}


def validate_poly(eqn_str):
    """Confirm the polynomial depresses (x -> x - B/nA) to a pure x^n + const."""
    expr = sp.sympify(eqn_str)
    p = sp.Poly(expr, x)
    n = p.degree()
    A, B = p.all_coeffs()[0], p.all_coeffs()[1]
    dep = sp.expand(expr.subs(x, x - B / (n * A)))
    mid = sp.Poly(dep, x).all_coeffs()[1:-1]          # x^1 .. x^{n-1}
    bad = [m for m in mid if sp.simplify(m) != 0]
    return not bad, bad


def main():
    # --- enumerate + validate every bucket -------------------------------
    buckets = {}
    print("buckets (symbol permutations of each template):")
    for name, n_slots, builder, kind in BUCKETS:
        eqns = [builder(*c) for c in itertools.permutations(SYMBOLS, n_slots)]
        buckets[name] = eqns
        if kind == "poly":
            for s in eqns[:3]:
                ok, bad = validate_poly(s)
                assert ok, f"{name}: '{s}' does NOT depress cleanly -> {bad}"
        print(f"  {name:20s} {len(eqns):4d}  ({'validated' if kind=='poly' else 'exp form'})")

    # --- shared test split + per-bucket train pools (deterministic) -------
    rng = random.Random(SEED)
    test = []
    train_pool = {}
    for name, *_ in BUCKETS:
        eqns = buckets[name][:]
        rng.shuffle(eqns)
        n_test = max(2, round(TEST_FRAC * len(eqns)))
        test += eqns[:n_test]
        train_pool[name] = eqns[n_test:]
    rng.shuffle(test)

    # --- nested train sets (prefixes of the same shuffled pools) ----------
    print("datasets:")
    for ds, (n_gen, n_monic) in SIZE_SPEC.items():
        train = []
        for name, *_ in BUCKETS:
            pool = train_pool[name]
            k = n_monic if name in MONIC else n_gen
            train += pool if k is None else pool[:k]
        random.Random(SEED + 1).shuffle(train)
        d = OUT / ds
        d.mkdir(parents=True, exist_ok=True)
        _write(d / "train_eqns.txt", train, f"{ds} train")
        _write(d / "test_eqns.txt", test, "shared CoV test set (identical for cov_tiny/small/large)")
        print(f"  {ds:10s} train={len(train):4d}  test={len(test)}  -> {d}")


def _write(path, eqns, label):
    with open(path, "w") as f:
        f.write(f"# {len(eqns)} equations -- {label}\n")
        for e in eqns:
            f.write(e + "\n")


if __name__ == "__main__":
    main()
