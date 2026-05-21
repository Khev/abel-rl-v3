#!/usr/bin/env python3
"""Build cov_tiny / cov_small / cov_large -- pure change-of-variables datasets
for training pi_cov, WITH expert demonstration traces.

Design (decided 2026-05-20/21):
  * 7 coefficient symbols a..g  (matches the open_small / mixed_v2_easy pool).
  * 4 families x {general, monic} forms; exponential has a single form.
    Equations are symbol PERMUTATIONS of fixed templates -- no integer
    coefficients, so pi_cov must learn the *structural* substitution rule.
  * Every polynomial template depresses to a pure x^n + const by construction;
    re-verified here with sympy.
  * One shared held-out test set (~20%, stratified by family AND form).
  * Nested train sets: cov_tiny  subset  cov_small  subset  cov_large.
  * Expert traces: for each TRAIN equation, the covEnv (op,term) action
    sequence that builds the correct depression substitution. These seed the
    success-replay buffer (RL-from-demonstrations) -- pi_cov learns by
    refining correct demonstrations rather than blind discovery.
      general -> x - B/(nA):  SUB:B  DIV:n  DIV:A  STOP
      monic   -> x - b:       SUB:b  STOP
    (covEnv: first action sets base_op+cov; STOP / depth==max_depth finalizes
     f(x)=base_op(x,cov). Exponential needs a log op -- no trace yet.)

Outputs per dataset: train_eqns.txt, train_traces.txt (aligned), test_eqns.txt
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

# --- equation templates: general builders take 3 symbols, monic take 2 ------
def quad_general(A, B, C):    return f"{A}*x**2 + {B}*x + {C}"
def quad_monic(b, c):         return f"x**2 + 2*{b}*x + {c}"
def cubic_general(A, B, D):   return f"{A}*x**3 + {B}*x**2 + {B}**2*x/(3*{A}) + {D}"
def cubic_monic(b, d):        return f"x**3 + 3*{b}*x**2 + 3*{b}**2*x + {d}"
def quartic_general(A, B, E): return f"{A}*x**4 + {B}*x**3 + 3*{B}**2*x**2/(8*{A}) + {B}**3*x/(16*{A}**2) + {E}"
def quartic_monic(b, e):      return f"x**4 + 4*{b}*x**3 + 6*{b}**2*x**2 + 4*{b}**3*x + {e}"
def exp_form(A, B, C):        return f"{A}*exp(x) + {B}*exp(-x) + {C}"

# --- expert traces: the covEnv action sequence for the correct substitution -
def quad_general_tr(A, B, C):    return f"SUB:{B} DIV:2 DIV:{A} STOP"
def quad_monic_tr(b, c):         return f"SUB:{b} STOP"
def cubic_general_tr(A, B, D):   return f"SUB:{B} DIV:3 DIV:{A} STOP"
def cubic_monic_tr(b, d):        return f"SUB:{b} STOP"
def quartic_general_tr(A, B, E): return f"SUB:{B} DIV:4 DIV:{A} STOP"
def quartic_monic_tr(b, e):      return f"SUB:{b} STOP"
def exp_tr(A, B, C):             return "LOGX STOP"   # f(x)=log(x): a*e^x+b*e^-x+c -> a*x+b/x+c

# (name, n_slots, eqn_builder, trace_builder, kind)
BUCKETS = [
    ("quadratic_general", 3, quad_general,    quad_general_tr,    "poly"),
    ("quadratic_monic",   2, quad_monic,      quad_monic_tr,      "poly"),
    ("cubic_general",     3, cubic_general,   cubic_general_tr,   "poly"),
    ("cubic_monic",       2, cubic_monic,     cubic_monic_tr,     "poly"),
    ("quartic_general",   3, quartic_general, quartic_general_tr, "poly"),
    ("quartic_monic",     2, quartic_monic,   quartic_monic_tr,   "poly"),
    ("exponential",       3, exp_form,        exp_tr,             "exp"),
]
MONIC = {"quadratic_monic", "cubic_monic", "quartic_monic"}
SIZE_SPEC = {"cov_tiny": (7, 3), "cov_small": (50, 10), "cov_large": (None, None)}


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
    # --- enumerate + validate every bucket; items are (eqn, trace) ---------
    buckets = {}
    print("buckets (symbol permutations of each template):")
    for name, n_slots, builder, tr_builder, kind in BUCKETS:
        items = [(builder(*c), tr_builder(*c))
                 for c in itertools.permutations(SYMBOLS, n_slots)]
        buckets[name] = items
        if kind == "poly":
            for s, _ in items[:3]:
                ok, bad = validate_poly(s)
                assert ok, f"{name}: '{s}' does NOT depress cleanly -> {bad}"
        ntr = sum(1 for _, t in items if t)
        print(f"  {name:20s} {len(items):4d}  ({ntr} with expert trace)")

    # --- shared test split + per-bucket train pools (deterministic) -------
    rng = random.Random(SEED)
    test, train_pool = [], {}
    for name, *_ in BUCKETS:
        items = buckets[name][:]
        rng.shuffle(items)
        n_test = max(2, round(TEST_FRAC * len(items)))
        test += items[:n_test]
        train_pool[name] = items[n_test:]
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
        _write(d / "train_eqns.txt",   [e for e, _ in train], f"{ds} train")
        _write(d / "train_traces.txt", [t for _, t in train],
               f"{ds} expert traces (line-aligned with train_eqns.txt; blank = no demo)")
        _write(d / "test_eqns.txt",    [e for e, _ in test],
               "shared CoV test set (identical for cov_tiny/small/large)")
        ndemo = sum(1 for _, t in train if t)
        print(f"  {ds:10s} train={len(train):4d} ({ndemo} demos)  test={len(test)}  -> {d}")


def _write(path, items, label):
    with open(path, "w") as f:
        f.write(f"# {len(items)} lines -- {label}\n")
        for it in items:
            f.write(it + "\n")


if __name__ == "__main__":
    main()
