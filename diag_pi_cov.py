#!/usr/bin/env python3
"""Diagnostic for pi_cov: family-wise greedy (argmax) vs beam accuracy on the
held-out test set. Shows WHERE greedy fails -- the input ChatGPT's structured
CoV head needs.

Usage: python diag_pi_cov.py [model.zip] [dataset_dir]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass
import sympy as sp

MODEL  = sys.argv[1] if len(sys.argv) > 1 else "gemini/cov/pi_cov_best.zip"
DATASET = sys.argv[2] if len(sys.argv) > 2 else "equation_templates/cov_large"
TERM_BANK_STR = "a,b,c,d,e,f,g,2,3,4"
x = sp.Symbol("x")

from envs.env_cov import covEnv
from train_cov import _beam_search_one          # importable: argparse is guarded

# --- load model (ppo-mem -> try MaskablePPO then PPO) ----------------------
model = None
for modname, clsname in [("sb3_contrib", "MaskablePPO"),
                         ("stable_baselines3", "PPO")]:
    try:
        cls = getattr(__import__(modname, fromlist=[clsname]), clsname)
        model = cls.load(MODEL, device="cpu")
        print(f"loaded {MODEL} with {clsname}")
        break
    except Exception as e:
        print(f"  ({clsname} load failed: {type(e).__name__})")
if model is None:
    sys.exit("could not load model")


def family_of(expr):
    """Classify a test equation -> (family, mode)."""
    if expr.has(sp.exp):
        return "exponential", "-"
    try:
        p = sp.Poly(expr, x)
        deg = p.degree()
        name = {2: "quadratic", 3: "cubic", 4: "quartic"}.get(deg, f"deg{deg}")
        mode = "monic" if p.LC() == 1 else "general"
        return name, mode
    except Exception:
        return "?", "?"


def read_eqns(path):
    out = []
    with open(path) as f:
        for ln in f:
            if ln.strip() and not ln.startswith("#"):
                out.append(ln.strip())
    return out


tb = [sp.sympify(t) for t in TERM_BANK_STR.split(",")]
test_eqns = read_eqns(f"{DATASET}/test_eqns.txt")
print(f"test equations: {len(test_eqns)}\n")

# bucket -> [greedy_successes, beam_successes, total]
from collections import defaultdict
stats = defaultdict(lambda: [0, 0, 0])

for i, eqn_s in enumerate(test_eqns):
    eqn = sp.sympify(eqn_s)
    fam, mode = family_of(eqn)
    key = f"{fam}/{mode}"
    env = covEnv(eqn, tb, max_depth=3, step_penalty=0.1, f_penalty=0.0,
                 hist_len=10, multi_eqn=False, use_curriculum=False,
                 state_rep="integer_1d", dataset_path=DATASET)
    # greedy (argmax) rollout
    obs, _ = env.reset()
    done = trunc = False
    steps = 0
    info = {}
    while not (done or trunc) and steps < 6:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(int(a))
        steps += 1
    g_ok = (info.get("delta_complexity", 0) or 0) > 0
    # beam (width 5)
    env2 = covEnv(eqn, tb, max_depth=3, step_penalty=0.1, f_penalty=0.0,
                  hist_len=10, multi_eqn=False, use_curriculum=False,
                  state_rep="integer_1d", dataset_path=DATASET)
    env2.reset()
    best_delta, _ = _beam_search_one(env2, model, width=5, max_depth=3)
    b_ok = best_delta > 0
    s = stats[key]
    s[0] += int(g_ok); s[1] += int(b_ok); s[2] += 1

print(f"{'family/mode':22s} {'n':>4s} {'greedy':>8s} {'beam':>8s}")
print("-" * 46)
tg = tb_ = tn = 0
for key in sorted(stats):
    g, b, n = stats[key]
    tg += g; tb_ += b; tn += n
    print(f"{key:22s} {n:4d} {g/n:8.3f} {b/n:8.3f}")
print("-" * 46)
print(f"{'OVERALL':22s} {tn:4d} {tg/tn:8.3f} {tb_/tn:8.3f}")
