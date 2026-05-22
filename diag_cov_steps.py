#!/usr/bin/env python3
"""Per-step diagnostic: teacher-force each test equation through its expert
trace and check whether the policy's argmax matches the expert action at
each step. Shows WHICH step (and which family) greedy fails on.

Usage: python diag_cov_steps.py [model.zip] [dataset_dir]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass
import sympy as sp
from collections import defaultdict

MODEL  = sys.argv[1] if len(sys.argv) > 1 else "gemini/cov/pi_cov_best.zip"
DATASET = sys.argv[2] if len(sys.argv) > 2 else "equation_templates/cov_large"
TERM_BANK_STR = "a,b,c,d,e,f,g,2,3,4"
x = sp.Symbol("x")

from envs.env_cov import covEnv
from train_cov import _trace_to_action_indices
from stable_baselines3 import PPO

model = PPO.load(MODEL, device="cpu")
print(f"loaded {MODEL}\n")
tb = [sp.sympify(t) for t in TERM_BANK_STR.split(",")]


def family_and_trace(expr):
    """Return (family/mode, expert trace string)."""
    if expr.has(sp.exp):
        return "exponential/-", "LOGX STOP"
    p = sp.Poly(expr, x)
    n = p.degree()
    cs = p.all_coeffs()
    A, B = cs[0], cs[1]
    name = {2: "quadratic", 3: "cubic", 4: "quartic"}.get(n, f"deg{n}")
    if A == 1:                                  # monic: x^n + n*b*x^{n-1} + ...
        b = sp.simplify(B / n)
        return f"{name}/monic", f"SUB:{b} STOP"
    return f"{name}/general", f"SUB:{B} DIV:{n} DIV:{A} STOP"


def read_eqns(path):
    return [l.strip() for l in open(path)
            if l.strip() and not l.startswith("#")]


test_eqns = read_eqns(f"{DATASET}/test_eqns.txt")

# per family: list of per-step correctness (only the cov-building steps that
# actually matter -- excludes the final auto-finalize action)
step_hits = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # fam -> step -> [hit,tot]
all_correct = defaultdict(lambda: [0, 0])                      # fam -> [greedy-ok, tot]
first_wrong = defaultdict(lambda: defaultdict(int))           # fam -> step -> count

for eqn_s in test_eqns:
    eqn = sp.sympify(eqn_s)
    try:
        fam, trace = family_and_trace(eqn)
    except Exception:
        continue
    env = covEnv(eqn, tb, max_depth=3, step_penalty=0.1, f_penalty=0.0,
                 hist_len=10, multi_eqn=False, use_curriculum=False,
                 state_rep="integer_1d", dataset_path=DATASET)
    obs, _ = env.reset()
    try:
        idxs = _trace_to_action_indices(trace, env.actions)
    except Exception:
        continue
    # the steps that matter: all but the trailing STOP for monic/exp (1 cov
    # action), all but trailing for general (3 cov actions). Just check every
    # listed action except we stop teacher-forcing once the env terminates.
    n_check = len(idxs)
    ok_all = True
    fw = None
    for i, exp_a in enumerate(idxs):
        pred_a, _ = model.predict(obs, deterministic=True)
        hit = int(pred_a) == exp_a
        step_hits[fam][i][0] += int(hit)
        step_hits[fam][i][1] += 1
        if not hit and ok_all:
            ok_all = False
            fw = i
        out = env.step(exp_a)
        obs = out[0]
        if out[2] or out[3]:        # terminated/truncated
            break
    all_correct[fam][0] += int(ok_all)
    all_correct[fam][1] += 1
    if fw is not None:
        first_wrong[fam][fw] += 1

print(f"{'family/mode':20s} {'step1':>8s} {'step2':>8s} {'step3':>8s} "
      f"{'all-ok':>8s}  first-wrong-step")
print("-" * 72)
for fam in sorted(step_hits):
    cells = []
    for s in range(3):
        h, t = step_hits[fam].get(s, [0, 0])
        cells.append(f"{h/t:.2f}" if t else "  - ")
    ac_h, ac_t = all_correct[fam]
    fw = dict(first_wrong[fam])
    print(f"{fam:20s} {cells[0]:>8s} {cells[1]:>8s} {cells[2]:>8s} "
          f"{ac_h/ac_t:8.2f}  {fw}")
