#!/usr/bin/env python3
"""Compare pi_cov_general (analytic oracle, "pi_cov_perfect") vs the trained
slot-head pi_cov, equation by equation, on the cov datasets. Answers: are they
the same function on our distribution?

Usage: python compare_pi_cov.py [model.pt] [dataset_dir ...]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass
import numpy as np
import sympy as sp
import torch
import torch.nn as nn
from collections import defaultdict
from envs.env_cov import covEnv
from envs.env_multi_eqn import pi_cov_general

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemini/cov/pi_cov_slots_best.pt"
DATASETS = sys.argv[2:] if len(sys.argv) > 2 else ["equation_templates/cov_large"]
ENC_DS = "equation_templates/cov_large"   # feature_dict reference (training dataset)

x = sp.Symbol("x")
SYMS = list("abcdefg")
FAMS = ["quadratic", "cubic", "quartic", "exponential"]
MODES = ["general", "monic"]
N_OF = {"quadratic": 2, "cubic": 3, "quartic": 4}
TERM_BANK = [sp.sympify(t) for t in "a,b,c,d,e,f,g,2,3,4".split(",")]


def read_eqns(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


def slots_of(expr):
    if expr.has(sp.exp):
        return ("exponential", None, None, None)
    p = sp.Poly(expr, x)
    n = p.degree()
    cs = p.all_coeffs()
    A, B = cs[0], cs[1]
    fam = {2: "quadratic", 3: "cubic", 4: "quartic"}[n]
    if A == 1:
        return (fam, "monic", str(sp.simplify(B / n)), None)
    return (fam, "general", str(B), str(A))


def render(family, mode, num_sym, lead_sym):
    if family == "exponential":
        return sp.log(x)
    n = N_OF[family]
    B = sp.Symbol(num_sym)
    if mode == "monic":
        return x - B
    return x - B / (n * sp.Symbol(lead_sym))


class SlotModel(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.family = nn.Linear(hidden, 4)
        self.mode = nn.Linear(hidden, 2)
        self.num = nn.Linear(hidden, 7)
        self.lead = nn.Linear(hidden, 7)

    def forward(self, z):
        h = self.trunk(z)
        return self.family(h), self.mode(h), self.num(h), self.lead(h)


_env = covEnv(main_eqn=sp.sympify(read_eqns(f"{ENC_DS}/train_eqns.txt")[0]),
              term_bank=TERM_BANK, max_depth=3, step_penalty=0.1, f_penalty=0.0,
              hist_len=10, multi_eqn=False, use_curriculum=False,
              state_rep="integer_1d", dataset_path=ENC_DS)

def encode(expr):
    return np.asarray(_env.to_vec(expr, 0)[0], dtype=np.float32).flatten()


IN_DIM = len(encode(sp.sympify(read_eqns(f"{ENC_DS}/train_eqns.txt")[0])))
model = SlotModel(IN_DIM)
model.load_state_dict(torch.load(MODEL, map_location="cpu"))
model.eval()
print(f"loaded slot-head: {MODEL}")


def pi_cov_trained(expr):
    with torch.no_grad():
        lf, lm, ln, ll = model(torch.tensor(encode(expr)).unsqueeze(0))
    fam = FAMS[int(lf.argmax(1))]
    if fam == "exponential":
        return render(fam, None, None, None)
    mode = MODES[int(lm.argmax(1))]
    num = SYMS[int(ln.argmax(1))]
    lead = SYMS[int(ll.argmax(1))] if mode == "general" else None
    return render(fam, mode, num, lead)


# ----------------------------------------------------------- compare
eqns = []
for ds in DATASETS:
    for fn in ("train_eqns.txt", "test_eqns.txt"):
        p = f"{ds}/{fn}"
        if os.path.isfile(p):
            eqns += read_eqns(p)
eqns = sorted(set(eqns))
print(f"comparing on {len(eqns)} unique equations from {DATASETS}\n")

fam_stat = defaultdict(lambda: [0, 0])   # family/mode -> [match, total]
match = oracle_none = 0
mism = []
for s in eqns:
    e = sp.sympify(s)
    try:
        fam, mode, _, _ = slots_of(e)
        key = f"{fam}/{mode or '-'}"
    except Exception:
        key = "?"
    o = pi_cov_general(e)
    if o is None:
        oracle_none += 1
        continue
    t = pi_cov_trained(e)
    try:
        same = sp.simplify(o - t) == 0
    except Exception:
        same = False
    fam_stat[key][0] += int(same)
    fam_stat[key][1] += 1
    if same:
        match += 1
    else:
        mism.append((s, o, t))

n = len(eqns) - oracle_none
print(f"{'family/mode':22s} {'match':>10s}")
print("-" * 34)
for k in sorted(fam_stat):
    m, tot = fam_stat[k]
    print(f"{k:22s} {m:4d}/{tot:<4d} {m/tot:.3f}")
print("-" * 34)
print(f"{'OVERALL':22s} {match:4d}/{n:<4d} {match/n:.4f}")
if oracle_none:
    print(f"(oracle returned None on {oracle_none} equations)")
for s, o, t in mism[:12]:
    print(f"  MISMATCH: {s}\n    oracle ={o}\n    trained={t}")
