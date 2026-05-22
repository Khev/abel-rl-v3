#!/usr/bin/env python3
"""Localize the +1 symbol mismatch: train/test location, label, oracle,
model prediction, encoding -- all in ONE process."""
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
from envs.env_cov import covEnv
from envs.env_multi_eqn import pi_cov_general

x = sp.Symbol("x")
SYMS = list("abcdefg")
FAMS = ["quadratic", "cubic", "quartic", "exponential"]
MODES = ["general", "monic"]
N_OF = {"quadratic": 2, "cubic": 3, "quartic": 4}
TB = [sp.sympify(t) for t in "a,b,c,d,e,f,g,2,3,4".split(",")]
DS = "equation_templates/cov_large"


def read(p):
    return [l.strip() for l in open(p) if l.strip() and not l.startswith("#")]


def slots_of(e):
    if e.has(sp.exp):
        return ("exponential", None, None, None)
    p = sp.Poly(e, x); n = p.degree(); cs = p.all_coeffs(); A, B = cs[0], cs[1]
    fam = {2: "quadratic", 3: "cubic", 4: "quartic"}[n]
    if A == 1:
        return (fam, "monic", str(sp.simplify(B / n)), None)
    return (fam, "general", str(B), str(A))


def render(f, m, nu, le):
    if f == "exponential":
        return sp.log(x)
    n = N_OF[f]; B = sp.Symbol(nu)
    if m == "monic":
        return x - B
    return x - B / (n * sp.Symbol(le))


class SlotModel(nn.Module):
    def __init__(s, d, h=256):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        s.family = nn.Linear(h, 4); s.mode = nn.Linear(h, 2)
        s.num = nn.Linear(h, 7); s.lead = nn.Linear(h, 7)

    def forward(s, z):
        h = s.trunk(z)
        return s.family(h), s.mode(h), s.num(h), s.lead(h)


_env = covEnv(main_eqn=sp.sympify(read(f"{DS}/train_eqns.txt")[0]), term_bank=TB,
              max_depth=3, step_penalty=0.1, f_penalty=0.0, hist_len=10,
              multi_eqn=False, use_curriculum=False, state_rep="integer_1d",
              dataset_path=DS)

def encode(e):
    return np.asarray(_env.to_vec(e, 0)[0], dtype=np.float32).flatten()


IN = len(encode(sp.sympify(read(f"{DS}/train_eqns.txt")[0])))
m = SlotModel(IN)
m.load_state_dict(torch.load("gemini/cov/pi_cov_slots_best.pt", map_location="cpu"))
m.eval()


def predict(e):
    with torch.no_grad():
        lf, lm, ln, ll = m(torch.tensor(encode(e)).unsqueeze(0))
    return (FAMS[int(lf.argmax())], MODES[int(lm.argmax())],
            SYMS[int(ln.argmax())], SYMS[int(ll.argmax())])


tr = set(read(f"{DS}/train_eqns.txt"))
te = set(read(f"{DS}/test_eqns.txt"))
print(f"IN_DIM={IN}  train={len(tr)} test={len(te)}")

for s in ["a*x**2 + b*x + c", "a*x**2 + d*x + b", "a*x**2 + e*x + b"]:
    e = sp.sympify(s)
    loc = "TRAIN" if s in tr else ("TEST" if s in te else "NEITHER")
    pr = predict(e)
    print(f"\n{s}  [{loc}]")
    print(f"  slots_of       = {slots_of(e)}")
    print(f"  pi_cov_general = {pi_cov_general(e)}")
    print(f"  model raw pred = {pr}")

# model accuracy on the 192 test, IN THIS PROCESS
te_l = read(f"{DS}/test_eqns.txt")
ok_sl = ok_or = 0
for s in te_l:
    e = sp.sympify(s)
    pr = predict(e)
    rp = render(pr[0], pr[1], pr[2], pr[3] if pr[1] == "general" else None)
    if sp.simplify(rp - render(*slots_of(e))) == 0:
        ok_sl += 1
    o = pi_cov_general(e)
    if o is not None and sp.simplify(rp - o) == 0:
        ok_or += 1
print(f"\n[this process] model vs slots_of-label on 192 test: {ok_sl}/192")
print(f"[this process] model vs pi_cov_general  on 192 test: {ok_or}/192")

# and on the 774 train
tr_l = read(f"{DS}/train_eqns.txt")
ok_tr = 0
for s in tr_l:
    e = sp.sympify(s)
    pr = predict(e)
    rp = render(pr[0], pr[1], pr[2], pr[3] if pr[1] == "general" else None)
    o = pi_cov_general(e)
    if o is not None and sp.simplify(rp - o) == 0:
        ok_tr += 1
print(f"[this process] model vs pi_cov_general  on 774 train: {ok_tr}/774")
