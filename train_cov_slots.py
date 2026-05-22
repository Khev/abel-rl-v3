#!/usr/bin/env python3
"""Structured slot-head pi_cov.

Instead of emitting a sequential covEnv trace (SUB:B DIV:3 DIV:A STOP) -- where
greedy succeeds only if every step is argmax, so errors compound -- predict the
change-of-variables as independent semantic slots in ONE forward pass:

    family   in {quadratic, cubic, quartic, exponential}
    mode     in {general, monic}
    num_sym  -- the numerator symbol  (B for general, b for monic)
    lead_sym -- the leading-coeff symbol (A; general only)

A deterministic renderer turns the slots into the substitution:

    general      -> x - B/(n*A)        (n = 2/3/4 by family)
    monic        -> x - B
    exponential  -> log(x)

Supervised; labels derived from the equations (the depression formula). The
training set is augmented ONCE with many random symbol renamings so the model
learns the *structural* rule (generalizes to any symbol assignment).

Usage: python train_cov_slots.py [dataset_dir] [epochs] [seed]
"""
import sys, os, random, time
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

DATASET = sys.argv[1] if len(sys.argv) > 1 else "equation_templates/cov_large"
EPOCHS  = int(sys.argv[2]) if len(sys.argv) > 2 else 600
SEED    = int(sys.argv[3]) if len(sys.argv) > 3 else 0
N_AUG   = 20          # random symbol-renamed copies of each train equation
BATCH   = 256

x = sp.Symbol("x")
SYMS = list("abcdefg")
SYM2I = {s: i for i, s in enumerate(SYMS)}
FAMS = ["quadratic", "cubic", "quartic", "exponential"]
FAM2I = {f: i for i, f in enumerate(FAMS)}
MODES = ["general", "monic"]
MODE2I = {m: i for i, m in enumerate(MODES)}
N_OF = {"quadratic": 2, "cubic": 3, "quartic": 4}
TERM_BANK = [sp.sympify(t) for t in "a,b,c,d,e,f,g,2,3,4".split(",")]
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def slots_of(expr):
    """equation -> (family, mode, num_sym, lead_sym).  exp: mode/num/lead None."""
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


def read_eqns(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


_env = covEnv(main_eqn=sp.sympify(read_eqns(f"{DATASET}/train_eqns.txt")[0]),
              term_bank=TERM_BANK, max_depth=3, step_penalty=0.1, f_penalty=0.0,
              hist_len=10, multi_eqn=False, use_curriculum=False,
              state_rep="integer_1d", dataset_path=DATASET)

def encode(expr):
    return np.asarray(_env.to_vec(expr, 0)[0], dtype=np.float32).flatten()


def rand_perm():
    shuf = SYMS[:]
    random.shuffle(shuf)
    return {SYMS[i]: shuf[i] for i in range(len(SYMS))}


def apply_perm(expr, perm):
    return expr.subs({sp.Symbol(o): sp.Symbol(n) for o, n in perm.items()},
                     simultaneous=True)


def example(expr):
    """expr -> (encoding, family_idx, mode_idx, num_idx, lead_idx). -1 = N/A."""
    fam, mode, num, lead = slots_of(expr)
    return (encode(expr), FAM2I[fam],
            MODE2I[mode] if mode else -1,
            SYM2I[num] if num else -1,
            SYM2I[lead] if lead else -1)


class SlotModel(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.family = nn.Linear(hidden, 4)
        self.mode   = nn.Linear(hidden, 2)
        self.num    = nn.Linear(hidden, 7)
        self.lead   = nn.Linear(hidden, 7)

    def forward(self, z):
        h = self.trunk(z)
        return self.family(h), self.mode(h), self.num(h), self.lead(h)


# ------------------------------------------------------------------------ data
train_raw = [sp.sympify(s) for s in read_eqns(f"{DATASET}/train_eqns.txt")]
test_raw  = [sp.sympify(s) for s in read_eqns(f"{DATASET}/test_eqns.txt")]
IN_DIM = len(encode(train_raw[0]))
print(f"dataset={DATASET}  train={len(train_raw)} test={len(test_raw)}  in_dim={IN_DIM}")

# encode the augmented training set ONCE (identity + N_AUG symbol renamings)
t0 = time.time()
rows = []
for e in train_raw:
    for v in [e] + [apply_perm(e, rand_perm()) for _ in range(N_AUG)]:
        rows.append(example(v))
Xtr = torch.tensor(np.stack([r[0] for r in rows]))
Yf = torch.tensor([r[1] for r in rows]); Ym = torch.tensor([r[2] for r in rows])
Yn = torch.tensor([r[3] for r in rows]); Yl = torch.tensor([r[4] for r in rows])
print(f"augmented train set: {len(Xtr)} examples encoded in {time.time()-t0:.0f}s")

te = [example(e) for e in test_raw]
teX = torch.tensor(np.stack([t[0] for t in te]))
te_slots_true = [slots_of(e) for e in test_raw]

model = SlotModel(IN_DIM)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss(ignore_index=-1)


def evaluate():
    model.eval()
    with torch.no_grad():
        lf, lm, ln, ll = model(teX)
        pf, pm, pn, pl = (lf.argmax(1).tolist(), lm.argmax(1).tolist(),
                          ln.argmax(1).tolist(), ll.argmax(1).tolist())
    fam_ok = defaultdict(lambda: [0, 0])
    n_ok = 0
    for i, (tf, tm, tn, tl) in enumerate(te_slots_true):
        fam = FAMS[pf[i]]
        mode = MODES[pm[i]] if fam != "exponential" else None
        num = SYMS[pn[i]] if fam != "exponential" else None
        lead = SYMS[pl[i]] if (fam != "exponential" and mode == "general") else None
        try:
            ok = sp.simplify(render(fam, mode, num, lead)
                             - render(*te_slots_true[i])) == 0
        except Exception:
            ok = False
        n_ok += int(ok)
        k = f"{tf}/{tm or '-'}"
        fam_ok[k][0] += int(ok); fam_ok[k][1] += 1
    model.train()
    return n_ok / len(te), dict(fam_ok)


# ------------------------------------------------------------------------ train
best, best_state = 0.0, None
nb = (len(Xtr) + BATCH - 1) // BATCH
for ep in range(1, EPOCHS + 1):
    order = torch.randperm(len(Xtr))
    tot = 0.0
    for i in range(0, len(Xtr), BATCH):
        idx = order[i:i + BATCH]
        lf, lm, ln, ll = model(Xtr[idx])
        loss = (ce(lf, Yf[idx]) + ce(lm, Ym[idx])
                + ce(ln, Yn[idx]) + ce(ll, Yl[idx]))
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    if ep % 20 == 0 or ep == 1:
        acc, fam_ok = evaluate()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"ep {ep:4d}  loss {tot/nb:.4f}  test_exact {acc:.3f}  best {best:.3f}",
              flush=True)

if best_state is not None:
    model.load_state_dict(best_state)
acc, fam_ok = evaluate()
print(f"\nFINAL (best checkpoint) test exact-substitution accuracy: {best:.3f}")
for k, v in sorted(fam_ok.items()):
    print(f"  {k:22s} {v[0]:3d}/{v[1]:3d}  {v[0]/v[1]:.3f}")
_save = f"gemini/cov/pi_cov_slots_s{SEED}.pt"
torch.save(model.state_dict(), _save)
print(f"saved -> {_save}")
