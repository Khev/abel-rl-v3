#!/usr/bin/env python3
"""Compare the grammar-decoder pi_cov vs pi_cov_general (the analytic oracle)
on the cov datasets. Usage: python compare_grammar.py [model.pt] [dataset ...]
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

torch.set_num_threads(1)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemini/cov/pi_cov_grammar_best.pt"
DATASETS = sys.argv[2:] if len(sys.argv) > 2 else ["equation_templates/cov_large"]
ENC_DS = "equation_templates/cov_large"

x = sp.Symbol("x")
SYMS = list("abcdefg")
TERM_BANK = [sp.sympify(t) for t in "a,b,c,d,e,f,g,2,3,4".split(",")]
TOKENS = ["BOS", "ADD", "SUB", "MUL", "DIV", "LOG", "XVAR",
          "INT2", "INT3", "INT4", "COPY", "PAD"]
T2I = {t: i for i, t in enumerate(TOKENS)}
ARITY = {"ADD": 2, "SUB": 2, "MUL": 2, "DIV": 2, "LOG": 1,
         "XVAR": 0, "INT2": 0, "INT3": 0, "INT4": 0, "COPY": 0}
MAXLEN = 7


def read_eqns(p):
    return [l.strip() for l in open(p) if l.strip() and not l.startswith("#")]


def parse_prefix(toks, cps):
    pos = [0]

    def rec():
        if pos[0] >= len(toks):
            raise ValueError
        t = TOKENS[toks[pos[0]]]; c = cps[pos[0]]; pos[0] += 1
        if t == "XVAR":
            return x
        if t in ("INT2", "INT3", "INT4"):
            return sp.Integer(int(t[3]))
        if t == "COPY":
            return sp.Symbol(SYMS[c])
        if t == "LOG":
            return sp.log(rec())
        if t == "ADD":
            return rec() + rec()
        if t == "SUB":
            a = rec(); b = rec(); return a - b
        if t == "MUL":
            return rec() * rec()
        if t == "DIV":
            a = rec(); b = rec(); return a / b
        raise ValueError

    try:
        return rec()
    except Exception:
        return None


class GrammarDecoder(nn.Module):
    def __init__(self, in_dim, hidden=256, emb=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.tok_emb = nn.Embedding(len(TOKENS), emb)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.h_tok = nn.Linear(hidden, len(TOKENS))
        self.h_copy = nn.Linear(hidden, 7)

    def forward(self, enc_vec, in_toks):
        h0 = self.enc(enc_vec).unsqueeze(0)
        out, _ = self.gru(self.tok_emb(in_toks), h0)
        return self.h_tok(out), self.h_copy(out)


_env = covEnv(main_eqn=sp.sympify(read_eqns(f"{ENC_DS}/train_eqns.txt")[0]),
              term_bank=TERM_BANK, max_depth=3, step_penalty=0.1, f_penalty=0.0,
              hist_len=10, multi_eqn=False, use_curriculum=False,
              state_rep="integer_1d", dataset_path=ENC_DS)

def encode(e):
    return np.asarray(_env.to_vec(e, 0)[0], dtype=np.float32).flatten()


IN = len(encode(sp.sympify(read_eqns(f"{ENC_DS}/train_eqns.txt")[0])))
model = GrammarDecoder(IN)
model.load_state_dict(torch.load(MODEL, map_location="cpu"))
model.eval()
print(f"loaded grammar decoder: {MODEL}")


def decode(enc_vec):
    with torch.no_grad():
        h = model.enc(enc_vec.unsqueeze(0)).unsqueeze(0)
        inp = torch.tensor([[T2I["BOS"]]])
        toks, cps, debt = [], [], 1
        for _ in range(MAXLEN):
            out, h = model.gru(model.tok_emb(inp), h)
            tl = model.h_tok(out[:, -1]).clone()
            tl[0, T2I["BOS"]] = -1e9
            tl[0, T2I["PAD"]] = -1e9
            ti = int(tl.argmax())
            ci = int(model.h_copy(out[:, -1]).argmax())
            toks.append(ti)
            cps.append(ci if TOKENS[ti] == "COPY" else -1)
            inp = torch.tensor([[ti]])
            debt += ARITY.get(TOKENS[ti], 0) - 1
            if debt == 0:
                break
        return toks, cps


def family_of(expr):
    if expr.has(sp.exp):
        return "exponential/-"
    p = sp.Poly(expr, x); n = p.degree()
    nm = {2: "quadratic", 3: "cubic", 4: "quartic"}.get(n, "?")
    return f"{nm}/{'monic' if p.LC() == 1 else 'general'}"


eqns = []
for ds in DATASETS:
    for fn in ("train_eqns.txt", "test_eqns.txt"):
        p = f"{ds}/{fn}"
        if os.path.isfile(p):
            eqns += read_eqns(p)
eqns = sorted(set(eqns))
print(f"comparing on {len(eqns)} unique equations\n")

fam = defaultdict(lambda: [0, 0])
match = total = 0
mism = []
for s in eqns:
    e = sp.sympify(s)
    o = pi_cov_general(e)
    if o is None:
        continue
    total += 1
    toks, cps = decode(torch.tensor(encode(e)))
    pred = parse_prefix(toks, cps)
    try:
        same = pred is not None and sp.simplify(pred - o) == 0
    except Exception:
        same = False
    k = family_of(e)
    fam[k][0] += int(same); fam[k][1] += 1
    match += int(same)
    if not same:
        mism.append((s, o, pred))

print(f"{'family/mode':22s} {'match':>10s}")
print("-" * 34)
for k in sorted(fam):
    m, t = fam[k]
    print(f"{k:22s} {m:4d}/{t:<4d} {m/t:.3f}")
print("-" * 34)
print(f"{'OVERALL':22s} {match:4d}/{total:<4d} {match/total:.4f}")
for s, o, p in mism[:10]:
    print(f"  MISMATCH: {s}\n    oracle ={o}\n    grammar={p}")
