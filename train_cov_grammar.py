#!/usr/bin/env python3
"""Tabula-rasa pi_cov: a grammar-constrained AST decoder.

Where the slot-head fills a fixed template (a renderer hard-codes x - B/(nA)),
this model PREDICTS the substitution expression itself, token by token, in
prefix notation:

    x - B/(n*A)  ->  SUB XVAR DIV COPY[B] MUL INT_n COPY[A]
    x - b        ->  SUB XVAR COPY[b]
    log(x)       ->  LOG XVAR

A GRU decoder emits the token sequence; a copy head picks which input symbol
each COPY refers to. The model genuinely decides SUB-vs-ADD, decides to divide,
picks the integer -- no depression template baked in. Grammar arities make the
decode self-delimiting.

Supervised (targets from the known substitutions), per-epoch symbol-renaming
augmentation. Usage: python train_cov_grammar.py [dataset] [epochs] [seed]
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

torch.set_num_threads(1)   # 1 thread/process -> no oversubscription when 8 run in parallel

DATASET = sys.argv[1] if len(sys.argv) > 1 else "equation_templates/cov_large"
EPOCHS  = int(sys.argv[2]) if len(sys.argv) > 2 else 600
SEED    = int(sys.argv[3]) if len(sys.argv) > 3 else 0
N_AUG, BATCH = 20, 256

x = sp.Symbol("x")
SYMS = list("abcdefg")
SYM2I = {s: i for i, s in enumerate(SYMS)}
N_OF = {"quadratic": 2, "cubic": 3, "quartic": 4}
TERM_BANK = [sp.sympify(t) for t in "a,b,c,d,e,f,g,2,3,4".split(",")]

TOKENS = ["BOS", "ADD", "SUB", "MUL", "DIV", "LOG", "XVAR",
          "INT2", "INT3", "INT4", "COPY", "PAD"]
T2I = {t: i for i, t in enumerate(TOKENS)}
ARITY = {"ADD": 2, "SUB": 2, "MUL": 2, "DIV": 2, "LOG": 1,
         "XVAR": 0, "INT2": 0, "INT3": 0, "INT4": 0, "COPY": 0}
INT_TOK = {2: "INT2", 3: "INT3", 4: "INT4"}
MAXLEN = 7
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def read_eqns(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


def slots_of(expr):
    if expr.has(sp.exp):
        return ("exponential", None, None, None)
    p = sp.Poly(expr, x); n = p.degree(); cs = p.all_coeffs(); A, B = cs[0], cs[1]
    fam = {2: "quadratic", 3: "cubic", 4: "quartic"}[n]
    if A == 1:
        return (fam, "monic", str(sp.simplify(B / n)), None)
    return (fam, "general", str(B), str(A))


def render(family, mode, num, lead):
    if family == "exponential":
        return sp.log(x)
    n = N_OF[family]
    B = sp.Symbol(num)
    if mode == "monic":
        return x - B
    return x - B / (n * sp.Symbol(lead))


def target_seq(family, mode, num, lead):
    """canonical prefix: token-ids[MAXLEN], copy-symbol-ids[MAXLEN] (-1 = none)."""
    if family == "exponential":
        toks, cps = ["LOG", "XVAR"], [-1, -1]
    elif mode == "monic":
        toks, cps = ["SUB", "XVAR", "COPY"], [-1, -1, SYM2I[num]]
    else:
        n = N_OF[family]
        toks = ["SUB", "XVAR", "DIV", "COPY", "MUL", INT_TOK[n], "COPY"]
        cps = [-1, -1, -1, SYM2I[num], -1, -1, SYM2I[lead]]
    L = len(toks)
    toks = [T2I[t] for t in toks] + [T2I["PAD"]] * (MAXLEN - L)
    cps = cps + [-1] * (MAXLEN - L)
    return toks, cps


def parse_prefix(toks, cps):
    """token-id list + copy-id list -> sympy expr, or None if malformed."""
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


# --------------------------------------------------------------------- encoder
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


# ----------------------------------------------------------------------- model
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
        h0 = self.enc(enc_vec).unsqueeze(0)        # [1,B,hidden]
        out, _ = self.gru(self.tok_emb(in_toks), h0)
        return self.h_tok(out), self.h_copy(out)   # [B,L,V], [B,L,7]


def decode(model, enc_vec):
    """greedy autoregressive decode -> (token-ids, copy-ids), arity-delimited."""
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


# ------------------------------------------------------------------------ data
train_raw = [sp.sympify(s) for s in read_eqns(f"{DATASET}/train_eqns.txt")]
test_raw  = [sp.sympify(s) for s in read_eqns(f"{DATASET}/test_eqns.txt")]
IN_DIM = len(encode(train_raw[0]))
print(f"dataset={DATASET}  train={len(train_raw)} test={len(test_raw)}  in_dim={IN_DIM}")

t0 = time.time()
Xtr, TOKtr, CPStr = [], [], []
for e in train_raw:
    for v in [e] + [apply_perm(e, rand_perm()) for _ in range(N_AUG)]:
        toks, cps = target_seq(*slots_of(v))
        Xtr.append(encode(v)); TOKtr.append(toks); CPStr.append(cps)
Xtr = torch.tensor(np.stack(Xtr))
TOKtr = torch.tensor(TOKtr, dtype=torch.long)
CPStr = torch.tensor(CPStr, dtype=torch.long)
print(f"augmented train set: {len(Xtr)} examples encoded in {time.time()-t0:.0f}s")

teX = torch.tensor(np.stack([encode(e) for e in test_raw]))
te_truth = [render(*slots_of(e)) for e in test_raw]
te_fam = [slots_of(e)[0] + "/" + (slots_of(e)[1] or "-") for e in test_raw]

model = GrammarDecoder(IN_DIM)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss(ignore_index=-1)
ce_tok = nn.CrossEntropyLoss(ignore_index=T2I["PAD"])
BOS = torch.full((len(Xtr), 1), T2I["BOS"], dtype=torch.long)
in_toks_all = torch.cat([BOS, TOKtr[:, :MAXLEN - 1]], dim=1)


def evaluate():
    model.eval()
    fam_ok = defaultdict(lambda: [0, 0])
    n_ok = 0
    for i in range(len(test_raw)):
        toks, cps = decode(model, teX[i])
        pred = parse_prefix(toks, cps)
        try:
            ok = pred is not None and sp.simplify(pred - te_truth[i]) == 0
        except Exception:
            ok = False
        n_ok += int(ok)
        fam_ok[te_fam[i]][0] += int(ok); fam_ok[te_fam[i]][1] += 1
    model.train()
    return n_ok / len(test_raw), dict(fam_ok)


# ----------------------------------------------------------------------- train
best, best_state = 0.0, None
nb = (len(Xtr) + BATCH - 1) // BATCH
for ep in range(1, EPOCHS + 1):
    order = torch.randperm(len(Xtr))
    tot = 0.0
    for i in range(0, len(Xtr), BATCH):
        idx = order[i:i + BATCH]
        tl, cl = model(Xtr[idx], in_toks_all[idx])
        loss = (ce_tok(tl.reshape(-1, len(TOKENS)), TOKtr[idx].reshape(-1))
                + ce(cl.reshape(-1, 7), CPStr[idx].reshape(-1)))
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    if ep % 40 == 0 or ep == 1:
        acc, fam_ok = evaluate()
        if acc > best:
            best = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"ep {ep:4d}  loss {tot/nb:.4f}  test_exact {acc:.3f}  best {best:.3f}",
              flush=True)

if best_state is not None:
    model.load_state_dict(best_state)
acc, fam_ok = evaluate()
print(f"\nFINAL (best checkpoint) test exact-substitution accuracy: {best:.3f}")
for k, v in sorted(fam_ok.items()):
    print(f"  {k:22s} {v[0]:3d}/{v[1]:3d}  {v[0]/v[1]:.3f}")
torch.save(model.state_dict(), f"gemini/cov/pi_cov_grammar_s{SEED}.pt")
print(f"saved -> gemini/cov/pi_cov_grammar_s{SEED}.pt")
