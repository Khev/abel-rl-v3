#!/usr/bin/env python3
"""Pointer pi_cov -- a tabula-rasa model that generalizes to UNSEEN symbols.

The slot-head and grammar decoder pick symbols with fixed 7-way heads, so they
cannot emit a symbol they never trained on. This model instead:

  * tokenizes the equation as an AST sequence; every coefficient symbol is the
    SAME generic "SYM" token  -> the model literally cannot use symbol identity,
    only structural position;
  * a Transformer encodes the token sequence -> per-token vectors;
  * a GRU decoder emits the substitution AST (SUB/DIV/MUL/INT/COPY ...);
  * for a COPY token, a pointer attends over the encoder's SYM-token positions
    and copies whatever symbol sits there.

Because symbols are invisible to the weights and copied by position, a test
equation using a brand-new symbol is handled structurally. Supervised; trained
with symbol-renaming augmentation.

Usage: python train_cov_ptr.py [train_dataset] [epochs] [seed] [test_dataset]
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

torch.set_num_threads(1)

TRAIN_DS = sys.argv[1] if len(sys.argv) > 1 else "equation_templates/cov_large"
EPOCHS   = int(sys.argv[2]) if len(sys.argv) > 2 else 400
SEED     = int(sys.argv[3]) if len(sys.argv) > 3 else 0
TEST_DS  = sys.argv[4] if len(sys.argv) > 4 else TRAIN_DS
N_AUG, BATCH = 30, 256   # heavy symbol-renaming augmentation regularizes the Transformer
EQ_LEN = 48          # max equation token-sequence length
MAX_SYM = 16         # max coefficient-symbol slots per equation
DEC_LEN = 7          # max substitution prefix length

x = sp.Symbol("x")
N_OF = {"quadratic": 2, "cubic": 3, "quartic": 4}
# augmentation symbol pool -- a..t, so train/test can use disjoint subsets
AUG_POOL = list("abcdefghijklmnopqrst")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# equation-AST token vocab (SYM is generic: symbol identity is invisible)
EQ_TOK = ["PAD", "ADD", "MUL", "POW", "EXP", "XVAR", "SYM",
          "NEG1", "N0", "N1", "N2", "N3", "N4", "N6", "N8", "N16", "NUM"]
EQ2I = {t: i for i, t in enumerate(EQ_TOK)}
NUM2TOK = {-1: "NEG1", 0: "N0", 1: "N1", 2: "N2", 3: "N3",
           4: "N4", 6: "N6", 8: "N8", 16: "N16"}

# decoder (substitution) token vocab
TOKENS = ["BOS", "ADD", "SUB", "MUL", "DIV", "LOG", "XVAR",
          "INT2", "INT3", "INT4", "COPY", "PAD"]
T2I = {t: i for i, t in enumerate(TOKENS)}
ARITY = {"ADD": 2, "SUB": 2, "MUL": 2, "DIV": 2, "LOG": 1,
         "XVAR": 0, "INT2": 0, "INT3": 0, "INT4": 0, "COPY": 0}
INT_TOK = {2: "INT2", 3: "INT3", 4: "INT4"}


def read_eqns(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


def tokenize_eq(expr):
    """expr -> (token-ids[EQ_LEN], sym_positions[MAX_SYM], sym_strings list)."""
    toks, sym_pos, sym_str = [], [], []

    def rec(node):
        if len(toks) >= EQ_LEN:
            return
        if node == x:
            toks.append(EQ2I["XVAR"])
        elif getattr(node, "is_Symbol", False):
            sym_pos.append(len(toks)); sym_str.append(str(node))
            toks.append(EQ2I["SYM"])
        elif getattr(node, "is_Number", False):
            key = int(node) if getattr(node, "is_Integer", False) else None
            toks.append(EQ2I[NUM2TOK.get(key, "NUM")])
        elif node.is_Add:
            toks.append(EQ2I["ADD"])
            for a in sorted(node.args, key=sp.default_sort_key):
                rec(a)
        elif node.is_Mul:
            toks.append(EQ2I["MUL"])
            for a in sorted(node.args, key=sp.default_sort_key):
                rec(a)
        elif node.is_Pow:
            toks.append(EQ2I["POW"])
            for a in node.args:
                rec(a)
        elif node.func.__name__ == "exp":
            toks.append(EQ2I["EXP"])
            for a in node.args:
                rec(a)
        else:
            toks.append(EQ2I["NUM"])

    rec(sp.sympify(expr))
    tok_ids = toks[:EQ_LEN] + [EQ2I["PAD"]] * (EQ_LEN - len(toks))
    pos = (sym_pos[:MAX_SYM] + [0] * (MAX_SYM - len(sym_pos)))
    mask = [1] * min(len(sym_pos), MAX_SYM) + [0] * (MAX_SYM - len(sym_pos))
    return tok_ids, pos, mask, sym_str[:MAX_SYM]


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
    """prefix token-ids[DEC_LEN] + copy-symbol-string per slot ('' = no copy)."""
    if family == "exponential":
        toks, cps = ["LOG", "XVAR"], ["", ""]
    elif mode == "monic":
        toks, cps = ["SUB", "XVAR", "COPY"], ["", "", num]
    else:
        toks = ["SUB", "XVAR", "DIV", "COPY", "MUL", INT_TOK[N_OF[family]], "COPY"]
        cps = ["", "", "", num, "", "", lead]
    L = len(toks)
    toks = [T2I[t] for t in toks] + [T2I["PAD"]] * (DEC_LEN - L)
    cps = cps + [""] * (DEC_LEN - L)
    return toks, cps


def parse_prefix(toks, copy_syms):
    """decoder token-ids + copied symbol strings -> sympy expr or None."""
    pos = [0]

    def rec():
        if pos[0] >= len(toks):
            raise ValueError
        t = TOKENS[toks[pos[0]]]; cs = copy_syms[pos[0]]; pos[0] += 1
        if t == "XVAR":
            return x
        if t in ("INT2", "INT3", "INT4"):
            return sp.Integer(int(t[3]))
        if t == "COPY":
            return sp.Symbol(cs)
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


def rand_perm(pool):
    src = sorted(set(pool))
    dst = src[:]
    random.shuffle(dst)
    return {s: d for s, d in zip(src, dst)}


def apply_perm(expr, perm):
    return expr.subs({sp.Symbol(o): sp.Symbol(n) for o, n in perm.items()},
                     simultaneous=True)


# ----------------------------------------------------------------------- model
class PointerCov(nn.Module):
    def __init__(self, hidden=192, heads=4, layers=2, demb=64):
        super().__init__()
        self.tok_emb = nn.Embedding(len(EQ_TOK), hidden)
        self.pos_emb = nn.Embedding(EQ_LEN, hidden)
        enc_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2,
                                               batch_first=True, dropout=0.2)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.dec_emb = nn.Embedding(len(TOKENS), demb)
        self.gru = nn.GRU(demb, hidden, batch_first=True)
        self.h_tok = nn.Linear(hidden, len(TOKENS))
        self.q_ptr = nn.Linear(hidden, hidden)        # decoder -> pointer query
        self.k_ptr = nn.Linear(hidden, hidden)        # encoder sym -> pointer key
        self.hidden = hidden

    def encode(self, eq_toks):
        B, L = eq_toks.shape
        pos = torch.arange(L).unsqueeze(0).expand(B, L)
        h = self.tok_emb(eq_toks) + self.pos_emb(pos)
        pad_mask = (eq_toks == EQ2I["PAD"])
        return self.encoder(h, src_key_padding_mask=pad_mask)   # [B,L,hidden]

    def decode_step_logits(self, enc_h, sym_pos, sym_mask, in_toks):
        """teacher-forced: enc_h [B,L,H], sym_pos/sym_mask [B,MAX_SYM],
        in_toks [B,DEC_LEN] -> token logits [B,DEC_LEN,V], ptr logits
        [B,DEC_LEN,MAX_SYM]."""
        B = enc_h.shape[0]
        ctx = enc_h.mean(1, keepdim=True).transpose(0, 1)        # [1,B,H]
        out, _ = self.gru(self.dec_emb(in_toks), ctx)            # [B,DEC_LEN,H]
        tok_logits = self.h_tok(out)
        # gather encoder hiddens at the symbol positions
        idx = sym_pos.unsqueeze(-1).expand(-1, -1, self.hidden)  # [B,MAX_SYM,H]
        sym_h = torch.gather(enc_h, 1, idx)                      # [B,MAX_SYM,H]
        q = self.q_ptr(out)                                      # [B,DEC_LEN,H]
        k = self.k_ptr(sym_h)                                    # [B,MAX_SYM,H]
        ptr = torch.bmm(q, k.transpose(1, 2)) / (self.hidden ** 0.5)  # [B,DEC,MAX_SYM]
        ptr = ptr.masked_fill(~sym_mask.bool().unsqueeze(1), -1e9)
        return tok_logits, ptr


# ------------------------------------------------------------------------ data
TRAIN_POOL = list("abcdefghijklm")        # augmentation renames into a-m
UNSEEN = {c: list("nopqrst")[i] for i, c in enumerate("abcdefg")}  # a-g -> n-t


def build(eqns):
    EQ, SP, SM, DT, IT, CM = [], [], [], [], [], []
    for s in eqns:
        e0 = sp.sympify(s)
        for e in [e0] + [apply_perm(e0, rand_perm(TRAIN_POOL)) for _ in range(N_AUG)]:
            tok, pos, mask, sym_str = tokenize_eq(e)
            dtok, csym = target_seq(*slots_of(e))
            cmask = [[0] * MAX_SYM for _ in range(DEC_LEN)]
            for t in range(DEC_LEN):
                if csym[t]:
                    for p in range(len(sym_str)):
                        if sym_str[p] == csym[t]:
                            cmask[t][p] = 1
            EQ.append(tok); SP.append(pos); SM.append(mask)
            DT.append(dtok); IT.append([T2I["BOS"]] + dtok[:DEC_LEN - 1])
            CM.append(cmask)
    return (torch.tensor(EQ), torch.tensor(SP),
            torch.tensor(SM, dtype=torch.float), torch.tensor(DT),
            torch.tensor(IT), torch.tensor(CM, dtype=torch.float))


def decode(expr):
    tok, pos, mask, sym_str = tokenize_eq(expr)
    eq_t = torch.tensor([tok]); pos_t = torch.tensor([pos])
    mask_t = torch.tensor([mask], dtype=torch.float)
    with torch.no_grad():
        enc_h = model.encode(eq_t)
        h = enc_h.mean(1, keepdim=True).transpose(0, 1)
        idx = pos_t.unsqueeze(-1).expand(-1, -1, model.hidden)
        k = model.k_ptr(torch.gather(enc_h, 1, idx))
        inp = torch.tensor([[T2I["BOS"]]])
        toks, csyms, debt = [], [], 1
        for _ in range(DEC_LEN):
            out, h = model.gru(model.dec_emb(inp), h)
            tl = model.h_tok(out[:, -1]).clone()
            tl[0, T2I["BOS"]] = -1e9; tl[0, T2I["PAD"]] = -1e9
            ti = int(tl.argmax()); toks.append(ti)
            if TOKENS[ti] == "COPY":
                q = model.q_ptr(out[:, -1])
                pl = (q.unsqueeze(1) @ k.transpose(1, 2)).squeeze(1) / (model.hidden ** 0.5)
                pl = pl.masked_fill(~mask_t.bool(), -1e9)
                slot = int(pl.argmax())
                csyms.append(sym_str[slot] if slot < len(sym_str) else "")
            else:
                csyms.append("")
            inp = torch.tensor([[ti]])
            debt += ARITY.get(TOKENS[ti], 0) - 1
            if debt == 0:
                break
        return toks, csyms


def eval_on(eqns, rename=None):
    model.eval()
    fam = defaultdict(lambda: [0, 0]); ok = 0
    for s in eqns:
        e = sp.sympify(s)
        if rename:
            e = apply_perm(e, rename)
        toks, csyms = decode(e)
        pred = parse_prefix(toks, csyms)
        try:
            good = pred is not None and sp.simplify(pred - render(*slots_of(e))) == 0
        except Exception:
            good = False
        ok += int(good)
        sl = slots_of(e)
        fam[f"{sl[0]}/{sl[1] or '-'}"][0] += int(good)
        fam[f"{sl[0]}/{sl[1] or '-'}"][1] += 1
    model.train()
    return ok / len(eqns), dict(fam)


train_eqns = read_eqns(f"{TRAIN_DS}/train_eqns.txt")
test_eqns  = read_eqns(f"{TEST_DS}/test_eqns.txt")
print(f"train={len(train_eqns)} test={len(test_eqns)}")
t0 = time.time()
EQ, SP, SM, DT, IT, CM = build(train_eqns)
print(f"train tensors: {len(EQ)} examples in {time.time()-t0:.0f}s")

model = PointerCov()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
ce_tok = nn.CrossEntropyLoss(ignore_index=T2I["PAD"])

best, best_state = 0.0, None
nb = (len(EQ) + BATCH - 1) // BATCH
for ep in range(1, EPOCHS + 1):
    order = torch.randperm(len(EQ)); tot = 0.0
    for i in range(0, len(EQ), BATCH):
        idx = order[i:i + BATCH]
        enc_h = model.encode(EQ[idx])
        tl, ptr = model.decode_step_logits(enc_h, SP[idx], SM[idx], IT[idx])
        tok_loss = ce_tok(tl.reshape(-1, len(TOKENS)), DT[idx].reshape(-1))
        is_copy = (DT[idx] == T2I["COPY"]).float()
        marg = (torch.softmax(ptr, dim=-1) * CM[idx]).sum(-1).clamp_min(1e-9)
        copy_loss = (-torch.log(marg) * is_copy).sum() / is_copy.sum().clamp_min(1)
        loss = tok_loss + copy_loss
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    if ep % 20 == 0 or ep == 1:
        acc, _ = eval_on(test_eqns)
        if acc > best:
            best = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"ep {ep:4d}  loss {tot/nb:.4f}  test {acc:.3f}  best {best:.3f}", flush=True)

if best_state:
    model.load_state_dict(best_state)
acc, fam = eval_on(test_eqns)
acc_u, fam_u = eval_on(test_eqns, rename=UNSEEN)
print(f"\nFINAL in-distribution test: {best:.3f}")
for k, v in sorted(fam.items()):
    print(f"  {k:22s} {v[0]:3d}/{v[1]:3d}  {v[0]/v[1]:.3f}")
print(f"UNSEEN-symbol test (a-g -> n-t, never trained): {acc_u:.3f}")
for k, v in sorted(fam_u.items()):
    print(f"  {k:22s} {v[0]:3d}/{v[1]:3d}  {v[0]/v[1]:.3f}")
torch.save(model.state_dict(), f"gemini/cov/pi_cov_ptr_s{SEED}.pt")
print(f"saved -> gemini/cov/pi_cov_ptr_s{SEED}.pt")
