# π_cov — overnight run, 2026-05-21 night

*Autonomous session. Mandate: (1) fix the +1 symbol bug, (2) build the
tabula-rasa version. Both done; a third piece (unseen-symbol pointer model)
attempted as a bonus.*

## TL;DR

- **The +1 bug is fixed and verified.** The slot-head `π_cov` now equals the
  analytic oracle `pi_cov_general` on **966/966** cov equations, reproducibly
  (separate process). It was an encoding non-determinism, not a model flaw.
- **The tabula-rasa grammar decoder is done.** It predicts the substitution
  *AST itself* (no baked-in template) — **8/8 seeds → 1.000**, verified
  **966/966 == oracle**.
- **The pointer model** (unseen-symbol generalization) is symbol-invariant by
  construction — confirmed empirically — and its accuracy is being finalized.

## 1. The +1 symbol bug — fixed

**Symptom:** the slot-head scored 100% in its own training process but only
96% against the real oracle, with a systematic +1 alphabet shift on the
general polynomials.

**Root cause:** `sympy_expression_to_list` (utils/utils_env.py) *mutated* the
feature_dict during encoding — numeric atoms were assigned IDs lazily, in
encounter-order. That order differs per process (Python hash randomization
upstream), so a model trained in one process saw a different encoding when
reloaded → non-portable. (`debug_slots.py`: 192/192 in-process, 183/192 in a
fresh process.)

**Fix (committed `5c92c2f`):** `make_feature_dict_integer_1d_multi` now
pre-loads *every* numeric atom in the dataset; the encoder only looks up,
never mutates. Verified deterministic across processes; 8/8 retrained
slot-head seeds → 1.000; `compare_pi_cov.py` → **966/966 == pi_cov_general**.

## 2. Tabula-rasa grammar decoder — done

`train_cov_grammar.py`. Instead of the slot-head's fixed renderer (which
hard-codes `x − B/(nA)`), the grammar decoder **predicts the substitution AST
itself**, token by token in prefix notation:

```
x − B/(n·A)  ->  SUB XVAR DIV COPY MUL INT_n COPY
x − b        ->  SUB XVAR COPY
log(x)       ->  LOG XVAR
```

A GRU decoder emits the tokens; a copy head picks the symbols; grammar
arities make the decode self-delimiting. The model genuinely decides
SUB-vs-ADD, decides to divide, picks the integer — no depression template
baked in.

**Result (`cov_large`, committed `371d65c`):** 8/8 seeds → **1.000** on the
192-eqn held-out test; `compare_grammar.py` → **966/966 == pi_cov_general**.
Model: `gemini/cov/pi_cov_grammar_best.pt`.

## 3. Pointer model — unseen-symbol generalization

`train_cov_ptr.py`. Both models above use fixed 7-way symbol heads, so they
cannot emit a symbol never seen in training. The pointer model fixes this:
the equation is tokenized as an AST sequence in which **every coefficient
symbol is the same generic `SYM` token** — symbol identity is invisible to
the weights. A Transformer encodes it; a **copy-pointer** attends over the
SYM-token positions and copies whatever symbol sits there.

**Symbol-invariance — confirmed.** Smoke test: in-distribution accuracy and
the unseen-symbol accuracy (test equations renamed to symbols `n–t`, never
trained) were **identical to the decimal**. The architecture generalizes to
any symbol set by construction.

**Accuracy:** the first full run *overfit* — the Transformer memorized the
training set (train loss → 0) and generalized inconsistently (0.70–0.96 per
seed). Relaunched with regularization (dropout 0.2, 30× symbol-renaming
augmentation, weight decay). *[result pending — will be appended]*

## 4. Where π_cov stands

| model | tabula-rasa? | test acc | == oracle | unseen symbols |
|---|---|---|---|---|
| slot-head | fills a template | 1.000 | 966/966 ✓ | no (7-way heads) |
| grammar decoder | predicts the AST | 1.000 | 966/966 ✓ | no (7-way copy) |
| pointer model | predicts the AST | *pending* | *pending* | **yes** (by construction) |

**Bottom line:** the two requested deliverables are done. `π_cov` is a
genuine learned change-of-variables policy — deterministic, **no oracle at
inference**, and verified equal to `pi_cov_general` on every cov equation.
The grammar decoder is the honest "the model learns the CoV structure"
version; the pointer model additionally generalizes to unseen symbols.

## 5. Commits this session

- `5c92c2f` — fix encoding non-determinism; slot-head == oracle 966/966
- `371d65c` — grammar decoder verified == oracle; pointer model added
- *(earlier: `2112591`, `aa62ca1`, `68aa978`, `806506e`)*
