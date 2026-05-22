# π_cov — Plan & Idea Log

*Updated 2026-05-21. Goal: a standalone **learned** change-of-variables policy
`π_cov` — given an equation it deterministically outputs the correct CoV
substitution `x ↦ f(x)`, with **no oracle at inference**, replacing the
analytic `pi_cov_general`.*

---

## 0. Status in one line

The **structured slot-head `π_cov`** reaches **100% greedy** (deterministic
argmax, no beam search) on the 192-equation held-out test set. The greedy/beam
gap is closed. Remaining work is **rigor** (does it extrapolate to unseen
symbols / families?) and **integration** into the closed-equation solver.

---

## 1. Where we are — the journey

| approach | test_greedy | test_beam | verdict |
|---|---|---|---|
| plain PPO RL on covEnv | ~0.0 | ~0.40 | fails — never *discovers* the exact substitution |
| RL + success-buffer seeding | ~0.0 | ~0.40 | fails — BC↔PPO tug-of-war churns the policy |
| BC-pretrain covEnv policy | 0.64 | 0.85 | works, but the **sequential decoder compounds** errors |
| **structured slot-head** | **1.00** | — | **one-shot prediction, no compounding — solved** |

Key fixes shipped this push (committed): `LOGX` action (exp solvable);
expert traces + `--bc_pretrain_iters`; **process-stable feature_dict**
(`sorted()` not `list(set())` — saved models were un-reloadable); eval helpers
given `dataset_path`.

**Why the slot-head wins.** The covEnv decoder emits a 2–4 action sequence;
greedy succeeds only if *every* step is argmax, so `test_greedy ≈
per-step-acc ^ trace-len`. The per-step diagnostic (`diag_cov_steps.py`)
showed cubic/general at `0.62 × 0.76 × 0.60`. The slot-head predicts the
substitution's *semantic fields* (`family, mode, num_sym, lead_sym`) with
independent heads in **one forward pass** — no sequence, no compounding — and
a deterministic renderer formats `x − B/(nA)`.

---

## 2. What is proven — and what is NOT

**Proven:** the slot-head perfectly learns the CoV *structural rule* for the
7 families (quad/cubic/quartic × {general, monic} + exp) over the 7-symbol
pool `a–g`. 100% deterministic, reproducible, oracle-free at inference.

**Not yet proven (the honest gaps):**
1. **Baked-in template.** The renderer encodes the depression *form*
   (`x − B/(nA)`); the model only learns to recognise `family` + the
   role-symbols. Legit structured prediction — but a more *scaffolded* claim
   than the covEnv "build it from primitive ops" policy.
2. **Same-symbol / same-family test.** Symbol-renaming augmentation means the
   192 test permutations lie inside the *learned* distribution. 100% proves
   "the rule is learned", not extrapolation to **new symbols** or **new
   families**. Fixed 7-way symbol heads *cannot* emit an unseen symbol — true
   new-symbol generalization needs **pointer heads**.

Closing gap (1) → grammar decoder (Phase C). Closing gap (2) → pointer heads
+ held-out splits (Phase B). Those two phases turn "100% on a friendly test"
into a **research-quality** generalization result.

---

## 3. Execution model — 8-core parallel waves

Machine: 10 cores / 32 GB. **Use 8**, leave 2 for headroom (see
[[compute-ram-constraint]]). A slot-head run is fast (~33 s encode + ~1 min
train ≈ **1.5 min**), single-process, light RAM — so experiments run in
**waves of 8 concurrent runs**. Each wave below is sized to exactly 8 jobs.

Launch pattern: `nohup … train_cov_slots.py … & ` ×8, record PIDs, one
watcher polls to completion, then collate. Never exceed 8 heavy procs.

---

## 4. The plan (phased)

### Phase A — Harden the slot-head  *(2 waves × 8 = 16 runs, ~10 min)*
- **Wave A1:** slot-head on `cov_large`, **seeds 0–7** → mean ± std of the
  100%; confirm it is not seed luck.
- **Wave A2:** 8 ablations — `N_AUG ∈ {0, 2, 5, 10, 40}`, `hidden ∈
  {64, 512}`, `lr 3e-3` — isolate what actually drives the result
  (hypothesis: augmentation is the key lever; capacity is not).

### Phase B — Extrapolation: pointer heads + held-out splits  *(the rigor)*
- **B0:** build the **pointer-head** variant — predict `num`/`lead` as a
  pointer over the symbol nodes *present in the equation*, not a fixed 7-way
  class. Prerequisite for any unseen-symbol test.
- **Wave B1 (8):** **new-symbol** splits — train on symbol subsets, test on
  held-out symbols (e.g. train `a–e`, test needs `f,g`); 8 distinct splits.
- **Wave B2 (8):** **new-family** holdout — train on 3 families, test on the
  4th, each family in turn; + **larger symbol pools** (10 / 12 / 15 symbols,
  regenerate datasets) for scale stress.
- Acceptance: pointer model ≥ 0.9 greedy on genuinely unseen symbols.

### Phase C — Less baked-in: AST encoder + grammar decoder  *(stronger claim)*
- **C1 — AST/GNN encoder** (`utils/sympy_ast_graph.py`, message-passing GNN):
  feed the equation *tree*, not a flat integer vector. Node features: node
  type, symbol kind, integer value, depth, child index.
- **C2 — grammar-constrained decoder:** predict the substitution **AST**
  under a grammar (`Var(x) | CopySymbol(node) | Int | Add/Sub/Mul/Div/Pow |
  Log`), invalid actions masked to `−inf`. Removes the baked-in renderer
  template → genuine "learned CoV", and extends past the 7 fixed families.
- Run as budget sweeps (`10k/30k/100k/300k` steps × seeds), 8-wide.

### Phase D — Integration  *(the actual end goal)*
- Wire the slot-head (or pointer/grammar) `π_cov` into the closed-equation
  env, **replacing `pi_cov_general`**. Produce the end-to-end open-equation
  result with **no oracle at inference**.
- Re-run the open_small / open_large evaluations with learned `π_cov`.

### Phase E — Verifier / propose-and-verify loop  *(optional robustness)*
- `π_cov` proposes K candidate substitutions; a **verifier** (complexity
  drop, degree reduction, no blow-up — *not* the oracle answer) picks the
  best. Catches whatever a single argmax misses. Enables **expert iteration**
  (search improves labels → model imitates improved labels → repeat).

---

## 5. Full idea catalog (status)

| idea | status | note |
|---|---|---|
| Plain PPO / SAC RL | ✗ rejected | discovery problem; do not revisit |
| Success-replay buffer seeding | ✗ rejected | BC↔PPO tug-of-war |
| BC-from-demonstrations (covEnv) | ✓ done | 0.85 beam / 0.64 greedy baseline |
| `LOGX` action for exponential CoV | ✓ done | exp now solvable |
| Process-stable feature_dict | ✓ done | critical bug fix |
| Structured slot-head | ✓ done | **100% greedy** |
| Symbol-renaming augmentation | ✓ done | folded into slot-head |
| Per-step / family diagnostics | ✓ done | `diag_pi_cov.py`, `diag_cov_steps.py` |
| Multi-seed + ablations | ▶ Phase A | confirm robustness |
| Pointer heads (symbol pointers) | ▶ Phase B | unseen-symbol generalization |
| Held-out symbol / family splits | ▶ Phase B | the real generalization test |
| Larger symbol pools (10–15) | ▶ Phase B | scale stress |
| AST / GNN encoder | ▶ Phase C | structured input |
| Grammar-constrained decoder | ▶ Phase C | removes baked-in template |
| Integration into closed env | ▶ Phase D | replace `pi_cov_general` |
| Verifier / propose-and-verify | ▶ Phase E | robustness |
| Expert iteration | ▶ Phase E | search-improved labels |
| DAgger | ✗ rejected | needs train-rollout drift; ours is test-generalization, short unique traces |
| Bigger flat MLP only | ✗ rejected | `bc_loss→0` — capacity is not the bottleneck |

---

## 6. What NOT to do

- No long PPO/SAC runs — the bottleneck is supervised symbolic translation,
  not sparse-reward exploration.
- No more BC iterations after train loss hits zero.
- No scaling the flat-observation MLP without changing representation.
- No VAE-style latent generation before exact deterministic decoding works.
- Do not exceed 8 concurrent heavy processes (RAM panic risk).

---

## 7. Definition of done

- **Minimum (done):** slot-head ≥ 0.95 greedy on `cov_large` held-out test.
- **Strong:** pointer-head model ≥ 0.9 greedy on **genuinely unseen symbols**.
- **Research-quality:** AST + grammar decoder generalizes to unseen symbols
  *and* unseen families/templates.
- **Deployment:** learned `π_cov` wired into the closed solver; the
  open-equation results reproduced with **no oracle at inference**.
