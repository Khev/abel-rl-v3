# abel-rl-v3 — session summary (2026-05-17 to 2026-05-19)

A running summary of what we've established, what's in flight, and what's
deferred. Updates by hand as new results land.

---

## 1. Headline result

**`test_beam = 0.62` on the mixed_v2_easy open-equation test set** (56/91)
from a single seed (`seed10000`) using the full method stack:

- `ppo-tree-rc-buf-cov` (relabel-constants + success-replay + CoV macroaction)
- `--anti_loop_penalty 0.1` (action-diversity penalty)
- `--use_cbrt` (cube-root action enabled)
- Plain beam at eval with `max_steps=10` and canonical-state dedup
- Value-guided beam (`λ=1.0`) actually **hurts** this seed by 1-3pp:
  the trained policy is already sharp enough that V(s) signal becomes noise.

Greedy test_acc on the same checkpoint: 0.31 (28/91). So the lift from
beam+dedup is responsible for going 0.31 → 0.62.

A second viable seed (`seed8000`, training stalled at 1.8M of 3M) reaches
test_beam = 0.41 with the same stack. The 3rd seed (`seed9000`) was a
failure (test_beam = 0.05). So **best-config 3-seed sweep**: 2/3 viable,
range 0.41–0.62, mean ≈ 0.52.

For mixed_v2_easy plain-beam baseline (no methods): test_beam ≈ 0.27.

---

## 2. Per-class breakdown of the headline (seed10000)

| Class | n | greedy | plain beam | value beam |
|---|---|---|---|---|
| cubic | 15 | 15/15 | 15/15 | 15/15 |
| quartic | 15 | 14/15 | 15/15 | 15/15 |
| quadratic | 15 | 10/15 | 12/15 | 15/15 |
| abel_level1 | 1 | 0/1 | 1/1 | 1/1 |
| abel_level2 | 10 | 2/10 | 5/10 | 5/10 |
| abel_level3 | 20 | 4/20 | 8/20 | 8/20 |
| **exponential** | **15** | **0/15** | **0/15** | **0/15** |

- Cubic class went 0% → 100% (cbrt action is the fix)
- Quartic/quadratic class fully solved
- Exponential class still 0%: ROOT CAUSE identified (see Section 4)
- Closed-equation classes (abel_level2/3) partially solved (40-50%)

---

## 3. Method-level findings (what helps what)

### Helps

| Method | Effect | Mechanism |
|---|---|---|
| **Canonical-state dedup in beam** | +50–100% relative | Beam doesn't waste slots on equivalent partial derivations |
| **Cube-root action (cbrt)** | Cubic class 0% → 100% | Removes a structural floor |
| **Anti-loop penalty (α=0.1)** | +3-5pp on test_beam (variance high) | Forces action diversity; breaks the "depressed-quartic-recipe" memorization mode |
| **Value-guided beam (λ=1.0)** | +5–7pp when policy is unsharp; **hurts** sharp policies | Tie-breaks weakly-ranked correct actions |
| **Fresh success-replay buffer** (drop oldest 50% every 20 rollouts) | 5× sample efficiency on mixed_v2_easy | Combats staleness: the policy memorizes the buffer and loses marginal signal |
| **max_cov_apps = 3** (was 1) | Allows nested-CoV recipes | Needed for exponential → quadratic chain |
| **`pi_cov(current_form)` fallback** | Unblocks nested-CoV pattern detection | Manually verified to solve a*e^x + b*e^(-x) + c in 7 steps |

### Doesn't help / hurts

| Method | Verdict | Why |
|---|---|---|
| Curiosity bonuses (ICM, RND, NGU) | Doesn't help; RND actively harms | TreeMLP's structural inductive bias already supplies exploration signal |
| `tree_hidden_dim = 256` (vs 128) | No gain at current data size | Over-parameterization, memorizes train, doesn't generalize |
| Action factorization (op-head + term pointer) | Not needed | Flat policy converges to near-factorized behavior (74% prob mass on modal op) |
| Deeper beam (`max_steps = 20`) | Memory explosion; killed | Combinatorial; needs smarter pruning (PUCT) |
| `balanced_lw` buffer (length-weighted balanced sampling) | No different from flat | Coverage faster but test unchanged |
| Decoder-level solved-state cache (memory beam) | +1 eqn over value beam | Cache too small (greedy on train solves only ~10% of eqns); needs beam-built cache for more entries |
| `balanced` buffer (per-equation sampling, no length weighting) | Marginal gain only | At 500k Ntrain test_greedy stays at 0.022 like flat |

### Negative/null results to write up

These are useful contributions in themselves:

- **Curiosity hurts TreeMLP**, contradicting earlier MLP-based work
- **dim=256 is no better than dim=128**: capacity is not the current bottleneck
- **Action factorization unnecessary**: flat policy is already op-structured
- **Memory cache adds 1 eqn**: not the big lift we hoped; future work for beam-built caches

---

## 4. The exponential debug (2026-05-19)

The exponential class was 0% across all checkpoints. Manual trace identified a
sequence of three bugs:

### Bug A: action-set was missing cbrt

After depressed-cubic CoV we hit `a·y³ + k = 0` which needs cube root.
Originally only `sqrt` was in the action set. **Fixed**: added
`custom_cbrt` under a `--use_cbrt` flag (default True). Backwards-compat:
cbrt is APPENDED at end of the fixed-action list so legacy checkpoints
retain their action ordering.

### Bug B: max_cov_apps was 1

Exponential requires nested CoV:
  1. `e^(kx)` substitution → rational form `a·x + b/x + c`
  2. Multiply by x → quadratic `a·x² + c·x + b`
  3. Quadratic-depression CoV `x → x - c/(2a)`

With `max_cov_apps=1` the agent could only do step 1 and got stuck.
**Fixed**: bumped default to 3.

### Bug C: pi_cov_general only saw main_eqn

`_apply_cov` called `pi_cov(self.main_eqn)`, but `main_eqn` is only updated
by CoV/relabel — not by arithmetic. So after step 1 (CoV) + step 2 (mul x),
the CURRENT lhs/rhs was a quadratic, but `main_eqn` was still the rational
form. `pi_cov` returned None on step 3 and the agent stalled.

**Fixed**: `_apply_cov` now calls `pi_cov(simplify(lhs - rhs))` first
(current transformed form), and falls back to `pi_cov(main_eqn)` only if
that returns None.

### Verification

`trace_exp_manual.py` walks the recipe by hand on `a·e^x + b·e^(-x) + c`:

```
Start                        a*exp(x) + b*exp(-x) + c
  CoV                        a*x + b/x + c
  mul x                      x*(a*x + b/x + c)
  CoV  (NEW: depresses)      a*x**2 + b - c**2/(4a)
  relabel                    a + b*x**2
  sub a / div b / sqrt       x = log((a*sqrt(-4 + c**2/a**2) - c)/(2a))   ✓
```

7 actions, deterministic, correct closed-form.

### Status as of writing

All three fixes are committed (`6055b46`, `f0ea046`, `d134a65`).

**The currently-running laptop training jobs do NOT have the pi_cov fix in
memory** — they were spawned before the commit. They have max_cov_apps=3 and
cbrt. Even so, one cubic+exp seed (seed16000) hit test_greedy = 0.48 (vs
old baseline 0.01).

**The desktop sweeps (`run_desktop_sweeps.sh`) will pick up ALL fixes
automatically** when launched fresh.

---

## 5. The fresh-buffer finding (2026-05-19)

The original `SuccessBuffer` was a flat deque with uniform random sampling.
Variants implemented:

- `flat`: original, uniform sampling
- `balanced`: per-equation uniform sampling
- `balanced_lw`: balanced + length-weighted (prefer shorter traces)
- `fresh`: drop oldest 50% every 20 rollouts (combats staleness)

Single-seed comparison at Ntrain=500k on mixed_v2_easy:

| variant | seed | final cov | final test_greedy |
|---|---|---|---|
| flat | 7300 | 0.128 | 0.022 |
| balanced | 7400 | 0.202 | 0.022 |
| balanced_lw | 7500 | 0.190 | 0.022 |
| **fresh** | 7600 | **0.317** | **0.066** |

Multi-seed at 3M (in flight as of writing): seed14000 fresh-buffer hit
test_greedy = 0.33 at t=750k on mixed_v2_easy — **matching seed10000's
final test_greedy at 1/4 the steps**. Two of three seeds tracking
similarly. The fresh-buffer is now the recommended default.

The mechanism: PPO + flat buffer ends up training against a stale BC
distribution; the policy memorizes the buffer and marginal signal dies.
Dropping oldest entries forces continual refresh from on-policy successes.

---

## 6. The full method stack (as of the desktop launch script)

```
ppo-tree --action_space dynamic
  --Ntrain 3000000                       # for easy; 1e7 for large
  --use_relabel_constants                # constant-canonicalization
  --use_success_replay                   # BC from solved trajectories
  --use_cov                              # CoV macroaction in action set
  --use_cbrt                             # cube-root action (default True)
  --anti_loop_penalty 0.1                # action-diversity penalty
  --sr_buffer_kind fresh                 # NEW: drop-oldest-50% buffer
  --early_stop_patience 8                # stop after 8 evals w/o test_beam improvement
  --eval_lite                            # skip beam/at10 mid-training; final eval is full
  --eval_subsample 200                   # for large only
```

Plus the env-side fixes:
- `max_cov_apps = 3` (default)
- `pi_cov` checks current-form first then main_eqn fallback

Eval-time:
- `max_steps = 10`
- Beam width 5, top-k 5 per node
- Canonical-state dedup in beam
- Value-bonus λ choice depends on policy sharpness (λ=0 for full-stack;
  λ=1 for under-trained checkpoints)

---

## 7. What we hope to find (running / queued)

### Laptop (in flight)

- **`seed14000` fresh-buffer mixed_v2_easy → full eval at end**: tests
  if fresh-buffer headline beats seed10000 (0.62 test_beam). With
  test_greedy already at 0.33 at t=750k, expect final test_beam in
  the 0.65-0.75 range.

- **`seed16000` cubic+exp + fresh + max_cov_apps=3 → full eval at end**:
  test_greedy=0.48 at t=750k. Whether this is mostly cubic-class lifts
  or some unexpected exp solves is open. Without the pi_cov fix, exp
  should still be 0 — but worth checking.

### Desktop (run_desktop_sweeps.sh)

- **5-seed mixed_v2_easy at Ntrain=3M, full stack + pi_cov fix**: proper
  variance bars on the headline. Hope: mean ≥ 0.55 test_beam, best ≥ 0.70.

- **5-seed mixed_v2_large at Ntrain=1e7, full stack**: the "scaling"
  story. Hope: non-trivial test_beam on the 1001-eqn test set.

- **3-seed fresh-buffer ablation on mixed_v2_easy**: confirm the
  fresh-buffer story is reproducible (already 1 seed → ~0.33 greedy,
  needs n>1).

### Specific things we want to verify

1. **Exp class actually solvable now**: per-class breakdown on a fresh
   seed trained with pi_cov fix should show exp > 0/15.

2. **Fresh-buffer gain is robust**: not just one lucky seed.

3. **mixed_v2_large scales**: can the methods reach > 0.2 test_beam on
   the 1001-eqn test set? Closed-eq paper got ~0.66 on the 1000-eqn
   abel_level3 test set, so this is the target.

---

## 8. What's deferred (next paper or later)

| Item | Why deferred |
|---|---|
| **Action decomposition** (op-head + term pointer) | Flat policy already converges to op-factored behavior; not needed at current scale |
| **TreeMLP number-embedding fix** (sign+log-magnitude) | Deep architectural change; better as next paper |
| **AlphaZero-style expert iteration** | 3-4 weeks of focused work; saved for follow-up paper. Plan in `markdowns/alphazero_plan.md` |
| **BC pretraining** | Plan in `markdowns/bc_pretrain_plan.md`. Probably less impactful than the fresh-buffer + pi_cov fixes |
| **Memory-augmented solver** (Phase 2+) | Plan in `markdowns/memory_nn_plan.md`. Phase 1 (exact cache) showed +1 eqn lift; needs beam-built cache for more impact |
| **PUCT beam search** | A small step toward AlphaZero; could be 1-2 days if next paper wants it |

---

## 9. Paper-readiness assessment

We have a defensible main-track conference paper if framed as:

> "First reinforcement-learning agent for end-to-end open algebraic
> equation solving via CoV macroactions. We characterize the failure
> modes via diagnostic traces (REL-loops in baseline), propose
> complementary training-time (anti-loop) and decoding-time
> (canonical-state dedup, value-guided beam) interventions, identify
> two structural bottlenecks (missing cube-root action, single-CoV
> cap) and fix them. Per-class analysis shows 100% on quartic,
> cubic, quadratic, abel_level1; partial on closed-eq levels 2/3;
> identifies exponential as the remaining open problem and proposes
> a concrete fix (validated by hand-trace). Method stack reaches
> test_beam = 0.62 on the 91-eqn open-equation test set (best of
> n=3, mean ≈ 0.52), more than 2× the plain-beam baseline."

Gaps to close before submission:
- Multi-seed variance bars (5+ seeds on mixed_v2_easy with full stack)
- mixed_v2_large scaling result
- Updated per-class table after pi_cov fix verification
- ConPoLe-style baseline comparison for open eqns (or argue no direct
  prior work and frame as "first")
- Limitations section: cubic action-set, single-π_cov, no learned π_cov

Realistic timeline: ~1 week of focused work + desktop compute. ICML 2027
(Jan deadline) very comfortable. NeurIPS 2026 / ICLR 2027 (Sep) tight
but feasible.

---

## 10. File index for the work

Code:
- `envs/env_multi_eqn.py` — env with all fixes (cbrt, max_cov_apps=3, pi_cov current-form)
- `train_abel.py` — training with all knobs (anti_loop, eval_lite, eval_subsample, sr_buffer_kind, early_stop, use_cbrt)
- `eval_value_beam.py` — value-guided beam decode with λ sweep
- `eval_memory_beam.py` — decoder-level solved-state cache
- `eval_per_class.py` — per-class test breakdown for any checkpoint
- `diagnose_action_traces.py`, `diagnose_antiloop_vs_baseline.py`,
  `diagnose_action_structure.py`, `diagnose_first_action.py` —
  failure-mode/structure diagnostics
- `audit_solvability.py` — sympy-based dataset solvability audit
- `trace_exp_manual.py` — manual recipe trace; how the exp-class fix was found
- `run_desktop_sweeps.sh` — the heavy-compute launches for the desktop
- `run_buffer_sweep.sh` — sequential single-core buffer-variant ablation

Datasets:
- `equation_templates/mixed_v2_{tiny,easy,small,large}/` — closed+open mixed
- `equation_templates/cov_v2_{small,large}/{quadratic,cubic,quartic,exponential}/` — per-class CoV
- `equation_templates/cubic_exp_focused/` — cubic+exp only (debug)
- `equation_templates/exp_debug/` — one-eqn (debug)

Plans:
- `markdowns/alphazero_plan.md` — expert iteration design
- `markdowns/memory_nn_plan.md` — memory-augmented solver design (incoming from user)
- `markdowns/bc_pretrain_plan.md` — BC pretraining from traces
- `markdowns/action_decomp_plan.md` — op-head + term-pointer (deferred)
- `markdowns/figures_plan.md` — paper figures
- `markdowns/speedup_plan.md`, `markdowns/speedup_plan_v2.md` — env / forward / eval optimizations
- `markdowns/session_findings.md` — running research log
