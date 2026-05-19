# Session findings — 2026-05-17

Log of what we learned during the first hands-on debugging pass. Goal: pin down
bugs and identify the bottleneck for the CoV paper before scaling experiments.

## Bugs fixed

1. **Beam search in `train_cov.py` was triple-buggy.** Indentation collapsed
   the beam to its last entry's children at every depth; `action_history` was
   reset to `[-1]*hist_len` at every depth so the policy hit OOD states past
   d=0; the cov state was built compositionally (`x ⊕₁ τ₁ ⊕₂ τ₂`) while the
   env composes via `base_op(x, cov_inner)` only at termination — so beam
   evaluated a different `f(x)` than the env. Rewrote to track per-entry
   `(score, depth, base_op, cov_inner, action_history)` matching env semantics.
   Verified: on the saved linear-eqn model, `beam == greedy == 1.0`.

2. **`integer_encoding_1d` silently dropped all numeric constants.** SymPy's
   `sp.Integer` doesn't match `isinstance(node, (int, float))`, so SymPy
   numeric atoms fell through to the `is_Atom: continue` branch. Effect:
   `2*b*x`, `4*b*x`, and `b*x` all encoded to identical input vectors.
   Different equations requiring different substitutions looked identical to
   the policy. Fixed by routing numeric atoms (or anything already in
   feature_dict) through the lookup path before the generic-atom skip.

3. **`--use_cov` in `train_abel.py` was a dead flag.** Defined in argparse,
   never threaded to `make_env`. The closed-equation env was never exposing
   the CoV macroaction. Plumbed through `make_env → make_train_vec_env →
   run_trial → main`.

4. **Worker tracebacks were swallowed.** `run_parallel` only logged
   `str(exception)`, so "Invalid NaN comparison" had no file/line. Patched
   `run_trial_wrapper` to print a full traceback inside the worker before
   re-raising, plus the parent prints another traceback when collecting the
   result.

## Bugs found but not yet fixed

5. **`env.reset()` doesn't reset `self.base_op` in covEnv.** A previous
   episode's `base_op` leaks into the next. If the new episode picks STOP at
   depth 0 with stale `base_op == MUL` and `cov_inner == 0`, the composed
   `f(x) = mul(x, 0) = 0` — silently rewritten to `f(x) = x` by a
   defensive `if cov == 0` check. Hides a degenerate state.

6. **`SuccessReplayCallback` in `train_cov.py` (not train_abel) crashes on
   dict obs.** `np.stack` on a list of dicts fails. Latent — only triggered
   if someone runs `ppo-tree-mem` on covEnv. train_abel's version handles
   dicts correctly.

7. **Feature-dict ID collisions** in `make_feature_dict_integer_1d_multi`:
   `feature_dict['=']=0, feature_dict['pow']=0`, etc. Different tokens share
   IDs. The agent learns positionally so it's likely not catastrophic, but
   it's noisy.

## Experimental findings

### Direct π_cov training (unsupervised) — `train_cov.py`

| Setup | Result |
|---|---|
| Linear, single eqn `a*x + b` | ✅ greedy=1 at t=39k (rediscovered `f(x) = x - b/a`) |
| Single quadratic `x²+2bx+c` | ❌ briefly beam=1 at t=37k-49k then policy collapsed |
| Tiny multi-quadratic (50 eqns, v2 with integer scaling) | ❌ flat at 0 over 100k steps with fixed encoder |

The linear case works because the optimal trace is 3 actions deep, giving
PPO ~1/50³ ≈ 1/125k random-discovery chance per episode. The quadratic
optimal trace is 2 actions but the reward landscape rewards
zero-complexity-change paths that PPO falls into and never escapes.

### Warm-up: π_manipulate + perfect π_cov — `train_abel.py --use_cov`

The "perfect CoV button" is `pi_cov_general`, an analytic dispatcher that
returns the depression substitution for any quadratic/cubic/quartic/exp.

| Setup | Result |
|---|---|
| 5-seed ppo (MLP) × cov_level5 × 200k | All 0.0 across coverage and test |
| 5-seed ppo-tree × cov_level5 × 500k (killed at 75k) | All 0.0 — slow start |
| 2-seed ppo-tree × cov_quad1 × 200k (single eqn) | Seed 7100: coverage=1 at t=30k once, then `test_greedy` stayed 0 |

The single-eqn seed proves **the architecture works** — PPO + perfect CoV
can solve the post-CoV form. The drift-away-from-argmax pattern is the
same one we saw with direct π_cov training: PPO discovers a solving
trajectory rarely, doesn't reinforce it strongly enough to lock in.

**Diagnosis.** Intermediate rewards on the optimal solve path are all
*negative* (each step has -0.01 normalized step penalty, complexity delta
is zero until the final sqrt step). So PPO sees no partial-credit signal
between random discovery (rare) and final solve (big positive). The
gradient from one solve event isn't enough to override accumulated
negative-signal experience.

### Hypothesis for next experiment

Adding `--use_success_replay` should fix this. When a solve happens, the
trace gets stored and BC-trained between PPO rollouts, effectively making
each solve sticky. Now running: 5-seed ppo-tree + use_cov + use_success_replay
on cov_level3 (19 quadratics, simpler than cov_level5).

## Engineering shipped this session

- `eqn_gen/` package — per-class equation generators (`Quadratic`, `Cubic`,
  `Quartic`, `Exponential`) with intended-CoV validation and canonical-form
  dedup. Drop-in extensible for new classes.
- `make_eqns_v2.py` — CLI driver, produces per-class train/test directories.
- Three dataset scales: tiny (50/20), small (1000/200), large (5000/1000) —
  all per class, all at full yield.
- `trace_cov.py` — loads a CoV model and prints solution traces.
- `plot_seeds.py` — multi-seed learning-curve overlay; auto-detects
  train_cov vs train_abel CSV formats.
- `train_cov.py` gains `--seed`, `--verbose`, `--dataset_path` flags.
- `train_abel.py` gains `--use_cov` plumbing (was a dead flag).

## Decisions parked

- Adding success replay to the unsupervised π_cov pipeline is the obvious
  next fix if direct π_cov training is still on the table after this paper.
- AlphaZero / factorized actions / GFlowNets — see `project.md` Future Work.
- Wiring trained π_cov into the closed-equation env (replacing pi_cov_general)
  — needed for end-to-end paper results, deferred until π_cov reliably solves
  its own dataset.

---

# Session findings — 2026-05-18 (continuation)

After deciding the π_cov direct-training arm was too brittle, we pivoted to
the **mixed-dataset** framing: a single PPO policy with the CoV macroaction
(calling an analytic `pi_cov_general`) inside the closed-equation env. This
section captures what we discovered in that arm.

## Headline results

| Config | test_beam (best) | notes |
|---|---|---|
| mixed_v2_easy seed8001 (baseline, plain beam, max_steps=5) | 0.31 | original training-time eval |
| mixed_v2_easy seed8001 (plain beam, max_steps=10, NO dedup) | 0.275 | deeper but no dedup actually hurts |
| mixed_v2_easy seed8001 + value-beam λ=1.0 | 0.352 | +28% over plain |
| mixed_v2_easy seed8001 + value-beam λ=1.0 + dedup | **0.396** | **+44% over plain, no retraining** |
| mixed_v2_easy seed9100 (anti-loop α=0.1, plain beam) | 0.341 (peak), 0.275 final | one seed of 3 succeeded |
| mixed_v2_easy seed9100 + value-beam (λ>0) | hurts | value head miscalibrated when policy already strong |
| mixed_v2_small seed7000 final | 0.242 | takes 2× more steps than easy-bootstrap |

Best overall: **seed8001 + value-beam λ=1.0 + dedup = 0.396 test_beam**. The
search-side improvement is bigger than the training-side anti-loop gain and
applies retroactively to every existing checkpoint.

## What we built

- **`envs/env_multi_eqn.py`**: caught `HeuristicGCDFailed` (+ siblings) at
  three sites (`_apply_cov`, the CoV-unwind on solve, `_check_eqn_solved_local`);
  added `anti_loop_penalty` per-step reward shaping; switched cache=True default
  and bounded `make_actions_cache` to 10k entries with FIFO eviction.
- **`train_abel.py`**: added `sys.set_int_max_str_digits(0)` (sympy can emit
  huge ints that hit the Python 3.11 cap and crash printing); added
  `--early_stop_patience`, `--anti_loop_penalty`, `--eval_subsample`,
  `--eval_lite`; lowered `sr_iters_per_rollout` default 10→5; added
  `_policy_value` + `_state_complexity` helpers; refactored `beam_solve_one`
  to support `--beam_lambda --beam_alpha --beam_beta` and canonical-state
  dedup.
- **`diagnose_first_action.py` / `diagnose_action_traces.py` /
  `diagnose_antiloop_vs_baseline.py`**: failure-mode analysis tools. Showed
  baseline locks into one quartic recipe and falls into REL-loops on simpler
  eqns; anti-loop α=0.1 cuts the average failed-trace repeat-fraction from
  0.86 → 0.48.
- **`eval_value_beam.py`**: standalone re-eval. Takes any checkpoint, runs
  plain + value-guided beam at a λ-sweep on the test set. This is how we
  retroactively lift Table 1.
- **`equation_templates/mixed_v2_{easy,tiny}/build.py`**: stratified
  datasets (916/91 and 100/10).
- **`run_mass_value_beam.sh`**: wrapper that re-evaluates every interesting
  checkpoint with both plain and value-beam, building a CSV for Table 1.

## What surprised us

1. **The action cache barely matters** (~5% on env step in calibration).
   Each step has a unique `(lhs, rhs)`, so cache hits are rare during one
   episode; only resets benefit. The big wins live in the eval phase, not
   env step.
2. **Test-eval phase was the real bottleneck on mixed_v2_large**: with 1001
   test eqns × greedy+beam+at10, each eval cycle could take 30+ min. The
   `--eval_subsample` + `--eval_lite` flags drop this to seconds.
3. **The greedy/beam gap is huge**: seed8001 greedy=0.011 vs beam=0.275 vs
   value-beam=0.396. The policy has the right ideas but bad argmax — search
   matters far more than typical RL papers suggest.
4. **Anti-loop and value-beam target the same problem from different ends**.
   Anti-loop forces action diversity during *training*; value-beam corrects
   bad policy decisions at *decode*. On the baseline they both reach the
   same 0.35-0.40 ceiling; they don't naturally stack on the same trained
   model (anti-loop's reward shaping miscalibrates the value head).
5. **SymPy crashes were quietly killing workers**: HeuristicGCDFailed
   (non-deterministic GCD heuristic), and `int_max_str_digits` (printing
   huge ints from `simplify`) each cost us a full mixed_v2_large run before
   we noticed.

## Bugs fixed this session

- Killing one worker in a `ProcessPoolExecutor` cascades to all peers via
  `BrokenProcessPool`. Lesson: can't kill a stuck seed without losing
  productive ones in the same pool. Worked around by running independent
  pools per experiment.
- `eval_value_beam.py`'s initial version passed raw strings as equations;
  the env's `set_equation` doesn't sympify, so test_acc was 0 across all
  λ. Fixed by sympifying in the loader.

## Open questions / next pass

- Anti-loop seed7100 plateaued at test_beam=0.15 (early-stopped). Why is
  seed-variance so high under anti-loop? Maybe the penalty interacts badly
  with success-replay if early successes are themselves repetitive.
- Does value-beam + α=0.3 (length penalty) help? Earlier α/β grid was run
  BEFORE the dedup change; re-run with dedup.
- Does BC pretraining from existing solved traces stack with value-beam?
  (Untested.)
- Action decomposition (op-head + term-pointer pointer) — ChatGPT's #2.
  Bigger refactor, deferred to next paper.
