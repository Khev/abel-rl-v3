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
