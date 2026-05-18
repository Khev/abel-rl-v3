# Speedup plan (no execution yet)

Diagnostic: mixed_v2_large is running at ~45 steps/sec vs expected ~100. Two big causes (action-list rebuild + per-eval test-set size) account for most of the gap.

## Ranked by ROI (rough estimates)

| # | Change | Est. speedup | Effort | Notes |
|---|---|---|---|---|
| **1** | Enable `cache=True` for actions in `multiEqn.__init__` | **1.5–2×** | trivial (1 line) | `cache=False` is the current default; `make_actions_cache` exists, just unused |
| **2** | Subsample test set during in-training evals (e.g. 100 of 1001) | **5–10×** on eval phase | small | mixed_v2_large has 1001 test eqns × greedy+beam+at10 → 10+ min per eval; subsample for curves, full set only at end |
| **3** | Skip `beam_accuracy` + `success_at_n` mid-training (only at end) | **3×** on eval phase | small | greedy alone is enough for early-stop signal; beam/at10 add 80% of eval cost |
| **4** | Lower `sr_iters_per_rollout` from 10 → 3 | **1.1–1.2×** | trivial | 10 BC updates per rollout is excessive when buffer is mostly fresh |
| **5** | Pre-allocate graph_encoding torch tensors per env | **1.2–1.5×** on env step | medium | every step does `torch.tensor(...)` + `torch.cat()` afresh |
| **6** | Short-circuit `_check_eqn_solved_local` | **1.1×** | small | skip `ratsimp`/`powdenest`/`simplify` when expr has no rational/power terms |
| **7** | Identity-detect in `relabel_with_existing_constants` | **1.1×** in REL-heavy seeds | small | if no mapping needed, skip the simplify call entirely |
| 8 | Compile/onnx the TreeMLP forward | 2× | high | parallel inference for beam |

## Bottleneck #1: action-list rebuilding (every step rebuilds ~3×)

In `env_multi_eqn.py:step_base`:
- Line 357-360: `make_actions` (or `make_actions_cache` if `self.cache=True`)
- Called inside `step_base()` AND `setup()` AND wrapped envs call `get_valid_action_mask()` before every predict
- `make_actions` runs `get_ordered_sub_expressions(lhs)` which BFS-walks the sympy expression tree and calls `sort_key()` on every node — expensive on big quartic-depressed forms

**Fix:** Set `cache=True` as default. The cache key `(lhs, rhs)` already exists — hashing sympy exprs is structural but cached by sympy itself, so it's not free but is much cheaper than rebuilding.

Risk: cache memory grows unbounded across episodes. Add cap to `self.action_cache` with LRU eviction (~10k entries should be plenty).

## Bottleneck #2: eval phase is expensive on the large set

`_log_eval` (TrainingLogger:831) runs THREE evals every interval:
1. `greedy_accuracy`: 1001 eqns × up to 5 steps × ~0.5s budget = up to **8 min** worst case (timeout-bounded)
2. `beam_accuracy`: 1001 eqns × 5 steps × beam_width=5 × topk=5 = up to **20 min**
3. `success_at_n`: 1001 eqns × 10 trials × 5 steps = up to **15 min**

So each eval can cost up to **45 min** on mixed_v2_large. With eval every 500k steps and 1e7 total, that's **20 evals × 45 min = 15 hours of pure eval**. The training itself is comparable.

**Fix:**
- **(a)** For in-training evals, **sample** 100 eqns uniformly from test set (deterministic seed). Run full 1001-eqn eval only at end.
- **(b)** Skip `beam_accuracy` and `success_at_n` during training; just track `greedy_accuracy`. Compute beam/at10 once at the end. Loses some signal (best_test_beam relies on beam) — can swap to `best_test_greedy` for early-stop.

`EVAL_TIMEOUT_DET = 0.75s` per eqn × 1001 eqns × (greedy + beam + at10) ≈ 22 min worst case. Subsampling to 100 eqns drops to 2.2 min.

## Bottleneck #3: SymPy simplify chains in solved-check + CoV

`_check_eqn_solved_local` (line 585-606): does up to 4 simplify-family calls. With my recent try-protected refactor, each is wrapped but still runs.

Idea: short-circuit on **cheap structural checks** first:
- If `sol.free_symbols` is empty AND `sol` evaluates numerically to ~0 → solved
- If `sol` is a `Poly` with all-zero coefficients → solved
- Otherwise fall through to canonicalizers

Currently `expand` IS the first check, but `expand` is heavyweight too (it canonicalizes the full polynomial). A `nsimplify(sol, tolerance=1e-10) == 0` check might be cheaper.

## Bottleneck #4: success-replay extra forward passes

`SuccessReplayCallback` (line 219+): runs `sr_iters_per_rollout=10` BC updates per rollout. Each update = `sr_batch_size=256` × model forward + backward.

With TreeMLP that's ~2.5k extra forwards per rollout. PPO does ~10 epochs of n_steps=2048 forwards = 20k forwards anyway, so success-replay adds ~12% overhead. Cutting iters 10→3 saves ~8%.

## Bottleneck #5: per-step graph_encoding rebuild

`utils_env.py:graph_encoding` (line 1137): every step rebuilds:
- New torch tensors for node features
- `torch.cat` for merging LHS/RHS subgraphs
- Edge index manipulations

For a 41-node graph that's ~50 small tensor ops per step. Not the dominant cost but adds 10-20% on top of pure sympy.

**Fix:** allocate per-env `node_buf`, `edge_buf` tensors of shape `(MAX_NODES, F)` and `(2, MAX_EDGES)`. Use slicing to fill in-place. Saves allocator churn.

## Proposed action list (in order)

**Now (no risk, 1-line changes):**
- Change `multiEqn.__init__` default `cache=False` → `cache=True`
- Bound `self.action_cache` size (add LRU eviction at 10k entries)
- Reduce `sr_iters_per_rollout` from 10 to 5 in train_abel.py CLI default

**Soon (small refactor):**
- Add `--eval_subsample N` flag; if set, `_log_eval` uses N random test eqns
- Add `--eval_skip_beam_during_training` flag; only run greedy during in-training evals; full beam/at10 at end

**Future (larger refactor):**
- Pre-allocate graph_encoding buffers per env
- Short-circuit solved-check with cheap structural tests
- Compile TreeMLP forward path

## Validation

After enabling #1-3, expected throughput on mixed_v2_large:
- 45 → ~80–100 steps/sec from action cache + sr_iters reduction
- Eval phase 45 min → 5 min from subsampling + skip beam during training
- Net effect: 1e7-step trial wall-clock drops from ~36h to ~10–12h

Worth verifying with a 1M-step calibration run before rolling out to the long mixed_v2_large run.
