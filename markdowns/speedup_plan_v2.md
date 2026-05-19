# Deeper speedup pass — v2

After implementing the first three speedups (cache, eval_subsample, eval_lite)
and seeing they only moved the needle ~5% on tiny, I did a deeper read. Here
are speedups I missed on the first pass, ranked by ROI.

## Top new wins (not yet implemented)

### 1. ❗ `step_base` does a needless sympify roundtrip on relabel_const

**File**: `envs/env_multi_eqn.py:375`

```python
if self.use_relabel_constants and operation.__name__ == 'relabel_const_custom':
    lhs_new, rhs_new, map_constants = operation(lhs_old, rhs_old)
    self.map_constants = map_constants
    if self.main_eqn_original is None:
        self.main_eqn_original = self.main_eqn
    self.main_eqn = sympify(str(lhs_new) + ' - ' + str(rhs_new))   # <-- BAD
    self.map_constants_history.append(map_constants)
```

This converts two sympy expressions to strings and re-parses them, just to
produce `lhs - rhs`. The agent picks `relabel_const` ~17% of the time per
the first-action diagnostic. Should be:

```python
self.main_eqn = lhs_new - rhs_new   # or expand(lhs_new - rhs_new) if needed
```

**Estimated speedup**: ~2× on relabel-heavy episodes. **Effort**: 1 line.
**Risk**: low; need to confirm whether downstream code depends on canonical
expanded form. The `sympify(str(x))` doesn't guarantee that anyway.

### 2. ❗ TreeMLP forward recomputes constants every K-iter

**File**: `utils/utils_env.py:389-411`

```python
for _ in range(self.K):
    # Flatten safely (handles non-contiguous tensors)
    src_flat   = src.reshape(-1)                 # CONSTANT across iters
    dst_flat   = dst.reshape(-1)                 # CONSTANT
    batch_flat = batch_idx.reshape(-1)           # CONSTANT
    mask_flat  = edge_mask.reshape(-1).bool()    # CONSTANT
    src_flat   = src_flat[mask_flat]             # CONSTANT
    dst_flat   = dst_flat[mask_flat]             # CONSTANT
    batch_flat = batch_flat[mask_flat]           # CONSTANT
    src_idx = batch_flat * N + src_flat          # CONSTANT
    dst_idx = batch_flat * N + dst_flat          # CONSTANT
    h_flat   = h.reshape(B * N, self.hidden_dim) # changes (h updated)
    messages = h_flat[src_idx]                   # changes
    agg_flat = torch.zeros_like(h_flat)          # could preallocate
    agg_flat.index_add_(0, dst_idx, messages)
    ...
```

All the `*_flat` and `*_idx` tensors are constant across the K=3 iterations
but recomputed every time. Hoist them above the loop.

Also `agg_flat = torch.zeros_like(h_flat)` allocates per iter; preallocate
once before the loop and use `agg_flat.zero_()`.

**Estimated speedup**: 1.3-1.6× on forward pass. **Effort**: 15 minutes.
**Risk**: low (purely a reordering); add a torch.allclose sanity check.

Since `forward()` is called every action step + every beam node expansion
+ every BC step, this propagates into a meaningful end-to-end speedup
(maybe 15-25% overall).

### 3. `sympy_expression_to_graph` recomputes encoding-type per node

**File**: `utils/utils_env.py:1112-1116`

```python
while stack:
    ...
    # Get feature vector
    # Detect encoding type (1D int or 2D list)
    first_val = next(iter(feature_dict.values()))   # CONSTANT per call
    is_1d = isinstance(first_val, int)              # CONSTANT
    default_val = 0 if is_1d else [99, 99]          # CONSTANT
```

Hoist out of the while loop. **Effort**: 1 minute. **Speedup**: tiny but
free.

### 4. Add last-call memoize at the env level

**File**: `envs/env_multi_eqn.py:step_base`

Even with `cache=True`, every step:
1. The action-masker wrapper calls `get_valid_action_mask()` → `make_actions_cache(lhs, rhs, ...)` → dict lookup
2. `step_base` then calls `make_actions_cache(lhs, rhs, ...)` again → another dict lookup

Two dict lookups + key hashing on the same `(lhs, rhs)`. Add a one-slot
memo:

```python
if (lhs_old, rhs_old) == self._last_actions_key:
    action_list, action_mask = self._last_actions_value
else:
    action_list, action_mask = make_actions_cache(...)
    self._last_actions_key = (lhs_old, rhs_old)
    self._last_actions_value = (action_list, action_mask)
```

**Effort**: 5 minutes. **Speedup**: ~5-10% per step (skip the second dict
lookup + hash).

### 5. torch.compile the TreeMLP forward

**File**: `utils/utils_env.py:368`

```python
@torch.compile(mode="reduce-overhead", dynamic=True)
def forward(self, obs: dict) -> torch.Tensor:
    ...
```

torch.compile typically gives 1.2-1.5× for free on small-batch
forward-heavy code. Requires Python 3.10+ and a warm-up pass (the first
call is slow).

**Effort**: 5 minutes to add the decorator. **Risk**: might not interact
well with sb3's policy machinery; needs a smoke test.

### 6. Lazy `is_solved` short-circuit on `_apply_cov`

**File**: `envs/env_multi_eqn.py:526-538`

`_apply_cov` is called when the agent picks the CoV macroaction.
`pi_cov_general` is called to find the substitution. If `pi_cov` returns
None, we early-return — good. But pi_cov itself runs sympy pattern matching
on the equation, which is non-trivial.

If we've already applied K CoVs (`len(self.cov_inv) >= max_cov_apps`),
`step_base` already short-circuits at line 379. Good.

But within one episode, if the agent picks CoV twice in a row on equations
where pi_cov returns None twice, we re-run pi_cov each time. Cache the
None-result for the current `main_eqn` to avoid recomputation.

**Effort**: 15 minutes. **Speedup**: depends on agent's CoV-spamming
behavior, but possibly 5-10% on CoV-heavy seeds (looks like ~70% of test
first-actions are CoV per diagnostic).

## Already-known but-not-yet-done items from v1 plan

### 7. Pre-allocate graph_encoding tensors

Still on the table — every step does `torch.tensor(...)` + `torch.cat` to
build node features. Switch to numpy preallocation + `torch.from_numpy`.

Estimated 1.2-1.4× on env step. Medium effort (30-60 min).

### 8. Short-circuit `_check_eqn_solved_local`

ACK that I already added `if lhs != v: return False` short-circuit, which
catches most non-solved states cheaply. **Already done.** No further work
needed here.

## Already implemented

- ✅ action cache (cache=True default + bounded eviction at 10k)
- ✅ --eval_subsample
- ✅ --eval_lite (skip beam+at10 mid-training)
- ✅ sr_iters_per_rollout 10→5
- ✅ Catch HeuristicGCDFailed
- ✅ Disable Python int-string limit

## Recommended next steps

In priority order (effort vs payoff):

| # | Change | Estimated speedup | Effort |
|---|---|---|---|
| 1 | Remove `sympify(str(...)+...)` in step_base | 2× on REL-heavy episodes | 1 line |
| 2 | Hoist constants out of TreeMLP K-loop | 1.3-1.6× on forward | 15 min |
| 4 | Last-call memo at env level | +5-10% on env step | 5 min |
| 5 | torch.compile TreeMLP | 1.2-1.5× on forward | 5 min |
| 7 | Pre-allocate graph_encoding | 1.2-1.4× on env step | 30-60 min |
| 3 | Hoist constants in sympy_expression_to_graph | <5%, free | 1 min |
| 6 | Cache pi_cov None-result | 5-10% on CoV-heavy seeds | 15 min |

Doing #1, #2, #4, #5 — all surgical changes — could give a combined
**~2× total speedup** on mixed_v2_easy and mixed_v2_large training. That's
a meaningful improvement worth chasing once current runs finish.

Defer #3 and #6 to a polish pass; #7 only if we're recompiling everything.

## Validation plan

Same as v1: run a 1M-step calibration on mixed_v2_easy (or tiny) with the
new code, compare steps/sec vs the existing baseline. If new code is
≥1.5× faster on env step, ship. Otherwise dig deeper into profiling.
