# CoV Agent — MDP

As currently implemented in `envs/env_cov.py` (`covEnv`).

## State

Observation = (equation encoding, depth, action history).

- **Equation encoding** of the *current main equation* `(lhs, rhs=0)`:
  - `integer_1d`: preorder-traversal token IDs, padded to `max_expr_length=20`.
  - `graph_integer_1d`: dict with `node_features`, `edge_index`, `node_mask`, `edge_mask`.
- **Depth**: scalar in `[0, max_depth]`, number of CoV ops applied so far.
- **Action history**: last `hist_len=10` action indices (pad −1).

Note: the encoded equation does *not* change during an episode — only the depth and action history evolve. The agent builds the substitution `f(x)` against a static view of the input equation.

## Actions

Action set = `(op, term)` pairs ∪ `(STOP, None)`.

- `op ∈ {ADD, SUB, MUL, DIV}`
- `term ∈ term_bank` — default `{a, b, c, d, e, 2, 3, 4}`
- One special `(STOP, None)` action

`|A| = 4 × |term_bank| + 1 = 33` with defaults.

Action mask blocks `(DIV, 0)`.

## Transitions

Internal substitution-under-construction `cov ∈ Sympy expr`. Update rule:

- **Step 1** (`depth=0`): `base_op ← op`, `cov ← term`. Depth → 1.
- **Step k>1**: `cov ← simplify(cov ⊕ term)` where `⊕ ∈ {+, −, ×, ÷}` per `op`. Depth → k+1.

On `(STOP, None)` *or* `depth == max_depth`:
- Compose with starting `x`: `cov ← base_op(x, cov)`. So if `base_op=SUB`, final `f(x) = x − cov_inner`.
- Substitute into the main equation: `eqn_after ← main_eqn.subs(x, cov)`.
- Episode terminates.

## Reward

- **Per non-terminal step**: `−step_penalty` (default `−0.1`).
- **Terminal step** (after CoV applied):
  - `Δ = C(main_eqn) − C(eqn_after)` where `C` = polynomial term count or `count_ops` fallback.
  - `reward = Δ − f_penalty · cost(cov)`.
  - If `reward > 0`: `reward *= 10` (positive amplification + bookkeeping for `solve_counts`).
  - If `reward < 0`: clipped to `−1`.
  - If `reward == 0` and `depth < 2`: forced to `−1` (discourage trivial early stops).

## Terminal condition

`(STOP, None)` selected, OR `depth ≥ max_depth` (default `max_depth=3`).

## Episode budget

Default `max_depth=3`, which is enough for quadratics (`x → x − b/(2a)` needs 3 ops: SUB b, DIV 2, DIV a). For cubic/quartic depression the same depth works (`x → x − b/(na)`). Need to bump for radical (`x → x²`, 2 ops) and reciprocal-symmetric (`x → x + 1/x`, more ops).

## What this MDP rewards

**Complexity reduction**, not solvability. The agent gets positive reward when the post-CoV equation has fewer polynomial terms than the original — which for our special forms coincides with "depresses to a pure form."

Implicit assumption: if `Δ > 0`, the post-CoV equation is now closed-agent-solvable. This holds for our four current classes by construction (special-form polynomials, exponential). For new classes we add, we need to verify this property at dataset-generation time.

## Open MDP questions

1. **State should arguably include the partial `cov` itself.** Right now the agent sees only the original equation, depth, and action history — it must infer `cov` from action history. Adding `cov` encoding to the state would make the policy Markovian on the substitution-construction process. Worth A/B.
2. **Reward sparsity.** The complexity-delta shaping is dense but may misdirect (a substitution that *looks* simpler but doesn't actually solve). Worth testing pure outcome reward (`+1` if `Δ > 0`, `−1` otherwise) as ablation.
3. **`STOP` as a separate action vs. forced termination at `max_depth`.** Current env supports both; might be cleaner to drop `STOP` and always run to `max_depth`. The "STOP early" optionality probably hurts exploration.
