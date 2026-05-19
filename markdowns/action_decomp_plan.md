# Action decomposition refactor plan (deferred)

ChatGPT recommendation #2: replace the flat 50-action discrete space with a
factorized op-head + term-pointer policy. This is "next paper" material —
sketching here so we can pick it up cleanly.

## What's wrong with the current action space

```python
self.actions_fixed = [
    (custom_expand, None), (custom_collect, x), (custom_square, None),
    (custom_sqrt, None), (custom_log, None), (custom_exp, None),
    (custom_sin, None), (inverse_sin, None), (custom_cos, None),
    (inverse_cos, None), (mul, -1),
    # + (cov_action, None) if use_cov
    # + (relabel_const, None) if use_relabel_constants
]
# Then dynamic: (op, term) for op in {add, sub, mul, truediv} and
# term in get_ordered_sub_expressions(lhs)
# Total padded to 50 with custom_identity.
```

Problems:

- **Arbitrary 50-action cap.** Equations with many subterms get truncated;
  the agent never sees those term choices.
- **Term identity is positional**: action index 14 means "multiply by the
  third term in sort-order" — completely policy-dependent and unstable as
  the expression changes.
- **No generalization across structurally identical positions.** Action 14
  for eqn A means something totally different than action 14 for eqn B.

## What we want

Two heads:

```python
op_logits     = OpHead(state_embedding)              # shape (1, |OP_VOCAB|)
term_logits   = TermPointer(state_embedding, op)     # shape (1, max_nodes)
                                                     # masked by valid nodes
```

where:

- `OP_VOCAB = {add, sub, mul, truediv, square, sqrt, log, exp, sin, cos, ...}`
  -- around 15 op symbols. Fixed and small.
- `term_logits` is an attention over expression-tree nodes, using TreeMLP's
  per-node embeddings as keys. Some ops (like `square`, `sqrt`, `log`) don't
  need a term, so the term pointer is conditionally bypassed.

## Architecture changes

`utils/tree_mlp.py` currently produces a pooled state embedding and feeds
into PPO's policy head (a small MLP → 50-dim logits). New design:

```
TreeMLPExtractor:
    features_dim = D_state + max_nodes * D_node      # concat state + per-node
ActorHead:
    op_head      = MLP(D_state → |OP_VOCAB|)
    term_query   = MLP(D_state + D_op_embed → D_node) # query for attention
    term_pointer = dot(term_query, per_node_keys) → (1, max_nodes)
CriticHead:
    value_head   = MLP(D_state → 1)
```

This requires writing a custom `MaskableMultiHeadCategorical` distribution
that emits action tuples `(op_idx, term_idx)` and computes
`log_prob = log π(op | s) + log π(term | s, op)`.

## PPO interface impacts

`MaskablePPO` (from sb3-contrib) supports single-discrete-action with
masking. Factored actions need either:

1. **Subclass `MaskablePPO`** with a custom action space and policy that
   handles the factored log-prob computation. Most flexible but most code.
2. **Flatten to (op_idx * max_nodes + term_idx)** with masking, and recover
   factored structure inside the policy. Easier to wire in sb3 but loses
   the architectural elegance.

Option 1 is cleaner. Estimate 3-4 days of focused work just for the policy
side, plus 1-2 days for the env-side changes (apply factored action).

## Env-side changes

`step_base` currently does:

```python
operation, term = action_list[action_index]
# ... apply operation(lhs, term) ...
```

Need to change to:

```python
op_idx, term_idx = action_tuple
operation = OP_VOCAB[op_idx]
if operation in NO_TERM_OPS:
    term = None
else:
    term = expression_tree_nodes[term_idx]  # pointer to actual subtree
```

The `expression_tree_nodes` list is what TreeMLP's encoder uses internally
— we need to expose it.

## Action-mask changes

Currently `valid_action_mask` is a boolean array of length 50. New mask
shape: `(|OP_VOCAB|,)` for op + `(max_nodes,)` for term, with term mask
depending on the chosen op.

Two-stage masking: sample `op` from `op_logits[op_mask]`, then sample
`term` from `term_logits[term_mask(op)]`.

## Estimated effort

| Task | Days | Risk |
|---|---|---|
| `MaskableFactoredCategorical` distribution | 1 | medium (sb3 internals) |
| TreeMLP head split (op + pointer) | 1 | low |
| Custom policy class subclassing `MaskableActorCriticPolicy` | 1 | high (PPO advantages, value targets) |
| Env-side: expose node list to action layer | 0.5 | low |
| First-run debugging (action distribution sanity check) | 1 | high |
| Re-train all our baselines under new arch | (compute) | n/a |
| Ablation table: flat vs factored | 0.5 | low |

**Total: ~5 days of focused work**, plus compute to re-train baselines.

## What this should unlock

- Equations with arbitrary subterm counts (no 50-cap truncation).
- Better generalization across equations: "multiply by 3rd subterm" is no
  longer a fixed action — the policy points at a meaningful node.
- Cleaner inputs for value-guided beam (pointer mass is more interpretable
  than flat-action mass).

If we believe ChatGPT's analysis, this is **the** big architectural lever.
But: it's a multi-day refactor with rewrite-PPO-internals risk. Only attempt
if we have ≥1 week of focused time and value-beam + anti-loop + BC haven't
maxed out the existing arch.
