# AlphaZero-style expert iteration plan (next paper)

For when we decide to push past test-time beam search and treat symbolic
equation solving as **search with a learned heuristic, trained via
self-improvement**.

## Why this is the natural next step

Our env is essentially a *deterministic search problem* with verifiable
terminal states. RL with PPO and shaped rewards is one way to learn the
heuristic — but it has structural mismatches:

1. **PPO is trained for greedy decoding.** Its loss optimizes the policy to
   place mass on the modal action. At test time we then run beam search,
   which extracts more by considering non-modal actions. The policy was
   never *trained* to be a good search partner.

2. **The value head V(s) is mis-targeted.** Our V predicts cumulative shaped
   reward (mostly complexity-delta + step penalty). That correlates with
   "will this state lead to a solve?" but isn't the same. When we use
   V in beam scoring, we're really asking "go toward simpler states", not
   "go toward solvable states". The bias is masked but real.

3. **No feedback loop between search and policy.** Beam-search wins don't
   inform training. The policy keeps making the same first-action mistakes
   that beam routinely corrects.

Expert iteration closes all three loops at once:
- Policy is trained to *match search-improved targets*, becoming a strong
  search partner.
- Value is trained on *actual outcomes* (solved or not after T steps),
  becoming a true heuristic for solvability.
- Search produces training data; training data improves search.

## Three-step sequence (in dependency order)

### Step 1 — Retrain V on solvability (1-2 days)

Highest ROI, smallest scope. Today's V is trained as part of PPO's
advantage estimator; replace its target.

**New V target candidates:**

| Target | Meaning | Pros | Cons |
|---|---|---|---|
| `1.0 if solved within K steps else 0.0` | Bernoulli "will solve" | Simple, well-calibrated | Sparse signal |
| `-steps_to_solve / K` | Negative remaining cost-to-go | Dense | Needs `inf` handling for failed eqns |
| `exp(-steps/τ)` if solved else 0 | Smooth solvability | Dense + bounded | One extra hyperparam |

Recommendation: start with `1.0 if solved else 0.0`, train a fresh
critic-only head on the same TreeMLP backbone, evaluate as a beam-scoring
function. Compare against the PPO-trained V on the same checkpoint.

**Practical setup:**

```python
# new_value_train.py
trainer = ValueTrainer(
    backbone=copy.deepcopy(policy.features_extractor),  # freeze
    head=nn.Linear(features_dim, 1),
    target_kind="solvability",  # or "neg_steps"
)
# data source: trajectory buffer from existing PPO rollouts, labeled by
# whether the trajectory's terminal was is_solved
trainer.train(epochs=20, lr=1e-3)
torch.save(trainer.head.state_dict(), "v_solvability.pt")
```

**Eval test**: run `eval_value_beam.py` with the new V plugged in, sweep λ.
If test_beam lifts > 5% absolute over today's V at the same λ, the
mis-targeting hypothesis is confirmed and we proceed.

### Step 2 — MCTS at test time (2-3 days)

Build on the existing snapshot/restore infrastructure from beam_solve_one.

**Node structure:**
```python
@dataclass
class MCTSNode:
    snap: dict          # env state
    parent: Optional['MCTSNode']
    action_from_parent: Optional[int]
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0  # policy prior P(a | parent_state)
    children: dict[int, 'MCTSNode'] = field(default_factory=dict)
    is_terminal: bool = False
    solved: bool = False
```

**PUCT score:**
```
score(node, action) = Q(child) + c_puct * P(action | node.state) * sqrt(visits) / (1 + child.visits)
```
where `Q(child) = child.value_sum / max(1, child.visits)` and `V(s)` from
the retrained head is used as `value_sum` initialization at expansion.

**Simulation loop:**
1. From root, traverse using PUCT until reaching a leaf.
2. Expand: legal-action mask + policy priors.
3. Get V(leaf) as initial value estimate.
4. Backup along path: `node.visits += 1; node.value_sum += V_leaf`.
5. If any child is_solved, propagate solved=True up the tree.

After N simulations from root, the "improved policy" is `π(a|root) ∝ N(root, a)`.

**Eval harness:** new script `eval_mcts.py` mirroring `eval_value_beam.py`
but with MCTS instead of beam. Compare on the same test sets.

**Critical design choices:**
- **Canonicalization**: dedupe nodes by `simplify(lhs - rhs)` (not just
  `expand`) to avoid redundant subtree exploration.
- **Action mask handling**: PUCT only selects from legal children. The
  mask is applied at expansion time.
- **Cycle detection**: track visited (lhs, rhs) along the path; refuse to
  expand into a state already seen ancestor-side. Symbolic env has many
  cycles (e.g., expand then collect).
- **Compute budget**: 30-100 simulations per root state. Each sim does
  ~10-20 env-steps + forward passes. Adds ~10× to per-state eval time.

### Step 3 — Expert iteration training (1-2 weeks)

Loop:
```python
for iter in range(N_ITERS):
    # 1. Collect search-improved data
    buffer = []
    for eqn in train_eqns:
        root_visit_distribution, trajectory = mcts_search(eqn, policy, value, N_SIMS)
        buffer.append((state, root_visit_distribution, trajectory_outcome))

    # 2. Train policy to match search distribution
    for state, target_pi, _ in buffer:
        policy_loss = cross_entropy(policy(state), target_pi)

    # 3. Train value to match observed outcomes
    for state, _, outcome in buffer:
        value_loss = mse(value(state), outcome)

    # 4. Save checkpoint
    save(f"iter_{iter}.zip")
```

**Compute budget:** if each train eqn needs 50 sims and the train set is
1000 eqns, that's 50k sims per iter. At ~5s per sim, ~70h per iter. So
1 iter per day on 10 cores. 10 iters = 10 days. Same order as we're
already spending on multi-seed PPO sweeps, but with much higher expected
gain.

**Mitigations for compute:**
- Use the value-beam decoder as warm-start for the first iter (it's like
  pre-training on the existing policy).
- Reduce N_SIMS for the first few iters (10-20 sims) and ramp up.
- Parallelize MCTS across train eqns (embarrassingly parallel by eqn).

### Step 4 (optional) — Curriculum / self-play extension

If step 3 gives strong results, we can extend in two directions:

- **Difficulty-aware curriculum**: bias sampling toward eqns the agent
  recently failed. Closes another loop.
- **Eqn generation**: a separate generator proposes new training eqns
  matching the structure of frequently-failed ones. Adapted from
  Lample-Charton-style data augmentation but on-policy.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| V on solvability is too sparse on hard eqns | Use `exp(-steps/τ)` form for dense signal |
| MCTS explodes branching factor on big eqns | Tight action masks + canonical dedup; cap tree size to 5000 nodes per search |
| Cycles in symbolic env | Hash-based ancestor check at expansion |
| Compute eats deadlines | Start with step 1+2 only (~1 week); defer step 3 to follow-up paper |
| New target distribution diverges from old PPO policy | Use KL anchor for first iter; warm-start from PPO checkpoint |

## Existing related work to read before starting

- **AlphaZero** (Silver et al. 2018) — the canonical algorithm.
- **Polu & Sutskever 2020** "Generative Language Modeling for Automated
  Theorem Proving" — applied expert iteration to math, gives a working
  template for our setting.
- **HyperTree Proof Search** (Lample et al. 2022) — adapts MCTS for
  deduction trees in Lean theorem proving. Most directly analogous to our
  setting.
- **AlphaProof** (DeepMind 2024) — recent IMO-level proof search using
  expert iteration. Mostly a press piece but the methods section is
  worth scanning.
- **DSO / SymbolicGPT** for symbolic regression context — different
  problem but related search structure.

## When to start

Not now. Current paper depends on stability of existing experiments. Start
expert iteration AFTER:

- ICLR submission goes in (or whatever next deadline)
- We have at least 3 verified seed reproductions of the current best
  config (value-beam + anti-loop + cbrt)
- Mass-eval-style comparison tables are stable

Then dedicate a focused 2-3 week sprint to steps 1-2 (V retraining + MCTS
at test time). Evaluate; if results are strong, commit to step 3 (expert
iteration) for the follow-up paper.

## Estimated total compute and code budget

| Phase | Days code | Days compute | Risk |
|---|---|---|---|
| Step 1 (retrain V) | 1-2 | 0.5 | low |
| Step 2 (MCTS test-time) | 2-3 | 1 | medium |
| Step 3 (expert iteration) | 4-7 | 7-14 | high |
| Total to working AlphaZero variant | 7-12 | 8-16 | -- |

Roughly 3-4 weeks of focused work to a competitive AlphaZero baseline on
our problem. Worth it for the next paper. Not worth it for the current one.
