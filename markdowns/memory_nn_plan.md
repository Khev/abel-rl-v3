# memory_nn.md

# Memory-Augmented Neural Equation Solver

## One-line goal

Add an **episodic memory module** to the ABEL-RL / equation-solving agent so that once the system solves an equation state, it can reuse that successful trace, action, or value estimate instead of rediscovering the solution from scratch.

---

## Core hypothesis

Equation solving is not just policy learning. It is:

\[
\text{neural heuristic} + \text{symbolic search} + \text{memory of solved subproblems}.
\]

The key hypothesis is:

> A memory-augmented solver will outperform a pure neural policy and vanilla beam search because algebraic solution traces contain reusable substructure.

In plain language:

> If we solved a similar equation before, we should remember what worked.

---

## Why memory is natural here

Equation solving has many recurring motifs:

- subtract constant from both sides,
- divide by coefficient,
- collect like terms,
- expand then collect,
- multiply by denominator,
- simplify rational expression,
- isolate \(x\),
- apply inverse operation,
- solve repeated templates with different coefficients.

A neural network can learn these patterns slowly in its weights. An external memory can store them explicitly.

The system should remember:

1. exact solved states,
2. similar solved states,
3. successful next actions,
4. full solution traces,
5. estimated remaining steps to solution,
6. common macro-actions/tactics.

---

## Important framing

Do **not** start with a fancy differentiable memory network like a Neural Turing Machine.

Start with a practical memory system:

\[
\text{policy/value network} + \text{episodic memory} + \text{beam/PUCT search}.
\]

The first useful version is not end-to-end differentiable. It is an external datastore that improves action selection and search.

---

# High-level architecture

Given current equation state \(s\), the solver has:

1. neural policy:

\[
\pi_\theta(a \mid s)
\]

2. neural value:

\[
V_\theta(s)
\]

3. exact memory:

\[
M_{\rm exact}[\operatorname{hash}(s)]
\]

4. kNN memory:

\[
M_{\rm knn} = \{(h(s_i), a_i^\*, v_i, \tau_i)\}
\]

where:

- \(h(s_i)\) is the neural embedding of a solved/visited state,
- \(a_i^\*\) is the successful action taken from that state,
- \(v_i\) is the memory value, e.g. negative remaining steps or success probability,
- \(\tau_i\) is the remaining solution trace.

The final policy can be:

\[
\pi_{\rm final}(a \mid s)
=
(1-\alpha)\pi_\theta(a \mid s)
+
\alpha \pi_{\rm mem}(a \mid s)
\]

or in logit form:

\[
\ell_{\rm final}(a)
=
\ell_\theta(a)
+
\beta \ell_{\rm mem}(a).
\]

For beam search:

\[
\text{score}(s,a,s')
=
\log \pi_\theta(a \mid s)
+
\beta \log \pi_{\rm mem}(a \mid s)
+
\lambda V_\theta(s')
+
\gamma V_{\rm mem}(s')
-
\mu C(s').
\]

where \(C(s')\) is symbolic complexity.

---

# Desired result

The target empirical ladder is:

\[
\text{greedy policy}
<
\text{vanilla beam}
<
\text{value-guided beam}
<
\text{memory-guided beam}
<
\text{memory + PUCT beam}.
\]

A strong result would be:

> Memory-guided beam solves more hard equations than value-guided beam at the same expansion budget.

An even stronger result would be:

> kNN memory improves greedy or shallow-search performance on held-out templates, not just near-duplicate equations.

---

# Scope for first implementation

Do **not** implement everything at once.

Implement in this order:

1. exact solved-state cache,
2. trace replay from exact cache,
3. kNN action memory,
4. memory-guided beam scoring,
5. optional PUCT memory scoring,
6. macro-action memory later.

The first milestone should be useful even if kNN memory is not ready.

---

# Assumptions about existing code

The existing code likely has some or all of:

- an equation-solving environment,
- `reset()`, `step(action)`,
- `is_solved` or equivalent terminal check,
- legal action masks,
- trained PPO policy,
- value head,
- beam search or value-guided beam,
- symbolic complexity function,
- state encoder / graph encoder / vector encoder,
- evaluation harness over equation sets.

Do not rewrite the environment. Integrate memory with the existing training/evaluation pipeline.

---

# Code structure to add

Add a new memory module without disrupting existing code.

Suggested files:

```text
src/
    memory/
        __init__.py
        canonical.py
        exact_cache.py
        knn_memory.py
        memory_policy.py
        memory_beam.py
        trace_store.py
        metrics.py

scripts/
    build_memory.py
    eval_memory.py
    eval_memory_beam.py
    inspect_memory.py

tests/
    test_canonical.py
    test_exact_cache.py
    test_knn_memory.py
    test_memory_policy.py
    test_memory_beam.py

results/
    memory/
        .gitkeep
```

If the project uses a different structure, adapt names but keep these logical components.

---

# Phase 1: Exact solved-state cache

## Goal

If the solver sees an equation state it has solved before, retrieve and replay the known solution trace.

This is the easiest and probably immediately useful memory.

---

## What to store

For every solved rollout/beam trace, store each state along the successful path:

```python
{
    "canonical_key": str,
    "state_repr": optional,
    "best_action": action,
    "remaining_trace": list[action],
    "remaining_steps": int,
    "final_success": True,
    "final_equation": str,
    "complexity": float,
    "source_equation_id": str,
    "metadata": dict,
}
```

For a successful trajectory:

```text
s0 --a0--> s1 --a1--> s2 --a2--> ... --ak--> solved
```

store:

```python
memory[s0] = {
    "best_action": a0,
    "remaining_trace": [a0, a1, ..., ak],
    "remaining_steps": k+1,
}

memory[s1] = {
    "best_action": a1,
    "remaining_trace": [a1, ..., ak],
    "remaining_steps": k,
}
```

etc.

---

## Canonicalization

Implement:

```python
canonical_key(state) -> str
```

This is critical.

Use existing environment methods if available. Otherwise:

1. extract equation expression,
2. simplify left and right sides,
3. move all terms to one side if safe,
4. expand/collect if appropriate,
5. normalize sign if safe,
6. sort terms via SymPy canonical string,
7. include relevant environment metadata if needed.

Example:

\[
2x + 4 = 0
\]

and

\[
x + 2 = 0
\]

may or may not be equivalent depending on allowed normalization. Be conservative first.

Start with exact syntactic canonicalization. Add algebraic normalization later.

---

## Exact cache API

Create `src/memory/exact_cache.py`.

Implement:

```python
class ExactSolvedCache:
    def __init__(self):
        self.store = {}

    def add_trace(self, states, actions, final_success, metadata=None):
        ...

    def lookup(self, state):
        ...

    def contains(self, state):
        ...

    def save(self, path):
        ...

    @classmethod
    def load(cls, path):
        ...
```

`lookup(state)` should return either:

```python
None
```

or:

```python
{
    "best_action": ...,
    "remaining_trace": ...,
    "remaining_steps": ...,
    "metadata": ...
}
```

---

## Trace replay

Add a utility:

```python
def solve_with_exact_memory(env, cache, max_steps):
    ...
```

Logic:

1. canonicalize current state,
2. if exact cache hit, try replaying stored trace,
3. verify every action is legal,
4. stop if solved,
5. if replay fails, fall back to policy/beam.

Important:

- Do not blindly trust memory.
- Always verify action legality and state transitions.
- If replay fails, log it as stale/invalid memory.

---

## Phase 1 deliverables

Files:

```text
src/memory/canonical.py
src/memory/exact_cache.py
src/memory/trace_store.py
scripts/build_memory.py
scripts/eval_memory.py
tests/test_exact_cache.py
```

Commands:

```bash
python scripts/build_memory.py \
    --input results/solved_traces.pkl \
    --output results/memory/exact_cache.pkl

python scripts/eval_memory.py \
    --memory results/memory/exact_cache.pkl \
    --eval_set data/eval_easy.json
```

Metrics:

- exact hit rate,
- exact replay success rate,
- solved rate with exact memory,
- average steps saved,
- number of stale/invalid traces.

Success criterion:

> Exact memory improves solve rate or reduces average steps on evaluation equations that share subproblems with training traces.

---

# Phase 2: kNN action memory

## Goal

Retrieve similar solved states and use their successful actions to bias the current policy.

Exact memory only handles states we have seen before. kNN memory should help on structurally similar but unseen states.

---

## What to store

For each state on a successful trace:

```python
{
    "embedding": np.ndarray,
    "action": action,
    "remaining_steps": int,
    "success": True,
    "canonical_key": str,
    "equation_str": str,
    "metadata": dict,
}
```

The embedding should come from the existing neural encoder if possible.

Potential embedding sources:

1. policy network hidden state,
2. value network hidden state,
3. graph encoder output,
4. vector observation itself,
5. hand-crafted symbolic feature vector.

Start with whatever is easiest and stable.

---

## kNN memory API

Create `src/memory/knn_memory.py`.

Implement:

```python
class KNNActionMemory:
    def __init__(self, k=16, metric="cosine"):
        ...

    def add(self, embedding, action, remaining_steps, metadata=None):
        ...

    def build_index(self):
        ...

    def query(self, embedding, k=None):
        ...

    def save(self, path):
        ...

    @classmethod
    def load(cls, path):
        ...
```

If FAISS is available, use it optionally. Otherwise use NumPy / scikit-learn nearest neighbors.

Do not make FAISS a hard dependency for the first version.

---

## Memory action prior

Given nearest neighbors:

\[
\{(h_j, a_j, d_j, r_j)\}_{j=1}^k
\]

construct action weights:

\[
w_j = \exp(-d_j/\tau) \cdot \frac{1}{1+r_j}
\]

where \(r_j\) is remaining steps.

Then:

\[
\pi_{\rm mem}(a)
=
\frac{
\sum_{j: a_j=a} w_j
}{
\sum_j w_j
}.
\]

If actions are structured, convert them to an action id compatible with the current action space. If an action is illegal in the current state, mask it out.

---

## Memory policy API

Create `src/memory/memory_policy.py`.

Implement:

```python
def memory_action_prior(
    state,
    embedding,
    knn_memory,
    legal_action_mask,
    action_space,
    k=16,
    temperature=0.1,
):
    ...
```

Return:

```python
pi_mem: np.ndarray
diagnostics: dict
```

Diagnostics should include:

- nearest distances,
- neighbor actions,
- how many neighbor actions were legal,
- entropy of memory prior,
- top memory action,
- memory confidence.

---

## Combining neural policy and memory policy

Implement:

```python
def combine_policy_with_memory(
    policy_probs,
    memory_probs,
    legal_action_mask,
    alpha=0.3,
    mode="mixture",
):
    ...
```

Modes:

1. mixture:

\[
\pi = (1-\alpha)\pi_\theta + \alpha \pi_{\rm mem}
\]

2. logit_add:

\[
\ell = \ell_\theta + \beta \log(\pi_{\rm mem}+\epsilon)
\]

3. override_if_confident:

If memory confidence exceeds threshold, use memory top action.

Start with mixture.

---

## Phase 2 deliverables

Files:

```text
src/memory/knn_memory.py
src/memory/memory_policy.py
scripts/build_knn_memory.py
scripts/eval_knn_policy.py
tests/test_knn_memory.py
tests/test_memory_policy.py
```

Commands:

```bash
python scripts/build_knn_memory.py \
    --traces results/solved_traces.pkl \
    --model path/to/model.zip \
    --output results/memory/knn_memory.pkl

python scripts/eval_knn_policy.py \
    --memory results/memory/knn_memory.pkl \
    --model path/to/model.zip \
    --eval_set data/eval_easy.json \
    --alpha 0.3 \
    --k 16
```

Metrics:

- kNN hit rate,
- legal neighbor action rate,
- memory prior entropy,
- greedy solve rate with neural policy only,
- greedy solve rate with neural + memory,
- average solution length.

Success criterion:

> Neural + kNN memory improves greedy or shallow beam solve rate over neural-only on at least one held-out evaluation set.

---

# Phase 3: Memory-guided beam search

## Goal

Use memory to improve beam search action selection and state scoring.

This is the most important near-term research experiment.

---

## Beam score

Add a memory term to the existing beam score:

\[
\text{score}
=
\log \pi_\theta(a\mid s)
+
\beta \log \pi_{\rm mem}(a\mid s)
+
\lambda V_\theta(s')
+
\gamma V_{\rm mem}(s')
-
\mu C(s').
\]

where:

- \(s'\) is next state,
- \(C(s')\) is symbolic complexity,
- \(V_{\rm mem}(s')\) is estimated from kNN remaining steps.

Possible memory value:

\[
V_{\rm mem}(s)
=
-\min_{j\in \operatorname{kNN}(s)} r_j
\]

or

\[
V_{\rm mem}(s)
=
\sum_j w_j \frac{1}{1+r_j}.
\]

Start simple:

\[
V_{\rm mem}(s)
=
\max_j \frac{w_j}{1+r_j}.
\]

---

## Memory-guided beam API

Create `src/memory/memory_beam.py`.

Implement:

```python
def memory_guided_beam_search(
    env,
    model,
    exact_cache=None,
    knn_memory=None,
    beam_width=16,
    max_depth=20,
    alpha=0.3,
    beta=1.0,
    lambda_value=1.0,
    gamma_mem_value=1.0,
    complexity_penalty=0.1,
    use_exact_cache=True,
    use_knn_policy=True,
    use_knn_value=True,
):
    ...
```

Return:

```python
{
    "solved": bool,
    "actions": list,
    "states": list,
    "num_expanded": int,
    "final_complexity": float,
    "cache_hits": int,
    "knn_queries": int,
    "diagnostics": dict,
}
```

---

## Baselines to compare

Run all decoders on identical eval sets:

| Decoder | Description |
|---|---|
| greedy_policy | current policy only |
| vanilla_beam | existing beam score |
| value_beam | policy + value head |
| exact_cache_only | replay exact memory when possible |
| knn_policy_greedy | policy mixed with kNN action prior |
| memory_beam | beam with exact + kNN policy/value |
| memory_beam_no_exact | kNN only |
| memory_beam_no_knn | exact only |

---

## Phase 3 deliverables

Files:

```text
src/memory/memory_beam.py
scripts/eval_memory_beam.py
tests/test_memory_beam.py
```

Commands:

```bash
python scripts/eval_memory_beam.py \
    --model path/to/model.zip \
    --exact_cache results/memory/exact_cache.pkl \
    --knn_memory results/memory/knn_memory.pkl \
    --eval_set data/eval_hard.json \
    --beam_width 16 \
    --max_depth 20
```

Metrics:

- solve rate,
- mean solution length,
- median solution length,
- mean expansions,
- solve rate at fixed expansion budget,
- exact cache hit rate,
- kNN legal action rate,
- mean final complexity for failures,
- wall-clock time.

Success criterion:

> Memory-guided beam beats value-guided beam at the same beam width and max depth.

Best success criterion:

> Memory-guided beam solves equations that value-guided beam fails, especially on hard or longer-horizon cases.

---

# Phase 4: Memory dataset splits

## Goal

Prove memory is not just cheating by memorizing train examples.

Use multiple evaluation splits.

---

## Split A: same templates, different coefficients

Training:

\[
ax+b=c
\]

with coefficient range:

```python
a,b,c in [-10,10]
```

Test:

same template, new coefficients.

This tests interpolation.

---

## Split B: held-out templates

Train on:

- \(ax+b=c\),
- \(a/(x+b)=c\),
- \(c(ax+b)+d=e\).

Test on:

- \((ax+b)/(cx+d)=e\),
- nested rational variants,
- mixed forms.

This tests structural generalization.

---

## Split C: longer horizon

Train on equations solvable in \(\leq 5\) steps.

Test on equations solvable in \(6-12\) steps.

This tests extrapolation.

---

## Split D: adversarial variants

Create equations that look simple by complexity but require non-greedy moves.

Example categories:

- must expand before collecting,
- must multiply denominator before simplifying,
- simplification trap,
- equivalent forms with different syntax,
- distractor terms.

This tests whether memory helps escape local policy traps.

---

## Required table

Report:

| Method | Same template | Held-out templates | Longer horizon | Adversarial |
|---|---:|---:|---:|---:|
| greedy policy |  |  |  |  |
| beam |  |  |  |  |
| value beam |  |  |  |  |
| exact cache |  |  |  |  |
| kNN memory |  |  |  |  |
| memory beam |  |  |  |  |

---

# Phase 5: Macro-action memory

## Goal

Discover and reuse common solution subtraces as macro-actions/options.

Do this only after exact/kNN memory works.

---

## Macro extraction

From solved traces, mine common subsequences:

```text
subtract_const -> divide_coeff
expand -> collect -> divide
multiply_denominator -> simplify -> collect
factor -> cancel -> divide
```

Store macros:

```python
{
    "macro_id": str,
    "actions": list,
    "precondition_signature": optional,
    "success_count": int,
    "avg_remaining_steps_after": float,
}
```

---

## Macro execution

At a state, a macro is legal if:

1. first primitive action is legal,
2. sequence can be replayed without illegal action,
3. complexity does not blow up beyond threshold,
4. stop early if solved.

Add macros as candidate actions in beam search.

---

## Success criterion

> Macro-memory reduces solution length or expansion count on longer-horizon equations.

---

# Phase 6: Optional PUCT memory search

## Goal

Use AlphaZero-style PUCT scoring with memory-augmented priors.

This is optional for now.

PUCT score:

\[
U(s,a)
=
Q(s,a)
+
c_{\rm puct}
P_{\rm final}(a\mid s)
\frac{\sqrt{N(s)}}{1+N(s,a)}.
\]

where:

\[
P_{\rm final}
=
(1-\alpha)P_\theta + \alpha P_{\rm mem}.
\]

Use memory value to initialize leaf values:

\[
V_{\rm leaf}
=
\eta V_\theta(s)
+
(1-\eta)V_{\rm mem}(s).
\]

Deliverable:

```text
src/memory/memory_puct.py
scripts/eval_memory_puct.py
```

This is lower priority than memory-guided beam.

---

# Logging and diagnostics

Every memory experiment should log:

```python
{
    "equation_id": str,
    "method": str,
    "solved": bool,
    "num_steps": int,
    "num_expanded": int,
    "wall_time_sec": float,
    "exact_cache_hits": int,
    "knn_queries": int,
    "knn_legal_action_rate": float,
    "memory_confidence_mean": float,
    "memory_confidence_max": float,
    "final_complexity": float,
    "failure_reason": str,
}
```

Save per-equation traces for debugging:

```text
results/memory/traces/{method}/{equation_id}.json
```

Each trace should include:

- equation string at each step,
- selected action,
- neural policy top actions,
- memory top actions,
- value estimate,
- memory value estimate,
- complexity,
- cache hit/miss.

---

# Plots

Generate:

1. solve rate by method,
2. solve rate vs beam width,
3. solve rate vs max depth,
4. expansions per solved equation,
5. memory hit rate vs eval split,
6. kNN distance histogram,
7. legal-neighbor-action rate,
8. examples where memory solves and baseline fails.

---

# Paper-level results to look for

The memory experiment is valuable if it shows one of these:

## Result A: exact memory helps repeated subproblems

> Exact solved-state cache reduces average solution length and recovers known subtraces.

This is useful engineering, but not enough for a paper by itself.

## Result B: kNN memory improves generalization

> kNN action memory improves solve rate on unseen equations with related structure.

This is much stronger.

## Result C: memory-guided beam beats value-guided beam

> Memory-guided search solves hard equations that the policy/value network alone misses.

This is the best near-term result.

## Result D: macro memory discovers reusable tactics

> Repeated solution traces can be compressed into tactics that improve long-horizon solving.

This is potentially a strong research contribution.

---

# Go / no-go criteria

Continue investing in memory if at least one is true:

1. exact cache hit rate is nontrivial on evaluation traces,
2. kNN memory improves greedy solve rate by at least 5 percentage points,
3. memory-guided beam improves solve rate over value-guided beam at same expansion budget,
4. memory helps on held-out templates or longer-horizon equations.

Stop or deprioritize memory if:

1. kNN retrieved actions are usually illegal,
2. memory only helps exact train duplicates,
3. memory-guided beam is slower but not more accurate,
4. learned embeddings do not organize algebraic similarity.

---

# First Claude Code instruction

Start with **Phase 1 and Phase 2 only**.

Do not implement macro-actions or PUCT yet.

Immediate deliverables:

```text
src/memory/canonical.py
src/memory/exact_cache.py
src/memory/knn_memory.py
src/memory/memory_policy.py
scripts/build_memory.py
scripts/build_knn_memory.py
scripts/eval_memory.py
scripts/eval_knn_policy.py
tests/test_exact_cache.py
tests/test_knn_memory.py
tests/test_memory_policy.py
```

First milestone command:

```bash
python scripts/eval_knn_policy.py \
    --model path/to/model.zip \
    --memory results/memory/knn_memory.pkl \
    --eval_set data/eval_easy.json \
    --alpha 0.3 \
    --k 16
```

The first question to answer:

> Does neural policy + kNN memory outperform the same neural policy without memory?

If yes, proceed to memory-guided beam.

If no, inspect whether the failure is due to bad embeddings, illegal retrieved actions, or poor canonicalization.

---

# Minimal implementation notes

## Keep memory read-only during evaluation

For the first experiments, build memory from training traces, then freeze it during evaluation.

Do not update memory online during the eval set, otherwise train/test leakage becomes ambiguous.

---

## Verify action legality

Every memory-suggested action must pass the existing legal action mask.

If the remembered action is illegal, discard it and log it.

---

## Avoid train/test leakage

Do not store traces from the evaluation set in memory.

Use separate trace sources:

```text
memory_build_set
validation_set
test_set
```

---

## Keep outputs reproducible

Every script should accept:

```bash
--seed
--model
--memory
--eval_set
--output_dir
```

Save config and metrics with every run.

---

# Recommended final experiment table

Produce this as the main memory result:

| Method | Solve rate | Avg steps | Avg expansions | Wall time | Hard solve rate |
|---|---:|---:|---:|---:|---:|
| Greedy policy |  |  |  |  |  |
| Policy + exact cache |  |  |  |  |  |
| Policy + kNN memory |  |  |  |  |  |
| Beam |  |  |  |  |  |
| Value beam |  |  |  |  |  |
| Memory beam |  |  |  |  |  |

The key win condition:

\[
\text{Memory beam} > \text{Value beam}
\]

at fixed expansion budget.

