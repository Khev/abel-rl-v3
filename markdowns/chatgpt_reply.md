# RL Innovations to Try for Symbolic Mathematics

## Summary

I would **not** just swap PPO for a fancier RL algorithm. This problem is really a **deterministic symbolic search problem with an exact verifier**, dynamic legal actions, and reusable solution traces. That points toward:

> **neural-guided search + expert iteration + better action factorization**

more than simply using a bigger PPO variant.

The current draft already has strong ingredients: expression-tree states, dynamic `(operation, term)` actions capped at 50 with illegal-action masks, dense complexity rewards, TreeMLP, curriculum, relabel-constants, and success replay. It also says curiosity **hurts** with TreeMLP, so curiosity should probably not remain the headline unless newer experiments overturn that result.

---

## Top Recommendations

| Priority | Upgrade | Why it fits the paper | Effort | Likely payoff |
|---:|---|---|---:|---:|
| 1 | **AlphaZero-style search / expert iteration** | Equation solving is a tree search with exact terminal checks. Use TreeMLP as policy + value, then guide beam/MCTS. | Medium | Very high |
| 2 | **Factorized action policy: op head + term pointer head** | The current flattened `(op, term)` action cap is unnatural. Let the policy choose an operation and point to an expression-tree node/subexpression. | Medium | Very high |
| 3 | **Offline pretraining from generated solution traces** | Generate traces by SymPy/BFS/beam for easier equations, then fine-tune with RL. | Low-medium | High |
| 4 | **Value-guided beam/A\*** | The paper already evaluates beam search; make it learned-value-guided instead of raw-policy-guided. | Low | High |
| 5 | **GFlowNet / diverse trace generation** | Symbolic math has many valid solution paths. GFlowNets are built for sampling diverse high-reward compositional objects. | Medium-high | Medium-high |
| 6 | **Grammar-generated terms for open equations** | This addresses the stated limitation: closed equations only use existing subterms, but completing the square needs new terms. | High | Paper-defining |
| 7 | **Structured symbolic novelty instead of generic curiosity** | Curiosity may chase syntactic weirdness. Better: novelty over canonical equation states, operators, solved families, or tree-edit neighborhoods. | Low | Medium |

---

## 1. Biggest Win: AlphaZero-Style Symbolic Search

This is the first thing I would try.

The environment is perfect for it:

- state = current equation;
- actions = legal algebraic transformations;
- transition = exact SymPy rewrite;
- terminal = verified solved equation;
- reward = solved / shorter / simpler;
- search tree = possible derivations.

So instead of PPO alone, train a network with:

```latex
\pi_\theta(a \mid s), \qquad V_\theta(s)
```

where `π` gives action priors and `V` estimates probability of solving, negative distance-to-solution, or expected remaining steps.

Then at evaluation/training time, use **PUCT/MCTS** or **value-guided beam search**:

```text
score(s, a) = Q(s, a) + c * P(a | s) * sqrt(N(s)) / (1 + N(s, a))
```

Each successful search gives improved targets:

- train policy toward the search-improved action distribution;
- train value toward solved/not-solved or `-solution_length`;
- store successful traces in the replay buffer.

This is basically **expert iteration**.

For the paper, the clean story would be:

> PPO learns a local symbolic policy; search converts it into a stronger solver; successful searches become new training data.

That is much beefier than “PPO + curiosity.”

---

## 2. Replace Flat Actions with a Pointer / Factorized Policy

Right now the action is essentially flattened:

```latex
a = (\text{operation}, \text{term})
```

and the action space is capped at `|A| = 50`. That is fine for a first paper, but it is not the most natural architecture.

Use:

```text
π(op | equation)
π(term_node | equation, op)
```

So the policy first chooses:

```text
add / subtract / divide / log / exp / expand / collect / ...
```

then points to a node/subtree in the expression tree.

Advantages:

- no arbitrary action cap;
- better generalization to larger equations;
- term choice becomes a **pointer problem**, not a classification over fixed IDs;
- TreeMLP node embeddings can be used directly;
- masks become cleaner: invalid ops/terms are removed at the op-term level.

This would make the model feel much more modern and much more aligned with the structure of the task.

---

## 3. Add Offline Pretraining from Exact Traces

The success replay buffer is already a good idea. Make it more aggressive.

Generate traces using:

- BFS for small equations;
- beam search;
- SymPy-style scripted solvers;
- current trained agent successes;
- hand-coded traces for each equation family.

Then pretrain:

```text
policy loss: imitate action in successful trace
value loss: predict remaining steps to solution
```

Then fine-tune online with PPO/search.

This is not cheating. It is exactly what people do in theorem proving and combinatorial reasoning: use cheap generated data to bootstrap the policy, then use RL/search to exceed the bootstrapper.

A clean ablation:

| Agent | Pretrain? | Search? | Replay? |
|---|---:|---:|---:|
| PPO-TreeMLP | no | no | no |
| PPO-TreeMLP + replay | no | no | yes |
| BC → PPO | yes | no | yes |
| BC → PPO + beam | yes | yes | yes |
| Expert iteration | yes | yes | yes |

This would be a stronger results section than only comparing PPO variants.

---

## 4. Use Learned Value-Guided Beam Search

The draft already uses greedy and beam accuracy, and the large dataset seems to benefit from beam search relative to greedy.

Improve beam search with a learned value:

```text
beam_score(path) =
    log π(actions)
    + λ V(current_state)
    - α path_length
    - β complexity(current_state)
```

or use an A*-like score:

```text
f(s) = g(s) + hθ(s)
```

where:

- `g(s)` = steps taken so far;
- `hθ(s)` = learned remaining steps to solve.

This directly attacks non-optimal traces. The current agent sometimes does dumb extra operations, e.g. multiplying by `-1` unnecessarily. A learned value/cost-to-go can penalize that.

---

## 5. GFlowNets Are a Good Researchy Upgrade

Symbolic math has many solution paths. PPO tends to collapse onto one path. But diverse traces are useful:

```text
solve ax+b=0 by subtract b, divide a
solve ax+b=0 by subtract ax, multiply -1, divide a
solve rational equation by clear denominator first
solve rational equation by rearrange first
...
```

GFlowNets are designed to sample diverse high-reward compositional objects.

For this problem, a GFlowNet could sample **solution traces** with probability proportional to reward:

```latex
P(\tau) \propto R(\tau)
```

where `R` rewards:

- verified solution;
- short trace;
- low intermediate complexity;
- no illegal/degenerate transformations;
- generalizable canonical form.

This is probably not the first thing to implement, but it could become a distinctive section:

> Beyond optimal traces, we train a flow model to sample diverse valid algebraic derivations.

That sounds much cooler than curiosity.

---

## 6. For Open Equations, Add Generative Term Synthesis

This is the real frontier.

The current MDP handles **closed equations**, where required terms are already present in the expression/subexpression list. But **open equations** require inventing new terms. Completing the square is the classic example: the agent needs to generate something like:

```latex
\left(\frac{b}{2a}\right)^2
```

So the beefy innovation is not PPO. It is adding an action like:

```text
generate_term(grammar, context)
apply_operation(op, generated_term)
```

Example grammar:

```text
term :=
    constant
  | variable
  | subtree
  | term + term
  | term * term
  | term / term
  | term ^ 2
  | sqrt(term)
  | log(term)
  | exp(term)
```

Then the agent can invent:

```latex
\frac{b}{2a}, \qquad \left(\frac{b}{2a}\right)^2, \qquad e^{kx}, \qquad \log u
```

This would help with:

- completing the square;
- substitutions;
- trig identities;
- exponential/log transforms;
- rationalizing radicals;
- `u`-substitution for calculus later.

That moves the paper from:

> RL solves closed symbolic equations

into:

> RL discovers useful symbolic substitutions.

That is much more important.

---

## 7. Replace Generic Curiosity with Symbolic Novelty

The draft says curiosity worsened learning with TreeMLP. That makes sense. In symbolic domains, generic curiosity often rewards garbage:

```text
weird expression ≠ useful expression
novel syntax ≠ mathematical progress
```

Better exploration bonuses:

| Exploration bonus | Definition |
|---|---|
| **Canonical-state count** | Bonus for visiting new `simplify/canonicalize(equation)` states |
| **Family novelty** | Bonus for solving underrepresented equation templates |
| **Operator novelty** | Bonus for useful rare operations like `log`, `asin`, `collect` |
| **Tree-distance progress** | Reward for reducing distance to solved/canonical forms |
| **Unsolved-frontier bonus** | Prioritize states/equations where success probability is improving |

The curriculum should be based on **learning progress**, not just inverse solved count:

```text
sample equation i ∝ recent_improvement_i + uncertainty_i
```

That keeps the agent on equations that are neither trivial nor impossible.

---

## What I Would Not Do First

I would **not** jump to TD-MPC/MuZero/world models. The transition model is already known: SymPy. Learning a model wastes capacity.

I would **not** focus on bigger GNNs yet. The ablation says TreeMLP beats GNN and GraphSAGE on the small dataset.

I would **not** make diffusion policies a priority. This is a discrete symbolic search problem, not a continuous-action robotics problem.

---

## Blunt Recommendation

The next version should be:

> **TreeMLP/TreeTransformer policy + value head + factorized op/term pointer actions + value-guided beam/MCTS + expert iteration replay.**

Possible names:

- **Symbolic Expert Iteration for Algebraic Equation Solving**
- **Neural-Guided Search for Symbolic Equation Solving**
- **Expert Iteration for Reinforcement Learning in Symbolic Mathematics**

PPO can remain underneath, but the intellectual center should move from:

> PPO plus curiosity

into:

> learned symbolic search.

