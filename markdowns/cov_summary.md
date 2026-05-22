# π_cov — learned change-of-variables policy: status & open problem

*Self-contained summary for external feedback. Goal: raise the policy's
**greedy (argmax) accuracy** as high as possible.*

## 1. The problem

We are building an RL agent that solves symbolic algebra equations. "Closed"
equations are solved by rearranging existing sub-terms. **"Open" equations**
need a *generative* **change of variables (CoV)** — e.g. completing the square.
`π_cov` is the policy that, given an equation, must **produce the correct
substitution** `x ↦ f(x)`.

We currently have an analytic oracle (`pi_cov_general`) that pattern-matches
the equation and returns the exact substitution. We want to **replace the
oracle with a learned `π_cov`** so the system is honestly end-to-end (the
oracle is only a training teacher; at inference `π_cov` stands alone).

## 2. The environment (`covEnv`)

`π_cov` builds the substitution `f(x)` step by step:

- **Actions** = `(op, term)` pairs. `op ∈ {ADD, SUB, MUL, DIV}` (binary),
  `term ∈ term_bank = {a,b,c,d,e,f,g, 2,3,4}`. Plus two unary actions:
  `STOP` and `LOGX`. Total action space = 4×10 + 2 = **42**.
- **Building `f(x)`:** the first action sets `base_op` and `cov = term₁`;
  each subsequent action folds `cov ← op(cov, termᵢ)`. `STOP` (or hitting
  `max_depth = 3`) finalizes: **`f(x) = base_op(x, cov)`**. `LOGX` is a
  special unary base op giving `f(x) = log(x)`.
- **Reward:** sparse, at episode end — `Δcomplexity = C(eqn) − C(eqn|x→f(x))`;
  "success" iff `Δ > 0`.
- **State:** integer-encoded equation (flat 1-D vector) + current depth +
  last-10 action history.
- Episodes are short: **2–4 actions**.

The substitution families and their correct traces:

| family | equation (symbolic) | substitution | covEnv trace | # actions |
|---|---|---|---|---|
| quadratic, general | `A x² + B x + C` | `x − B/(2A)` | `SUB:B DIV:2 DIV:A STOP` | 3 |
| quadratic, monic | `x² + 2b x + c` | `x − b` | `SUB:b STOP` | 1 |
| cubic, general | `A x³ + B x² + (B²/3A) x + D` | `x − B/(3A)` | `SUB:B DIV:3 DIV:A STOP` | 3 |
| cubic, monic | `x³ + 3b x² + 3b² x + d` | `x − b` | `SUB:b STOP` | 1 |
| quartic (analogous) | … | `x − B/(4A)` / `x − b` | … | 3 / 1 |
| exponential | `A e^x + B e^{-x} + C` | `x → log(x)` | `LOGX STOP` | 1 |

So **monic + exp need 1 decisive action; general families need an exact
3-action sequence.**

## 3. Datasets

- Equations are **symbol permutations** of the templates above — coefficients
  are bare symbols from a 7-symbol pool `a–g` (no numeric coefficients), so
  the policy must learn the *structural* rule, not memorize symbols.
- Three nested sizes; we use **`cov_large`: 774 train / 192 test** equations.
- Test set is **held out** — disjoint symbol permutations, never seen in
  training.
- **We know the exact correct substitution for every equation** (generated
  from the depression formulas), so we have full supervision available.

## 4. Method — BC-from-demonstrations

Plain RL fails: PPO almost never *discovers* the exact substitution sequence
by exploration (sparse reward, precise 3-action targets), so there is nothing
for success-replay to reinforce. Mixing behavior-cloning into PPO also fails
(BC vs PPO tug-of-war churns the policy).

**What works:** a dedicated **supervised BC pretraining phase**:
1. Generate the expert `(obs, action)` demonstrations for all 774 train
   equations (run covEnv through the known-correct traces).
2. Pretrain the MLP policy with cross-entropy `−log p(expert_action | obs)`
   for ~300k gradient steps, to convergence (`bc_loss → ~0.0000`).
3. PPO *after* pretraining only **erodes** the policy, so `π_cov` is taken to
   be the BC-pretrained model (pure imitation).

Policy: a small MLP (hidden 128), PPO/SB3 infrastructure.

## 5. Results (cov_large, held-out test, 6 seeds)

| metric | value |
|---|---|
| `bc_loss` (train) | ~0.0000 (perfect imitation of train demos) |
| **`test_beam`** (beam search, width 5, depth 3) | **0.76 mean, 0.87 best** |
| **`test_greedy`** (deterministic argmax rollout) | **0.13 – 0.28** |

Progress this push: `test_beam` went 0.40 → 0.76 (adding demos, then the
`log` action for the exponential family). `bc_loss → 0` means the policy
reproduces every *training* demonstration essentially perfectly.

## 6. The open problem — the greedy/beam gap

**The policy ranks the correct substitution in the beam-top-5 ~80% of the
time, but makes it the argmax only ~20%.** We want `test_greedy` (the
deterministic, argmax accuracy) as high as possible — ideally near
`test_beam` — because a deployed `π_cov` should produce *one* correct
substitution without an internal search.

Key structural fact: a greedy rollout succeeds only if **every** action in
the trace is the argmax, so

```
test_greedy  ≈  (per-step argmax accuracy) ^ (trace length)
```

The 1-action families (monic, exp) are forgiving; the **3-action general
families compound errors** — per-step accuracy 0.85 → 0.85³ ≈ 0.61.

Diagnosis: this is a **generalization-sharpness / imitation-learning**
problem, not an optimization one (`bc_loss` is already ~0 on train, and it
is present *before* any PPO). Two suspected contributors:
- **Soft generalization:** on novel symbol permutations the policy spreads
  probability — the right action is high-ranked but a distractor occasionally
  edges it out of rank 1.
- **Train/rollout distribution mismatch:** BC trains only on the *expert's*
  states; a greedy rollout that makes one small early error then visits
  states the policy was never trained on, and derails.

## 7. What has been tried / known

- More BC iterations help then plateau (20k→0.53, 150k→0.64, 300k→0.77
  `test_beam`); past ~300k gives little.
- PPO finetuning after BC pretraining erodes the policy (does not refine it).
- The 7-symbol pool caps the dataset at 774 demos.
- `test_greedy` is low across all seeds and is measured *pre-PPO*.

## 8. What is flexible

Essentially everything can change: the `covEnv` action space / reward /
observation; the dataset (more symbols, more equations, curriculum); the
policy architecture; the training algorithm (imitation, RL, hybrid). The only
fixed point is the end goal: a standalone learned `π_cov` that, given an
equation, **deterministically outputs the correct CoV substitution**.

## 9. The question

**How do we close the greedy/beam gap — i.e. raise the deterministic
(argmax) accuracy of `π_cov` toward (or past) its beam-search accuracy?**

Our current candidate ideas (feedback welcome — better ones especially):
1. **DAgger** — roll out the policy, have the oracle relabel the states it
   actually visits, aggregate, retrain. Directly attacks the train/rollout
   mismatch.
2. **More symbols → more demonstrations** — enlarge the symbol pool from 7 to
   ~12 (thousands of demos) for sharper generalization.
3. **Structured / graph observation** — replace the flat integer encoding
   with a graph encoding of the equation tree, so "which symbol is the
   leading coefficient" is explicit rather than inferred.
4. **Larger / better policy network.**
5. Anything that raises **per-step argmax accuracy**, since it compounds over
   the 3-action general substitutions.
