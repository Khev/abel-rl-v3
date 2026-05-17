# Open Equations / Change-of-Variables Project

## Framing

**One-sentence pitch:** RL learns to propose change-of-variable substitutions that solve "open" equations — problems where the solution requires terms outside the equation's own expression tree, a form of generative algebraic reasoning that current LLMs and prior symbolic-RL methods (ConPoLe, Lample-Charton seq2seq) fail on.

The closed-equation work from the workshop paper becomes the warm-up; the contribution is CoV.

## Target

- **Venue:** ICLR 2027
- **Deadline:** ~Sept 25, 2026
- **Budget:** ~18 weeks from 2026-05-17

## Phase 1 — Scope & story (weeks 1–2)

- Lock the equation taxonomy. Decide on 4–6 CoV classes to claim. Candidates: quadratic completion, cubic depression, quartic depression, exponential→rational, reciprocal symmetric polynomials (`x → 1/x`), Weierstrass for trig. Last two are stretch.
- Write the MDP formally (currently empty in `paper.tex` §4). States, actions (op-sequence to build `f(x)`), transitions, terminal condition, reward. Do this before experiments so the design isn't post-hoc.
- Decide on the "what LLMs can't do" angle. Quick eyeball test: give Claude-4.6 / GPT-5 ~20 CoV problems from each class, see where they break. This is also the motivation figure.
- **Deliverable:** 2-page design doc + updated paper §4 MDP, no experiments yet.

## Phase 2 — Experimental scaffolding (weeks 3–4)

- Clean up the CoV codebase. `env_cov.py` and `train_cov.py` are usable but have rough edges. Standardize eval harness so all baselines run through one entry point.
- Equation generation pipeline. Generalize `make_eqns.py` to produce held-out test sets per CoV class. Target ~1k train / 200 test per class, strict no-leakage checks.
- Baseline implementations:
  - **Analytic dispatcher:** wrap existing `pi_cov_general` as a baseline (the "expert system" comparison).
  - **LLM zero-shot:** Claude API harness, prompt asks for substitution, parsed back into env.
  - **LLM with tool-use:** same model, given a SymPy REPL — fairer baseline.
  - **No-CoV PPO:** vanilla closed-equation agent on open problems (lower bound).
  - **Seq2seq (Lample-Charton style):** smallest viable implementation, 6-layer transformer trained on (equation, substitution) pairs.
- Multi-seed infrastructure. Default to 5 seeds everywhere. Bootstrap CIs in plot helpers.
- **Deliverable:** all baselines runnable, smoke-tested on 1 equation class.

## Phase 3 — Core experiments (weeks 5–10)

- Main results table. Each method × each equation class, greedy + beam + success@10, 5 seeds, error bars.
- **OOD generalization** (headline experiment): train on classes {A, B}, test on held-out class C. Show LLM + seq2seq degrade more than the RL agent. If they don't, the story is in trouble — pivot to sample efficiency or interpretability.
- Ablations on the RL agent: TreeMLP vs. MLP, with/without success replay, with/without curriculum, with/without action masking. Single benchmark, single table.
- Sample efficiency curves. Episodes-to-first-solve, plotted against seq2seq training-step-equivalents.
- Failure analysis. Cluster failure modes per class. Also informs the ODE follow-up.
- **Deliverable:** every figure and table in the final paper has data behind it.

## Phase 4 — Writing & polish (weeks 11–15)

- Rewrite abstract & intro around the CoV/generative-reasoning framing.
- Demote closed-equations section to ~1.5 pages — it's now warm-up.
- Expand §4 to 3–4 pages with MDP, results, ablations, failure analysis.
- Fill every `\cite{}` and `\ref{}` placeholder. Add Lample-Charton, Poesia, recent LLM-math work (MATH, miniF2F, GSM8K-symbolic), DreamCoder, PySR.
- Related work section (currently absent).
- 2 rounds of self-review + 1 external read.
- **Deliverable:** submission-ready PDF by week 15, with 3 weeks buffer.

## Decision points / risks

- **End of Phase 1:** if the LLM eyeball test shows GPT-5 solves >80% of CoV classes zero-shot, stop and rescope. Need to push to harder substitution patterns (Tschirnhaus, integrating-factor-like for ODEs) — at that point may as well jump straight to ODEs.
- **End of Phase 3:** if OOD generalization isn't there, not main-conference material. Fall back to a strong workshop submission and pivot to ODEs.
- **Scope creep risk:** temptation to add a 7th equation class or another baseline. Resist after week 6.

## Strategic context

This paper is foundation work for a longer research program on RL for exact equation solving (next target: ODEs). The CoV framing is the bridge — substitution is core to ODE techniques (separable, integrating factor, characteristic equations). Sharpening it now writes the related-work section of the ODE paper a year early.

## Future Work (deferred — likely next paper, NOT this one)

Captured from a ChatGPT review (`chatgpt_reply.md`, `chatgpt_tasks.md`) plus follow-up discussion. Held back from the current paper to keep the "fully-unsupervised RL discovers CoV" framing clean.

### Strong fits for the follow-up paper

- **AlphaZero / expert iteration.** The domain is a deterministic search problem with an exact verifier — textbook MCTS+PUCT territory. Train policy + value heads, search with PUCT, fold successful searches back as training data. Natural framing: "PPO learns a local symbolic policy; search converts it into a stronger solver; successful searches become new training data." Major engineering, paper-defining payoff. Especially apt for ODE solving where episode lengths grow.
- **Factorized action policy: op head + term-pointer head.** Replace the `|A|=50` flat action cap with `π(op | state)` then `π(term_node | state, op)` as a pointer over expression-tree nodes. More natural, removes the cap, generalizes to bigger equations, makes masking cleaner. Touches env + masks + policy + replay format; non-trivial refactor.
- **Value-guided beam search.** Score partial paths by `log π_sum + λ·V(state) − α·length − β·complexity`. Cheap upgrade if the value head exists. **Before doing this:** investigate why the current beam returns 0 even when greedy returns 1 — likely a plain bug, not a methodological gap.
- **Symbolic novelty bonuses (not generic curiosity).** Canonical-state count, family novelty, operator novelty, tree-distance progress. Curriculum based on learning-progress, not just inverse-solve-count. Cheap to add, fits any paper.

### Out of scope for the unsupervised story

- **Offline BC pretraining from BFS/scripted-solver traces.** ChatGPT proposed this as a strong upgrade; it would distill an oracle into the policy. **Directly conflicts with our current "unsupervised, no oracle distillation" framing** — same epistemological category as the LLM/seq2seq baselines we dropped. Only useful in a paper that explicitly studies search+distillation+RL together.
- **GFlowNets for diverse trace generation.** Interesting in principle (symbolic math has many valid solution paths). Separate research project. Skip until after ICLR submission.

### Reframing: things ChatGPT recommended that we're *already doing*

- "Grammar-generated terms for open equations" — this is exactly what CoV-as-MDP does. The op-sequence the CoV agent builds *is* a generative grammar over substitutions. ChatGPT framed it as a future innovation; we've already committed to this direction.

### Bigger items deliberately not pursued

- Bigger GNNs / GraphSAGE / Transformer policies (TreeMLP is competitive and lightweight).
- TD-MPC / MuZero / world models (the transition is known — SymPy).
- Diffusion policies (problem is discrete, not continuous).
