# Open Equations / CoV — Task List

**Direction:** push the fully-unsupervised TreeMLP-PPO approach as far as it goes on breadth of CoV classes. No LLM or seq2seq baselines. The story is "how far can unsupervised RL reach on generative algebraic substitutions."

Tasks are grouped by phase. Use `[ ]` / `[x]` to track. Each task has a rough size: S (≤1 day), M (2–4 days), L (1+ week).

## Phase 0 — Closed-equations paper cleanup (parallel, opportunistic)

The §3 results in `papers/paper.tex` are partly drafted on the *buggy* encoder (numeric atoms were silently dropped). With the fix shipped, all numbers need re-running, and several figures / table cells / placeholders are missing. This phase is parallel to the CoV work — runs can be queued in the background.

### Re-runs (all on the FIXED encoder)
- [ ] (L) Fill Table 1: 10 missing cells. Per-algorithm runs on `small` (1k/100 train/test) and `large` (10k/1k) closed-eq datasets, plus the existing `poesia` benchmark. Algorithms: ppo, ppo-tree, ppo-tree-rc, ppo-tree-rc-buf.
- [ ] (S) Fill missing `ppo` × `poesia` cell (shows `..` in the draft).
- [ ] (M) Beam search width: settle on a value and report consistently (draft has "beam search of XXX"). Audit `train_abel.py`'s beam implementation for bugs analogous to the ones we found in `train_cov.py`.

### Figures
- [ ] (L) `learning_curves_closed`: per-dataset × per-agent learning curves with min/max envelope across 3 seeds. Generate via `plot_seeds.py` on the re-run results.
- [ ] (M) `gnn-comparison`: TreeMLP vs GCN vs GraphSAGE on the small dataset. Need to run `ppo-gnn` and `ppo-sage` agents.
- [ ] (M) `curiosity-tree-hurts`: TreeMLP with/without curiosity (ICM/RND/etc.) on small dataset, to back the claim that "curiosity worsened learning with TreeMLP." **Currently UNSUPPORTED by any experiment in `data/`** — soften the claim until data lands. First run launched: `ppo-tree-RND × abel_level3 × 500k`.
- [x] (S) Representation-learning figure — embedding visualization done in two forms: `figures/tsne_l3_rcbuf.png` (single-method) and `figures/embed_l3_panel.png` (2×2: PCA + t-SNE × 2D + 3D, via `plot_embeddings.py`).
- [ ] (S) Generate the 4-panel embedding figure on abel_level4 and poesia too, for the appendix.

### Content / analysis
- [ ] (M) Failure analysis on closed-eq: cluster failure modes from a trained checkpoint, give representative traces. (TODO bullet line 268.)
- [ ] (S) "Comment: this is done by X" placeholder (line 269) — figure out what was intended.
- [ ] (S) Cite the prior curiosity-helps-with-MLP work (`\cite{}` line 208).

### Polish (paper.tex, all small)
- [ ] (S) Fix broken LaTeX in TreeMLP description (line 197): `\mathrm{MLP}!\left([\mathrm{emb}i ,|, \sum{j\to i} h^{(t)}_j]\right)` — should be `\mathrm{MLP}\!\left([\mathrm{emb}_i, \,\big\|\, \sum_{j\to i} h^{(t)}_j]\right)` or similar.
- [ ] (S) Pick a single name and stick to it: `ppo-tree-rc-buf` vs `ppo-tree-rc-buff`.
- [ ] (S) Fix typo: `ppp-tree` (line 236) → `ppo-tree`.
- [ ] (S) Verify the inline claims about Table 1 numbers ("ppo-tree-rc-buf beats ConPole") once the cells are filled.

### Methodological notes (don't forget)
- Existing pre-fix results in `data/dynamic_actions/abel_level*/` are obsolete for the headline numbers. Keep them around as a "buggy encoder" historical reference but report only the post-fix runs.
- The `papers/paper.tex` is the canonical version; `gemini/paper.tex` is a near-duplicate (3 lines differ).

## Phase 1 — Scope & data (weeks 1–2)

### Equation taxonomy
- [ ] (S) Lock the class list. Current candidates: quadratic, cubic, quartic, exponential, reciprocal-symmetric, radical. Stretch: trig (Weierstrass), logarithmic.
- [ ] (S) For each class, write the intended CoV `f(x)` in `markdowns/classes.md`.
- [ ] (S) Confirm `pi_cov_general` handles each class analytically; extend it for any class it doesn't already cover (this is also the analytic baseline).

### MDP
- [ ] (S) Write formal MDP for CoV agent in `markdowns/mdp.md`: states, action space, transitions, reward, terminal condition.
- [ ] (S) Decide CoV episode budget (max steps to build `f(x)`).
- [ ] (S) Decide reward shaping: pure outcome vs. complexity-delta during construction.
- [ ] (M) Update `paper.tex` §4 with the formal MDP.

### Dataset generator (`make_eqns.py` v2)
- [ ] (M) Design per-class generator interface (one class per file under `eqn_gen/`).
- [ ] (M) Implement `Quadratic`, `Cubic`, `Quartic`, `Exponential` generators (these line up with existing `pi_cov_general` patterns).
- [ ] (M) Implement `ReciprocalSymmetric` and `Radical` generators.
- [ ] (M) Canonical-form / dedup helper (relabel free symbols in canonical order; use to enforce no train/test leakage and no near-duplicates within a split).
- [ ] (S) Per-equation validation: (i) solvable by SymPy, (ii) solvable via intended CoV within step budget, (iii) free-symbol set is non-degenerate.
- [ ] (S) Emit datasets to `equation_templates/cov_v2/{class}/{train,test}.txt`, target 1k train / 200 test per class.
- [ ] (S) **Decision point:** if any class yields <200 valid distinct instances or all instances are trivially-solvable by the no-CoV agent, drop or redesign it.

### Stretch classes (after the 4 core classes ship)
- [ ] (M) `Trig` generator + Weierstrass CoV check.
- [ ] (M) `Logarithmic` generator + log substitution check.

## Phase 2 — Experimental scaffolding (weeks 3–4)

### Codebase cleanup
- [ ] (M) Audit `envs/env_cov.py`; document state/action semantics inline.
- [ ] (M) Refactor `train_cov.py` to share eval harness with `train_abel.py` (currently duplicated).
- [ ] (S) Move CoV eval functions out of `train_abel.py` into `utils/eval_cov.py`.
- [ ] (S) Delete or archive `backup_*.py` files.

### Baselines (no LLM, no seq2seq)
- [ ] (S) **Analytic dispatcher:** wrap `pi_cov_general` as an evaluator hooked into the same eval harness. This is the ceiling.
- [ ] (S) **No-CoV PPO:** run a vanilla closed-equation `train_abel.py` agent on each CoV test set, expect near-zero. This is the floor.
- [ ] (S) **Random substitution:** sample an action sequence of length k uniformly at random, evaluate. Floor with non-trivial sample budget.
- [ ] (S) **Greedy heuristic:** pick op-sequence that maximizes one-step complexity reduction. Weak baseline above random.

### Multi-seed infra
- [ ] (S) Standardize seed set: `[7000, 7001, 7002, 7003, 7004]`. Document in `markdowns/protocol.md`.
- [ ] (S) Add bootstrap CI helper to `utils/utils_general.py`.
- [ ] (S) Smoke-test PPO-tree + all 4 baselines end-to-end on quadratic class.

## Phase 3 — Core experiments (weeks 5–10)

### Main results: per-class accuracy
- [ ] (L) Train PPO-tree CoV agent on each core class (4), 5 seeds each.
- [ ] (L) Train one **multi-class agent** on the union of all classes, 5 seeds.
- [ ] (M) Run analytic, no-CoV PPO, random, and greedy baselines on every class test set.
- [ ] (M) Main table: method × class, greedy / beam / success@10, 5 seeds with CIs.
- [ ] (S) Statistical test: PPO-tree vs. analytic (paired bootstrap or Wilcoxon).
- [ ] (S) **Decision point:** if PPO-tree matches analytic on ≥3 classes within 5pp, story is strong. If <2 classes match, scope down or rethink.

### OOD generalization (headline experiment)
- [ ] (S) Define OOD splits (cross-class held-outs):
  - train {quad, cubic, quartic}, test exponential
  - train {quad, cubic, quartic, exponential}, test reciprocal
  - train {polynomial classes}, test radical
- [ ] (M) Train multi-class PPO-tree on each train split, eval on held-out class.
- [ ] (S) Compare against analytic dispatcher (which has no notion of "training" — it either handles the class or not).
- [ ] (S) **Decision point:** if multi-class agent never transfers to held-out class, weaken claim to "in-distribution unsupervised breadth" and reposition.
- [ ] (S) OOD figure + table.

### Ablations
- [ ] (M) TreeMLP vs. plain MLP on CoV (5 seeds, one representative class).
- [ ] (M) With/without success replay.
- [ ] (M) With/without curriculum.
- [ ] (M) With/without action masking.
- [ ] (M) Reward shaping: complexity-delta vs. sparse outcome-only.
- [ ] (S) Consolidated ablation table.

### Sample efficiency
- [ ] (S) Log episodes-to-first-solve per equation for PPO-tree.
- [ ] (S) Compare to random-baseline solve rate at matched budget.
- [ ] (S) Plot curves.

### Failure analysis
- [ ] (M) Collect ~100 failure cases per class.
- [ ] (M) Manually cluster failure modes; tag each (e.g. "wrong substitution form," "right substitution but step-budget exceeded," "introduces extraneous root," "produces unsolvable post-CoV equation").
- [ ] (S) Write failure-mode subsection with representative examples.

### Solution-trace analysis (interpretability angle)
- [ ] (M) Extract learned action sequences per class; check whether the agent rediscovers the analytic CoV exactly, or finds equivalent alternatives.
- [ ] (S) Pick ~3 striking traces per class for the paper.

## Phase 4 — Writing & polish (weeks 11–15)

### Paper rewrite
- [ ] (M) Rewrite abstract: framing is fully-unsupervised RL discovering CoV substitutions for generative algebraic reasoning.
- [ ] (M) Rewrite intro; demote ConPoLe comparison from headline (no LLM comparison either — frame as "we study what unsupervised RL alone can reach").
- [ ] (M) Compress closed-equations section to ~1.5 pages.
- [ ] (L) Expand §4 (Open Equations) to 3–4 pages: MDP, per-class results, OOD experiment, ablations, failure analysis, representative traces.
- [ ] (M) Related Work. Must cite: Lample & Charton 2019 (deep symbolic math, neural-translation framing), Poesia 2021 (ConPoLe), DreamCoder, PySR, Dabelow & Ueda 2024. We can position cleanly against neural-translation as "supervised; we are unsupervised."

### Cleanup
- [ ] (S) Fill every `\cite{}` placeholder in `paper.tex`.
- [ ] (S) Fill every `\ref{}` placeholder.
- [ ] (S) Fill empty cells in Table 1.
- [ ] (S) Delete `\section{Notes}` TODO list.
- [ ] (S) Add Acknowledgements + Impact Statement.

### Figures
- [ ] (M) Per-class learning curves: PPO-tree vs. analytic vs. random, 5 seeds with CI shading.
- [ ] (S) OOD figure.
- [ ] (S) Failure-mode bar chart.
- [ ] (S) Tree / MDP schematic.

### Review
- [ ] (M) Self-review pass 1 (substance: claims supported, baselines fair).
- [ ] (M) Self-review pass 2 (writing).
- [ ] (M) External read by 1 colleague.
- [ ] (S) Final ICLR formatting check.
- [ ] (S) Submit by 2026-09-25.

## Decision points / risks

- **End of Phase 1 (dataset):** if any class can't yield ≥200 distinct, non-trivial, CoV-solvable instances, drop or redesign that class. If <3 classes survive, the breadth story is dead.
- **End of Phase 3 (in-distribution):** if PPO-tree matches the analytic dispatcher on <2 classes within 5pp, the unsupervised story is too weak; scope down to a workshop submission.
- **End of Phase 3 (OOD):** if the multi-class agent doesn't transfer to any held-out class, drop OOD as the headline and reposition around in-distribution breadth + sample efficiency.
- **Scope creep risk:** temptation to add a 7th class or a sub-ablation. Resist after week 7.

## Always-on / continuous

- [ ] Update `markdowns/project.md` if scope changes.
- [ ] Keep `paper.tex` building cleanly throughout.
- [ ] Commit experiment results to git as they land.
- [ ] Maintain `markdowns/results.md` log of runs (seed set + date + accuracy).
