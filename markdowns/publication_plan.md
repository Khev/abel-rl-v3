# Publication plan — RL for symbolic equation solving

*Updated 2026-05-21.*

## 1. The contribution (framing)

Prior work on **learn-to-solve-without-solution-traces** symbolic equation
solving stops at **linear** equations:
- Dabelow & Ueda (2024), *Symbolic equation solving via RL* — explicitly
  restricts the agent to linear equations `a0 + a1 x = a2 + a3 x`.
- Poesia et al. (2021), *ConPoLe* — RL/contrastive on Common Core
  (elementary-school) symbolic reasoning.

This work advances that frontier: **nonlinear** equations (radicals, exp,
trig) and, crucially, **open** equations requiring a *generative*
change-of-variables — with a fully learned `pi_cov` (no analytic oracle at
inference) and **no supervised solution traces**. That is the spine of the
paper; the intro/related-work must make it unmissable.

The "why not an LLM?" question is answered by framing: the result is not that
quadratics *can* be solved, but that an agent can *learn* the exact symbolic
transformations step-by-step, with no traces and no hallucination.

## 2. Target venues & timeline

| venue | deadline | status |
|---|---|---|
| NeurIPS 2026 | May 6, 2026 | **missed** |
| **AAAI 2027** | **abstract Jul 21 / paper Jul 28, 2026** | **primary target** (Feb 2027, Montreal) |
| ICLR 2027 | ~late Sep / early Oct 2026 (expected) | fallback / upgrade target |
| ICML 2027 | ~late Jan 2027 (typical) | far option |

**Plan: aim for AAAI 2027 (Jul 28).** The paper is contribution-complete now
— only rigor + polish are needed, which fits comfortably in ~9 weeks.
Submitting in July yields reviews by autumn and a clean resubmit path to
ICLR 2027 / ICML 2027 if needed. Choose ICLR 2027 instead only if expanding
scope (the generalization story below) into the strongest possible version.

## 3. Pre-submission checklist (rigor — required)

- [ ] **Fig 4 multi-seed** — 5-seed run in progress; mean + min-to-max band.
- [ ] **Per-class table → multi-seed mean** (currently single-seed 14000).
- [ ] **Rewrite the "single-seed headline" limitation** → the multi-seed
      result.
- [ ] **Late-training drift** — at minimum characterize it cleanly; ideally
      a one-paragraph explanation (currently "we do not understand it").
- [ ] Test-compile, proof-read, tighten the frontier framing in intro/RW.

## 4. Strengthening list — what would make it even better

Ranked by acceptance impact / effort.

1. **LLM positioning paragraph** *(~half a day, writing only)* — do NOT run a
   head-to-head accuracy benchmark: it is the wrong axis (the agent is not
   competing with an LLM at solving a quadratic) and a table where the LLM
   "wins" would obscure the contribution. Instead add one paragraph that
   acknowledges LLMs and states the orthogonality: this work is about an
   agent *learning* exact symbolic transformations, unsupervised, with no
   solution traces and no hallucination. Preempts the reviewer question
   without the misleading benchmark.
2. **Generalization beyond the training templates** *(highest payoff for
   top-tier, ~1 week)* — the pointer-style `pi_cov` (generic symbol tokens +
   copy-pointer) generalizes to **unseen symbols**, and ideally **unseen
   families**. Turns "solves 4 templates" into "learns the CoV rule and
   extrapolates." This is the main lever from "solid" to "strong."
3. **Scaling result** *(good "and it scales" story, compute-heavy)* — the
   `open_large` 10x dataset (`Ntrain` 1e7). Run in the background over the
   coming weeks; report even a 3-seed result.
4. **Frontier-comparison table in related work** *(low effort, high clarity)*
   — a small table: prior work vs equation class solved (linear → ours).
5. **Qualitative learned traces** *(low effort)* — show the agent
   rediscovering named techniques (completing the square, depressing a
   cubic); a figure or two. Interpretability is a genuine strength here.
6. **More open families** *(scope, medium effort)* — additional CoV types
   beyond the current four, if time permits.

## 4b. Explicitly OUT of scope for this submission (-> Paper 2)

Named in Future Work, built later -- do not cram into the AAAI submission:
- **AlphaZero-style expert iteration** (MCTS + value head + search-improved
  training data). Full plan already in `markdowns/alphazero_plan.md`;
  `north_star.md` scopes it as "Paper 2".
- **Learned skill / option discovery** — automatically discovering reusable
  action sub-routines instead of the current hand-defined macroactions. Same
  research cluster as expert iteration; uncertain payoff; would dilute the
  clean frontier-advance contribution if rushed.

## 5. Suggested two-week schedule

- **Week 1:** finish Fig 4 + per-class multi-seed; update the paper
  (headline, table, limitation). LLM baseline. Characterize the drift.
- **Week 2:** generalization story (fix the pointer model's overfitting —
  smaller encoder — then unseen-symbol / unseen-family eval). Tighten
  related-work / frontier framing. Qualitative traces.
- **Background (whole window):** `open_large` scaling run.

Buffer: ~6 weeks remain after that for writing polish and internal review
before the AAAI deadline.
