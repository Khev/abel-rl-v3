# Table 1 (Closed Equations) — current state

Pulled from `results_all.csv` (output of `aggregate_results.py`).

All numbers are **single seed**, **on the pre-fix encoder**.

## What we have

| Algorithm | small (abel_level3) | large (abel_level4) | poesia-full |
|---|---|---|---|
| ConPoLe-local (Poesia 2021) | n/a | n/a | 0.765 |
| ConPoLe (Poesia 2021) | — | — | 0.925 |
| ppo (dyn) | **0.040** | **0.018** | missing |
| ppo-tree (dyn, no rc, no buf) | missing¹ | missing¹ | missing |
| ppo-tree-rc (dyn+rc) | **0.902** (beam 0.881) | missing | **0.955** |
| ppo-tree-rc-buf (dyn+rc+buf) | **0.801** (beam 0.899) | **0.441** | (have `buf+dyn` at 0.865; not `rc+buf`) |

¹ We have `ppo-tree` on `abel_level3` from older `data/abel_level3_hidden_dim64_nenvs1/` (non-`dynamic_actions/`) at greedy=0.55, beam=0.80 — but the paper's runs use `--action_space dynamic`. Possible to report this, or run fresh.

## Still genuinely missing

Five cells:
1. ppo × poesia
2. ppo-tree (dyn) × abel_level3
3. ppo-tree (dyn) × abel_level4
4. ppo-tree (dyn) × poesia
5. ppo-tree-rc × abel_level4
6. ppo-tree-rc-buf × poesia (need full `rc+buf` combo)

The first one (ppo × poesia) probably gives a near-zero number consistent with "ppo without inductive biases struggles."

## Methodological notes

- Encoder bug affects equations with numeric coefficients. Looking at the abel_level3 templates earlier: most are purely symbolic (`a*sin(x) - c`, `cos(log(x/b))`, `a + cos(sqrt(x))`). For these, the bug doesn't change encoding. **The pre-fix numbers for abel_level3 are likely valid.**
- The exception is when equations include integers (e.g. `2*c + x/b`). Some abel_level3 equations have these. Strictly speaking, these specific instances were trained with a worse representation than the fixed-encoder version would give. Magnitude of effect: small fraction of the dataset, not catastrophic.
- For poesia: more numeric content, encoder fix likely matters more. Worth a re-run for that column when time permits.
- **None of the existing summaries are multi-seed** (`coverage_std` empty everywhere). For the paper we want ≥3 seeds + std. The cheapest way to get this: pull from `learning_curves.csv` end-of-training values, or accept single-seed reporting with a footnote.

## Recommended action

For each missing cell or single-seed cell:
- (a) Defer until after CoV paper experiments are designed; we'll do the closed-eq re-runs as part of paper-finalization sweep.
- (b) Run the 5 missing cells now in the background (~2-3 hours each, 5 × 2hr = 10 hours sequential or 5 hours parallel if we have cores).

Currently keeping just one in-flight run (`b10omzfr6`: ppo-tree-rc-buf × abel_level3 with fixed encoder) as a sanity check that the encoder fix doesn't regress the headline number.
