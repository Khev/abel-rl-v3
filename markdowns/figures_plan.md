# Paper figures plan

What we need for the camera-ready open-equations section, given current results.

## Already in the paper (closed-eqs side, unchanged)

- **Figure 2**: learning curves (coverage, test_greedy, test_beam) for
  ppo, ppo-tree, ppo-tree-rc, ppo-tree-buf, ppo-tree-rc-buf on
  CommonCore + small + large. Don't touch.
- **Figure 3 / 4**: representation visualizations (t-SNE, PCA). Already
  built. Possibly update with seed8001 or seed9100 embeddings if we want
  the "anti-loop policy embeds differently" story (probably out of scope).
- **Table 1**: closed-eq test accuracy. Already filled with figure-derived
  eyeballed numbers. **Open question: do we update with value-guided beam
  numbers?** That would change every column. Pro: bigger numbers. Con:
  inconsistent comparison with prior literature that used plain beam.
  Recommendation: keep the original Table 1 as "plain beam" and add a
  small extra table for value-beam (new row, "+ value-guided beam").

## New figures for the mixed/open-eqs section

### Figure 5: anti-loop motivation (failure-mode trace)

Side-by-side trace tables, like a code snippet visualization, showing the
baseline's REL-loop pattern next to anti-loop's diverse solve. Numbers
straight from `diagnose_antiloop_vs_baseline.py`. Three illustrative
equations:

| Equation | Baseline trace | Anti-loop trace |
|---|---|---|
| `log(x/c)**2` | REL REL REL REL REL... | SRT * * / / REL ✓ |
| `sqrt(-a + b*x)` | SQR CLT CLT CLT... | SQR + / ✓ |
| `a*sqrt(b*x)` | COV COV COV COV... | SQR * REL / ✓ |

Plus a small bar showing: "fraction of repeat-actions in failed traces"
baseline=0.86, anti-loop=0.48.

### Figure 6: mixed_v2_easy learning curves

Subplot grid:

- (a) coverage vs steps: baseline (mean of 4 seeds) vs anti-loop α=0.1
  (mean of 3 seeds). Shows anti-loop has lower coverage (slower
  memorization).
- (b) test_beam vs steps: same pair. Shows anti-loop's higher peak
  test_beam at fewer steps.
- (c) test_at10 vs steps: same pair. Similar shape.

Need to load all 7 seeds (7000, 7001, 8001, 9001 baseline; 7100, 8100, 9100
anti-loop) and shaded-error-bar them. seed8100 has no rows — drop it.

### Figure 7: value-guided beam λ-sweep

Single panel: x = λ, y = test_beam. Two lines:

- seed8001 (baseline): monotone rise to 0.40 at λ=0.5–1.0, then falls
- seed9100 (anti-loop): monotone fall from 0.35

Plus horizontal dashed line at 0.275 (plain beam baseline) for reference.

Caption: "Value-guided beam helps when the policy is unsharp but hurts when
it's already strong."

### Figure 8 (maybe): the "free lift" table

```
                          plain beam   + value-guided beam (λ=1.0, dedup)
ppo-tree-rc-buf small       0.55          ?
ppo-tree-rc-buf large       0.66          ?
mixed_v2_easy seed8001      0.28          0.40
mixed_v2_easy seed9100      0.34          0.34 (no gain — policy already strong)
abel_level3 seed7006        0.87          ?
```

The "?" rows fill in from the mass re-eval currently running. This is the
key "no retraining required" table.

## Things we DON'T need new figures for

- t-SNE of anti-loop vs baseline embeddings: nice to have but doesn't
  pay for itself.
- Per-class accuracy breakdown: data exists in
  `diagnose_action_traces.py` output; could be a small table in the
  appendix if space.
- The action-cache speedup: pure throughput, not a result.

## Asset locations / scripts

- `plot_seeds.py` — exists; can adapt for fig 6.
- `plot_ablation.py` — exists; can adapt for fig 7 / 8.
- `plot_embeddings.py`, `plot_tsne_panel.py` — for any new representation
  fig.
- New script needed: `plot_value_beam_curve.py` for fig 7.
- New script needed: `plot_failure_modes.py` (or hand-draw fig 5).

## Priority order

1. Fig 7 (value-beam λ-sweep) — single panel, cleanest result, smallest
   effort.
2. Fig 6 (mixed_v2_easy learning curves) — needs the seeds to finish
   training. seed9100 still has ~900k to go.
3. Table 1+ (free-lift table) — needs mass re-eval to finish.
4. Fig 5 (failure-mode trace) — data already in
   `diagnose_antiloop_vs_baseline.py` output; hand-format.

Defer fig 8 and any t-SNE updates until results stabilize.
