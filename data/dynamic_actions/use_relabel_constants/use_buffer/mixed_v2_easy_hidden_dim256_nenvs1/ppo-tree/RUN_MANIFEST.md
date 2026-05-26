# Run manifest — mixed_v2_easy / ppo-tree

The directory path encodes only `use_relabel_constants` + `use_success_replay`.
It does NOT encode `anti_loop_penalty`, `sr_buffer_kind`, `use_cbrt`, or
`use_cov`, and `metrics.json` records results only — not hyperparameters.
This file records the config of seeds whose config is otherwise unrecoverable.

## Open-equation figure seeds (Fig. 4)

| seed   | role                | config |
|--------|---------------------|--------|
| 8001   | baseline            | ppo-tree-rc-buf-cov; no anti-loop, flat buffer, pre-cbrt |
| 9100   | + anti-loop (ablation only) | ppo-tree-rc-buf-cov + anti_loop_penalty 0.1 |
| 14000  | full method stack   | full stack (see below) |
| 16000  | full method stack (replica) | full stack — launched 2026-05-19 |
| 17000  | full method stack (replica) | full stack — launched 2026-05-19 |

Seeds 8001 / 9100 configs are believed-correct from `plot_open_curves.py`
documentation; they predate this manifest and cannot be verified from disk.

## Full-stack config (seeds 14000, 16000, 17000)

Exact command for 16000 / 17000 (launched 2026-05-19, background):

```
python -u train_abel.py \
    --gen mixed_v2_easy --agent ppo-tree --action_space dynamic \
    --Ntrain 3000000 \
    --use_relabel_constants --use_success_replay --use_cov \
    --use_cbrt \
    --anti_loop_penalty 0.1 \
    --sr_buffer_kind fresh \
    --early_stop_patience 8 \
    --eval_lite \
    --base_seed 9000 --n_trials 2 --n_workers 2 --n_envs 1
```

Seed derivation: `seed = base_seed + 1000*t + 7000` (the `+7000` is the
`ppo-tree` agent bump). base_seed 9000, t in {0,1} -> seeds 16000, 17000.

`--eval_lite` means test_beam is NOT in `learning_curves.csv`; reconstruct it
afterwards with `eval_checkpoint_curve.py` (writes `test_beam_curve.csv`),
then rerun `plot_open_curves.py` to get the mean ± min/max band.
