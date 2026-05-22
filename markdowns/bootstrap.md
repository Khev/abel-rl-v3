# Bootstrap — resume after reboot

Last updated: 2026-05-19 ~22:10

## Why this file exists
The machine had 4 unkillable zombie `train_abel.py` processes — tiny
`abel_level1 --Ntrain 2000` runs that hung 2+ days ago and are stuck in
uninterruptible kernel sleep (`UE` state). `kill -9` cannot reap them; a
reboot is the only way to clear them. They use 0% CPU (cosmetic load
only), so a reboot is **optional**, but it gives a guaranteed-clean slate
before any big sweep. This file relaunches the in-flight work afterward.

## Environment
- Repo: `/Users/Kev/Documents/research/abel-rl-v3`
- Python: `/Users/Kev/mambaforge/envs/abel-rl/bin/python` (use this absolute
  path directly — `conda activate` fails due to a two-conda conflict)
- 10 cores; keep total workers <= 8.

## 1. Full-stack open-equation seeds (16000, 17000)
Two replica seeds of the headline full-stack config. Goal: turn the Fig. 4
full-stack curve into a 3-seed mean ± min/max band (with existing seed14000).
Config recorded in: `data/.../mixed_v2_easy_.../ppo-tree/RUN_MANIFEST.md`.

```
cd /Users/Kev/Documents/research/abel-rl-v3
P=data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree
rm -rf "$P/seed16000" "$P/seed17000"   # clear any partial dirs from the killed run
mkdir -p logs
nohup /Users/Kev/mambaforge/envs/abel-rl/bin/python -u train_abel.py \
    --gen mixed_v2_easy --agent ppo-tree --action_space dynamic \
    --Ntrain 3000000 --use_relabel_constants --use_success_replay --use_cov \
    --use_cbrt --anti_loop_penalty 0.1 --sr_buffer_kind fresh \
    --early_stop_patience 8 --eval_lite \
    --base_seed 9000 --n_trials 2 --n_workers 2 --n_envs 1 \
    > logs/fullstack_16000_17000.log 2>&1 &
```
Runtime ~3-5h (2 cores). When both seeds finish:
```
/Users/Kev/mambaforge/envs/abel-rl/bin/python eval_checkpoint_curve.py --run_dir "$P/seed16000" --gen mixed_v2_easy
/Users/Kev/mambaforge/envs/abel-rl/bin/python eval_checkpoint_curve.py --run_dir "$P/seed17000" --gen mixed_v2_easy
/Users/Kev/mambaforge/envs/abel-rl/bin/python plot_open_curves.py   # full-stack curve -> 3-seed band
```

## 2. abel_level4 decoder re-eval
Re-scores the existing abel_level4 `ppo-tree-rc-buf` checkpoints with the
value-beam decoder + best-checkpoint selection (no retraining). Scout for
the "recreate Figure 2" decision: does value-beam recover the late-training
drift (current abel_level4 test_beam mean ~0.55)?

```
cd /Users/Kev/Documents/research/abel-rl-v3
# smoke test first (~2 min on a clean machine):
/Users/Kev/mambaforge/envs/abel-rl/bin/python eval_abel4_decoder.py --smoke --workers 4
# if value-beam numbers look sane (~0.4-0.7, non-zero), run full (~1-1.5h, 6 cores):
nohup /Users/Kev/mambaforge/envs/abel-rl/bin/python eval_abel4_decoder.py \
    --workers 6 > logs/abel4_decoder.log 2>&1 &
```
Writes `abel4_decoder_curve.csv` into each seed dir + prints a summary.

Running #1 and #2 together = 2 + 6 = 8 cores. Fine.

## 3. Open decision — recreate Figure 2
Considering re-running the closed-equation figure as a clean, self-contained
sweep: 3 datasets x 4 agents (ppo-tree / +rc / +buf / +rc-buf) x 3 seeds =
36 runs, **5M-step cap + early-stopping + best-checkpoint** (the data shows
performance peaks ~3-5M then drifts down). This is a desktop sweep, not a
laptop job. Decide after the abel_level4 re-eval (#2) result.

## Paper state
- `papers/main_v1.tex` — live cleanup version, 13 pages, compiles clean.
  Results merged into one section; Fig 4 restyled (seaborn, matches Fig 2);
  Fig 7 (curiosity) regenerated at 3M horizon.
- `papers/main.tex` — pre-cleanup original (18 pages). `main_v0.tex` — backup.
- Known issue to reconcile: `ppo-tree-rc` poesia accuracy is 0.85 in Table 1
  but 0.96 in prose; abel_level4 test_beam stated as 0.66 but data mean ~0.55.
