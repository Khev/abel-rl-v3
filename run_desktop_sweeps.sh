#!/bin/bash
# Heavy-compute sweeps to run on the desktop (NOT the laptop).
# All scripts use the best-known config:
#   --use_relabel_constants --use_success_replay --use_cov --use_cbrt
#   --anti_loop_penalty 0.1 --early_stop_patience 8 --eval_lite
#
# Edit paths if running outside the repo. Adjust --n_workers to match
# desktop core count.
#
# Recommended order:
#   1) headline 5-seed sweep on mixed_v2_easy (3-5h per seed, ~25h total at 1 core/seed)
#   2) mixed_v2_large 5-seed sweep (24-48h per seed; the "scaling" story)
#   3) optional: fresh-buffer sweep if seed7600 result holds
#
# Total: ~3-5 days of compute depending on core count.

set -u
PY=python  # set to absolute path of your conda env's python on the desktop
N_CORES_AVAILABLE=${N_CORES_AVAILABLE:-12}  # override via env var

LOGDIR=results/desktop_logs
mkdir -p "$LOGDIR"

echo "============================================"
echo "Desktop sweep launcher"
echo "Cores available: $N_CORES_AVAILABLE"
echo "Logs: $LOGDIR"
echo "============================================"
echo ""

# ----------------------------------------------------------------------
# 1) Headline: 5-seed best-config on mixed_v2_easy
# ----------------------------------------------------------------------
# Existing seed10000 hit test_beam=0.615. Replicating with 5 fresh seeds
# gives proper error bars for the paper headline.
echo "[1/3] 5-seed best-config on mixed_v2_easy (Ntrain=3M each)"
$PY -u train_abel.py \
    --gen mixed_v2_easy --agent ppo-tree --action_space dynamic \
    --Ntrain 3000000 \
    --use_relabel_constants --use_success_replay --use_cov \
    --use_cbrt \
    --anti_loop_penalty 0.1 \
    --early_stop_patience 8 \
    --eval_lite \
    --base_seed 2000 --n_trials 5 --n_workers 5 --n_envs 1 \
    > "$LOGDIR/easy_5seeds.log" 2>&1 &
EASY_PID=$!
echo "  launched PID $EASY_PID"

# Wait a bit before launching the next pool so they don't fight on startup
sleep 60

# ----------------------------------------------------------------------
# 2) Scaling: 5-seed best-config on mixed_v2_large (Ntrain=1e7)
# ----------------------------------------------------------------------
# 10x more training equations and longer horizon. Provides the "and it
# scales" story. Each seed takes ~24h with the speedups; total wall-clock
# is gated by max parallelism.
echo "[2/3] 5-seed best-config on mixed_v2_large (Ntrain=1e7 each)"
$PY -u train_abel.py \
    --gen mixed_v2_large --agent ppo-tree --action_space dynamic \
    --Ntrain 10000000 \
    --use_relabel_constants --use_success_replay --use_cov \
    --use_cbrt \
    --anti_loop_penalty 0.1 \
    --early_stop_patience 8 \
    --eval_lite \
    --eval_subsample 200 \
    --base_seed 3000 --n_trials 5 --n_workers 5 --n_envs 1 \
    > "$LOGDIR/large_5seeds.log" 2>&1 &
LARGE_PID=$!
echo "  launched PID $LARGE_PID"

sleep 60

# ----------------------------------------------------------------------
# 3) Fresh-buffer ablation: 3 seeds with --sr_buffer_kind fresh
# ----------------------------------------------------------------------
# We saw single-seed evidence that the "fresh" buffer (drop oldest 50%
# every 20 rollouts) beats the flat buffer. Confirm with 3 seeds.
echo "[3/3] 3-seed fresh-buffer ablation on mixed_v2_easy"
$PY -u train_abel.py \
    --gen mixed_v2_easy --agent ppo-tree --action_space dynamic \
    --Ntrain 3000000 \
    --use_relabel_constants --use_success_replay --use_cov \
    --use_cbrt \
    --anti_loop_penalty 0.1 \
    --sr_buffer_kind fresh \
    --early_stop_patience 8 \
    --eval_lite \
    --base_seed 4000 --n_trials 3 --n_workers 3 --n_envs 1 \
    > "$LOGDIR/fresh_3seeds.log" 2>&1 &
FRESH_PID=$!
echo "  launched PID $FRESH_PID"

echo ""
echo "All sweeps launched. PIDs: easy=$EASY_PID large=$LARGE_PID fresh=$FRESH_PID"
echo "Total cores in use: ~13 (5+5+3). Adjust --n_workers if you have fewer."
echo ""
echo "Monitor with:"
echo "  ps aux | grep multiprocessing-fork | grep -v grep | wc -l"
echo "  tail -f $LOGDIR/*.log"
echo ""
echo "Final eval (after each finishes) with eval_value_beam.py and"
echo "eval_per_class.py — see those scripts for usage."
