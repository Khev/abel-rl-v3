#!/bin/bash
# Heavy-compute sweeps to run on the desktop (NOT the laptop).
# All scripts use the best-known config:
#   --use_relabel_constants --use_success_replay --use_cov --use_cbrt
#   --anti_loop_penalty 0.1 --early_stop_patience 8 --eval_lite
#   --sr_buffer_kind fresh        (the freshness-managed buffer)
#
# POST-FIX status (2026-05-22): the open-equation no-output bug is fixed —
# four CoV-path simplify() calls in env_multi_eqn.py recursed into
# trigsimp/exptrigsimp and hung uninterruptibly (commits da264b6, dc293f6).
# All CoV-path normalization is now expand(). Runs are hang-free.
# POST-FIX status (2026-05-23): vocab_max_id default raised 256 -> 600 in
# utils/utils_env.py (mixed_v2_large has feature_dict max ID 511; the old
# default crashed with IndexError in the embedding layer). mixed_v2_large
# sweeps now work out of the box.
# Today's 5-seed laptop run with the fixed code: 3-seed mean test_beam = 0.86
# (range 0.84-0.90), seeds 28000/30000/31000 -- supersedes the earlier
# single-seed 0.80 headline.
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
# 1) Headline: 5-seed full-stack (+ fresh buffer) on mixed_v2_easy
# ----------------------------------------------------------------------
# Replicates the laptop's 3-seed mean (0.86) with more seeds for tight
# error bars. base_seed 40000 chosen to avoid collision with existing
# seed dirs (laptop has seed28000-32000).
echo "[1/4] 5-seed headline config on mixed_v2_easy (Ntrain=3M each)"
$PY -u train_abel.py \
    --gen mixed_v2_easy --agent ppo-tree --action_space dynamic \
    --Ntrain 3000000 \
    --use_relabel_constants --use_success_replay --use_cov \
    --use_cbrt \
    --anti_loop_penalty 0.1 \
    --sr_buffer_kind fresh \
    --early_stop_patience 8 \
    --eval_lite \
    --base_seed 40000 --n_trials 5 --n_workers 5 --n_envs 1 \
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
echo "[2/4] 5-seed headline config on mixed_v2_large (Ntrain=1e7 each)"
$PY -u train_abel.py \
    --gen mixed_v2_large --agent ppo-tree --action_space dynamic \
    --Ntrain 10000000 \
    --use_relabel_constants --use_success_replay --use_cov \
    --use_cbrt \
    --anti_loop_penalty 0.1 \
    --sr_buffer_kind fresh \
    --early_stop_patience 8 \
    --eval_lite \
    --eval_subsample 200 \
    --base_seed 50000 --n_trials 5 --n_workers 5 --n_envs 1 \
    > "$LOGDIR/large_5seeds.log" 2>&1 &
LARGE_PID=$!
echo "  launched PID $LARGE_PID"

sleep 60

# ----------------------------------------------------------------------
# 3) Fresh-buffer ablation: 3 seeds with --sr_buffer_kind fresh
# ----------------------------------------------------------------------
# We saw single-seed evidence that the "fresh" buffer (drop oldest 50%
# every 20 rollouts) beats the flat buffer. Confirm with 3 seeds.
echo "[3/4] 3-seed fresh-buffer ablation on mixed_v2_easy"
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

sleep 60

# ----------------------------------------------------------------------
# 4) Cubic+exponential focused run — confirms the exp-class story on a
#    clean run with ALL fixes (cbrt, max_cov_apps=3, pi_cov current-form
#    using expand() + count_ops guard, fresh buffer).
# ----------------------------------------------------------------------
# Earlier cubic+exp runs were either pre-fix or hit the simplify() hang.
# This is the clean ablation: 3 seeds, focused 1000-eqn cubic+exp set.
# Expect cubic ~100% and exponential well above 0 (seed14000 on the mixed
# set already reached exp 13/15; this isolates the two hard classes).
echo "[4/4] 3-seed cubic+exp focused run (all fixes)"
$PY -u train_abel.py \
    --gen cubic_exp_focused --agent ppo-tree --action_space dynamic \
    --Ntrain 3000000 \
    --use_relabel_constants --use_success_replay --use_cov \
    --use_cbrt \
    --anti_loop_penalty 0.1 \
    --sr_buffer_kind fresh \
    --early_stop_patience 8 \
    --eval_lite \
    --base_seed 5000 --n_trials 3 --n_workers 3 --n_envs 1 \
    > "$LOGDIR/cubic_exp_3seeds.log" 2>&1 &
CUBEXP_PID=$!
echo "  launched PID $CUBEXP_PID"

echo ""
echo "All sweeps launched. PIDs: easy=$EASY_PID large=$LARGE_PID fresh=$FRESH_PID cubexp=$CUBEXP_PID"
echo "Total cores in use: ~16 (5+5+3+3). Adjust --n_workers if you have fewer."
echo ""
echo "Monitor with:"
echo "  ps aux | grep multiprocessing-fork | grep -v grep | wc -l"
echo "  tail -f $LOGDIR/*.log"
echo ""
echo "Final eval (after each finishes) with eval_value_beam.py and"
echo "eval_per_class.py — see those scripts for usage."
