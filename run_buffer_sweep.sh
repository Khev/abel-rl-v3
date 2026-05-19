#!/bin/bash
# Sequential single-core sweep over success-buffer variants.
set -u
PY=/Users/Kev/mambaforge/envs/abel-rl/bin/python
LOGDIR=/tmp/buffer_sweep_logs
mkdir -p $LOGDIR

for KIND in flat balanced balanced_lw fresh; do
  case "$KIND" in
    flat) SEED=300 ;;
    balanced) SEED=400 ;;
    balanced_lw) SEED=500 ;;
    fresh) SEED=600 ;;
  esac
  echo "=== Running buffer_kind=$KIND seed=$SEED ==="
  $PY -u train_abel.py \
    --gen mixed_v2_easy --agent ppo-tree --action_space dynamic \
    --Ntrain 500000 \
    --use_relabel_constants --use_success_replay --use_cov \
    --use_cbrt \
    --anti_loop_penalty 0.1 \
    --sr_buffer_kind "$KIND" \
    --early_stop_patience 8 \
    --eval_lite \
    --base_seed "$SEED" --n_trials 1 --n_workers 1 --n_envs 1 \
    > "$LOGDIR/${KIND}_seed${SEED}.log" 2>&1
  echo "  done: $KIND"
done
echo "All sweeps complete. Logs in $LOGDIR"
