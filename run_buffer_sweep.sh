#!/bin/bash
# Sequential single-core sweep over success-buffer variants.
# Compares: flat (baseline) vs balanced vs balanced_lw vs fresh.
# 1 seed each (hardcoded for reproducibility), mixed_v2_easy, Ntrain=500k.
# ~2.2 h per variant on 1 core; 4 variants = ~9 h total (fits overnight).
set -u
PY=/Users/Kev/mambaforge/envs/abel-rl/bin/python
LOGDIR=/tmp/buffer_sweep_logs
mkdir -p $LOGDIR

# Different base_seed per variant so the new training-data folders don't collide
declare -A SEEDS=(
  [flat]=300
  [balanced]=400
  [balanced_lw]=500
  [fresh]=600
)

for KIND in flat balanced balanced_lw fresh; do
  SEED=${SEEDS[$KIND]}
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
echo "All buffer-kind sweeps complete. Logs in $LOGDIR"
echo ""
echo "Run eval comparison with:"
echo "  for KIND in flat balanced balanced_lw fresh; do"
echo "    echo \"\$KIND:\""
echo "    python eval_value_beam.py --ckpt data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/seed\${SEEDS[\$KIND]}/checkpoints/latest.zip --lambdas 0.0,1.0 --gen mixed_v2_easy"
echo "  done"
