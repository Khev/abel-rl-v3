#!/bin/bash
# Watcher: polls seed14000 (mixed_v2_easy fresh-buffer) until training is
# done, then runs the full eval suite and launches the pi_cov-fixed
# cubic+exp run.
set -u
PY=/Users/Kev/mambaforge/envs/abel-rl/bin/python
CKPT_DIR=data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/seed14000
CSV=$CKPT_DIR/learning_curves.csv
RESULTS_DIR=/tmp/seed14000_results
mkdir -p $RESULTS_DIR

# Wait until training finished (step >= 2.85M or csv stale > 20 min)
echo "[$(date +%H:%M:%S)] watch_seed14000: starting poll loop"
while true; do
  if [ ! -f "$CSV" ]; then
    echo "[$(date +%H:%M:%S)] csv missing, waiting..."; sleep 300; continue
  fi
  last_step=$(tail -1 "$CSV" | cut -d, -f1)
  if [[ "$last_step" =~ ^[0-9]+$ ]] && [ "$last_step" -ge 2850000 ]; then
    echo "[$(date +%H:%M:%S)] training done (step=$last_step)"; break
  fi
  age_min=$(( ($(date +%s) - $(stat -f %m "$CSV")) / 60 ))
  if [ "$age_min" -gt 20 ]; then
    echo "[$(date +%H:%M:%S)] csv stale ($age_min min); assume done at step=$last_step"; break
  fi
  echo "[$(date +%H:%M:%S)] step=$last_step  age=${age_min}min  ...still training"
  sleep 300
done

# === Eval suite ===
echo "[$(date +%H:%M:%S)] === Running eval suite on seed14000 ==="
CKPT=$CKPT_DIR/checkpoints/latest.zip

# 1. value-beam sweep (greedy + plain beam + value beam at λ=0.5, 1.0) at max_steps=10
echo "[$(date +%H:%M:%S)] value-beam sweep (max_steps=10)"
$PY eval_value_beam.py --ckpt "$CKPT" --gen mixed_v2_easy \
    --lambdas 0.0,0.5,1.0 --max_steps 10 --include_greedy \
    > $RESULTS_DIR/value_beam_max10.txt 2>&1

# 2. value-beam at max_steps=15 (deeper search, mid-ground vs the OOM'd 20)
echo "[$(date +%H:%M:%S)] value-beam at max_steps=15"
$PY eval_value_beam.py --ckpt "$CKPT" --gen mixed_v2_easy \
    --lambdas 0.0,1.0 --max_steps 15 --include_greedy \
    > $RESULTS_DIR/value_beam_max15.txt 2>&1

# 3. per-class breakdown
echo "[$(date +%H:%M:%S)] per-class breakdown"
# Edit eval_per_class.py to point at seed14000, then run
python <<'PYEND' > $RESULTS_DIR/per_class.txt 2>&1
import sys
try: sys.set_int_max_str_digits(0)
except: pass
sys.path.insert(0, '.')
import eval_per_class
eval_per_class.CKPTS = {
  'seed10000_old_full_stack': 'data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/seed10000/checkpoints/latest.zip',
  'seed14000_fresh_buffer':   'data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/seed14000/checkpoints/latest.zip',
}
eval_per_class.main()
PYEND

echo "[$(date +%H:%M:%S)] === Eval done. Results in $RESULTS_DIR ==="
ls -la $RESULTS_DIR/

# === Launch the pi_cov-fixed cubic+exp run ===
echo "[$(date +%H:%M:%S)] === Launching cubic+exp v2 (pi_cov fix) ==="
nohup $PY -u train_abel.py \
    --gen cubic_exp_focused --agent ppo-tree --action_space dynamic \
    --Ntrain 3000000 \
    --use_relabel_constants --use_success_replay --use_cov \
    --use_cbrt \
    --sr_buffer_kind fresh \
    --anti_loop_penalty 0.1 \
    --early_stop_patience 8 \
    --eval_lite \
    --base_seed 17000 --n_trials 3 --n_workers 3 --n_envs 1 \
    > /tmp/cubic_exp_v3_picov_fix.log 2>&1 &
echo "[$(date +%H:%M:%S)] cubic+exp v3 launched PID $!"
echo "[$(date +%H:%M:%S)] watch_seed14000: DONE"
