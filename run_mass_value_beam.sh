#!/bin/bash
# Mass re-eval: run value-guided beam (lambda=1.0) on all interesting checkpoints.
# Also runs plain beam (lambda=0) for direct comparison.
# Outputs to /tmp/value_beam_results.csv

set -u
OUT=/tmp/value_beam_results.csv
PY=/Users/Kev/mambaforge/envs/abel-rl/bin/python

echo "tag,gen,ckpt,plain_beam,value_beam_l1" > "$OUT"

run_one() {
  local tag="$1"
  local gen="$2"
  local ckpt="$3"
  if [ ! -f "$ckpt" ]; then
    echo "  SKIP: $tag (no ckpt at $ckpt)"
    return
  fi
  echo "  Running: $tag"
  local result
  result=$($PY eval_value_beam.py --ckpt "$ckpt" --gen "$gen" --lambdas 0.0,1.0 2>/dev/null | tail -3 | head -2 | awk '{print $4}' | tr '\n' ',' | sed 's/,$//')
  echo "$tag,$gen,$ckpt,$result" >> "$OUT"
  echo "    → $result"
}

# mixed_v2_easy: baseline (anti_loop=0) and anti-loop (anti_loop=0.1) seeds
for s in seed7000 seed7001 seed8001 seed9001; do
  run_one "easy_baseline_$s" mixed_v2_easy \
    "data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/$s/checkpoints/latest.zip"
done
for s in seed7100 seed9100; do
  run_one "easy_antiloop_$s" mixed_v2_easy \
    "data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/$s/checkpoints/latest.zip"
done

# abel_level3 (Table 1 in paper)
for s in seed7001 seed7006 seed8001; do
  run_one "abel3_$s" abel_level3 \
    "data/dynamic_actions/use_relabel_constants/use_buffer/abel_level3_hidden_dim256_nenvs1/ppo-tree/$s/checkpoints/latest.zip"
done

# mixed_v2_small
run_one "small_seed7000" mixed_v2_small \
  "data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_small_hidden_dim256_nenvs1/ppo-tree/seed7000/checkpoints/latest.zip"

# Curiosity (abel_level3)
run_one "curiosity_RND" abel_level3 \
  "data/dynamic_actions/abel_level3_hidden_dim256_nenvs1/ppo-tree-RND/seed4000/checkpoints/model_step3000000.zip"
run_one "curiosity_ICM" abel_level3 \
  "data/dynamic_actions/abel_level3_hidden_dim256_nenvs1/ppo-tree-ICM/seed1000/checkpoints/model_step3000000.zip"
run_one "curiosity_NGU" abel_level3 \
  "data/dynamic_actions/abel_level3_hidden_dim256_nenvs1/ppo-tree-NGU/seed6000/checkpoints/model_step3000000.zip"

echo
echo "Done. Results in $OUT"
column -t -s, "$OUT"
