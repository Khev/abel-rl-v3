#!/bin/bash
# v2: fixed parsing. Captures BOTH plain beam (λ=0) AND value-guided beam (λ=1)
# per checkpoint. Outputs to /tmp/value_beam_results_v2.csv.

set -u
OUT=/tmp/value_beam_results_v2.csv
PY=/Users/Kev/mambaforge/envs/abel-rl/bin/python

echo "tag,gen,ckpt,plain_beam,value_beam_l1,delta" > "$OUT"

run_one() {
  local tag="$1"
  local gen="$2"
  local ckpt="$3"
  if [ ! -f "$ckpt" ]; then
    echo "  SKIP: $tag (no ckpt at $ckpt)"
    return
  fi
  echo "  Running: $tag"

  # Capture full output, then grep for data rows: lines that start with
  # optional whitespace followed by a digit. Column 4 is test_beam.
  local out
  out=$($PY eval_value_beam.py --ckpt "$ckpt" --gen "$gen" --lambdas 0.0,1.0 2>/dev/null)
  local plain value
  plain=$(echo "$out" | awk '/^[[:space:]]*0\.00[[:space:]]/{print $4; exit}')
  value=$(echo "$out" | awk '/^[[:space:]]*1\.00[[:space:]]/{print $4; exit}')

  if [ -z "$plain" ] || [ -z "$value" ]; then
    echo "    PARSE FAIL: plain=[$plain] value=[$value]"
    return
  fi

  # Compute delta with bc
  local delta
  delta=$(echo "$value $plain" | awk '{printf "%+.4f", $1-$2}')
  echo "$tag,$gen,$ckpt,$plain,$value,$delta" >> "$OUT"
  echo "    → plain=$plain value=$value Δ=$delta"
}

# Same list as v1
for s in seed7000 seed7001 seed8001 seed9001; do
  run_one "easy_baseline_$s" mixed_v2_easy \
    "data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/$s/checkpoints/latest.zip"
done
for s in seed7100 seed9100; do
  run_one "easy_antiloop_$s" mixed_v2_easy \
    "data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/$s/checkpoints/latest.zip"
done
for s in seed7001 seed7006 seed8001; do
  run_one "abel3_$s" abel_level3 \
    "data/dynamic_actions/use_relabel_constants/use_buffer/abel_level3_hidden_dim256_nenvs1/ppo-tree/$s/checkpoints/latest.zip"
done
run_one "small_seed7000" mixed_v2_small \
  "data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_small_hidden_dim256_nenvs1/ppo-tree/seed7000/checkpoints/latest.zip"
run_one "curiosity_RND" abel_level3 \
  "data/dynamic_actions/abel_level3_hidden_dim256_nenvs1/ppo-tree-RND/seed4000/checkpoints/model_step3000000.zip"
run_one "curiosity_ICM" abel_level3 \
  "data/dynamic_actions/abel_level3_hidden_dim256_nenvs1/ppo-tree-ICM/seed1000/checkpoints/model_step3000000.zip"
run_one "curiosity_NGU" abel_level3 \
  "data/dynamic_actions/abel_level3_hidden_dim256_nenvs1/ppo-tree-NGU/seed6000/checkpoints/model_step3000000.zip"

echo
echo "Done. Results in $OUT"
column -t -s, "$OUT"
