#!/bin/bash
# ----------------------------------------------------------------------
# Fig-2 closed-equation sweep — clean, self-contained recreation.
#
#   3 datasets x 4 agents x 3 seeds = 36 runs
#   5M-step cap + early-stop (patience 8, metric=test_beam) + best.zip
#
# Datasets : abel_level3 (small), abel_level4 (large), poesia-full
# Agents   : ppo, ppo-tree, ppo-tree-rc, ppo-tree-rc-buf
#            (rc = --use_relabel_constants, buf = --use_success_replay,
#             flat buffer — the closed-equation defaults, NOT the open
#             stack's --use_cov / --anti_loop_penalty / fresh buffer)
#
# Seeds    : base_seed 3000  ->  ppo {3000,4000,5000}
#                                ppo-tree {10000,11000,12000}
#            chosen to not collide with any existing seed dir, so the new
#            runs are identifiable. (Old seed dirs are left in place;
#            decide archive-vs-filter for plot_closed_headline.py once
#            the sweep finishes.)
#
# Memory safety (this machine panicked from RAM exhaustion 2026-05-20):
#   - waits for the fullstack run to finish first (no process overlap)
#   - then runs ONE combo at a time; each combo = 3 train_abel workers.
#   Peak heavy procs = 3. Do not parallelize combos without watching RSS.
# ----------------------------------------------------------------------
set -u

PY=/Users/Kev/mambaforge/envs/abel-rl/bin/python
REPO=/Users/Kev/Documents/research/abel-rl-v3
cd "$REPO" || exit 1

LOGDIR=logs/fig2_sweep
mkdir -p "$LOGDIR"
SWEEP_LOG="$LOGDIR/_sweep.log"

NTRAIN=5000000
BASE_SEED=3000
NSEEDS=3
WAIT_PID=${WAIT_PID:-15423}    # fullstack PID to drain before starting

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SWEEP_LOG"; }

# --- wait for the fullstack run to drain -------------------------------
if [ -n "$WAIT_PID" ] && kill -0 "$WAIT_PID" 2>/dev/null; then
    log "waiting for fullstack PID $WAIT_PID to finish before starting..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
    log "fullstack PID $WAIT_PID done — starting sweep"
else
    log "fullstack PID $WAIT_PID not running — starting sweep immediately"
fi

# --- combo runner ------------------------------------------------------
# args: <gen> <agents-arg> <extra-flags> <tag>
run_combo () {
    local gen="$1" agent="$2" extra="$3" tag="$4"
    local lg="$LOGDIR/${tag}.log"
    log "START $tag  (gen=$gen agent=$agent extra='$extra')"
    $PY -u train_abel.py \
        --gen "$gen" --agents $agent --action_space dynamic \
        --Ntrain $NTRAIN $extra \
        --early_stop_patience 8 \
        --eval_subsample 150 \
        --base_seed $BASE_SEED --n_trials $NSEEDS --n_workers $NSEEDS --n_envs 1 \
        > "$lg" 2>&1
    local rc=$?
    log "END   $tag  (exit=$rc)"
}

# --- 12 combos: abel_level3 (fast) -> poesia-full -> abel_level4 (slow)-
for gen in abel_level3 poesia-full abel_level4; do
    run_combo "$gen" "ppo"      ""                                              "${gen}__ppo"
    run_combo "$gen" "ppo-tree" ""                                              "${gen}__ppo-tree"
    run_combo "$gen" "ppo-tree" "--use_relabel_constants"                       "${gen}__ppo-tree-rc"
    run_combo "$gen" "ppo-tree" "--use_relabel_constants --use_success_replay"  "${gen}__ppo-tree-rc-buf"
done

log "ALL 12 COMBOS DONE"
