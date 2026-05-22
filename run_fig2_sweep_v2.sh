#!/bin/bash
# ----------------------------------------------------------------------
# Fig-2 sweep v2 — 2-wide concurrency (replaces sequential run_fig2_sweep.sh).
#
# Combo abel_level3__ppo is already running (PID 69774) from the original
# sequential launcher. This v2 launcher runs the remaining 11 combos two at
# a time, counting that in-progress combo as occupying a slot until it ends.
#
# Peak heavy procs = MAX_COMBOS * 3 = 6 (closed-equation workers, ~1-3 GB
# each — the CoV memory leak does NOT affect this path). Do not raise
# MAX_COMBOS past 2 without watching RSS: 3 -> 9 workers approaches the
# 2026-05-20 panic zone. See memory: compute-ram-constraint.
#
# Restart-safe: a combo that exits 0 drops a <tag>.done marker and is
# skipped on re-run.
# ----------------------------------------------------------------------
set -u

PY=/Users/Kev/mambaforge/envs/abel-rl/bin/python
REPO=/Users/Kev/Documents/research/abel-rl-v3
cd "$REPO" || exit 1

LOGDIR=logs/fig2_sweep
mkdir -p "$LOGDIR"
SWEEP_LOG="$LOGDIR/_sweep_v2.log"

NTRAIN=5000000
BASE_SEED=3000
NSEEDS=3
MAX_COMBOS=${MAX_COMBOS:-2}
COMBO1_PID=${COMBO1_PID:-69774}   # abel_level3__ppo, already running

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SWEEP_LOG"; }

run_combo(){
    local gen="$1" agent="$2" extra="$3" tag="$4"
    local lg="$LOGDIR/${tag}.log"
    if [ -f "$LOGDIR/${tag}.done" ]; then log "SKIP  $tag (already done)"; return; fi
    log "START $tag  (gen=$gen agent=$agent extra='$extra')"
    $PY -u train_abel.py \
        --gen "$gen" --agents $agent --action_space dynamic \
        --Ntrain $NTRAIN $extra \
        --early_stop_patience 8 --eval_subsample 150 \
        --base_seed $BASE_SEED --n_trials $NSEEDS --n_workers $NSEEDS --n_envs 1 \
        > "$lg" 2>&1
    local rc=$?
    log "END   $tag  (exit=$rc)"
    [ "$rc" -eq 0 ] && touch "$LOGDIR/${tag}.done"
}

PIDS=""
# Block until fewer than MAX_COMBOS combos are running (prunes dead PIDs in
# the current shell so PID reuse can't overcount). Counts combo-1 too.
wait_for_slot(){
    while : ; do
        local n=0 p alive=""
        for p in $PIDS; do
            if kill -0 "$p" 2>/dev/null; then n=$((n+1)); alive="$alive $p"; fi
        done
        PIDS="$alive"
        if kill -0 "$COMBO1_PID" 2>/dev/null; then n=$((n+1)); fi
        [ "$n" -lt "$MAX_COMBOS" ] && return
        sleep 60
    done
}

# 11 remaining combos: gen|agent|extra|tag
COMBOS="
abel_level3|ppo-tree||abel_level3__ppo-tree
abel_level3|ppo-tree|--use_relabel_constants|abel_level3__ppo-tree-rc
abel_level3|ppo-tree|--use_relabel_constants --use_success_replay|abel_level3__ppo-tree-rc-buf
poesia-full|ppo||poesia-full__ppo
poesia-full|ppo-tree||poesia-full__ppo-tree
poesia-full|ppo-tree|--use_relabel_constants|poesia-full__ppo-tree-rc
poesia-full|ppo-tree|--use_relabel_constants --use_success_replay|poesia-full__ppo-tree-rc-buf
abel_level4|ppo||abel_level4__ppo
abel_level4|ppo-tree||abel_level4__ppo-tree
abel_level4|ppo-tree|--use_relabel_constants|abel_level4__ppo-tree-rc
abel_level4|ppo-tree|--use_relabel_constants --use_success_replay|abel_level4__ppo-tree-rc-buf
"

log "v2 launcher start — 11 combos, MAX_COMBOS=$MAX_COMBOS, combo-1 PID=$COMBO1_PID"
while IFS='|' read -r gen agent extra tag; do
    [ -z "${gen:-}" ] && continue
    wait_for_slot
    run_combo "$gen" "$agent" "$extra" "$tag" &
    PIDS="$PIDS $!"
done <<< "$COMBOS"
wait
log "ALL REMAINING COMBOS DONE"
