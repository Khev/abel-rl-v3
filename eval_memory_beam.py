#!/usr/bin/env python3
"""Decoder-level solved-state cache (Phase 1 of memory_nn_plan.md).

Pipeline:
  1. Load a trained checkpoint.
  2. Build a solved-state cache by greedy-solving every TRAIN equation:
     for each state on a successful trace, store the remaining action sequence
     keyed by the canonical form of that state.
  3. Evaluate the TEST set under three decoders:
       (a) plain beam  (lambda=0, no cache)
       (b) value beam  (lambda=1, no cache)
       (c) memory beam (lambda=1, with cache lookup at each expansion)
  4. Report solve counts and how often cache hits drove the win.

The cache is built ONLY from training data. Test eqns benefit only when
their canonical form (after the policy applies relabel_const etc.) matches
a known training canonical. This is fair generalization-via-canonicalization,
not memorization.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass

import numpy as np
import sympy as sp
import torch
from sb3_contrib import MaskablePPO

from envs.env_multi_eqn import multiEqn
from train_abel import (
    _start_on_eqn, _snapshot_env, _restore_env,
    _policy_action_probs, _policy_value, _state_complexity,
    beam_solve_one,  # we'll bypass and write a memory-aware variant
    EVAL_TIMEOUT_DET, EvalTimeout, time_limit,
)
from utils.utils_env import make_actions
from heapq import nlargest


def canonical_key(env):
    """String key for cache lookup. Uses sp.expand for canonical polynomial form."""
    u = getattr(env, "unwrapped", env)
    try:
        lhs_c = sp.expand(u.lhs)
        rhs_c = sp.expand(u.rhs)
    except Exception:
        lhs_c, rhs_c = u.lhs, u.rhs
    # Include cov-depth and relabel-depth so different canonical "contexts"
    # don't collide (matches the dedup in beam_solve_one).
    cov_d = len(getattr(u, "cov_inv", []) or [])
    rel_d = len(getattr(u, "map_constants_history", []) or [])
    return (str(lhs_c), str(rhs_c), cov_d, rel_d)


def greedy_trace_with_states(env, model, eqn, max_steps=10, per_eqn_seconds=1.0):
    """Run greedy on eqn, returning (solved, list_of_state_keys, list_of_actions).
    The Nth state_key corresponds to the env BEFORE the Nth action."""
    try:
        with time_limit(per_eqn_seconds):
            _start_on_eqn(env, eqn)
            keys = [canonical_key(env)]
            actions = []
            info = {}
            for _ in range(max_steps):
                obs = getattr(env, "unwrapped", env).state
                mask = env.get_valid_action_mask() if hasattr(env, "get_valid_action_mask") else None
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                action = int(action)
                obs, _, terminated, truncated, info = env.step(action)
                actions.append(action)
                keys.append(canonical_key(env))
                if info.get("is_solved"):
                    return True, keys, actions
                if terminated or truncated:
                    break
            return False, keys, actions
    except (EvalTimeout, Exception):
        return False, [], []


def build_cache(env, model, train_eqns, max_steps=10, verbose=False):
    """Greedy-rollout every train eqn; for each state on a successful trace,
    store the remaining action sequence under its canonical key."""
    cache = {}
    n_solved = 0
    for i, eqn in enumerate(train_eqns):
        solved, keys, actions = greedy_trace_with_states(env, model, eqn, max_steps=max_steps)
        if solved:
            n_solved += 1
            # For each prefix-state on the trace, store the suffix-actions
            for j, k in enumerate(keys[:-1]):
                suffix = tuple(actions[j:])
                if k not in cache or len(suffix) < len(cache[k]):
                    cache[k] = suffix
        if verbose and (i + 1) % 100 == 0:
            print(f"  build_cache: {i+1}/{len(train_eqns)} done, cache size {len(cache)}, solved {n_solved}")
    return cache, n_solved


def replay_cached(env, action_seq, max_steps=10):
    """Try to replay action_seq from current env state. Returns True if solved."""
    for a in action_seq[:max_steps]:
        try:
            obs, _, terminated, truncated, info = env.step(int(a))
            if info.get("is_solved"):
                return True
            if terminated or truncated:
                return False
        except Exception:
            return False
    return False


def memory_beam_solve_one(model, env, eqn, *, beam_width=5, topk_per_node=5,
                         max_steps=10, per_eqn_seconds=EVAL_TIMEOUT_DET,
                         beam_lambda=1.0, cache=None):
    """Beam search with cache lookup. At each expansion, check the canonical
    key of the resulting state; if it's in `cache`, attempt to replay the
    stored action_seq and finalize on success.
    """
    cache_hit_count = 0
    try:
        with time_limit(per_eqn_seconds):
            obs0 = _start_on_eqn(env, eqn)
            base = _snapshot_env(env)
            beam = [(0.0, base, obs0, False, 0)]
            for depth in range(max_steps):
                new_beam = []
                for score, snap, obs, solved, dp in beam:
                    if solved:
                        new_beam.append((score, snap, obs, True, dp))
                        continue
                    _restore_env(env, snap)
                    obs = getattr(getattr(env, "unwrapped", env), "state", obs)
                    mask = env.get_valid_action_mask() if hasattr(env, "get_valid_action_mask") else None
                    probs = _policy_action_probs(model, obs, action_mask=mask)
                    if probs.ndim != 1: probs = probs.reshape(-1)
                    if probs.sum() <= 1e-12: continue
                    if topk_per_node < len(probs):
                        top_idx = np.argpartition(probs, -topk_per_node)[-topk_per_node:]
                        top_idx = top_idx[np.argsort(-probs[top_idx])]
                    else:
                        top_idx = np.argsort(-probs)
                    for a in top_idx:
                        if probs[a] <= 0.0: continue
                        _restore_env(env, snap)
                        obs_next, _, terminated, truncated, info = env.step(int(a))
                        solved_next = bool(info.get("is_solved", False))

                        # === MEMORY LOOKUP ===
                        if not solved_next and cache is not None:
                            key = canonical_key(env)
                            if key in cache:
                                cache_hit_count += 1
                                # Try to replay the cached action sequence
                                snap_before_replay = _snapshot_env(env)
                                if replay_cached(env, cache[key], max_steps=max_steps - dp - 1):
                                    return True, cache_hit_count
                                # Restore if replay failed
                                _restore_env(env, snap_before_replay)
                        # === end memory lookup ===

                        log_pi = float(np.log(probs[a] + 1e-12))
                        score_next = score + log_pi
                        if beam_lambda != 0.0 and not solved_next:
                            try:
                                v = _policy_value(model, obs_next)
                                score_next += beam_lambda * v
                            except Exception:
                                pass
                        new_beam.append((score_next, _snapshot_env(env), obs_next, solved_next, dp + 1))
                if not new_beam:
                    return False, cache_hit_count
                # Dedup by canonical key + same context
                seen = {}
                for i, (sc, snp, ob, sol, dp_) in enumerate(new_beam):
                    if sol:
                        key2 = (id(snp), True)
                    else:
                        try:
                            lhs_s = str(sp.expand(snp.get("lhs")))
                            rhs_s = str(sp.expand(snp.get("rhs")))
                        except Exception:
                            lhs_s = str(snp.get("lhs")); rhs_s = str(snp.get("rhs"))
                        cov_d = len(snp.get("cov_inv") or [])
                        rel_d = len(snp.get("map_constants_history") or [])
                        key2 = (lhs_s, rhs_s, cov_d, rel_d)
                    if key2 not in seen or new_beam[seen[key2]][0] < sc:
                        seen[key2] = i
                deduped = [new_beam[i] for i in seen.values()]
                beam = nlargest(beam_width, deduped, key=lambda x: x[0])
                if any(s for _, _, _, s, _ in beam):
                    return True, cache_hit_count
            return any(s for _, _, _, s, _ in beam), cache_hit_count
    except EvalTimeout:
        return False, cache_hit_count
    except Exception:
        return False, cache_hit_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gen", default="mixed_v2_easy")
    p.add_argument("--beam_width", type=int, default=5)
    p.add_argument("--topk_per_node", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--per_eqn_seconds", type=float, default=1.0)
    args = p.parse_args()

    print(f"Loading {args.ckpt}")
    env = multiEqn(
        gen=args.gen,
        state_rep="graph_integer_1d",
        use_cov=True,
        use_relabel_constants=True,
        use_success_replay=True,
        use_cbrt=False,  # match legacy ckpts
    )
    model = MaskablePPO.load(args.ckpt, env=env, device="cpu")

    # Load train and test eqns
    train_eqns = env.train_eqns if hasattr(env, "train_eqns") else []
    test_eqns = env.test_eqns if hasattr(env, "test_eqns") else []
    print(f"Train set: {len(train_eqns)} eqns,  Test set: {len(test_eqns)} eqns")

    # Build cache from train
    print("\nBuilding cache from train successes...")
    cache, n_train_solved = build_cache(env, model, train_eqns,
                                         max_steps=args.max_steps, verbose=True)
    print(f"  train solved (greedy): {n_train_solved}/{len(train_eqns)}")
    print(f"  cache entries (unique canonical keys): {len(cache)}\n")

    # Eval test under three decoders
    print("Evaluating test set...")
    plain_solved = 0
    value_solved = 0
    mem_solved = 0
    total_hits = 0
    for i, eqn in enumerate(test_eqns):
        # Plain beam
        ok_plain = beam_solve_one(model, env, eqn, beam_width=args.beam_width,
                                  topk_per_node=args.topk_per_node,
                                  max_steps=args.max_steps,
                                  per_eqn_seconds=args.per_eqn_seconds, beam_lambda=0.0)
        plain_solved += int(ok_plain)
        # Value beam
        ok_value = beam_solve_one(model, env, eqn, beam_width=args.beam_width,
                                  topk_per_node=args.topk_per_node,
                                  max_steps=args.max_steps,
                                  per_eqn_seconds=args.per_eqn_seconds, beam_lambda=1.0)
        value_solved += int(ok_value)
        # Memory beam
        ok_mem, hits = memory_beam_solve_one(model, env, eqn,
                                             beam_width=args.beam_width,
                                             topk_per_node=args.topk_per_node,
                                             max_steps=args.max_steps,
                                             per_eqn_seconds=args.per_eqn_seconds,
                                             beam_lambda=1.0,
                                             cache=cache)
        mem_solved += int(ok_mem)
        total_hits += hits

    n = len(test_eqns)
    print(f"\n=== Results on test set ({n} eqns) ===")
    print(f"  plain beam (lambda=0):              {plain_solved}/{n}  ({100*plain_solved/n:.1f}%)")
    print(f"  value-guided beam (lambda=1):       {value_solved}/{n}  ({100*value_solved/n:.1f}%)")
    print(f"  value beam + memory cache:          {mem_solved}/{n}  ({100*mem_solved/n:.1f}%)")
    print(f"\nDeltas:")
    print(f"  memory vs value: {mem_solved - value_solved:+d}")
    print(f"  memory vs plain: {mem_solved - plain_solved:+d}")
    print(f"  total cache hits attempted: {total_hits}")


if __name__ == "__main__":
    main()
