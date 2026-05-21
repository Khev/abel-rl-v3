#!/usr/bin/env python3
"""Reproduce the --use_cov memory blow-up in isolation.

Drives multiEqn(mixed_v2_easy, use_cov=True) through many CoV-heavy episodes
(always take the CoV macroaction when it's legal) and tracks current RSS,
peak RSS, expression size, and SymPy cache size to localize the leak.

Self-kills at 8 GB so it cannot panic the machine.
"""
import sys, os, resource, random, gc, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass
import numpy as np
from sympy import count_ops
from envs.env_multi_eqn import multiEqn

PID = os.getpid()


def cur_rss_gb():
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(PID)])
    return int(out.strip()) / (1024 ** 2)        # ps rss is KB on macOS


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def sympy_cache_len():
    try:
        from sympy.core.cache import CACHE
        return sum(len(getattr(c, "cache_info", lambda: [0])().__class__ and c) if False else 1
                   for c in CACHE)
    except Exception:
        return -1


random.seed(0)
np.random.seed(0)

env = multiEqn(gen="mixed_v2_easy", state_rep="graph_integer_1d",
               use_cov=True, use_relabel_constants=True,
               use_success_replay=True, use_cbrt=True)
cov_idx = getattr(env, "action_index_cov", None)
max_steps = getattr(env, "max_steps", 20) or 20
print(f"cov_idx={cov_idx}  max_cov_apps={env.max_cov_apps}  max_steps={max_steps}", flush=True)
print(f"{'ep':>6} {'curRSS':>8} {'peakRSS':>8} {'ops(main)':>10} {'maxOps':>8} {'covEps':>7}", flush=True)

N_EP = 2000
max_ops = 0
cov_eps = 0
for ep in range(1, N_EP + 1):
    obs, info = env.reset()
    used_cov = False
    for _ in range(max_steps):
        try:
            mask = np.asarray(env.get_valid_action_mask(), dtype=bool)
        except Exception:
            break
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            break
        if cov_idx is not None and cov_idx < len(mask) and mask[cov_idx]:
            a = int(cov_idx)
            used_cov = True
        else:
            a = int(random.choice(valid))
        try:
            obs, r, term, trunc, info = env.step(a)
        except Exception as e:
            print(f"  ep{ep} step raised {type(e).__name__}: {e}", flush=True)
            break
        if term or trunc:
            break
    cov_eps += int(used_cov)
    try:
        ops = int(count_ops(env.main_eqn))
    except Exception:
        ops = -1
    max_ops = max(max_ops, ops)

    if ep % 25 == 0:
        gc.collect()
        cur = cur_rss_gb()
        print(f"{ep:6d} {cur:7.2f}G {peak_rss_gb():7.2f}G {ops:10d} {max_ops:8d} {cov_eps:7d}", flush=True)
        if cur > 8.0:
            print(f"SELF-KILL: RSS {cur:.2f} GB > 8 GB at episode {ep}", flush=True)
            break

print("DONE", flush=True)
