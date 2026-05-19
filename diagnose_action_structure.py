#!/usr/bin/env python3
"""Quick diagnostic: is the flat policy already informally factorized by op?

For each test eqn, get the policy's action distribution over the 50 flat
slots. Categorize each action by its operation type. Then compute:
  - marginal_op[op] = sum of probs over all (op, *) actions
  - top_op = argmax(marginal_op)
  - p_top_op = marginal_op[top_op]
  - H_op = entropy of marginal_op (over the present ops; lower = more concentrated)
  - H_term_within_top_op = entropy over term-choices within the top op

If the policy is op-structured, p_top_op should be high (>0.5) and H_op
low. If actions are picked roughly uniformly across ops, factorization
would help.
"""
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
from utils.utils_env import make_actions
from utils.utils_custom_functions import operation_names

CKPT = "data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/seed9100/checkpoints/latest.zip"
TEST_FILE = "equation_templates/mixed_v2_easy/test_eqns.txt"


def load_eqns(path):
    with open(path) as f:
        return [sp.sympify(ln.strip()) for ln in f
                if ln.strip() and not ln.startswith("#")]


def op_label(op):
    """Bucket op functions into a small set of human-readable labels."""
    name = getattr(op, "__name__", str(op))
    return operation_names.get(op, name)


def get_action_probs(model, obs, action_mask):
    """Return masked & normalized prob distribution over the 50 action slots."""
    with torch.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_t)
        probs = dist.distribution.probs.squeeze(0).detach().cpu().numpy()
    probs = probs * np.asarray(action_mask, dtype=probs.dtype)
    s = probs.sum()
    return probs / s if s > 1e-12 else probs


def reset_to(env, eqn):
    env.reset(seed=0)
    env.main_eqn = eqn
    env.lhs = eqn
    env.rhs = 0
    env.setup()
    env.state, _ = env.to_vec(env.lhs, env.rhs)
    return env.state


def diagnose_eqn(env, model, eqn):
    """Return (per_op_marginal, p_top, top_label, h_op, h_term_in_top)."""
    obs = reset_to(env, eqn)
    mask = env.get_valid_action_mask()
    probs = get_action_probs(model, obs, mask)

    # Reconstruct the action list to know each slot's (op, term)
    action_list, _ = make_actions(env.lhs, env.rhs, env.actions_fixed, env.action_dim)

    by_op = defaultdict(float)
    terms_under_top = defaultdict(float)
    for i, (op, term) in enumerate(action_list):
        if probs[i] > 0:
            lbl = op_label(op)
            by_op[lbl] += float(probs[i])

    if not by_op:
        return {}, 0.0, None, 0.0, 0.0

    top_op = max(by_op, key=by_op.get)
    p_top = by_op[top_op]

    # Entropy over the marginal op distribution
    vals = np.array(list(by_op.values()))
    vals = vals / vals.sum()
    h_op = -float((vals * np.log(vals + 1e-12)).sum())

    # Entropy over the terms WITHIN the top op
    term_dist = []
    for i, (op, term) in enumerate(action_list):
        if op_label(op) == top_op and probs[i] > 0:
            term_dist.append(float(probs[i]))
    if term_dist:
        td = np.array(term_dist)
        td = td / td.sum()
        h_term = -float((td * np.log(td + 1e-12)).sum())
    else:
        h_term = 0.0

    return dict(by_op), p_top, top_op, h_op, h_term


def main():
    env = multiEqn(
        gen="mixed_v2_easy",
        state_rep="graph_integer_1d",
        use_cov=True,
        use_relabel_constants=True,
        use_success_replay=True,
        use_cbrt=False,  # match the trained ckpt's action ordering
    )
    print(f"Loading {CKPT}")
    model = MaskablePPO.load(CKPT, env=env, device="cpu")

    test = load_eqns(TEST_FILE)
    print(f"Test set: {len(test)} eqns\n")

    p_tops = []
    h_ops = []
    h_terms = []
    top_op_counter = Counter()
    op_marginals_all = defaultdict(list)
    for eqn in test:
        try:
            by_op, p_top, top, h_op, h_term = diagnose_eqn(env, model, eqn)
        except Exception:
            continue
        if top is None:
            continue
        p_tops.append(p_top)
        h_ops.append(h_op)
        h_terms.append(h_term)
        top_op_counter[top] += 1
        for k, v in by_op.items():
            op_marginals_all[k].append(v)

    print(f"=== Aggregate over {len(p_tops)} test eqns ===")
    print(f"Mean p(top_op):           {np.mean(p_tops):.3f}  (1.0 = perfectly op-concentrated; 1/N ~= flat)")
    print(f"Mean entropy over ops:    {np.mean(h_ops):.3f}  (0 = single op; log(N_ops) ~= flat)")
    print(f"Mean entropy over terms within top op: {np.mean(h_terms):.3f}")
    print()
    print(f"Most-common top_op across test eqns:")
    for op, cnt in top_op_counter.most_common(8):
        print(f"  {op:25s} {cnt:3d}  ({100*cnt/len(p_tops):.1f}%)")
    print()
    print(f"Mean marginal probability per op (averaged over eqns where the op appears):")
    ranked = sorted(op_marginals_all.items(), key=lambda kv: -np.mean(kv[1]))
    for op, vals in ranked[:10]:
        print(f"  {op:25s} mean_marginal={np.mean(vals):.3f}  n_eqns={len(vals)}")
    print()

    # Interpretation hint
    p_mean = np.mean(p_tops)
    if p_mean > 0.6:
        print(f"INTERPRETATION: p(top_op) mean = {p_mean:.2f} > 0.6.")
        print("  The policy is highly op-structured. Factorization would add little.")
    elif p_mean > 0.35:
        print(f"INTERPRETATION: p(top_op) mean = {p_mean:.2f}, moderate concentration.")
        print("  Factorization MIGHT help; worth a decode-time test.")
    else:
        print(f"INTERPRETATION: p(top_op) mean = {p_mean:.2f}, low concentration.")
        print("  Policy spreads probability across ops; factorization likely helps.")


if __name__ == "__main__":
    main()
