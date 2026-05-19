#!/usr/bin/env python3
"""Trace seed10000 on one exponential equation to confirm the 'CoV blocked' hypothesis."""
import sys
try: sys.set_int_max_str_digits(0)
except: pass
import sympy as sp
from sb3_contrib import MaskablePPO
from envs.env_multi_eqn import multiEqn
from utils.utils_env import make_actions
from utils.utils_custom_functions import operation_names

CKPT = "data/dynamic_actions/use_relabel_constants/use_buffer/mixed_v2_easy_hidden_dim256_nenvs1/ppo-tree/seed10000/checkpoints/latest.zip"

env = multiEqn(gen="mixed_v2_easy", state_rep="graph_integer_1d",
               use_cov=True, use_relabel_constants=True, use_success_replay=True)
model = MaskablePPO.load(CKPT, env=env, device="cpu")

for eqn_str in ["-d + 2*e*exp(2*x) + 2*f*exp(-2*x)",
                "4*a - 2*e*exp(-2*x) + 4*f*exp(2*x)"]:
    print(f"\n=== TRACE: {eqn_str} ===")
    eqn = sp.sympify(eqn_str)
    env.reset(seed=0)
    env.main_eqn = eqn
    env.lhs = eqn
    env.rhs = 0
    env.setup()
    env.state, _ = env.to_vec(env.lhs, env.rhs)

    for step in range(10):
        mask = env.get_valid_action_mask()
        action, _ = model.predict(env.state, deterministic=True, action_masks=mask)
        action = int(action)
        action_list, _ = make_actions(env.lhs, env.rhs, env.actions_fixed, env.action_dim)
        op, term = action_list[action]
        op_name = operation_names.get(op, op.__name__ if hasattr(op, "__name__") else str(op))
        # Track cov budget
        cov_used = len(env.cov_inv)
        print(f"  step {step+1}: action={op_name}({term})  cov_used={cov_used}/{env.max_cov_apps}")
        obs, r, term_done, trunc, info = env.step(action)
        print(f"     -> lhs={env.lhs}  rhs={env.rhs}  solved={info.get('is_solved')}")
        if term_done:
            break
