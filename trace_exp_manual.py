#!/usr/bin/env python3
"""Manual trace: try the textbook recipe for a*exp(x) + b*exp(-x) + c = 0.
If this fails, the env is buggy. If it succeeds, training is the issue."""
import sys
try: sys.set_int_max_str_digits(0)
except: pass
import sympy as sp
from envs.env_multi_eqn import multiEqn

env = multiEqn(gen="exp_debug", state_rep="graph_integer_1d",
               use_cov=True, use_relabel_constants=True,
               use_success_replay=False, use_cbrt=True)
env.reset(seed=0)
env.main_eqn = sp.sympify("a*exp(x) + b*exp(-x) + c")
env.lhs = env.main_eqn
env.rhs = 0
env.setup()
env.state, _ = env.to_vec(env.lhs, env.rhs)
print(f"Start: {env.lhs} = {env.rhs}")
print(f"max_cov_apps = {env.max_cov_apps}")

# Action mapping for diagnostics
from utils.utils_env import make_actions
from utils.utils_custom_functions import operation_names

def find_action(env, op_name, term_match=None):
    """Find the action index for (op, term) in the current action list."""
    action_list, mask = make_actions(env.lhs, env.rhs, env.actions_fixed, env.action_dim)
    for i, (op, term) in enumerate(action_list):
        name = operation_names.get(op, op.__name__ if hasattr(op, '__name__') else str(op))
        if name == op_name:
            if term_match is None or str(term) == str(term_match):
                if mask[i]:
                    return i, op, term
    return None, None, None

def step_diag(env, op_name, term_match=None, label=""):
    idx, op, term = find_action(env, op_name, term_match)
    if idx is None:
        print(f"  [FAIL] Could not find legal action {op_name} term={term_match}")
        return False
    print(f"  step: {label}  -> action #{idx}: ({op_name}, {term})")
    obs, r, term_done, _, info = env.step(idx)
    print(f"     -> lhs = {env.lhs}")
    print(f"     -> rhs = {env.rhs}")
    print(f"     -> solved={info.get('is_solved')}  cov_used={len(env.cov_inv)}/{env.max_cov_apps}")
    return info.get('is_solved', False)

# Recipe attempt
print("\n--- Recipe attempt ---")
# Step 1: CoV
done = step_diag(env, "cov", label="CoV (should apply x -> log(x))")
if done: print("Already solved?!"); sys.exit(0)

# Step 2: multiply by x (the new variable)
done = step_diag(env, "multiply", term_match="x", label="mul x")
if done: print("solved"); sys.exit(0)

# Step 3: Apply CoV again to depress the quadratic
done = step_diag(env, "cov", label="CoV #2 (should depress quadratic)")
if done: print("solved"); sys.exit(0)

# Step 4: relabel
done = step_diag(env, "relabel_const", label="relabel")
if done: print("solved"); sys.exit(0)

# Step 5+: finish the depressed quadratic
done = step_diag(env, "subtract", term_match="a", label="sub a")
if done: print("SOLVED!"); sys.exit(0)
done = step_diag(env, "divide", term_match="b", label="div b")
if done: print("SOLVED!"); sys.exit(0)
done = step_diag(env, "sqrt", label="sqrt")
if done: print("🎯 SOLVED!"); sys.exit(0)

print("\n--- Inspecting actions available now ---")
action_list, mask = make_actions(env.lhs, env.rhs, env.actions_fixed, env.action_dim)
print(f"Legal actions ({mask.sum()}):")
for i, (op, term) in enumerate(action_list[:20]):
    if mask[i]:
        name = operation_names.get(op, getattr(op, '__name__', '?'))
        print(f"  #{i}: ({name}, {term})")
