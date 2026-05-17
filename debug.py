from sympy import symbols, sympify, simplify
from envs.env_multi_eqn import multiEqn
from utils.utils_env import make_actions
from operator import add, sub, mul, truediv
from utils.utils_custom_functions import custom_sqrt

a, b, c, x = symbols('a b c x')

# CoV policy: complete-the-square when quadratic in x:
# returns sub_expr to assign into x (reuse x):  x := x - B/(2A)
def pi_cov_quadratic(main_eqn):
    poly = (simplify(main_eqn)).as_poly(x)
    if poly is None or poly.degree() != 2:
        return None
    A, B = poly.all_coeffs()[0], poly.all_coeffs()[1]
    return x - B/(2*A)

env = multiEqn(
    gen='abel_level1',
    state_rep='graph_integer_1d',
    use_cov=True,
    pi_cov=pi_cov_quadratic,
    use_relabel_constants=True,
    use_curriculum=False,   # deterministic for the demo
)

env.reset()
env.set_equation(sympify("a*x**2 + b*x + c"))

manual_solve = False
if manual_solve:
    print("\n[Before CoV]")
    print(" main_eqn:", env.main_eqn)
    print(" solve_var:", env.solve_var)
    print(" cov_depth:", len(env.cov_inv))

    # find cov action index
    cov_idx = None
    for i, (op, term) in enumerate(env.actions):
        if op is getattr(env, '_cov_op', None):
            cov_idx = i
            break
    if cov_idx is None:
        raise RuntimeError("CoV action not found; ensure use_cov=True and pi_cov provided.")

    # 1) Apply CoV: x := x - b/(2a)
    obs, rew, terminated, truncated, info = env.step(cov_idx)

    print("\n[After CoV]")
    print(" main_eqn:", env.main_eqn)    # should be a*x**2 + c - b**2/(4*a)
    print(" solve_var:", info.get("solve_var"))
    print(" cov_depth:", info.get("cov_depth"))
    print(" is_valid_eqn:", info.get("is_valid_eqn"))
    print(" is_solved:", info.get("is_solved"))

    # Helper to find index of (op, term) in current env.actions
    # 2) Recompute actions on-the-fly (only if you really need a fresh list)
    def find_action_recompute(env, op_fn, term_value=None):
        actions_temp, _mask = make_actions(env.lhs, env.rhs, env.actions_fixed, env.action_dim)
        for j, (opj, tj) in enumerate(actions_temp):
            same_op = (opj is op_fn)
            same_term = True
            if term_value is not None:
                if hasattr(tj, "equals"):
                    same_term = bool(tj.equals(term_value))
                else:
                    same_term = (str(tj) == str(term_value))
            if same_op and same_term:
                return j
        return None


    # 2) Add b**2/(4*a) to both sides: a*x**2 + c  =  b**2/(4*a)
    add_b2_over_4a = find_action_recompute(env, sub, term_value=("-b**2/(4*a)"))
    assert add_b2_over_4a is not None, "add (b**2/(4*a)) not available"
    obs, rew, terminated, truncated, info = env.step(add_b2_over_4a)
    print(f'{env.lhs} = {env.rhs}')

    # 3) Subtract c from both sides: a*x**2  =  b**2/(4*a) - c
    sub_c = find_action_recompute(env, sub, term_value=c)
    assert sub_c is not None, "sub c not available"
    obs, rew, terminated, truncated, info = env.step(sub_c)
    print(f'{env.lhs} = {env.rhs}')

    # 4) Divide both sides by a: x**2  =  b**2/(4*a**2) - c/a
    div_a = find_action_recompute(env, truediv, term_value = a)
    assert div_a is not None, "divide by a not available"
    obs, rew, terminated, truncated, info = env.step(div_a)
    print(f'{env.lhs} = {env.rhs}')

    # 5) Take sqrt on both sides: x = sqrt(b**2/(4*a**2) - c/a)
    do_sqrt = find_action_recompute(env, custom_sqrt, term_value=None)
    assert do_sqrt is not None, "sqrt op not available"
    obs, rew, terminated, truncated, info = env.step(do_sqrt)
    print(f'{env.lhs} = {env.rhs}')

    print("\n[After algebraic isolation]")
    print(" lhs:", info["lhs"])
    print(" rhs:", info["rhs"])
    print(" is_valid_eqn:", info["is_valid_eqn"])
    print(" is_solved:", info["is_solved"])

