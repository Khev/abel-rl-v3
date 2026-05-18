"""Wrap a trained π_cov PPO model into a callable matching multiEqn's pi_cov(main_eqn) -> sub_expr contract."""
import sympy as sp
from pathlib import Path
from operator import add, sub, mul, truediv

from stable_baselines3 import PPO
from envs.env_cov import covEnv


_x = sp.symbols("x")
_OPS = {"ADD": add, "SUB": sub, "MUL": mul, "DIV": truediv, "IDENTITY": lambda a, b: a}


def make_pi_cov_learned(
    model_path,
    dataset_path,
    *,
    term_bank=None,
    max_depth=3,
    state_rep="graph_integer_1d",
    hist_len=10,
    device="cpu",
    deterministic=True,
):
    """Return a callable pi_cov(main_eqn) -> sub_expr.

    Loads a trained PPO π_cov model and a placeholder covEnv (needed only for
    the action set and encoding). On each call:
      1. Pin the env to the given main_eqn.
      2. Roll out the model greedily for up to max_depth + 1 steps.
      3. Compose f(x) = base_op(x, cov_inner) and return it.
      4. Return None if the result is degenerate (cov == 0 or NaN).
    """
    if term_bank is None:
        term_bank = [sp.sympify(t) for t in "a,b,c,d,e,2,3,4".split(",")]
    elif isinstance(term_bank, str):
        term_bank = [sp.sympify(t) for t in term_bank.split(",")]

    # Placeholder env — main_eqn gets overwritten per call
    env = covEnv(
        main_eqn=sp.sympify("a*x + b"),
        term_bank=term_bank,
        max_depth=max_depth,
        step_penalty=0.1,
        state_rep=state_rep,
        hist_len=hist_len,
        multi_eqn=True,
        use_curriculum=False,
        dataset_path=str(dataset_path),
    )
    model = PPO.load(str(model_path), env=env, device=device, print_system_info=False)

    def pi_cov(main_eqn):
        """Predict a CoV substitution f(x) for main_eqn. Returns sympy Expr or None."""
        try:
            obs, _ = env.reset()
            env.main_eqn = sp.sympify(main_eqn)
            env.base_cmplx = env.__class__.__mro__[0].__dict__  # placeholder, overwritten below
            from envs.env_cov import C as _C
            env.base_cmplx = _C(env.main_eqn)
            env.obs = env.to_vec(env.main_eqn, 0)[0]
            obs = env._augment_obs(env.obs)

            base_op = None
            cov_inner = sp.Integer(0)
            for _ in range(max_depth + 1):
                action, _ = model.predict(obs, deterministic=deterministic)
                op, tau = env.actions[int(action)]
                obs, _, terminated, _, info = env.step(int(action))
                if op == "STOP" or terminated:
                    # env composed it for us; pull from info
                    if "cov" in info:
                        f = sp.sympify(info["cov"])
                        if f == 0:
                            return None
                        return f
                    break
            # Fallback: compose manually if env didn't expose it
            if base_op is None:
                return None
            return _OPS[base_op](_x, cov_inner)
        except Exception:
            return None

    return pi_cov
