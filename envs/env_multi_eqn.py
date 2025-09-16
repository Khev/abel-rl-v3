# envs/env_multi_eqn.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np
from sympy import sympify, symbols, simplify, Eq, solve, powdenest, ratsimp
import re
from gymnasium import spaces, Env
from operator import add, sub, mul, truediv
from utils.utils_env import *
from utils.utils_custom_functions import *
from collections import defaultdict, deque
#import faiss  # pip install faiss-cpu

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CoV placeholder op (we intercept this in step() to do a two-sided transform)
# ──────────────────────────────────────────────────────────────────────────────
def cov_action_placeholder(expr, term):
    # never used directly; step() intercepts and calls _apply_cov(lhs, rhs)
    return expr

operation_names[cov_action_placeholder] = 'cov'

a, b, c, x = symbols('a b c x')
def pi_cov_quadratic(main_eqn):
    poly = (simplify(main_eqn)).as_poly(x)
    if poly is None or poly.degree() != 2:
        return None
    A, B = poly.all_coeffs()[0], poly.all_coeffs()[1]
    return x - B/(2*A)


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")


class multiEqn(Env):
    """
    Environment for solving multiple equations using RL, with a simple curriculum
    that samples equations inversely proportional to how often they've been solved.

    Change-of-Variables (CoV) support (reusing x):
      - Pass use_cov=True and a pi_cov callable: pi_cov(main_eqn_expr) -> sub_expr or None
      - When triggered, we perform an in-place substitution  x := sub_expr
      - We also compute and push the inverse map inv_expr (x_prev in terms of new x)
      - On success, we unwind both relabel-constants and CoV stacks so the final
        solution is reported in the original variable.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 state_rep='integer_1d', 
                 normalize_rewards=True, 
                 verbose=False,
                 cache=False, 
                 level=4, 
                 gen='abel_level_2',
                 generalization=None,
                 sparse_rewards=False,
                 use_relabel_constants=False,
                 use_success_replay=True,
                 use_memory=False,
                 use_curriculum=True,
                 use_cov=False,
                 pi_cov = pi_cov_quadratic,
                 max_cov_apps = 1,
                 train_eqns=None,
                 ) -> None:
        super().__init__()

        # Static parts
        self.max_expr_length = 20
        self.max_steps = 5   # used to be 5
        self.action_dim = 50
        self.observation_dim = 2*self.max_expr_length + 1
        self.current_steps = 0

        # Rewards
        self.reward_solved = +100
        self.reward_invalid_equation = -100
        self.reward_illegal_action = -1
        self.reward_step = -1

        # Optimizing
        self.cache = cache
        if self.cache:
            self.action_cache = {}

        # Pars
        self.normalize_rewards = normalize_rewards
        self.state_rep = state_rep
        self.verbose = verbose
        self.generalization = generalization

        # Inductive biases
        self.sparse_reward = sparse_rewards
        self.use_curriculum = use_curriculum
        self.use_memory = use_memory  # backwards compatibility
        self.use_success_replay = use_success_replay
        self.use_relabel_constants = use_relabel_constants
        self.use_cov = bool(use_cov)
        self.pi_cov = pi_cov
        self.solve_var = symbols('x')           # current solve variable (stays 'x' with CoV reuse)
        self.cov_inv = []                       # stack of inverse maps (x_prev in terms of current x)
        self.main_eqn_original_cov = None       # remember original main_eqn before first CoV
        self.max_cov_apps = int(max_cov_apps)     # <-- and store it

        # For relabel constants
        self.map_constants = None
        self.map_constants_history = []
        self._timeout_count = 0
        self.main_eqn_original = None

        eqn_dirn = f"equation_templates"
        if train_eqns is None:
            self.train_eqns, self.test_eqns = load_train_test_equations(eqn_dirn, "", generalization=gen)
        else:
            self.train_eqns, self.test_eqns = train_eqns, train_eqns

        self.train_eqns_str = [str(eq) for eq in self.train_eqns]
        self.test_eqns_str = [str(eq) for eq in self.test_eqns]

        # Overwrite
        if gen == 'abel_level4':
            cutoff_train = 10000
            cutoff_test = cutoff_train // 10
            gens = ['abel_level1','abel_level2','abel_level3','abel_level4']
            train_eqns_all = []
            test_eqns_all = []
            for gen_name in gens:
                tr, te = load_train_test_equations(eqn_dirn, "", generalization=gen_name)
                if tr:
                    train_eqns_all.extend(list(tr))
                if te:
                    test_eqns_all.extend(list(te))
            self.train_eqns = train_eqns_all[:cutoff_train]
            self.test_eqns  = test_eqns_all[:cutoff_test]

            self.train_eqns_str = [str(eq) for eq in self.train_eqns][:cutoff_train]
            self.test_eqns_str  = [str(eq) for eq in self.test_eqns][:cutoff_test]

        # Tracking counts
        self.solve_counts = defaultdict(int)
        self.sample_counts = defaultdict(int)

        # Random initial eqn
        if generalization == 'poesia-full':
            self.main_eqn = self.sample_poesia_equation()
        else:
            eqn_str = np.random.choice(self.train_eqns_str)
            self.main_eqn = sympify(eqn_str)
        self.lhs = self.main_eqn
        self.rhs = 0
        self.x = symbols('x')

        # Make feature_dict, actions etc
        self.setup()

        # RL env setup
        self.state, _ = self.to_vec(self.lhs, self.rhs)
        self.action_space = spaces.Discrete(self.action_dim)

        if state_rep == 'integer_1d':
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float64)
        elif state_rep == 'integer_2d':
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim, 2), dtype=np.float64)
        elif state_rep == 'graph_integer_1d':
            self.observation_space = spaces.Dict({
                "node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float64),
                "edge_index": spaces.Box(low=0, high=self.observation_dim, shape=(2, 2*self.observation_dim), dtype=np.int32),
                "node_mask": spaces.Box(low=0, high=1, shape=(self.observation_dim,), dtype=np.int32),
                "edge_mask": spaces.Box(low=0, high=1, shape=(2*self.observation_dim,), dtype=np.int32),
            })
        elif state_rep == 'graph_integer_2d':
            self.observation_space = spaces.Dict({
                "node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim, 2), dtype=np.float64),
                "edge_index": spaces.Box(low=0, high=self.observation_dim, shape=(2, 2*self.observation_dim), dtype=np.int32),
                "node_mask": spaces.Box(low=0, high=1, shape=(self.observation_dim,), dtype=np.int32),
                "edge_mask": spaces.Box(low=0, high=1, shape=(2*self.observation_dim,), dtype=np.int32),
            })
        else:
            raise ValueError(f"Unsupported state representation: {state_rep}")

        self.traj_obs = []
        self.traj_act = []

    # ──────────────────────────────────────────────────────────────────────────
    # Build features + action set
    # ──────────────────────────────────────────────────────────────────────────
    def setup(self):
        # Build feature dict from all train/test eqns
        self.feature_dict = make_feature_dict_multi(
            self.train_eqns, self.test_eqns, self.state_rep
        )

        # Define some fixed 'global' transformations
        self.actions_fixed = [
            (custom_expand, None),
            (custom_collect, self.x),
            (custom_square, None),
            (custom_sqrt, None),
            (custom_log, None),
            (custom_exp, None),
            (custom_sin, None),
            (inverse_sin, None),
            (custom_cos, None),
            (inverse_cos, None),
            (mul, -1),
        ]

        # Add CoV placeholder if enabled
        if self.use_cov and callable(self.pi_cov):
            self._cov_op = cov_action_placeholder
            self.actions_fixed.append((self._cov_op, None))
            self.action_index_cov = len(self.actions_fixed) - 1  # Track index

        # Add relabel action if enabled
        if self.use_relabel_constants:
            def relabel_const_custom(lhs, rhs):
                return relabel_with_existing_constants(lhs, rhs, self.x, terms=[symbols('a'), symbols('b'), symbols('c')], strategy="partial", include_expr_constants=True)
            self.actions_fixed.append((relabel_const_custom, None))
            operation_names[relabel_const_custom] = 'relabel_const'
            self.action_index_relabel = len(self.actions_fixed) - 1  # Track index

        if self.cache:
            self.actions, self.action_mask = make_actions_cache(
                self.lhs, self.rhs, self.actions_fixed, 
                self.action_dim, self.action_cache
            )
        else:
            self.actions, self.action_mask = make_actions(self.lhs, self.rhs, self.actions_fixed, self.action_dim)

    # ──────────────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action_index: int):
        lhs_old, rhs_old, obs_old = self.lhs, self.rhs, self.state

        # (re)build current action list
        if self.cache:
            action_list, action_mask = make_actions_cache(lhs_old, rhs_old, self.actions_fixed, self.action_dim, self.action_cache)
        else:
            action_list, action_mask = make_actions(lhs_old, rhs_old, self.actions_fixed, self.action_dim)

        self.actions, self.action_mask = action_list, action_mask
        operation, term = action_list[action_index]

        if self.use_success_replay:
            self.traj_obs.append(obs_old.copy())
            self.traj_act.append(int(action_index))

        # Apply chosen action
        if self.use_relabel_constants and operation.__name__ == 'relabel_const_custom':
            lhs_new, rhs_new, map_constants = operation(lhs_old, rhs_old)
            self.map_constants = map_constants
            if self.main_eqn_original is None:
                self.main_eqn_original = self.main_eqn
            self.main_eqn = sympify(str(lhs_new) + ' - ' + str(rhs_new))
            self.map_constants_history.append(map_constants)

        elif self.use_cov and operation is getattr(self, '_cov_op', None):
            if len(self.cov_inv) < self.max_cov_apps:
                lhs_new, rhs_new = self._apply_cov(lhs_old, rhs_old)
                cov_blocked = False
            else:
                lhs_new, rhs_new = lhs_old, rhs_old
                cov_blocked = True
        else:
            lhs_new, rhs_new = operation(lhs_old, term), operation(rhs_old, term)

        obs_new, _ = self.to_vec(lhs_new, rhs_new)

        # Book keeping with variable-aware checks
        is_valid_eqn, lhs_new, rhs_new = self._check_valid_eqn_local(lhs_new, rhs_new)
        is_solved = self._check_eqn_solved_local(lhs_new, rhs_new)

        reward = self.find_reward(lhs_old, rhs_old, lhs_new, rhs_new, is_valid_eqn, is_solved)

        too_many_steps = (self.current_steps >= self.max_steps)
        terminated = bool(is_solved or too_many_steps or not is_valid_eqn)
        truncated = False

        if is_solved:
            # Unwind relabel-constants (if any)
            if self.map_constants_history:
                for m in reversed(self.map_constants_history):
                    lhs_new = lhs_new.subs(m)
                    rhs_new = rhs_new.subs(m)
                self.main_eqn = self.main_eqn_original

            # Unwind CoV inverse stack (compose from last to first)
            if self.cov_inv:
                rhs_unw = rhs_new
                for inv in reversed(self.cov_inv):
                    if inv is not None:
                        rhs_unw = simplify(inv.subs(self.solve_var, rhs_unw))
                lhs_new, rhs_new = self.solve_var, rhs_unw
                if self.main_eqn_original_cov is not None:
                    self.main_eqn = self.main_eqn_original_cov

            obs_new, _ = self.to_vec(lhs_new, rhs_new)
            eqn_str = str(self.main_eqn)
            self.solve_counts[eqn_str] += 1

        # update state
        self.lhs, self.rhs, self.state = lhs_new, rhs_new, obs_new
        self.current_steps += 1

        info = {
            "is_solved":      is_solved,
            "is_valid_eqn":   is_valid_eqn,
            "too_many_steps": too_many_steps,
            "lhs":            self.lhs,
            "rhs":            self.rhs,
            "action_taken":   f'{operation_names[operation]} {term}',
            "main_eqn":       self.main_eqn,
            "eqn_id":         str(self.main_eqn),
            "action_mask":    self.action_mask,
            "map_constants":  self.map_constants,
            "map_constants_history": list(self.map_constants_history),
            "solve_var":      self.solve_var,
            "cov_depth":      len(self.cov_inv),
            'cov_blocked':    'cov_blocked' in locals() and cov_blocked
        }

        if is_solved and self.use_success_replay:
            info["traj_obs"] = list(self.traj_obs)
            info["traj_act"] = list(self.traj_act)

        #self.verbose = True
        if self.verbose:
            print(f"{self.lhs} = {self.rhs}. (Operation, term): ({operation_names[operation]}, {term})")

        #if self.use_cov and action_index == self.action_index_cov:
        #    print(f"{self.lhs} = {self.rhs}. (Operation, term): ({operation_names[operation]}, {term})")

        if terminated or truncated:
            self.traj_obs = []
            self.traj_act = []

        return obs_new, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    # CoV: reuse x; push inverse mapping
    # ──────────────────────────────────────────────────────────────────────────

    # def _apply_cov(self, lhs, rhs):
    #     if not callable(self.pi_cov):
    #         return lhs, rhs

    #     print('line 1')

    #     v = self.solve_var
    #     # ⬇️ ensure SymPy objects
    #     lhs_s = sympify(lhs)
    #     rhs_s = sympify(rhs)
    #     main_s = sympify(self.main_eqn)

    #     sub_expr = self.pi_cov(main_s)
    #     if sub_expr is None:
    #         return lhs, rhs

    #     print('line 2')

    #     lhs2  = simplify(lhs_s.subs(v, sub_expr))
    #     rhs2  = simplify(rhs_s.subs(v, sub_expr))
    #     main2 = simplify(main_s.subs(v, sub_expr))

    #     if self.main_eqn_original_cov is None:
    #         self.main_eqn_original_cov = self.main_eqn
    #     self.main_eqn = main2

    #     print('line 3')

    #     inv_expr = None
    #     inv_expr = sympify(sub_expr)  # forward map: x_old := sub_expr(x_new)
    #     self.cov_inv.append(inv_expr)
    #     return lhs2, rhs2

    def _apply_cov(self, lhs, rhs, timeout_seconds=1):
        """
        Apply completion of the square with a timeout.
        
        Args:
            lhs: Left-hand side of the equation.
            rhs: Right-hand side of the equation.
            timeout_seconds: Maximum time (in seconds) to allow for computation.
        
        Returns:
            Tuple of (lhs, rhs) after CoV, or original (lhs, rhs) if timed out or invalid.
        """
        if not callable(getattr(self, 'pi_cov', None)):
            logger.warning("pi_cov is not callable")
            return lhs, rhs
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            v = self.solve_var
            # Ensure SymPy objects
            lhs_s = sympify(lhs)
            rhs_s = sympify(rhs)
            main_s = sympify(self.main_eqn)
            sub_expr = self.pi_cov(main_s)
            if sub_expr is None:
                logger.warning("pi_cov returned None")
                return lhs, rhs
            lhs2 = simplify(lhs_s.subs(v, sub_expr))
            rhs2 = simplify(rhs_s.subs(v, sub_expr))
            main2 = simplify(main_s.subs(v, sub_expr))
            if self.main_eqn_original_cov is None:
                self.main_eqn_original_cov = self.main_eqn
            self.main_eqn = main2
            inv_expr = sympify(sub_expr)  # Forward map: x_old := sub_expr(x_new)
            self.cov_inv.append(inv_expr)
            return lhs2, rhs2
        except TimeoutException:
            logger.warning(f"_apply_cov timed out after {timeout_seconds} seconds")
            return lhs, rhs
        finally:
            signal.alarm(0)  # Disable the alarm


    # ──────────────────────────────────────────────────────────────────────────
    # Variable-aware validity / solved checks (use self.solve_var)
    # ──────────────────────────────────────────────────────────────────────────
    def _check_valid_eqn_local(self, lhs, rhs):
        """Ensure the variable of interest is on one side; swap if needed."""
        v = self.solve_var
        lhs_has_v = getattr(lhs, 'has', lambda *_: False)(v)
        rhs_has_v = getattr(rhs, 'has', lambda *_: False)(v)

        if not lhs_has_v and not rhs_has_v:
            return False, lhs, rhs
        if not lhs_has_v and rhs_has_v:
            lhs, rhs = rhs, lhs
        return True, lhs, rhs

    def _check_eqn_solved_local(self, lhs, rhs):
        """(lhs, rhs) = (x, const-without-x) and verifies main_eqn.subs(x, rhs) == 0."""
        v = self.solve_var
        # require lhs == x (or Abs(x) could be allowed if desired)
        if lhs != v:
            return False
        if v in getattr(rhs, 'free_symbols', set()):
            return False

        # verify by substitution into *current* main_eqn
        sol = self.main_eqn.subs(v, rhs)
        if sol == 0 or getattr(sol, 'is_zero', False):
            return True
        if getattr(sol, 'expand', None) and sol.expand() == 0:
            return True
        if powdenest(sol, force=True) == 0:
            return True
        if ratsimp(sol) == 0:
            return True
        if simplify(sol) == 0:
            return True
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Sampling helpers / reset
    # ──────────────────────────────────────────────────────────────────────────
    def sample_poesia_equation(self, *, seed=None):
        if seed is not None:
            np.random.seed(seed)

        template = np.random.choice(self.templates)          # raw line
        placeholders = set(re.findall(r'-?\d+', template))   # e.g. "-2", "5"

        eq_str = template
        for ph in placeholders:
            rnd = np.random.randint(-10, 11)                # may be 0
            eq_str = eq_str.replace(ph, _int_to_symbol(rnd))

        # convert "ax" → "a*x", "bx" → "b*x"
        eq_str = re.sub(r'([A-Za-z])x\b', r'\1*x', eq_str)

        # "lhs = rhs"  →  "lhs - (rhs)"
        if '=' in eq_str:
            lhs, rhs = map(str.strip, eq_str.split('=', 1))
            eq_str = f'{lhs} - ({rhs})'

        return sympify(eq_str)

    def reset(self, seed=None, options=None):
        # Sample an equation
        if self.generalization == 'poesia-full':
            self.main_eqn = self.sample_poesia_equation()
        else:
            if options is None:
                if self.use_curriculum:
                    eqn_probs = np.array([1.0 / (1 + self.solve_counts[s]) for s in self.train_eqns_str], dtype=np.float64)
                    eqn_probs /= eqn_probs.sum()
                    chosen_eqn_str = np.random.choice(self.train_eqns_str, p=eqn_probs)
                    self.main_eqn = sympify(chosen_eqn_str)
                    self.sample_counts[chosen_eqn_str] += 1
                else:
                    chosen_eqn_str = np.random.choice(self.train_eqns_str)
                    self.main_eqn = sympify(chosen_eqn_str)
            elif options == 'train':
                chosen_eqn_str = np.random.choice(self.train_eqns_str)
                self.main_eqn = sympify(chosen_eqn_str)
            elif options == 'test':
                chosen_eqn_str = np.random.choice(self.test_eqns_str)
                self.main_eqn = sympify(chosen_eqn_str)

        # Reset CoV and relabel state
        self.solve_var = symbols('x')
        self.cov_inv = []
        self.main_eqn_original_cov = None
        self.map_constants = None
        self.map_constants_history = []
        self._timeout_count = 0
        self.main_eqn_original = None

        self.current_steps = 0
        self.lhs, self.rhs = self.main_eqn, 0
        obs, _ = self.to_vec(self.lhs, self.rhs)
        self.state = obs

        # Recompute actions, masks, etc.
        self.setup()

        self.traj_obs = []
        self.traj_act = []

        return obs, {}

    def to_vec(self, lhs, rhs):
        if self.state_rep == 'integer_1d':
            return integer_encoding_1d(lhs, rhs, self.feature_dict, self.max_expr_length)
        elif self.state_rep == 'integer_2d':
            return integer_encoding_2d(lhs, rhs, self.feature_dict, self.max_expr_length)
        elif self.state_rep == 'graph_integer_1d':
            return graph_encoding_1d(lhs, rhs, self.feature_dict, self.max_expr_length)  
        elif self.state_rep == 'graph_integer_2d':
            return graph_encoding(lhs, rhs, self.feature_dict, self.max_expr_length)  
        else:
            raise ValueError(f"Unknown state representation: {self.state_rep}")

    def find_reward(self, lhs_old, rhs_old, lhs_new, rhs_new, is_valid_eqn, is_solved):
        """
        Reward = 
          +100 if solved
          -100 if invalid eqn
          else ( oldComplexity - newComplexity )
               optionally normalized to [-1, 1].
        """
        if not is_valid_eqn:
            reward = self.reward_invalid_equation
        elif is_solved:
            reward = self.reward_solved
        else:
            if not self.sparse_reward:
                old_complex = get_complexity_expression(lhs_old) + get_complexity_expression(rhs_old)
                new_complex = get_complexity_expression(lhs_new) + get_complexity_expression(rhs_new)
                reward = old_complex - new_complex
            else:
                reward = 0

        # Add on step
        if not self.sparse_reward:
            reward += self.reward_step

        if self.normalize_rewards:
            # rescale reward to [-1, 1] with min=-100, max=+100
            min_r, max_r = self.reward_invalid_equation, self.reward_solved
            reward = 2.0 * (reward - min_r) / float(max_r - min_r) - 1.0            

        return reward

    def render(self, mode="human"):
        print(f"{self.lhs} = {self.rhs}")

    def get_valid_action_mask(self):
        # ensure dtype is strictly boolean for ActionMasker / MaskablePPO
        return np.asarray(self.action_mask, dtype=np.bool_)

    def set_equation(self, main_eqn):
        self.main_eqn, self.lhs, self.rhs = main_eqn, main_eqn, 0
        # Reset CoV + relabel state
        self.solve_var = symbols('x')
        self.cov_inv = []
        self.main_eqn_original_cov = None
        self.main_eqn_original = None
        self.map_constants = None
        self.map_constants_history = []
        obs, _ = self.to_vec(self.lhs, self.rhs)
        self.state = obs


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test: run `python envs/env_multi_eqn.py`
# Demonstrates: CoV (complete-the-square), algebraic isolation, solved check
# ──────────────────────────────────────────────────────────────────────────────
# if __name__ == '__main__':
#     from sympy import symbols

#     a, b, c, x = symbols('a b c x')

#     # CoV policy: complete-the-square when quadratic in x:
#     # returns sub_expr to assign into x (reuse x):  x := x - B/(2A)
#     def pi_cov_quadratic(main_eqn):
#         poly = (simplify(main_eqn)).as_poly(x)
#         if poly is None or poly.degree() != 2:
#             return None
#         A, B = poly.all_coeffs()[0], poly.all_coeffs()[1]
#         return x - B/(2*A)

#     env = multiEqn(
#         gen='abel_level1',
#         state_rep='graph_integer_1d',
#         use_cov=True,
#         pi_cov=pi_cov_quadratic,
#         use_relabel_constants=True,
#         use_curriculum=False,   # deterministic for the demo
#     )

#     env.reset()
#     env.set_equation(sympify("a*x**2 + b*x + c"))

#     print("\n[Before CoV]")
#     print(" main_eqn:", env.main_eqn)
#     print(" solve_var:", env.solve_var)
#     print(" cov_depth:", len(env.cov_inv))

#     # find cov action index
#     cov_idx = None
#     for i, (op, term) in enumerate(env.actions):
#         if op is getattr(env, '_cov_op', None):
#             cov_idx = i
#             break
#     if cov_idx is None:
#         raise RuntimeError("CoV action not found; ensure use_cov=True and pi_cov provided.")

#     # 1) Apply CoV: x := x - b/(2a)
#     obs, rew, terminated, truncated, info = env.step(cov_idx)

#     print("\n[After CoV]")
#     print(" main_eqn:", env.main_eqn)    # should be a*x**2 + c - b**2/(4*a)
#     print(" solve_var:", info.get("solve_var"))
#     print(" cov_depth:", info.get("cov_depth"))
#     print(" is_valid_eqn:", info.get("is_valid_eqn"))
#     print(" is_solved:", info.get("is_solved"))

#     # Helper to find index of (op, term) in current env.actions
#     # 2) Recompute actions on-the-fly (only if you really need a fresh list)
#     def find_action_recompute(env, op_fn, term_value=None):
#         actions_temp, _mask = make_actions(env.lhs, env.rhs, env.actions_fixed, env.action_dim)
#         for j, (opj, tj) in enumerate(actions_temp):
#             same_op = (opj is op_fn)
#             same_term = True
#             if term_value is not None:
#                 if hasattr(tj, "equals"):
#                     same_term = bool(tj.equals(term_value))
#                 else:
#                     same_term = (str(tj) == str(term_value))
#             if same_op and same_term:
#                 return j
#         return None


#     # Refresh actions after CoV
#     env.actions, env.action_mask = make_actions(env.lhs, env.rhs, env.actions_fixed, env.action_dim)

#     # 2) Add b**2/(4*a) to both sides: a*x**2 + c  =  b**2/(4*a)
#     add_b2_over_4a = find_action_recompute(add, sympify("b**2/(4*a)"))
#     assert add_b2_over_4a is not None, "add (b**2/(4*a)) not available"
#     obs, rew, terminated, truncated, info = env.step(add_b2_over_4a)

#     # 3) Subtract c from both sides: a*x**2  =  b**2/(4*a) - c
#     sub_c = find_action_recompute(sub, c)
#     assert sub_c is not None, "sub c not available"
#     obs, rew, terminated, truncated, info = env.step(sub_c)

#     # 4) Divide both sides by a: x**2  =  b**2/(4*a**2) - c/a
#     div_a = find_action_recompute(truediv, a)
#     assert div_a is not None, "divide by a not available"
#     obs, rew, terminated, truncated, info = env.step(div_a)

#     # 5) Take sqrt on both sides: x = sqrt(b**2/(4*a**2) - c/a)
#     do_sqrt = find_action_recompute(custom_sqrt, None)
#     assert do_sqrt is not None, "sqrt op not available"
#     obs, rew, terminated, truncated, info = env.step(do_sqrt)

#     print("\n[After algebraic isolation]")
#     print(" lhs:", info["lhs"])
#     print(" rhs:", info["rhs"])
#     print(" is_valid_eqn:", info["is_valid_eqn"])
#     print(" is_solved:", info["is_solved"])

#     # If solved, env has already unwound CoV (and relabels) in info["lhs"], info["rhs"]
#     # For symbolic a,b,c this should pass the substitution check in _check_eqn_solved_local.
