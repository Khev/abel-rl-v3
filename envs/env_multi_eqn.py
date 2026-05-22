# envs/env_multi_eqn.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np
from sympy import sympify, symbols, simplify, Eq, solve, powdenest, ratsimp
from sympy import symbols, simplify, exp, log, Wild, expand
import re
from gymnasium import spaces, Env
from operator import add, sub, mul, truediv
from utils.utils_env import *
from utils.utils_custom_functions import *
from collections import defaultdict, deque
from typing import Optional
#import faiss  # pip install faiss-cpu

import signal
import time
from sympy import count_ops

logger = logging.getLogger(__name__)

# Pre-check ceiling for the expand() in _apply_cov. If the raw lhs - rhs has
# more than this many ops, skip the expand-based current-form path (it could
# be slow) and fall back to the main_eqn pattern match. count_ops is cheap.
COV_EXPAND_OPS_LIMIT = 60

letter_map = {i: chr(ord('a') + i - 1) for i in range(1, 27)}

def _int_to_symbol(n: int) -> str:
    """
    0   → 'k'
    1   → 'a'
    …   → …
    -3  → '-c'
    10  → 'j'
    """
    if n == 0:
        return 'k'
    sign = '-' if n < 0 else ''
    n = abs(n)
    base = letter_map.get(n, f'c{n}')   # fallback name if |n|>26
    return f'{sign}{base}'

# ──────────────────────────────────────────────────────────────────────────────
# CoV placeholder op (we intercept this in step() to do a two-sided transform)
# ──────────────────────────────────────────────────────────────────────────────
def cov_action_placeholder(expr, term):
    # never used directly; step() intercepts and calls _apply_cov(lhs, rhs)
    return expr

operation_names[cov_action_placeholder] = 'cov'

a, b, c, x = symbols('a b c x')
def pi_cov_quadratic(main_eqn):
    poly = (expand(main_eqn)).as_poly(x)   # expand, not simplify: simplify can hang (trigsimp)
    if poly is None or poly.degree() != 2:
        return None
    A, B = poly.all_coeffs()[0], poly.all_coeffs()[1]
    return x - B/(2*A)


def pi_cov_general(main_eqn):
    """
    Return a change-of-variables sub-expression to assign into `x`
    (i.e., the env will do `x := <return value>`), choosing among:
      • Quadratic  ax^2+bx+c            →  x ↦ x - B/(2A)  (complete the square)
      • Cubic      ax^3+bx^2+cx+d       →  x ↦ x - B/(3A)  (depress the cubic)
      • Quartic    ax^4+bx^3+cx^2+dx+e  →  x ↦ x - B/(4A)  (depress the quartic)
      • Exponential a e^{k x} + b e^{-k x} + c →  x ↦ log(x)/k
        (so the equation becomes a*x + b/x + c = 0 in the new variable)

    Returns None if no supported pattern is detected.
    """
    # expand(), NOT simplify(): simplify() recurses into trigsimp/exptrigsimp
    # and can hang indefinitely on messy transformed equations -- stuck in
    # SymPy internals, past the SIGALRM step-timeout. expand() is safe and is
    # sufficient here: as_poly() and the exp Wild-match both work on the
    # expanded form. (This was the cause of open-equation runs producing no
    # output -- a worker invoking CoV would hang forever.)
    s = expand(main_eqn)

    # ----- Polynomial cases (deg = 2,3,4): use shift to kill the next-highest term
    poly = s.as_poly(x)
    if poly is not None:
        deg = poly.degree()
        if deg in (2, 3, 4):
            coeffs = poly.all_coeffs()
            A = coeffs[0]          # leading
            B = coeffs[1]          # next (x^(deg-1)) coefficient
            if A != 0:
                if deg == 2:       # quadratic: kill linear term
                    return x - B/(2*A)
                elif deg == 3:     # cubic: kill x^2 term
                    return x - B/(3*A)
                elif deg == 4:     # quartic: kill x^3 term
                    return x - B/(4*A)

    # ----- Exponential case: a*exp(k*x) + b*exp(-k*x) + c
    # Use Wilds to detect the structure and read off k
    aW = Wild('aW', exclude=[x])
    bW = Wild('bW', exclude=[x])
    cW = Wild('cW', exclude=[x])
    kW = Wild('kW', exclude=[x])

    # try exact 3-term form (up to simplification)
    m = s.match(aW*exp(kW*x) + bW*exp(-kW*x) + cW)
    if m and kW in m and m[kW] != 0:
        k = m[kW]
        # Substitute x := (1/k)*log(x). Then exp(k*x) → x and exp(-k*x) → 1/x.
        return log(x)/k

    # No supported pattern detected
    return None


class TimeoutException(BaseException):
    # Subclasses BaseException (not Exception) so the many `except Exception`
    # handlers on the CoV path cannot silently swallow a step() timeout --
    # only step()'s explicit `except TimeoutException` catches it.
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
                 cache=True,
                 level=4,
                 gen='abel_level_2',
                 sparse_rewards=False,
                 use_relabel_constants=False,
                 use_success_replay=True,
                 use_memory=False,
                 use_curriculum=True,
                 use_cov=False,
                 #pi_cov = pi_cov_quadratic,
                 pi_cov = pi_cov_general,
                 max_cov_apps = 3,  # was 1; bumped to allow nested CoV (e.g. exp -> quadratic depression). See trace_exp.py for the failure mode at max=1.
                 max_eqn_ops = 250,  # count_ops ceiling per step; over this the episode aborts (runaway-expression guard)
                 step_timeout: float = 0.5,  # abort slow sympy steps fast; 3.0 wasted ~6x compute per bad step (the straggler cause). The leak is fixed structurally (BaseException + no _apply_cov signal-cancel), not by this value.
                 train_eqns=None,
                 anti_loop_penalty: float = 0.0,
                 use_cbrt: bool = True,
                 ) -> None:
        super().__init__()

        # Static parts
        self.max_expr_length = 20
        self.max_steps = 10   # used to be 5
        self.action_dim = 50
        self.observation_dim = 2*self.max_expr_length + 1
        self.current_steps = 0
        self.gen = gen
        self.step_timeout = step_timeout
        self.anti_loop_penalty = float(anti_loop_penalty)
        self.last_action: Optional[int] = None
        self.use_cbrt = bool(use_cbrt)

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
        self.max_eqn_ops = int(max_eqn_ops)       # runaway-expression guard ceiling

        # For relabel constants
        self.map_constants = None
        self.map_constants_history = []
        self._timeout_count = 0
        self.main_eqn_original = None

        # Load in train/test equations
        eqn_dirn = f"equation_templates"
        if train_eqns is None:
            self.train_eqns, self.test_eqns = load_train_test_equations(eqn_dirn, "", generalization=gen)
        else:
            self.train_eqns, self.test_eqns = train_eqns, train_eqns

        self.train_eqns_str = [str(eq) for eq in self.train_eqns]
        self.test_eqns_str = [str(eq) for eq in self.test_eqns]

        if gen == 'poesia-full':
            TEMPLATE_PATH = "equation_templates/poesia/equations-ct.txt"
            def fetch_templates():
                with open(TEMPLATE_PATH, 'r') as f:
                    templates = [line.strip() for line in f if line.strip() and not line.startswith('!')]
                print(f"Loaded {len(templates)} templates from local file.")
                return templates
            self.templates = fetch_templates()  # Load templates once

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
        if self.gen == 'poesia-full':
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

        # Define some fixed 'global' transformations.
        # When use_cbrt=False we preserve the ORIGINAL action ordering exactly
        # so checkpoints trained before the cbrt feature still load with the
        # same action-index semantics.
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
        # Append cbrt at the END (not inserted mid-list) to keep old action
        # indices stable. Cubic class requires this to be solvable.
        if self.use_cbrt:
            self.actions_fixed.append((custom_cbrt, None))

        # Add CoV placeholder if enabled
        if self.use_cov and callable(self.pi_cov):
            self._cov_op = cov_action_placeholder
            self.actions_fixed.append((self._cov_op, None))
            self.action_index_cov = len(self.actions_fixed) - 1  # Track index
            #print('here')

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


    def sample_poesia_equation(self, *, seed=None):
        if seed is not None:
            np.random.seed(seed)

        template = np.random.choice(self.templates)          # raw line
        placeholders = set(re.findall(r'-?\d+', template))   # e.g. "-2", "5"

        eq_str = template
        for ph in placeholders:
            rnd = np.random.randint(-10, 11)                # may be 0
            eq_str = eq_str.replace(ph, _int_to_symbol(rnd))

        # convert "ax" → "a*x", "bx" → "b*x",  "0x" is impossible now
        eq_str = re.sub(r'([A-Za-z])x\b', r'\1*x', eq_str)

        # "lhs = rhs"  →  "lhs - (rhs)"
        if '=' in eq_str:
            lhs, rhs = map(str.strip, eq_str.split('=', 1))
            eq_str = f'{lhs} - ({rhs})'

        return sympify(eq_str)

    # ──────────────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────────────
    def step_base(self, action_index: int):
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
            # Was: self.main_eqn = sympify(str(lhs_new) + ' - ' + str(rhs_new))
            # The sympify(str(...)) roundtrip parses+reparses each side just to subtract.
            # Direct subtraction is identical and ~10x faster on relabel-heavy episodes.
            self.main_eqn = lhs_new - rhs_new
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

        # Runaway-expression guard. A degenerate action loop (e.g. alternating
        # multiply/expand, which the anti-loop penalty does not catch) can
        # compound lhs/rhs into a huge expression; the next unguarded SymPy op
        # (simplify in the solved-check, observation encoding) then recurses for
        # minutes and balloons RSS -- this is the --use_cov "memory leak".
        # count_ops is cheap and bounded: if we blow past the ceiling, discard
        # the monster and end the episode as an invalid step.
        eqn_blew_up = False
        try:
            if count_ops(lhs_new) + count_ops(rhs_new) > self.max_eqn_ops:
                eqn_blew_up = True
        except Exception:
            pass
        if eqn_blew_up:
            logger.warning(f"step: expression blow-up after "
                           f"{operation_names.get(operation, '?')} "
                           f"(> {self.max_eqn_ops} ops) -- aborting episode")
            lhs_new, rhs_new = lhs_old, rhs_old

        obs_new, _ = self.to_vec(lhs_new, rhs_new)

        # Book keeping with variable-aware checks
        is_valid_eqn, lhs_new, rhs_new = self._check_valid_eqn_local(lhs_new, rhs_new)
        is_solved = self._check_eqn_solved_local(lhs_new, rhs_new)
        if eqn_blew_up:
            is_valid_eqn, is_solved = False, False   # force episode end, invalid penalty

        reward = self.find_reward(lhs_old, rhs_old, lhs_new, rhs_new, is_valid_eqn, is_solved)

        # Anti-loop penalty: subtract α if this action repeats the previous one.
        # Targets the "REL REL REL..." failure mode (see diagnose_action_traces.py).
        if self.anti_loop_penalty > 0 and self.last_action is not None and action_index == self.last_action:
            reward -= self.anti_loop_penalty
        self.last_action = action_index

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
                        substituted = inv.subs(self.solve_var, rhs_unw)
                        try:
                            rhs_unw = expand(substituted)   # expand, not simplify: simplify can hang (trigsimp)
                        except Exception:
                            # SymPy heuristic-GCD/poly failures: fall back to unsimplified.
                            rhs_unw = substituted
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

        self.verbose = False
        if self.verbose:
            print(f"{self.lhs} = {self.rhs}. (Operation, term): ({operation_names[operation]}, {term})")
            for a in action_list:
                print(a)

        #if self.use_cov and action_index == self.action_index_cov:
        #    print(f"{self.lhs} = {self.rhs}. (Operation, term): ({operation_names[operation]}, {term})")

        if terminated or truncated:
            self.traj_obs = []
            self.traj_act = []

        return obs_new, reward, terminated, truncated, info


    def step(self, action_index: int):
        """Time-limited step wrapper. Calls step_base(action) with a wall-clock limit.
        On timeout: returns (state, small step reward, terminated=False, truncated=True)."""
        timeout = getattr(self, "step_timeout", 0.0)

        # No timeout or platform doesn't support SIGALRM → run normally
        if timeout is None or timeout <= 0 or os.name == "nt":
            return self.step_base(action_index)

        # Install a temporary alarm
        def _handler(signum, frame):
            raise TimeoutException("step() timed out")

        prev = signal.signal(signal.SIGALRM, _handler)
        # Use ITIMER_REAL so it measures wall-clock time even during pure Python
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            return self.step_base(action_index)
        except TimeoutException:
            # book-keeping
            self._timeout_count = getattr(self, "_timeout_count", 0) + 1
            self.traj_obs = []
            self.traj_act = []

            # Choose a minimal per-step reward consistent with your scheme
            if self.sparse_reward:
                raw_r = 0.0
            else:
                raw_r = self.reward_step  # your usual per-step shaping (-1)

            if self.normalize_rewards:
                # same normalization used in find_reward()
                min_r, max_r = self.reward_invalid_equation, self.reward_solved
                reward = 2.0 * (raw_r - min_r) / float(max_r - min_r) - 1.0
            else:
                reward = raw_r

            # Do NOT change state; just signal a truncation so SB3 resets the env
            info = {
                "timed_out": True,
                "action_taken": None,
                "main_eqn": self.main_eqn,
                "eqn_id": str(self.main_eqn),
            }
            return self.state, reward, False, True, info
        finally:
            # Always clear the alarm & restore previous handler
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, prev)


    # ──────────────────────────────────────────────────────────────────────────
    # CoV: reuse x; push inverse mapping
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_cov(self, lhs, rhs, timeout_seconds=3):
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

        # NOTE: no signal handling here. The enclosing step() owns a single
        # wall-clock timer for the whole step; _apply_cov must NOT install or
        # cancel its own alarm (doing so left the rest of step_base unprotected
        # and was the cause of the --use_cov memory blow-up).
        try:
            v = self.solve_var
            # Ensure SymPy objects
            lhs_s = sympify(lhs)
            rhs_s = sympify(rhs)
            main_s = sympify(self.main_eqn)
            # Try the CURRENT transformed equation first (expand(lhs - rhs)).
            # This is critical for nested-CoV recipes: e.g. for exp eqns, the
            # agent applies CoV (x->log(x)) and then mul-by-x to get a quadratic;
            # pi_cov on main_eqn (which is the rational form a*x + b/x + c) would
            # return None, but pi_cov on the expanded current form (a*x^2 + c*x + b)
            # correctly detects the quadratic and returns the depression substitution.
            # IMPORTANT: use expand(), NOT simplify(). simplify() can hang for
            # minutes on cubic/exponential forms and does not reliably respond
            # to the SIGALRM timeout (it gets stuck in SymPy C code). expand()
            # is cheap, bounded, and sufficient for the polynomial / exp pattern
            # matching that pi_cov_general performs.
            #
            # Belt-and-suspenders: a cheap count_ops() pre-check. If the raw
            # lhs - rhs is already very large, skip the expand entirely (it
            # could still be slow) and fall through to the main_eqn path.
            # count_ops is fast and does not hang.
            sub_expr = None
            try:
                raw_form = sympify(lhs_s - rhs_s)
                if count_ops(raw_form) <= COV_EXPAND_OPS_LIMIT:
                    current_form = expand(raw_form)
                    sub_expr = self.pi_cov(current_form)
            except Exception:
                sub_expr = None
            if sub_expr is None:
                # Fallback to main_eqn for the standard case (first-CoV on original)
                sub_expr = self.pi_cov(main_s)
            if sub_expr is None:
                #logger.warning("pi_cov returned None")
                return lhs, rhs
            # expand, NOT simplify: simplify() on the post-substitution form
            # (e.g. exp(log(x)) terms) recurses into trigsimp/exptrigsimp and
            # hangs in SymPy internals, past the SIGALRM step-timeout. This was
            # the open-equation no-output hang.
            lhs2 = expand(lhs_s.subs(v, sub_expr))
            rhs2 = expand(rhs_s.subs(v, sub_expr))
            main2 = expand(main_s.subs(v, sub_expr))
            if self.main_eqn_original_cov is None:
                self.main_eqn_original_cov = self.main_eqn
            self.main_eqn = main2
            inv_expr = sympify(sub_expr)  # Forward map: x_old := sub_expr(x_new)
            self.cov_inv.append(inv_expr)
            return lhs2, rhs2
        except Exception as e:
            # SymPy can raise HeuristicGCDFailed, PolynomialError, ZeroDivisionError,
            # CoercionFailed, etc. inside simplify/cancel during weird substitutions.
            # Treat any such failure as a no-op CoV so the worker survives.
            # (A step() timeout raises TimeoutException, a BaseException, so it
            # is NOT caught here -- it propagates to step() as intended.)
            return lhs, rhs


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
        # Each canonicalizer may raise (HeuristicGCDFailed, PolynomialError, etc.).
        # Try them independently so a single failure doesn't reject a true solution.
        for canon in (
            lambda s: s.expand() if getattr(s, 'expand', None) else s,
            lambda s: powdenest(s, force=True),
            lambda s: ratsimp(s),
            lambda s: simplify(s),
        ):
            try:
                if canon(sol) == 0:
                    return True
            except Exception:
                continue
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
        if self.gen == 'poesia-full':
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
        self.last_action = None
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
if __name__ == '__main__':
    from sympy import symbols

    a, b, c, x = symbols('a b c x')
    env = multiEqn(gen='test_cov0')
    
    actions = [11,21,17,3]
    for action in actions:
        state, reward, _, _, info = env.step(action)
        lhs, rhs = info['lhs'], info['rhs']
        if info['is_solved'] == False:
            print(f'{lhs} = {rhs}')
        else:
            print(f'Solved: {lhs} = {rhs}')
    
    breakpoint()
    
    
