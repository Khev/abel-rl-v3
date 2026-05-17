import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np
from sympy import sympify
import re
from sympy import sympify, symbols
from gymnasium import spaces, Env
from operator import add, sub, mul, truediv
from utils.utils_env import *
from utils.utils_custom_functions import *
from collections import defaultdict, deque
import faiss                    # pip install faiss-cpu

logger = logging.getLogger(__name__)


class multiEqn(Env):
    """
    Environment for solving multiple equations using RL, 
    with a simple curriculum that samples equations inversely 
    proportional to how often they've been solved.
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
                 train_eqns = None
                 ) -> None:
        super().__init__()

        # Static parts
        self.max_expr_length = 20
        self.max_steps = 5
        self.action_dim = 50
        self.observation_dim = 2*self.max_expr_length + 1

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
        self.use_memory= use_memory  #backwards compatibility
        self.use_success_replay = use_success_replay
        self.use_relabel_constants = use_relabel_constants

        # For relabel constants
        self.map_constants = None
        self.map_constants_history = []
        self._timeout_count = 0
        self.main_eqn_original = None

        eqn_dirn = f"equation_templates"
        if train_eqns == None:
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

        # Load train/test equations
        # self.level = level
        # self.generalization = generalization
        # eqn_dirn = f"equation_templates"
        # self.train_eqns, self.test_eqns = load_train_test_equations(
        #     eqn_dirn, level, generalization=generalization
        # )

        # print(f'{generalization}: {len(self.train_eqns)} train eqns, {len(self.test_eqns)} test eqns')

        # if self.generalization == 'poesia-full':
        #     TEMPLATE_PATH = "equation_templates/poesia/equations-ct.txt"
        #     def fetch_templates():
        #         with open(TEMPLATE_PATH, 'r') as f:
        #             templates = [line.strip() for line in f if line.strip() and not line.startswith('!')
        #             ]
        #         print(f"Loaded {len(templates)} templates from local file.")
        #         return templates
        #     self.templates = fetch_templates()  # Load templates once

        # Tracking how many times we've solved each eqn
        # Use a dict with keys = the actual sympy expression or string
        self.solve_counts = defaultdict(int)
        self.sample_counts = defaultdict(int)

        # Convert each eqn to a canonical string so we can store counts easily
        self.train_eqns_str = [str(eq) for eq in self.train_eqns]
        self.test_eqns_str = [str(eq) for eq in self.test_eqns]

        # Random initial eqn
        if generalization == 'poesia-full':
            self.main_eqn = self.sample_poesia_equation()
        else:
            eqn_str = np.random.choice(self.train_eqns_str)
            self.main_eqn = sympify(eqn_str)
        self.lhs = self.main_eqn
        self.rhs = 0
        self.x = symbols('x')

        #  Make feature_dict, actions etc
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


    def setup(self):
        # Build feature dict from all train eqns
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


    def step(self, action_index: int):
        lhs_old, rhs_old, obs_old = self.lhs, self.rhs, self.state

        # ── (re)build the current action list ────────────────────────────────────
        if self.cache:
            action_list, action_mask = make_actions_cache(lhs_old, rhs_old, self.actions_fixed,self.action_dim, self.action_cache)
        else:
            action_list, action_mask = make_actions(lhs_old, rhs_old, self.actions_fixed, self.action_dim)

        self.actions, self.action_mask = action_list, action_mask
        operation, term = action_list[action_index]

        if self.use_success_replay:
            self.traj_obs.append(obs_old.copy())
            self.traj_act.append(int(action_index))

        # ── apply chosen action ----------------------------------------------------
        if self.use_relabel_constants and operation.__name__ == 'relabel_const_custom':
            lhs_new, rhs_new, map_constants = operation(lhs_old, rhs_old)
            self.map_constants = map_constants
            if self.main_eqn_original is None:
                self.main_eqn_original = self.main_eqn
            self.main_eqn = sympify(str(lhs_new) + ' - ' + str(rhs_new))
            self.map_constants_history.append(map_constants)
        else:
            lhs_new, rhs_new = operation(lhs_old, term), operation(rhs_old, term)

        obs_new, _ = self.to_vec(lhs_new, rhs_new)

        # ── environment bookkeeping ------------------------------------------------
        is_valid_eqn, lhs_new, rhs_new = check_valid_eqn(lhs_new, rhs_new)
        is_solved = check_eqn_solved(lhs_new, rhs_new, self.main_eqn)

        reward = self.find_reward(lhs_old, rhs_old, lhs_new, rhs_new,
                                is_valid_eqn, is_solved)

        too_many_steps = (self.current_steps >= self.max_steps)
        terminated = bool(is_solved or too_many_steps or not is_valid_eqn)
        truncated = False


        if is_solved:
            if self.map_constants_history:
                for m in reversed(self.map_constants_history):
                    lhs_new = lhs_new.subs(m)
                    rhs_new = rhs_new.subs(m)
                self.main_eqn = self.main_eqn_original
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
            "map_constants": self.map_constants,
            "map_constants_history": list(self.map_constants_history)
        }

        if is_solved and self.use_success_replay:
            info["traj_obs"] = list(self.traj_obs)
            info["traj_act"] = list(self.traj_act)

        if self.verbose:
            print(f"{self.lhs} = {self.rhs}. "
                f"(Operation, term): ({operation_names[operation]}, {term})")

        if terminated or truncated:
            self.traj_obs = []
            self.traj_act = []

        return obs_new, reward, terminated, truncated, info


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


    def reset(self, seed=None, options=None):

        # Need to randomly sample each time
        if self.generalization == 'poesia-full':
            self.main_eqn = self.sample_poesia_equation()

        else:
            # Sample eqn in a 'curriculum' fashion:
            # pick eqn with probability ~ 1/(1+solve_counts)
            eqn_probs = []
            if options == None:
                if self.use_curriculum == True:
                    for eqn_str in self.train_eqns_str:
                        eqn_probs.append( 1.0 / (1 + self.solve_counts[eqn_str]) )
                    eqn_probs = np.array(eqn_probs, dtype=np.float64)
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

    # def get_valid_action_mask(self):
    #     return self.action_mask

    def get_valid_action_mask(self):
        # ensure dtype is strictly boolean for ActionMasker / MaskablePPO
        return np.asarray(self.action_mask, dtype=np.bool_)


    def set_equation(self, main_eqn):
        self.main_eqn, self.lhs, self.rhs = main_eqn, main_eqn, 0
        self.main_eqn_original = None
        self.map_constants = None
        self.map_constants_history = []
        obs, _ = self.to_vec(self.lhs, self.rhs)
        self.state = obs
