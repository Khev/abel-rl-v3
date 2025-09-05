import gymnasium as gym
from gymnasium import spaces
import sympy as sp
import numpy as np
from operator import add, sub, mul, truediv
import torch

from utils.utils_env import (
    integer_encoding_1d,
    integer_encoding_2d,
    graph_encoding,          # kept for other reps if you use them elsewhere
    make_feature_dict,
    make_feature_dict_multi,
    sympy_expression_to_graph
)

# Symbols
x, a, b, c = sp.symbols('x a b c')


# ------------------------------
# Graph encoder for covEnv (1-D)
# ------------------------------
def graph_encoding_cov_1d(lhs, rhs, feature_dict, max_length):
    """
    1-D node-feature graph encoding for covEnv only.

    IMPORTANT: Shapes MUST match env.observation_space when state_rep='graph_integer_1d'
      - MAX_NODES = max_length         (env uses self.max_expr_length)
      - MAX_EDGES = 2 * MAX_NODES      (env uses 2 * max_nodes)

    Returns a dict with:
      node_features: (MAX_NODES,) float32
      edge_index:    (2, MAX_EDGES) int64
      node_mask:     (MAX_NODES,) bool
      edge_mask:     (MAX_EDGES,) bool
    """
    # align to observation_space
    MAX_NODES = int(max_length)
    MAX_EDGES = int(2 * MAX_NODES)

    # Build raw graphs (expects tensors lhs_graph.x: (Nl,), lhs_graph.edge_index: (2, El))
    lhs_graph = sympy_expression_to_graph(lhs, feature_dict)
    rhs_graph = sympy_expression_to_graph(rhs, feature_dict)

    # equality node as scalar (1-D)
    eq_node_feature = torch.tensor([0.0], dtype=torch.float)  # shape: (1,)
    eq_node_idx = lhs_graph.x.shape[0]

    # shift RHS indices by (lhs nodes + 1 eq node)
    if rhs_graph.edge_index.numel() > 0:
        rhs_graph.edge_index = rhs_graph.edge_index + (eq_node_idx + 1)

    # concat node features (all 1-D)
    xfeat = torch.cat([lhs_graph.x, eq_node_feature, rhs_graph.x], dim=0)  # (N,)

    # edges: concat + link LHS_last -> "=" -> RHS_first
    if lhs_graph.edge_index.numel() > 0 and rhs_graph.edge_index.numel() > 0:
        edge_index = torch.cat([lhs_graph.edge_index, rhs_graph.edge_index], dim=1)
    elif lhs_graph.edge_index.numel() > 0:
        edge_index = lhs_graph.edge_index.clone()
    elif rhs_graph.edge_index.numel() > 0:
        edge_index = rhs_graph.edge_index.clone()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    lhs_last_idx = eq_node_idx - 1 if eq_node_idx > 0 else 0
    rhs_first_idx = eq_node_idx + 1
    if xfeat.shape[0] > rhs_first_idx:
        eq_edges = torch.tensor([[lhs_last_idx, eq_node_idx],
                                 [eq_node_idx,  rhs_first_idx]], dtype=torch.long)
        edge_index = torch.cat([edge_index, eq_edges], dim=1)

    # keep only edges that reference nodes within MAX_NODES
    if edge_index.shape[1] > 0:
        valid = (edge_index[0] < MAX_NODES) & (edge_index[1] < MAX_NODES)
        edge_index = edge_index[:, valid]

    # truncate to caps
    if xfeat.shape[0] > MAX_NODES:
        xfeat = xfeat[:MAX_NODES]
    if edge_index.shape[1] > MAX_EDGES:
        edge_index = edge_index[:, :MAX_EDGES]

    # pad to fixed sizes
    padded_x = torch.full((MAX_NODES,), 99.0, dtype=torch.float)
    real_nodes = min(xfeat.shape[0], MAX_NODES)
    if real_nodes > 0:
        padded_x[:real_nodes] = xfeat[:real_nodes]

    padded_e = torch.zeros((2, MAX_EDGES), dtype=torch.long)
    real_edges = min(edge_index.shape[1], MAX_EDGES)
    if real_edges > 0:
        padded_e[:, :real_edges] = edge_index[:, :real_edges]

    node_mask = torch.zeros((MAX_NODES,), dtype=torch.bool)
    node_mask[:real_nodes] = True
    edge_mask = torch.zeros((MAX_EDGES,), dtype=torch.bool)
    edge_mask[:real_edges] = True

    vec_dict = {
        "node_features": padded_x.numpy(),
        "edge_index":    padded_e.numpy(),
        "node_mask":     node_mask.numpy(),
        "edge_mask":     edge_mask.numpy(),
    }
    return vec_dict, 0


# ------------------------------
# Helpers
# ------------------------------
def load_equations(file_path):
    equations = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                equations.append(sp.sympify(line.strip()))
    except FileNotFoundError:
        raise FileNotFoundError(f"Equation file {file_path} not found. Ensure equations are generated.")
    return equations

def custom_identity(expr, term):
    return expr

def subs_x_with_f(lhs_rhs, f, xsym=x):
    if isinstance(lhs_rhs, tuple):
        L, R = lhs_rhs
        L = sp.sympify(L); R = sp.sympify(R)
        return (sp.simplify(L.subs(xsym, f)), sp.simplify(R.subs(xsym, f)))
    return sp.simplify(sp.sympify(lhs_rhs).subs(xsym, f))

def C(expr_or_pair):
    if isinstance(expr_or_pair, tuple):
        L, R = expr_or_pair
        expr = sp.expand(L - R)
    else:
        expr = sp.expand(expr_or_pair)
    try:
        P = sp.Poly(expr, x)
    except sp.PolynomialError:
        return int(sp.count_ops(expr))
    return len(P.terms())


# ------------------------------
# Environment
# ------------------------------
class covEnv(gym.Env):
    def __init__(
        self,
        main_eqn,
        term_bank,
        max_depth=5,
        step_penalty=0.0,
        f_penalty=0.0,
        hist_len=10,
        multi_eqn=True,
        use_curriculum=False,
        gen=3,
        state_rep='integer_1d'
    ):
        super().__init__()
        self.x = x
        self.main_eqn = main_eqn
        self.term_bank = list(term_bank)
        self.max_depth = max_depth
        self.step_penalty = float(step_penalty)
        self.f_penalty = float(f_penalty)
        self.state_rep = state_rep
        self.max_expr_length = 20
        self.hist_len = hist_len
        self.use_curriculum = use_curriculum
        self.multi_eqn = multi_eqn
        self.gen = gen

        # actions
        self.ops = ["ADD", "SUB", "MUL", "DIV", "STOP"]
        self.base_op = 'IDENTITY'
        self.actions = [(op, t) for op in self.ops[:-1] for t in self.term_bank] + [("STOP", None)]
        self.action_space = spaces.Discrete(len(self.actions))

        # data
        train_file = f"equation_templates/cov_level{self.gen}/train_eqns.txt"
        test_file  = f"equation_templates/cov_level{self.gen}/test_eqns.txt"
        self.train_eqns = load_equations(train_file)
        self.test_eqns  = load_equations(test_file)
        self.solve_counts  = {train_eqn: 0 for train_eqn in self.train_eqns}
        self.sample_counts = {train_eqn: 0 for train_eqn in self.train_eqns}
        self.coverage = 0

        # features
        if self.train_eqns:
            self.feature_dict = make_feature_dict_multi(self.train_eqns, self.test_eqns, state_rep)
        else:
            self.feature_dict = make_feature_dict(main_eqn, self.state_rep)

        # initial obs
        self.obs = self.to_vec(self.main_eqn, 0)[0]

        # observation space
        if self.state_rep == 'integer_1d':
            self.obs_dim = int(np.asarray(self.obs).size)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.obs_dim + 1 + self.hist_len,),
                dtype=np.float32
            )
        elif self.state_rep == 'graph_integer_1d':
            max_nodes = self.max_expr_length
            max_edges = 2 * max_nodes
            self.observation_space = spaces.Dict({
                "node_features":  spaces.Box(low=-np.inf, high=np.inf, shape=(max_nodes,), dtype=np.float32),
                "edge_index":     spaces.Box(low=0, high=max_nodes, shape=(2, max_edges), dtype=np.int32),
                "node_mask":      spaces.Box(low=0, high=1, shape=(max_nodes,), dtype=np.int32),
                "edge_mask":      spaces.Box(low=0, high=1, shape=(max_edges,), dtype=np.int32),
                "depth":          spaces.Box(low=0, high=self.max_depth, shape=(1,), dtype=np.float32),
                "action_history": spaces.Box(low=-1, high=len(self.actions), shape=(self.hist_len,), dtype=np.float32)
            })
        elif self.state_rep == 'graph_integer_2d':
            # same outer shapes to keep policy happy (features still treated as 1-D scalar ids)
            max_nodes = self.max_expr_length
            max_edges = 2 * max_nodes
            self.observation_space = spaces.Dict({
                "node_features":  spaces.Box(low=-np.inf, high=np.inf, shape=(max_nodes,), dtype=np.float32),
                "edge_index":     spaces.Box(low=0, high=max_nodes, shape=(2, max_edges), dtype=np.int32),
                "node_mask":      spaces.Box(low=0, high=1, shape=(max_nodes,), dtype=np.int32),
                "edge_mask":      spaces.Box(low=0, high=1, shape=(max_edges,), dtype=np.int32),
                "depth":          spaces.Box(low=0, high=self.max_depth, shape=(1,), dtype=np.float32),
                "action_history": spaces.Box(low=-1, high=len(self.actions), shape=(self.hist_len,), dtype=np.float32)
            })
        else:
            raise ValueError(f"Unsupported state_rep: {self.state_rep}")

        # episode tracking
        self.num_successful_episodes = 0

    # ------------------------------
    # Gym API
    # ------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        if self.multi_eqn:
            if self.use_curriculum:
                probs = [1 / (self.solve_counts[eqn] + 1) for eqn in self.train_eqns]
                s = sum(probs)
                probs = [p / s for p in probs]
                self.main_eqn = np.random.choice(self.train_eqns, p=probs)
            else:
                self.main_eqn = np.random.choice(self.train_eqns)
            self.sample_counts[self.main_eqn] += 1

        self.ep_obs = []
        self.ep_act = []
        self.cov = 0
        self.depth = 0
        self.history = []
        self.last_op_id = -1
        self.base_cmplx = C(self.main_eqn)
        self.obs = self.to_vec(self.main_eqn, 0)[0]
        self.action_history = [-1] * self.hist_len
        return self._augment_obs(self.obs), {}

    def _augment_obs(self, obs_core):
        if self.state_rep == 'integer_1d':
            return np.concatenate([
                np.asarray(obs_core, dtype=np.float32).flatten(),
                np.array([self.depth], dtype=np.float32),
                np.array(self.action_history, dtype=np.float32)
            ])
        elif self.state_rep.startswith('graph_'):
            graph_obs = obs_core  # dict
            graph_obs['depth'] = np.array([self.depth], dtype=np.float32)
            graph_obs['action_history'] = np.array(self.action_history, dtype=np.float32)
            return graph_obs
        else:
            raise ValueError(f"Unsupported state_rep: {self.state_rep}")

    def action_mask(self):
        mask = np.ones(self.action_space.n, dtype=np.int32)
        for i, (op, t) in enumerate(self.actions):
            if op == "DIV" and (t == 0 or t == 0*x):
                mask[i] = 0
        return mask

    def apply_op_to_cov(self, op, tau):
        if op == "DIV" and sp.simplify(tau) == 0:
            return False
        if self.depth == 0:
            self.base_op = op
            self.cov = tau
        else:
            if op == "ADD": newf = self.cov + tau
            elif op == "SUB": newf = self.cov - tau
            elif op == "MUL": newf = self.cov * tau
            elif op == "DIV": newf = self.cov / tau
            else: return False
            self.cov = sp.simplify(newf)
        self.history.append((op, tau))
        self.depth += 1
        return True

    def cost(self):
        try:
            if hasattr(self.cov, "is_polynomial") and self.cov.is_polynomial(self.x):
                deg = sp.degree(sp.Poly(self.cov, self.x))
            else:
                deg = 0
        except Exception:
            deg = 0
        return len(self.history) + len(list(getattr(self.cov, "free_symbols", set()))) + (deg if deg is not None else 0)

    def to_vec(self, lhs, rhs):
        if self.state_rep == 'integer_1d':
            return integer_encoding_1d(lhs, rhs, self.feature_dict, self.max_expr_length)
        elif self.state_rep == 'integer_2d':
            return integer_encoding_2d(lhs, rhs, self.feature_dict, self.max_expr_length)
        elif self.state_rep == 'graph_integer_1d':
            # emit arrays sized to (max_nodes=self.max_expr_length, max_edges=2*max_nodes)
            return graph_encoding_cov_1d(lhs, rhs, self.feature_dict, self.max_expr_length)
        else:
            raise ValueError(f"Unsupported state_rep: {self.state_rep}")

    def step(self, action_idx):
        operation, term = self.actions[action_idx]
        terminated = False
        truncated = False
        reward = 0.0

        if self.depth < self.max_depth:
            if self.state_rep == 'integer_1d':
                self.ep_obs.append(self._augment_obs(self.obs.copy()))
            else:
                self.ep_obs.append(self._augment_obs(self.obs))
            self.ep_act.append(int(action_idx))
            self.action_history = self.action_history[1:] + [action_idx]

        if operation == "STOP" or self.depth >= self.max_depth:
            op_dict = {'ADD': add, 'SUB': sub, 'MUL': mul, 'DIV': truediv, 'IDENTITY': custom_identity}
            self.cov = op_dict[self.base_op](x, self.cov)
            if self.cov == 0:
                self.cov = x
            main_eqn_after_cov = subs_x_with_f(self.main_eqn, self.cov, self.x)
            delta = self.base_cmplx - C(main_eqn_after_cov)
            reward = float(delta) - self.f_penalty * self.cost()
            terminated = True

            if reward > 0:
                reward *= 10
                self.num_successful_episodes += 1
                if self.multi_eqn:
                    self.solve_counts[self.main_eqn] += 1
                    if self.solve_counts[self.main_eqn] == 1:
                        self.coverage += 1.0 / len(self.train_eqns)
            if reward < 0:
                reward = -1
            if reward == 0 and self.depth < 2:
                reward = -1
        else:
            self.apply_op_to_cov(operation, term)
            reward = -self.step_penalty

        self.obs = self.to_vec(self.main_eqn, 0)[0]
        obs = self._augment_obs(self.obs)

        info = {"main_eqn": self.main_eqn, "cov": self.cov, "base_complexity": self.base_cmplx}
        if terminated:
            info["equation_after"] = subs_x_with_f(self.main_eqn, self.cov, self.x)
            info["after_complexity"] = C(info["equation_after"])
            info["delta_complexity"] = self.base_cmplx - info["after_complexity"]
            info['coverage'] = self.coverage
            success = bool(info.get("delta_complexity", 0) > 0)
            if success:
                info["traj_obs"] = (np.array(self.ep_obs, dtype=np.float32)
                                    if self.state_rep == 'integer_1d' else self.ep_obs)
                info["traj_act"] = np.array(self.ep_act, dtype=np.int64)
                info["success"] = success

        return obs, reward, terminated, truncated, info
