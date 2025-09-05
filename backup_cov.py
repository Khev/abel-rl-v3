import sympy as sp
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import torch
import time
import sys
import os
from collections import deque
from operator import add, sub, mul, truediv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
from utils.utils_env import *

# Set global seed for reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Fallback
sys.path.append(os.getcwd())  # Add current notebook folder explicitly

# Symbols
x, a, b, c = sp.symbols('x a b c')

def custom_identity(expr, term):
    return expr

def get_device():
    """Returns the appropriate device (CUDA or CPU)."""
    if torch.cuda.is_available():
        print('Found CUDA: using GPU')
        cur_proc_identity = mp.current_process()._identity
        if cur_proc_identity:
            return (cur_proc_identity[0] - 1) % torch.cuda.device_count()
        return 0
    print('CUDA not found: using CPU')
    return 'cpu'

class EntropyAnnealCallback(BaseCallback):
    def __init__(self, start=0.01, end=0.0, total_timesteps=100_000, by_rollout=False, verbose=0):
        super().__init__(verbose)
        self.start = float(start)
        self.end = float(end)
        self.total_timesteps = int(total_timesteps)
        self.by_rollout = bool(by_rollout)

    def _progress(self):
        """Calculate progress as a fraction of total timesteps, clamped to [0, 1]."""
        t = self.model.num_timesteps
        return min(1.0, max(0.0, t / max(1, self.total_timesteps)))

    def _set_ent(self):
        p = self._progress()
        current = self.start + (self.end - self.start) * p
        self.model.ent_coef = float(current)

    def _on_step(self) -> bool:
        if not self.by_rollout:
            self._set_ent()
        return True

    def _on_rollout_end(self) -> None:
        if self.by_rollout:
            self._set_ent()

class SuccessBuffer:
    def __init__(self, capacity=5000):
        self.obs = deque(maxlen=capacity)
        self.act = deque(maxlen=capacity)

    def add_episode(self, traj_obs, traj_act):
        for o, a in zip(traj_obs, traj_act):
            self.obs.append(o)
            self.act.append(a)

    def sample(self, batch_size):
        n = len(self.obs)
        if n == 0:
            return np.empty((0,)), np.empty((0,), dtype=np.int64)
        if n >= batch_size:
            idx = random.sample(range(n), k=batch_size)  # Without replacement
        else:
            idx = np.random.randint(0, n, size=batch_size).tolist()  # With replacement
        obs_b = np.stack([self.obs[i] for i in idx], axis=0).astype(np.float32)
        act_b = np.array([self.act[i] for i in idx], dtype=np.int64)
        return obs_b, act_b

    def __len__(self):
        return len(self.obs)

class SuccessReplayCallback(BaseCallback):
    """Mix 10–20% supervised policy updates from a success buffer after each rollout."""
    def __init__(self, mix_ratio=0.15, batch_size=256, iters_per_rollout=1, capacity=10000, verbose=0):
        super().__init__(verbose)
        self.mix_ratio = float(mix_ratio)
        self.batch_size = int(batch_size)
        self.iters_per_rollout = int(iters_per_rollout)
        self.buf = SuccessBuffer(capacity=capacity)
        self.unique_eqns = set()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            if info.get("success"):
                main_eqn = info.get('main_eqn')
                if main_eqn not in self.unique_eqns:
                    self.unique_eqns.add(main_eqn)
                    print(f't={self.num_timesteps} | Solved {info.get("main_eqn")} | Coverage = {info.get("coverage"):.3f}')
                traj_obs = info.get("traj_obs", None)
                traj_act = info.get("traj_act", None)
                if traj_obs is not None and traj_act is not None and len(traj_obs) == len(traj_act):
                    self.buf.add_episode(traj_obs, traj_act)
        return True

    @torch.no_grad()
    def _policy_logits(self, obs_tensor):
        dist = self.model.policy.get_distribution(obs_tensor)
        return dist.distribution.logits

    def _supervised_update(self, obs_batch, act_batch):
        device = self.model.policy.device
        obs_t = torch.as_tensor(obs_batch, device=device)
        act_t = torch.as_tensor(act_batch, device=device)
        self.model.policy.optimizer.zero_grad(set_to_none=True)
        dist = self.model.policy.get_distribution(obs_t)
        logp = dist.log_prob(act_t)
        loss_bc = -logp.mean()
        ent = dist.entropy().mean()
        loss = loss_bc - 0.001 * ent
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 0.5)
        self.model.policy.optimizer.step()
        return float(loss_bc.item()), float(ent.item())

    def _on_rollout_end(self) -> None:
        if len(self.buf) == 0:
            return
        n_envs = getattr(self.training_env, "num_envs", 1)
        n_steps = getattr(self.model, "n_steps", 2048)
        target_supervised_minibatches = max(1, int(self.mix_ratio * (n_envs * n_steps) / self.batch_size))
        total_iters = max(self.iters_per_rollout, target_supervised_minibatches)
        bc_losses, ents = [], []
        for _ in range(total_iters):
            obs_b, act_b = self.buf.sample(self.batch_size)
            lbc, ent = self._supervised_update(obs_b, act_b)
            bc_losses.append(lbc)
            ents.append(ent)
        if self.verbose:
            print(f"[SuccessReplay] size={len(self.buf)} bc_steps={total_iters} "
                  f"bc_loss={np.mean(bc_losses):.4f} ent={np.mean(ents):.3f}")

def eval_avg_reward(env, model, n_episodes=20, max_steps=5, deterministic=True, test_eqns=None):
    if test_eqns is None:
        totals = []
        for i in range(n_episodes):
            obs, info = env.reset(seed=SEED+i)
            done = False
            truncated = False
            ep_total = 0.0
            steps = 0
            while not (done or truncated) and steps < max_steps:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = env.step(action)
                ep_total += float(reward)
                steps += 1
            totals.append(ep_total)
        return float(np.mean(totals)), float(np.std(totals))
    else:
        success_count = 0
        for eqn in test_eqns:
            temp_env = covEnv(
                eqn, env.term_bank, max_depth=env.max_depth, step_penalty=env.step_penalty,
                f_penalty=env.f_penalty, hist_len=env.hist_len, multi_eqn=False, use_curriculum=False
            )
            totals = []
            for i in range(n_episodes):
                obs, info = temp_env.reset(seed=SEED+i)
                done = False
                truncated = False
                ep_total = 0.0
                steps = 0
                while not (done or truncated) and steps < max_steps:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, done, truncated, info = temp_env.step(action)
                    ep_total += float(reward)
                    steps += 1
                totals.append(ep_total)
            avg_rew = float(np.mean(totals))
            if avg_rew > 0:
                success_count += 1
        return success_count / len(test_eqns)

def test_greedy(env, model, max_steps=10, test_eqns=None):
    if test_eqns is None:
        obs, info = env.reset(seed=SEED)
        done, truncated = False, False
        steps, total_reward = 0, 0.0
        print('Greedy rollout')
        while not (done or truncated) and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        delta = info.get("delta_complexity", 0)
        success = (delta is not None) and (delta > 0)
        return success, info
    else:
        success_count = 0
        for eqn in test_eqns:
            temp_env = covEnv(
                eqn, env.term_bank, max_depth=env.max_depth, step_penalty=env.step_penalty,
                f_penalty=env.f_penalty, hist_len=env.hist_len, multi_eqn=False, use_curriculum=False
            )
            obs, info = temp_env.reset(seed=SEED)
            done, truncated = False, False
            steps, total_reward = 0, 0.0
            while not (done or truncated) and steps < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = temp_env.step(action)
                total_reward += reward
                steps += 1
            delta = info.get("delta_complexity", 0)
            if (delta is not None) and (delta > 0):
                success_count += 1
        return success_count / len(test_eqns)

def test_10(env, model, n_trials=10, max_steps=10, test_eqns=None):
    if test_eqns is None:
        success_any = False
        best_info = None
        best_delta = -1e9
        for t in range(n_trials):
            obs, info = env.reset(seed=SEED+t)
            done, truncated = False, False
            steps, total_reward = 0, 0.0
            while not (done or truncated) and steps < max_steps:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
            delta = info.get("delta_complexity", None)
            if delta is not None and delta > best_delta:
                best_delta = delta
                best_info = dict(info)
                best_info["trial"] = t
            if delta is not None and delta > 0:
                success_any = True
        return success_any, best_info
    else:
        success_count = 0
        for eqn in test_eqns:
            temp_env = covEnv(
                eqn, env.term_bank, max_depth=env.max_depth, step_penalty=env.step_penalty,
                f_penalty=env.f_penalty, hist_len=env.hist_len, multi_eqn=False, use_curriculum=False
            )
            success_any = False
            best_delta = -1e9
            for t in range(n_trials):
                obs, info = temp_env.reset(seed=SEED+t)
                done, truncated = False, False
                steps, total_reward = 0, 0.0
                while not (done or truncated) and steps < max_steps:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, done, truncated, info = temp_env.step(action)
                    total_reward += reward
                    steps += 1
                delta = info.get("delta_complexity", None)
                if delta is not None and delta > best_delta:
                    best_delta = delta
                if delta is not None and delta > 0:
                    success_any = True
            if success_any:
                success_count += 1
        return success_count / len(test_eqns)

def beam_search(env, model, width=3, max_depth=3, test_eqns=None):
    from heapq import nlargest
    if test_eqns is None:
        beam = [(0.0, 0, env.x, None)]
        best = (-1e9, None)
        for depth in range(max_depth + 1):
            cand = []
            for score, d, cov, _ in beam:
                after = subs_x_with_f(env.main_eqn, cov, env.x)
                delta = C(env.main_eqn) - C(after)
                if delta > best[0]:
                    best = (delta, cov)
                if d == max_depth:
                    continue
                _, _ = env.reset(seed=SEED+depth)
                env.cov = cov
                env.depth = d
                if hasattr(env, "hist_len"):
                    env.action_history = [-1] * env.hist_len
                obs_core = env.to_vec(env.main_eqn, 0)[0].astype(np.float32)
                obs = env._augment_obs(obs_core)
                obs_t = model.policy.obs_to_tensor(obs)[0]
                dist = model.policy.get_distribution(obs_t)
                logits = dist.distribution.logits.detach().cpu().numpy()
                logits = np.squeeze(logits)
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                topk = np.argsort(-probs)[:width]
                for a in topk:
                    op, tau = env.actions[a]
                    if op == "STOP":
                        continue
                    try:
                        if op == "ADD":
                            tmp = sp.simplify(cov + tau)
                        elif op == "SUB":
                            tmp = sp.simplify(cov - tau)
                        elif op == "MUL":
                            tmp = sp.simplify(cov * tau)
                        else:
                            tmp = sp.simplify(cov / tau)
                    except Exception:
                        continue
                    cand.append((score + float(probs[a]), d + 1, tmp, None))
                beam = nlargest(width, cand, key=lambda z: z[0])
                if not beam:
                    break
        return best
    else:
        success_count = 0
        for eqn in test_eqns:
            temp_env = covEnv(
                eqn, env.term_bank, max_depth=env.max_depth, step_penalty=env.step_penalty,
                f_penalty=env.f_penalty, hist_len=env.hist_len, multi_eqn=False, use_curriculum=False
            )
            beam = [(0.0, 0, temp_env.x, None)]
            best = (-1e9, None)
            for depth in range(max_depth + 1):
                cand = []
                for score, d, cov, _ in beam:
                    after = subs_x_with_f(temp_env.main_eqn, cov, temp_env.x)
                    delta = C(temp_env.main_eqn) - C(after)
                    if delta > best[0]:
                        best = (delta, cov)
                    if d == max_depth:
                        continue
                    _, _ = temp_env.reset(seed=SEED+depth)
                    temp_env.cov = cov
                    temp_env.depth = d
                    if hasattr(temp_env, "hist_len"):
                        temp_env.action_history = [-1] * temp_env.hist_len
                    obs_core = temp_env.to_vec(temp_env.main_eqn, 0)[0].astype(np.float32)
                    obs = temp_env._augment_obs(obs_core)
                    obs_t = model.policy.obs_to_tensor(obs)[0]
                    dist = model.policy.get_distribution(obs_t)
                    logits = dist.distribution.logits.detach().cpu().numpy()
                    logits = np.squeeze(logits)
                    probs = np.exp(logits - logits.max())
                    probs /= probs.sum()
                    topk = np.argsort(-probs)[:width]
                    for a in topk:
                        op, tau = temp_env.actions[a]
                        if op == "STOP":
                            continue
                        try:
                            if op == "ADD":
                                tmp = sp.simplify(cov + tau)
                            elif op == "SUB":
                                tmp = sp.simplify(cov - tau)
                            elif op == "MUL":
                                tmp = sp.simplify(cov * tau)
                            else:
                                tmp = sp.simplify(cov / tau)
                        except Exception:
                            continue
                        cand.append((score + float(probs[a]), d + 1, tmp, None))
                    beam = nlargest(width, cand, key=lambda z: z[0])
                    if not beam:
                        break
            delta, _ = best
            if delta > 0:
                success_count += 1
        return success_count / len(test_eqns)

def subs_x_with_f(lhs_rhs, f, xsym=x):
    """Substitute x -> f(x) into (lhs, rhs) or a single expr."""
    if isinstance(lhs_rhs, tuple):
        L, R = lhs_rhs
        L = sp.sympify(L)
        R = sp.sympify(R)
        return (sp.simplify(L.subs(xsym, f)), sp.simplify(R.subs(xsym, f)))
    return sp.simplify(sp.sympify(lhs_rhs).subs(xsym, f))

def C_old(expr_or_pair):
    """Simple complexity: count_ops on (lhs - rhs) if pair, else on expr."""
    if isinstance(expr_or_pair, tuple):
        L, R = expr_or_pair
        return int(sp.count_ops(sp.expand(L - R)))
    return int(sp.count_ops(sp.expand(expr_or_pair)))

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

def _nonfinite(expr):
    expr = sp.sympify(expr)
    return expr.has(sp.zoo) or expr.has(sp.oo) or expr.has(sp.nan) or expr.has(-sp.oo)

class covEnv(gym.Env):
    def __init__(self, main_eqn, term_bank, max_depth=5, step_penalty=0.0, f_penalty=0.0, hist_len=10, multi_eqn=False, use_curriculum=False):
        super().__init__()
        self.x = x
        self.main_eqn = main_eqn
        self.term_bank = list(term_bank)
        self.max_depth = max_depth
        self.step_penalty = float(step_penalty)
        self.f_penalty = float(f_penalty)
        self.state_rep = 'integer_1d'
        self.feature_dict = make_feature_dict(main_eqn, self.state_rep)
        self.max_expr_length = 20
        self.hist_len = hist_len
        self.use_curriculum = use_curriculum
        self.multi_eqn = multi_eqn
        self.ops = ["ADD", "SUB", "MUL", "DIV", "STOP"]
        self.base_op = 'IDENTITY'
        self.actions = [(op, t) for op in self.ops[:-1] for t in self.term_bank] + [("STOP", None)]
        self.action_space = spaces.Discrete(len(self.actions))
        self.obs = self.to_vec(self.main_eqn, 0)[0]
        self.obs_dim = int(np.asarray(self.obs).size)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim + 1 + self.hist_len,), dtype=np.float32)
        self.num_successful_episodes = 0
        self.train_eqns = [
            x**2 + 2*b*x + c,
            x**2 + 2*a*x + b,
            x**2 + 2*a*x + c,
            x**2 + b*x + c,
            a*x**2 + b*x + c,
            a*x**2 + 2*b*x + c,
            b*x**2 + a*x + c,
            x**2 + a*x + c,
            a*x**2 + c*x + b,
            b*x**2 + 2*a*x + c,
            x**3 + 3*b*x**2 + 3*b**2*x + c,
            a*x**3 + b*x**2 + (b**2/(3*a))*x + c,
            x**4 + 4*b*x**3 + ((16*3*b**2/8))*x**2 + (64*b**3/(16*a**2))*x + c,
            a*x**4 + b*x**3 + ((3*b**2/(8*a)))*x**2 + (b**3/(16*a**2))*x + c
        ]
        self.test_eqns = [
            x**2 + 2*c*x + b,
            x**2 + c*x + b,
            c*x**2 + b*x + a,
            c*x**2 + 2*b*x + a,
            b*x**2 + 2*c*x + a,
            x**3 + 3*b*x**2 + 3*b**2*x + c,
            a*x**3 + b*x**2 + (b**2/(3*a))*x + c,
            x**4 + 4*b*x**3 + ((16*3*b**2/8))*x**2 + (64*b**3/(16*a**2))*x + c,
            a*x**4 + b*x**3 + ((3*b**2/(8*a)))*x**2 + (b**3/(16*a**2))*x + c
        ]
        self.solve_counts = {train_eqn: 0 for train_eqn in self.train_eqns}
        self.sample_counts = {train_eqn: 0 for train_eqn in self.train_eqns}
        self.coverage = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        if self.multi_eqn:
            if self.use_curriculum:
                probs = [1 / (self.solve_counts[eqn] + 1) for eqn in self.train_eqns]
                probs = [p / sum(probs) for p in probs]
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
        return np.concatenate([
            np.asarray(obs_core, dtype=np.float32).flatten(),
            np.array([self.depth], dtype=np.float32),
            np.array(self.action_history, dtype=np.float32)
        ])

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
            if op == "ADD":
                newf = self.cov + tau
            elif op == "SUB":
                newf = self.cov - tau
            elif op == "MUL":
                newf = self.cov * tau
            elif op == "DIV":
                newf = self.cov / tau
            else:
                return False
            newf = sp.simplify(newf)
            self.cov = newf
        self.history.append((op, tau))
        self.depth += 1
        return True

    def cost(self):
        deg = sp.degree(sp.Poly(self.cov, self.x)) if self.cov.is_polynomial(self.x) else 0
        return len(self.history) + len(list(self.cov.free_symbols)) + (deg if deg is not None else 0)

    def to_vec(self, lhs, rhs):
        if self.state_rep == 'integer_1d':
            return integer_encoding_1d(lhs, rhs, self.feature_dict, self.max_expr_length)
        elif self.state_rep == 'integer_2d':
            return integer_encoding_2d(lhs, rhs, self.feature_dict, self.max_expr_length)
        elif self.state_rep in ['graph_integer_1d', 'graph_integer_2d']:
            return graph_encoding(lhs, rhs, self.feature_dict, self.max_expr_length)
        else:
            raise ValueError(f"Unknown state_rep: {self.state_rep}")

    def step(self, action_idx):
        operation, term = self.actions[action_idx]
        terminated = False
        truncated = False
        reward = 0.0
        if self.depth < self.max_depth:
            self.ep_obs.append(self._augment_obs(self.obs.copy()))
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
                info["traj_obs"] = np.array(self.ep_obs, dtype=np.float32)
                info["traj_act"] = np.array(self.ep_act, dtype=np.int64)
                info["success"] = success
        return obs, reward, terminated, truncated, info

# Args
Ntrain = 5 * 10**3
main_eqn = a * x**2 + b * x + c
term_bank = [a, b, c, sp.Integer(2), sp.Integer(3), sp.Integer(4)]
agents = ['ppo-mem']
multi_eqn = True
use_curriculum = True
curiosity = None
log_interval = Ntrain // 10
n_steps = 2 * 1024
ent_coef = 5 * 0.01

for agent in agents:
    t1 = time.time()
    print("\n" + "="*70)
    print(f"[Train] agent: {agent}")
    print("="*70)

    # Environment
    env = covEnv(
        main_eqn, term_bank, max_depth=3, step_penalty=0.1,
        use_curriculum=use_curriculum, multi_eqn=multi_eqn
    )
    print(f'\nTrain eqns, test eqns = {len(env.train_eqns)}, {len(env.test_eqns)}\n')

    # Model
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=256,
        learning_rate=3e-4,
        gamma=1.0,
        gae_lambda=0.95,
        ent_coef=ent_coef,
        verbose=0,
        seed=SEED,
    )

    # Callbacks
    ent_cb = EntropyAnnealCallback(start=0.01, end=0.0, total_timesteps=Ntrain, by_rollout=True)
    progress_cb = ProgressBarCallback()
    callbacks = [ent_cb, progress_cb]
    if 'mem' in agent:
        mix_ratio, iters_per_rollout = 0.5, 10
        cb_replay = SuccessReplayCallback(
            mix_ratio=mix_ratio, batch_size=256, iters_per_rollout=iters_per_rollout,
            capacity=20000, verbose=0
        )
        callbacks.append(cb_replay)

    # Training
    model.learn(total_timesteps=Ntrain, callback=callbacks)

    # Evaluation
    print("\n[Eval@end]")
    print(f"agent: {agent}")
    if not multi_eqn:
        ok, info = test_greedy(covEnv(main_eqn, term_bank, max_depth=3), model, max_steps=5, test_eqns=None)
        print("\n[Greedy]")
        print(f"success: {ok} (single equation)")
        print(f"f(x): {info.get('cov')} (single equation)")
        if "equation_after" in info:
            eq_after = info["equation_after"]
            print(f"transformed: {eq_after} (single equation)")
        print(f"Δcomplexity: {info.get('delta_complexity')} (single equation)")

        ok10, best = test_10(covEnv(main_eqn, term_bank, max_depth=3), model, n_trials=10, max_steps=5, test_eqns=None)
        print("\n[Success@10]")
        print(f"success_any: {ok10} (single equation)")
        if best is not None:
            print(f"best_trial: {best.get('trial')} (single equation)")
            print(f"f(x): {best.get('cov')} (single equation)")
        if best and "equation_after" in best:
            eq_after = best["equation_after"]
            print(f"transformed: {eq_after} (single equation)")
            print(f"Δcomplexity: {best.get('delta_complexity')} (single equation)")

        delta, best_cov = beam_search(env, model, test_eqns=None)
        print("\n[BeamSearch]")
        print(f"success: {delta>0} (single equation)")
        print(f"f(x): {best_cov} (single equation)")
        print(f"transformed: {subs_x_with_f(env.main_eqn, best_cov, env.x)} (single equation)")
        print(f"Δcomplexity: {delta} (single equation)")
    else:
        success_fraction_greedy = test_greedy(covEnv(main_eqn, term_bank, max_depth=3), model, max_steps=5, test_eqns=env.train_eqns)
        success_fraction_10 = test_10(covEnv(main_eqn, term_bank, max_depth=3), model, n_trials=10, max_steps=5, test_eqns=env.train_eqns)
        success_fraction_beam = beam_search(env, model, test_eqns=env.train_eqns)
        print(f"\ntrain_greedy_acc: {success_fraction_greedy:.3f}")
        print(f"train_10_acc: {success_fraction_10:.3f}")
        print(f"train_beam_acc: {success_fraction_beam:.3f}")

        success_fraction_greedy = test_greedy(covEnv(main_eqn, term_bank, max_depth=3), model, max_steps=5, test_eqns=env.test_eqns)
        success_fraction_10 = test_10(covEnv(main_eqn, term_bank, max_depth=3), model, n_trials=10, max_steps=5, test_eqns=env.test_eqns)
        success_fraction_beam = beam_search(env, model, test_eqns=env.test_eqns)
        print(f"\ntest_greedy_acc: {success_fraction_greedy:.3f}")
        print(f"test_10_acc: {success_fraction_10:.3f}")
        print(f"test_beam_acc: {success_fraction_beam:.3f}")

    t2 = time.time()
    print(f'\nTook {(t2-t1)/60.0:.2f} mins')