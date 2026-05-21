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
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from typing import Optional, Dict, Any
import pickle
from datetime import datetime as dt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env

from utils.utils_env import *
from utils.utils_train import *
from utils.utils_general import print_parameters
from envs.env_cov import covEnv

# --- Curiosity (rllte) ---
from rllte.xplore.reward import E3B, ICM, NGU, RE3, RIDE, RND

# Set global seed for reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

# Symbols
x, a, b, c = sp.symbols('x a b c')


def get_device():
    """Returns CUDA device index or 'cpu'."""
    if torch.cuda.is_available():
        print("Found CUDA: using GPU")
        cur_proc_identity = mp.current_process()._identity
        return (cur_proc_identity[0] - 1) % torch.cuda.device_count() if cur_proc_identity else 0
    else:
        print("CUDA not found: using CPU")
        return "cpu"

def get_intrinsic_reward(intrinsic_reward, vec_env):
    device = get_device()
    if intrinsic_reward == 'ICM':
        return ICM(vec_env, device=device)
    if intrinsic_reward == 'E3B':
        return E3B(vec_env, device=device)
    if intrinsic_reward == 'RIDE':
        return RIDE(vec_env, device=device)
    if intrinsic_reward == 'RND':
        return RND(vec_env, device=device)
    if intrinsic_reward == 'RE3':
        return RE3(vec_env, device=device)
    if intrinsic_reward == 'NGU':
        return NGU(vec_env, device=device)
    return None


class IntrinsicReward(BaseCallback):
    """Computes intrinsic rewards at the end of each rollout and adds to buffer."""
    def __init__(self, irs, verbose: int = 0, log_interval: int = 100):
        super().__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm):
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        # Minimal per-step hook; NGU.watch support can be added here if desired.
        return True

    def _on_rollout_end(self) -> None:
        try:
            device = self.irs.device
            obs_rb = self.buffer.observations  # shape: (n_steps, n_envs, ...)

            # Build next_obs by shifting along time axis:
            next_obs_rb = {}
            if isinstance(obs_rb, dict):
                for k, arr in obs_rb.items():  # arr shape (n_steps, n_envs, *obs_shape)
                    nxt = np.copy(arr)
                    nxt[:-1] = arr[1:]
                    nxt[-1]  = self.locals["new_obs"][k]  # shape (n_envs, *obs_shape)
                    next_obs_rb[k] = nxt
                obs_arr      = obs_to_curiosity_array(obs_rb)
                next_obs_arr = obs_to_curiosity_array(next_obs_rb)
            else:
                obs_arr = np.asarray(obs_rb, dtype=np.float32)
                next_obs_arr = np.copy(obs_arr)
                next_obs_arr[:-1] = obs_arr[1:]
                next_obs_arr[-1]  = np.asarray(self.locals["new_obs"], dtype=np.float32)

            actions = torch.as_tensor(self.buffer.actions,  device=device)
            rewards = torch.as_tensor(self.buffer.rewards,  device=device, dtype=torch.float32)
            starts  = torch.as_tensor(self.buffer.episode_starts, device=device, dtype=torch.float32)

            samples = dict(
                observations      = torch.as_tensor(obs_arr,      device=device, dtype=torch.float32),
                next_observations = torch.as_tensor(next_obs_arr, device=device, dtype=torch.float32),
                actions=actions, rewards=rewards,
                terminateds=starts, truncateds=starts
            )

            intrinsic = self.irs.compute(samples=samples, sync=True).detach().cpu().numpy()
            if intrinsic.ndim == 1:
                intrinsic = intrinsic[:, None]
            elif intrinsic.ndim > 2:
                intrinsic = intrinsic.reshape(intrinsic.shape[0], -1).mean(axis=1, keepdims=True)

            self.buffer.advantages += intrinsic
            self.buffer.returns    += intrinsic
            self.irs.update(samples=samples)
        except Exception as e:
            print(f"[Curiosity] rollout_end failed: {e}")

# ---------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------
def make_agent(
    agent: str,
    env,
    hidden_dim: int,
    n_steps: int,
    ent_coef: float,
    seed: int = 0,
    load_path: Optional[str] = None,
    tree_kwargs: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
):
    """Create or load a PPO agent (MlpPolicy or MultiInputPolicy with TreeMLPExtractor)."""
    if load_path and os.path.isfile(load_path):
        print(f"[{agent}] Loading model from: {load_path}")
        return PPO.load(load_path, env=env, device="auto", print_system_info=False)

    if 'tree' not in agent:
        policy_kwargs = dict(net_arch=[hidden_dim, hidden_dim])
        return PPO(
            'MlpPolicy',
            env=env,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=verbose,
            n_steps=n_steps,
            learning_rate=3e-4,
            gamma=1.0,
            gae_lambda=0.95,
            ent_coef=ent_coef
        )

    # Tree policy
    tkw = dict(
        embed_dim=32,
        hidden_dim=64,
        K=2,
        pooling="mean",
        vocab_min_id=-10,
        pad_id=99,
        pi_sizes=[128],
        vf_sizes=[128],
        max_nodes=20,  # From covEnv max_expr_length
        max_edges=40   # 2 * max_nodes
    )
    if tree_kwargs:
        tkw.update({k: v for k, v in tree_kwargs.items() if v is not None})

    kwargs = dict(
        features_extractor_class=TreeMLPExtractor,
        features_extractor_kwargs=dict(
            max_nodes=tkw["max_nodes"],
            max_edges=tkw["max_edges"],
            vocab_min_id=tkw["vocab_min_id"],
            pad_id=tkw["pad_id"],
            embed_dim=tkw["embed_dim"],
            hidden_dim=tkw["hidden_dim"],
            K=tkw["K"],
            pooling=tkw["pooling"],
        ),
        net_arch=dict(pi=tkw["pi_sizes"], vf=tkw["vf_sizes"]),
    )
    return PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=kwargs,
        verbose=verbose,
        seed=seed,
        n_steps=n_steps,
        learning_rate=3e-4,
        gamma=1.0,
        gae_lambda=0.95,
        ent_coef=ent_coef
    )

# ---------------------------------------------------------------------
# Logging callbacks
# ---------------------------------------------------------------------
class AccuracyLoggingCallback(BaseCallback):
    """
    Logs train/test accuracies every log_interval timesteps and saves
    the results to a CSV file at the end of training.
    """
    def __init__(self, env, model, algo, log_interval, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.model = model
        self.log_interval = max(1, int(log_interval))
        self.algo_name = algo

        # Create a unique directory for this run to save data. The PID is
        # included so parallel runs (a sweep) cannot collide on the same
        # second-resolution timestamp and clobber each other's best_model.zip
        # / accuracies.csv. A dataset tag is added for readability.
        timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        ds_tag = ""
        try:
            dp = getattr(env, "dataset_path", None)
            if dp:
                ds_tag = os.path.basename(str(dp).rstrip("/")) + "_"
        except Exception:
            ds_tag = ""
        dir_name = f"{self.algo_name}_{ds_tag}{timestamp}_{os.getpid()}"
        self.save_path = os.path.join("gemini", "cov", dir_name)
        os.makedirs(self.save_path, exist_ok=True)

        # Initialize lists to store data
        self.timesteps = []
        self.train_greedy_accs = []
        self.train_beam_accs = []
        self.test_greedy_accs = []
        self.test_beam_accs = []

    def _on_step(self) -> bool:
        # Log data at the specified interval
        if self.num_timesteps % self.log_interval == 0 and self.num_timesteps > 0:

            # Compute accuracies
            train_greedy_acc = test_greedy(self.env, self.model, max_steps=5, test_eqns=self.env.train_eqns)
            train_beam_acc = beam_search(self.env, self.model, test_eqns=self.env.train_eqns)
            test_greedy_acc = test_greedy(self.env, self.model, max_steps=5, test_eqns=self.env.test_eqns)
            test_beam_acc = beam_search(self.env, self.model, test_eqns=self.env.test_eqns)

            # Store accuracies
            self.timesteps.append(self.num_timesteps)
            self.train_greedy_accs.append(train_greedy_acc)
            self.train_beam_accs.append(train_beam_acc)
            self.test_greedy_accs.append(test_greedy_acc)
            self.test_beam_accs.append(test_beam_acc)

            step = self.num_timesteps
            timed_print(
                f"[{self.algo_name}] t={step}: "
                f"train_greedy={train_greedy_acc:.3f} | "
                f"train_beam={train_beam_acc:.3f} | "
                f"test_greedy={test_greedy_acc:.3f} | "
                f"test_beam={test_beam_acc:.3f}"
            )

            # Save CSV immediately
            data = {
                "timesteps": self.timesteps,
                "train_greedy_acc": self.train_greedy_accs,
                "train_beam_acc": self.train_beam_accs,
                "test_greedy_acc": self.test_greedy_accs,
                "test_beam_acc": self.test_beam_accs,
            }
            df = pd.DataFrame(data)
            filepath = os.path.join(self.save_path, "accuracies.csv")
            df.to_csv(filepath, index=False)

            # Best-checkpoint: keep the model at its peak test_beam. pi_cov
            # training tends to peak early then collapse, so the final-step
            # model is usually not the best one.
            if test_beam_acc > getattr(self, "_best_beam", -1.0):
                self._best_beam = test_beam_acc
                self.model.save(os.path.join(self.save_path, "best_model.zip"))
                timed_print(f"[{self.algo_name}] new best test_beam={test_beam_acc:.3f}"
                            f" -> best_model.zip")

        return True

    def _on_training_end(self) -> None:
        """
        Saves the collected accuracy data to a CSV file.
        """
        data = {
            "timesteps": self.timesteps,
            "train_greedy_acc": self.train_greedy_accs,
            "train_beam_acc": self.train_beam_accs,
            "test_greedy_acc": self.test_greedy_accs,
            "test_beam_acc": self.test_beam_accs,
        }
        df = pd.DataFrame(data)
        filepath = os.path.join(self.save_path, "accuracies.csv")
        df.to_csv(filepath, index=False)
        if self.verbose > 0:
            print("-" * 50)
            print(f"✅ Logged accuracies to: {filepath}")
            print("-" * 50)

class EntropyAnnealCallback(BaseCallback):
    def __init__(self, start=0.01, end=0.0, total_timesteps=100_000, by_rollout=False, verbose=0):
        super().__init__(verbose)
        self.start = float(start)
        self.end = float(end)
        self.total_timesteps = int(total_timesteps)
        self.by_rollout = bool(by_rollout)

    def _progress(self):
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

# ---------------------------------------------------------------------
# Success replay buffer (+ saver)
# ---------------------------------------------------------------------
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
            idx = random.sample(range(n), k=batch_size)
        else:
            idx = np.random.randint(0, n, size=batch_size).tolist()
        sample0 = self.obs[idx[0]]
        if isinstance(sample0, dict):
            # graph state_rep: stack per-key into dict of arrays
            obs_b = {k: np.stack([np.asarray(self.obs[i][k]) for i in idx], axis=0)
                     for k in sample0.keys()}
        else:
            obs_b = np.stack([self.obs[i] for i in idx], axis=0).astype(np.float32)
        act_b = np.array([self.act[i] for i in idx], dtype=np.int64)
        return obs_b, act_b

    def __len__(self):
        return len(self.obs)

    def save(self, path):
        """Save entire buffer to a pickle (shape-agnostic)."""
        payload = {
            "obs": list(self.obs),
            "act": list(self.act),
            "length": len(self.obs),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path

class SuccessReplayCallback(BaseCallback):
    def __init__(
        self,
        mix_ratio=0.15,
        batch_size=256,
        iters_per_rollout=1,
        capacity=10000,
        verbose=0,
    ):
        super().__init__(verbose)
        self.mix_ratio = float(mix_ratio)
        self.batch_size = int(batch_size)
        self.iters_per_rollout = int(iters_per_rollout)
        self.buf = SuccessBuffer(capacity=capacity)

        # For coverage (eqn-level) and trace dedup (eqn+actions)
        self.seen_eqns = set()
        self.seen_traces = set()  # keys are (str(main_eqn), tuple(traj_act))

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            if info.get("success"):
                main_eqn = info.get("main_eqn")
                if main_eqn not in self.seen_eqns:
                    self.seen_eqns.add(main_eqn)
                    print(f't={self.num_timesteps} | Solved {main_eqn} | Coverage = {info.get("coverage"):.3f}')

                traj_obs = info.get("traj_obs", None)
                traj_act = info.get("traj_act", None)

                # Dedup on (equation, action sequence)
                if traj_obs is not None and traj_act is not None and len(traj_obs) == len(traj_act):
                    key = (str(main_eqn), tuple(map(int, np.asarray(traj_act).tolist())))
                    if key not in self.seen_traces:
                        self.seen_traces.add(key)
                        self.buf.add_episode(traj_obs, traj_act)
        return True

    @torch.no_grad()
    def _policy_logits(self, obs_tensor):
        dist = self.model.policy.get_distribution(obs_tensor)
        return dist.distribution.logits

    def _supervised_update(self, obs_batch, act_batch):
        device = self.model.policy.device
        # Handle dict obs (graph state_rep) vs array obs (integer_1d)
        if isinstance(obs_batch, dict):
            obs_t = {k: torch.as_tensor(v, device=device) for k, v in obs_batch.items()}
        else:
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
            empty = (isinstance(obs_b, dict) and len(obs_b) == 0) or \
                    (hasattr(obs_b, "size") and obs_b.size == 0)
            if empty:
                break
            lbc, ent = self._supervised_update(obs_b, act_b)
            bc_losses.append(lbc)
            ents.append(ent)
        if self.verbose and bc_losses:
            print(f"[SuccessReplay] size={len(self.buf)} bc_steps={len(bc_losses)} "
                  f"bc_loss={np.mean(bc_losses):.4f} ent={np.mean(ents):.3f}")



class SuccessReplayCallbackOld(BaseCallback):
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
        # Handle dict obs (graph state_rep) vs array obs (integer_1d)
        if isinstance(obs_batch, dict):
            obs_t = {k: torch.as_tensor(v, device=device) for k, v in obs_batch.items()}
        else:
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

# ---------------------------------------------------------------------
# Evaluation helpers (BUGFIX: per-eqn branches now use temp_env.step)
# ---------------------------------------------------------------------
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
                f_penalty=env.f_penalty, hist_len=env.hist_len, multi_eqn=False, use_curriculum=False,
                state_rep=env.state_rep
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
                    obs, reward, done, truncated, info = temp_env.step(action)  # FIXED
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
        print(f"DEBUG: env.state_rep={env.state_rep}")
        print('Greedy rollout')
        while not (done or truncated) and steps < max_steps:
            # print(f"DEBUG: obs type={type(obs)}")
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
                f_penalty=env.f_penalty, hist_len=env.hist_len, multi_eqn=False, use_curriculum=False,
                state_rep=env.state_rep
            )
            obs, info = temp_env.reset(seed=SEED)
            done, truncated = False, False
            steps, total_reward = 0, 0.0
            while not (done or truncated) and steps < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = temp_env.step(action)  # FIXED
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
                f_penalty=env.f_penalty, hist_len=env.hist_len, multi_eqn=False, use_curriculum=False,
                state_rep=env.state_rep
            )
            success_any = False
            best_delta = -1e9
            for t in range(n_trials):
                obs, info = temp_env.reset(seed=SEED+t)
                done, truncated = False, False
                steps, total_reward = 0, 0.0
                while not (done or truncated) and steps < max_steps:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, done, truncated, info = temp_env.step(action)  # FIXED
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

# ---------------------------------------------------------------------
# Beam search — matches env's compose semantics: f(x) = base_op(x, cov_inner)
# Tracks action_history per beam entry so policy sees in-distribution states.
# ---------------------------------------------------------------------
_BEAM_OPS = {
    "ADD": lambda a, b: a + b,
    "SUB": lambda a, b: a - b,
    "MUL": lambda a, b: a * b,
    "DIV": lambda a, b: a / b,
}


def _beam_search_one(eqn_env, model, width, max_depth):
    """Run beam search on a single env (already pinned to one equation).

    Returns (best_delta, best_f_or_None).

    Beam entry = (score, depth, base_op, cov_inner, action_history_tuple).
      - base_op is None until the first action; once set, stays fixed (mirrors env).
      - cov_inner is built compositionally via base_op-agnostic ops on subsequent actions.
      - f(x) = base_op(x, cov_inner) at evaluation time.
      - action_history_tuple is the last hist_len actions taken on this path.
    """
    from heapq import nlargest
    xsym = eqn_env.x
    main_eqn = eqn_env.main_eqn
    base_cmplx = C(main_eqn)
    init_history = (-1,) * eqn_env.hist_len

    # initial beam: empty path
    beam = [(0.0, 0, None, sp.Integer(0), init_history)]
    best = (-1e9, None)

    for depth in range(max_depth + 1):
        cand = []
        for score, d, base_op, cov_inner, hist in beam:
            # Evaluate this path's f(x) and delta
            if base_op is None:
                f = xsym  # no actions => identity sub, delta=0
            elif base_op == "LOGX":
                f = sp.log(xsym)            # unary op: f(x) = log(x)
            else:
                f = _BEAM_OPS[base_op](xsym, cov_inner)
            try:
                after = sp.simplify(sp.sympify(main_eqn).subs(xsym, f))
                delta = base_cmplx - C(after)
            except Exception:
                delta = -1e9
            if delta > best[0]:
                best = (delta, f)
            if d == max_depth:
                continue
            # Query policy with state reflecting this path
            eqn_env.cov = cov_inner if base_op is not None else sp.Integer(0)
            eqn_env.depth = d
            eqn_env.action_history = list(hist)
            obs_core = eqn_env.to_vec(eqn_env.main_eqn, 0)[0]
            if not isinstance(obs_core, dict):
                obs_core = obs_core.astype(np.float32)
            obs = eqn_env._augment_obs(obs_core)
            obs_t = model.policy.obs_to_tensor(obs)[0]
            dist = model.policy.get_distribution(obs_t)
            logits = dist.distribution.logits.detach().cpu().numpy().squeeze()
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            topk = np.argsort(-probs)[:width]
            for a in topk:
                op, tau = eqn_env.actions[int(a)]
                if op == "STOP":
                    # STOP terminates without advancing cov; the path's current
                    # delta is already captured above. Skip expansion.
                    continue
                try:
                    if op == "LOGX":
                        # unary base op: f(x) = log(x); cov_inner unused
                        new_base_op = "LOGX"
                        new_cov_inner = sp.Integer(0)
                    elif base_op is None:
                        # first action: pin base_op and set cov_inner = tau
                        new_base_op = op
                        new_cov_inner = sp.sympify(tau)
                    elif base_op == "LOGX":
                        new_base_op = "LOGX"
                        new_cov_inner = sp.Integer(0)
                    else:
                        new_base_op = base_op
                        new_cov_inner = sp.simplify(_BEAM_OPS[op](cov_inner, tau))
                except Exception:
                    continue
                new_hist = tuple(list(hist)[1:] + [int(a)])
                new_score = score + float(np.log(probs[a] + 1e-12))
                cand.append((new_score, d + 1, new_base_op, new_cov_inner, new_hist))
        beam = nlargest(width, cand, key=lambda z: z[0])
        if not beam:
            break

    return best


def beam_search(env, model, width=3, max_depth=3, test_eqns=None):
    """Beam search over CoV action paths.

    - If `test_eqns is None`: searches on `env.main_eqn`, returns (best_delta, best_f).
    - Else: iterates each test equation in a single-equation env clone and returns
      the success rate (delta > 0).
    """
    if test_eqns is None:
        return _beam_search_one(env, model, width, max_depth)

    success_count = 0
    for eqn in test_eqns:
        temp_env = covEnv(
            eqn, env.term_bank, max_depth=env.max_depth, step_penalty=env.step_penalty,
            f_penalty=env.f_penalty, hist_len=env.hist_len, multi_eqn=False,
            use_curriculum=False, state_rep=env.state_rep,
            dataset_path=getattr(env, "dataset_path", None),
        )
        best_delta, _ = _beam_search_one(temp_env, model, width, max_depth)
        if best_delta > 0:
            success_count += 1
    return success_count / len(test_eqns)


# ---------------------------------------------------------------------
# Expert demonstrations (RL-from-demonstrations): seed the success-replay
# buffer with the known-correct CoV substitution traces.
# ---------------------------------------------------------------------
def _read_lines(path, keep_blank=False):
    out = []
    with open(path) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            s = ln.rstrip("\n")
            if s.strip() or keep_blank:
                out.append(s)
    return out


def _trace_to_action_indices(trace, actions):
    """Convert 'SUB:f DIV:3 DIV:a STOP' into covEnv action indices."""
    idxs = []
    for tok in trace.split():
        if ":" not in tok:          # unary actions: STOP, LOGX
            op, term = tok, None
        else:
            op, t = tok.split(":", 1)
            term = sp.sympify(t)
        match = None
        for i, (aop, aterm) in enumerate(actions):
            if aop != op:
                continue
            if (term is None and aterm is None) or \
               (term is not None and aterm is not None and aterm == term):
                match = i
                break
        if match is None:
            raise ValueError(f"trace token '{tok}' has no matching covEnv action")
        idxs.append(match)
    return idxs


def collect_expert_demos(dataset_path, term_bank, max_depth, step_penalty,
                         f_penalty, hist_len, state_rep):
    """Run covEnv through each train equation's expert trace; return the
    (traj_obs, traj_act) episodes for the ones that solve. One covEnv is
    reused -- its feature_dict matches the training env (same dataset_path)."""
    eqns   = _read_lines(f"{dataset_path}/train_eqns.txt")
    traces = _read_lines(f"{dataset_path}/train_traces.txt", keep_blank=True)
    if len(eqns) != len(traces):
        raise ValueError(f"train_eqns/train_traces length mismatch: "
                         f"{len(eqns)} vs {len(traces)}")
    env = covEnv(main_eqn=sp.sympify(eqns[0]), term_bank=term_bank,
                 max_depth=max_depth, step_penalty=step_penalty,
                 f_penalty=f_penalty, hist_len=hist_len, multi_eqn=False,
                 use_curriculum=False, state_rep=state_rep,
                 dataset_path=dataset_path)
    demos, n_ok, n_fail = [], 0, 0
    for eqn_s, trace in zip(eqns, traces):
        trace = trace.strip()
        if not trace:
            continue
        try:
            env.main_eqn = sp.sympify(eqn_s)
            env.reset()                       # multi_eqn=False -> keeps main_eqn
            idxs = _trace_to_action_indices(trace, env.actions)
            info = {}
            for ai in idxs:
                _, _, _, _, info = env.step(ai)
            if info.get("success"):
                demos.append((info["traj_obs"], info["traj_act"]))
                n_ok += 1
            else:
                n_fail += 1
        except Exception:
            n_fail += 1
    return demos, n_ok, n_fail


# ---------------------------------------------------------------------
# Train / Save
# ---------------------------------------------------------------------
def main(args):
    t1 = time.time()
    print("\n" + "="*70)
    print(f"[Train] agent: {args.agent}")
    print("="*70)

    # Parse main_eqn and term_bank
    main_eqn = sp.sympify(args.main_eqn)
    term_bank = [sp.sympify(t) for t in args.term_bank.split(',')]
    state_rep = 'integer_1d' if 'tree' not in args.agent else 'graph_integer_1d'

    # Environment
    env = covEnv(
        main_eqn=main_eqn,
        term_bank=term_bank,
        max_depth=args.max_depth,
        step_penalty=args.step_penalty,
        f_penalty=args.f_penalty,
        state_rep=state_rep,
        hist_len=args.hist_len,
        multi_eqn=args.multi_eqn,
        use_curriculum=args.use_curriculum,
        gen=args.gen,
        dataset_path=args.dataset_path,
    )
    print(f'\nTrain eqns, test eqns = {len(env.train_eqns)}, {len(env.test_eqns)}\n')

    # Model
    tree_kwargs = {
        'embed_dim': 32,
        'hidden_dim': 64,
        'K': 2,
        'pooling': 'mean',
        'vocab_min_id': -10,
        'pad_id': 99,
        'pi_sizes': [128],
        'vf_sizes': [128],
    }
    model = make_agent(agent=args.agent, env=env, hidden_dim=128, n_steps=args.n_steps,
                       ent_coef=args.ent_coef, seed=SEED, tree_kwargs=tree_kwargs,
                       verbose=args.verbose)

    # Callbacks
    cb_ent = EntropyAnnealCallback(start=args.ent_start, end=args.ent_end, total_timesteps=args.Ntrain, by_rollout=True)
    cb_progress = ProgressBarCallback()
    cb_accuracy = AccuracyLoggingCallback(env, model, args.agent,
                                          log_interval=(args.log_interval or max(1, args.Ntrain // 10)))
    callbacks = [cb_ent, cb_progress, cb_accuracy]

    cb_replay = None
    if 'mem' in args.agent:
        cb_replay = SuccessReplayCallback(
            mix_ratio=args.sr_mix_ratio, batch_size=args.sr_batch_size,
            iters_per_rollout=args.sr_iters, capacity=args.sr_capacity, verbose=0)
        callbacks.append(cb_replay)

    # RL-from-demonstrations: pre-seed the success buffer with expert traces.
    if getattr(args, "seed_demos", False) and cb_replay is not None and args.dataset_path:
        demos, n_ok, n_fail = collect_expert_demos(
            args.dataset_path, term_bank, args.max_depth, args.step_penalty,
            args.f_penalty, args.hist_len, state_rep)
        for traj_obs, traj_act in demos:
            cb_replay.buf.add_episode(traj_obs, traj_act)
        print(f"[seed_demos] seeded buffer with {n_ok} expert demos "
              f"({n_fail} traces failed to solve); buffer size = {len(cb_replay.buf)}")

        # BC pretraining: imitate the demos to convergence BEFORE PPO, so the
        # policy starts good instead of being churned by PPO from step 0.
        if args.bc_pretrain_iters > 0 and len(cb_replay.buf) > 0:
            cb_replay.model = model        # enable _supervised_update pre-learn()
            last = 0.0
            for it in range(args.bc_pretrain_iters):
                ob, ac = cb_replay.buf.sample(cb_replay.batch_size)
                last, _ = cb_replay._supervised_update(ob, ac)
                if it % 2000 == 0:
                    print(f"[bc_pretrain] iter {it:6d}  bc_loss={last:.4f}", flush=True)
            print(f"[bc_pretrain] done {args.bc_pretrain_iters} iters, "
                  f"final bc_loss={last:.4f}", flush=True)
            # Evaluate + save the pure BC-pretrained policy BEFORE any PPO,
            # so its true quality is recorded (PPO tends to erode it).
            _tg = test_greedy(env, model, max_steps=5, test_eqns=env.test_eqns)
            _tb = beam_search(env, model, test_eqns=env.test_eqns)
            print(f"[bc_pretrain] post-pretrain eval: "
                  f"test_greedy={_tg:.3f}  test_beam={_tb:.3f}", flush=True)
            try:
                model.save(os.path.join(cb_accuracy.save_path, "bc_pretrained.zip"))
            except Exception as _e:
                print(f"[bc_pretrain] save warning: {_e}", flush=True)

    if args.curiosity not in ['None','none'] and 'tree' not in args.agent:
        #irs = get_intrinsic_reward(args.curiosity, env)
        irs = get_intrinsic_reward(args.curiosity, make_vec_env(lambda: env, n_envs=1, seed=SEED))
        if irs:
            callbacks.append(IntrinsicReward(irs))

    # Training
    model.learn(total_timesteps=args.Ntrain, callback=callbacks)

    # -----------------------------
    # Save model + success buffer
    # -----------------------------
    # Use the same directory as accuracy logger (so everything for this run is together)
    save_dir = cb_accuracy.save_path  # created in AccuracyLoggingCallback
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "model.zip")
    model.save(model_path)

    if cb_replay is not None:
        buf_path = os.path.join(save_dir, "success_buffer.pkl")
        cb_replay.buf.save(buf_path)
        # quick CSV summary
        pd.DataFrame({"buffer_size": [len(cb_replay.buf)]}).to_csv(
            os.path.join(save_dir, "success_buffer_summary.csv"), index=False
        )

    t2 = time.time()
    print("-" * 50)
    print(f"✅ Saved model to: {model_path}")
    if cb_replay is not None:
        print(f"✅ Saved success buffer to: {buf_path} (size={len(cb_replay.buf)})")
    print(f"⏱️ Took {(t2 - t1)/60.0:.2f} mins")
    print(f"📁 Run artifacts directory: {save_dir}")
    print("-" * 50)
    return

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO agents for the symbolic math environment (covEnv).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core Training
    training_group = parser.add_argument_group('Core Training Settings')
    training_group.add_argument('--Ntrain', type=int, default=5*10**6,
                                help='Total number of training timesteps.')

    # Environment
    env_group = parser.add_argument_group('Environment Configuration')
    env_group.add_argument('--main_eqn', type=str, default='a*x**2 + b*x + c',
                           help='Primary equation to solve (in SymPy format).')
    env_group.add_argument('--multi_eqn', action='store_true',
                           help='If set, use a variety of equations instead of just the main one.')
    env_group.add_argument('--use_curriculum', action='store_true',
                           help='If set, use curriculum learning to gradually increase equation difficulty.')
    env_group.add_argument('--term_bank', type=str, default='a,b,c,d,e,2,3,4',
                           help='Comma-separated list of available terms/constants for the environment.')
    env_group.add_argument('--max_depth', type=int, default=3,
                           help='Maximum expression tree depth for generated equations.')
    env_group.add_argument('--gen', type=int, default=5,
                           help='Number of symbolic constants (a, b, c, ...) to use.')
    env_group.add_argument('--dataset_path', type=str, default=None,
                           help='Override gen-based lookup; load train/test_eqns.txt from this dir.')
    env_group.add_argument('--hist_len', type=int, default=10,
                           help='Number of previous states to include in the observation history.')
    env_group.add_argument('--step_penalty', type=float, default=0.1,
                           help='Negative reward applied at each step to encourage shorter solutions.')
    env_group.add_argument('--f_penalty', type=float, default=0.0,
                           help='Penalty for using complex functions (e.g., sin, log).')

    # Agent / Model
    agent_group = parser.add_argument_group('Agent & Model Configuration')
    agent_group.add_argument('--agent', type=str, default='ppo-mem',
                             choices=['ppo', 'ppo-mem', 'ppo-tree', 'ppo-tree-mem'],
                             help='The agent architecture to use.')
    agent_group.add_argument('--n_steps', type=int, default=2048,
                             help='Number of steps to run for each environment per PPO update (rollout buffer size).')
    agent_group.add_argument('--ent_coef', type=float, default=0.00,
                             help='Entropy coefficient for PPO (initial value if annealing).')
    agent_group.add_argument('--ent_start', type=float, default=0.02,
                             help='EntropyAnneal start coefficient.')
    agent_group.add_argument('--ent_end', type=float, default=0.0,
                             help='EntropyAnneal end coefficient; a non-zero floor keeps exploration alive.')
    agent_group.add_argument('--curiosity', type=str, default='none')

    # Success replay (only used by 'mem' agents)
    sr_group = parser.add_argument_group('Success Replay')
    sr_group.add_argument('--sr_mix_ratio', type=float, default=0.80,
                          help='Fraction of each update drawn from the success buffer (the frac_from_buffer knob).')
    sr_group.add_argument('--sr_batch_size', type=int, default=128)
    sr_group.add_argument('--sr_iters', type=int, default=40,
                          help='BC iterations per rollout from the success buffer.')
    sr_group.add_argument('--sr_capacity', type=int, default=100000)
    sr_group.add_argument('--seed_demos', action='store_true',
                          help='Pre-seed the success buffer with expert CoV traces (RL-from-demonstrations).')
    sr_group.add_argument('--bc_pretrain_iters', type=int, default=0,
                          help='Supervised BC pretraining iterations on the expert demos before PPO (0 = off).')

    # Logging
    log_group = parser.add_argument_group('Logging Settings')
    log_group.add_argument('--log_interval', type=int, default=None,
                           help='Interval for logging training progress. Defaults to Ntrain / 10.')
    log_group.add_argument('--verbose', type=int, default=1,
                           help='PPO verbosity (0=silent, 1=per-rollout stats, 2=debug). Default 1.')

    # Reproducibility
    repro_group = parser.add_argument_group('Reproducibility')
    repro_group.add_argument('--seed', type=int, default=0,
                             help='Master RNG seed (Python random, numpy, torch, env reset).')

    args = parser.parse_args()
    args.multi_eqn = True
    args.use_curriculum = True

    # Apply seed override
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if args.log_interval is None:
        args.log_interval = args.Ntrain // 50

    print_parameters(vars(args))
    main(args)
