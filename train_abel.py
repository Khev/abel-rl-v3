#!/usr/bin/env python3
import os
import sys
# Disable Python 3.11+ int-to-string conversion limit. SymPy can produce
# huge integers during simplification (e.g. via heuristic GCD), and printing
# them (e.g., in info["main_eqn"]=str(...)) raises ValueError under the cap.
try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass
import argparse
import json
import time
import signal
import datetime
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from heapq import nlargest
import os

import torch
from gymnasium import ObservationWrapper, spaces

# Keep CPU threads modest when running many workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

# --- Envs ---
from envs.env_single_eqn_fixed import singleEqn
from envs.env_multi_eqn_fixed import multiEqn
from envs.env_multi_eqn import multiEqn as multiEqnDynamic
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# --- SB3 ---
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from stable_baselines3.common.base_class import BaseAlgorithm

# --- Curiosity (rllte) ---
from rllte.xplore.reward import E3B, ICM, NGU, RE3, RIDE, RND

# --- Custom extractor ---
from utils.utils_env import TreeMLPExtractor, GCNExtractor, GraphSAGEExtractor
from utils.utils_general import print_parameters
from utils.utils_train import timed_print

# ==========================
# Global knobs
# ==========================
TRIAL_WALLCLOCK_LIMIT = 7 * 24 * 60 * 60  # 7 days hard cap per trial
EVAL_TIMEOUT_DET   = 0.75  # per-equation budget for greedy/beam
EVAL_TIMEOUT_STOCH = 1.00  # per-equation budget for success@N

class EvalTimeout(Exception):
    pass

def _unwrap_attr(env, name):
    # VecEnv path (SB3)
    if hasattr(env, "get_attr"):
        vals = env.get_attr(name)
        return vals[0] if vals else None
    # Gymnasium wrapper chain
    if hasattr(env, "get_wrapper_attr"):
        return env.get_wrapper_attr(name)
    # Plain gym env
    if hasattr(env, "unwrapped"):
        return getattr(env.unwrapped, name, None)
    return getattr(env, name, None)

@contextmanager
def time_limit(seconds: float):
    """Raise EvalTimeout if the with-block exceeds `seconds` (POSIX only)."""
    if os.name == "nt" or not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise EvalTimeout()

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)

def get_device():
    """Returns CUDA device index or 'cpu'."""
    if torch.cuda.is_available():
        print("Found CUDA: using GPU")
        cur_proc_identity = mp.current_process()._identity
        return (cur_proc_identity[0] - 1) % torch.cuda.device_count() if cur_proc_identity else 0
    else:
        #print("CUDA not found: using CPU")
        return "cpu"

def get_intrinsic_reward(intrinsic_reward: Optional[str], vec_env):
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

class NodeFeatureView(ObservationWrapper):
    """Extracts Box(node_features) from a Dict observation for curiosity modules."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["node_features"]
    def observation(self, obs):
        return np.asarray(obs["node_features"], dtype=np.float32)

def obs_to_curiosity_array(obs) -> np.ndarray:
    """Make observations (including Dict) into np.array for curiosity."""
    if isinstance(obs, dict):
        return np.asarray(obs["node_features"], dtype=np.float32)
    if isinstance(obs, (list, tuple)) and obs and isinstance(obs[0], dict):
        return np.stack([np.asarray(o["node_features"], dtype=np.float32) for o in obs], axis=0)
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32)
    if hasattr(obs, "keys") and "node_features" in obs:
        return np.asarray(obs["node_features"], dtype=np.float32)
    raise TypeError(f"Unsupported obs type for curiosity: {type(obs)}")

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
        return True

    def _on_rollout_end(self) -> None:
        try:
            device = self.irs.device
            obs_rb = self.buffer.observations

            next_obs_rb = {}
            if isinstance(obs_rb, dict):
                for k, arr in obs_rb.items():
                    nxt = np.copy(arr)
                    nxt[:-1] = arr[1:]
                    nxt[-1]  = self.locals["new_obs"][k]
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

# ==========================
# Success Replay (optional) — DEDUP VERSION
# ==========================
import random
from collections import deque

class SuccessBuffer:
    """Original flat buffer. Uniform random sampling across all (obs, action) pairs."""
    def __init__(self, capacity=20000):
        self.obs = deque(maxlen=capacity)
        self.act = deque(maxlen=capacity)

    def add_episode(self, traj_obs, traj_act, eqn_key=None):
        for o, a in zip(traj_obs, traj_act):
            self.obs.append(o)
            self.act.append(int(a))

    def sample(self, batch_size):
        n = len(self.obs)
        if n == 0:
            return None, None
        idx = random.sample(range(n), k=min(batch_size, n))
        obs_b = [self.obs[i] for i in idx]
        act_b = np.array([self.act[i] for i in idx], dtype=np.int64)
        return obs_b, act_b

    def __len__(self):
        return len(self.obs)

    def maintenance(self):
        """Called periodically by the callback. Subclasses can override."""
        pass


class BalancedSuccessBuffer:
    """Per-equation balanced sampling: pick an eqn uniformly, then a (obs, action)
    pair within that eqn's traces. Prevents over-represented classes (e.g., common
    abel_level3 forms) from dominating BC updates.

    Optional length-weighting: when sampling within an eqn's traces, prefer
    shorter ones (which represent cleaner recipes).
    """
    def __init__(self, capacity=20000, per_eqn_cap=200, length_weighted=True):
        self.capacity = int(capacity)
        self.per_eqn_cap = int(per_eqn_cap)
        self.length_weighted = bool(length_weighted)
        # Per-eqn dict: eqn_key -> list of (obs, action, trace_length) tuples
        from collections import defaultdict
        self._per_eqn = defaultdict(list)
        self._total = 0

    def add_episode(self, traj_obs, traj_act, eqn_key=None):
        if eqn_key is None:
            eqn_key = "__default__"
        L = len(traj_obs)
        entries = self._per_eqn[eqn_key]
        for o, a in zip(traj_obs, traj_act):
            entries.append((o, int(a), L))
            self._total += 1
        # Cap per-eqn size: keep most recent entries
        if len(entries) > self.per_eqn_cap:
            drop = len(entries) - self.per_eqn_cap
            self._total -= drop
            del entries[:drop]
        # Global capacity: drop entire eqns if we're over
        while self._total > self.capacity and self._per_eqn:
            # Drop the eqn with the most entries (most over-represented)
            drop_key = max(self._per_eqn, key=lambda k: len(self._per_eqn[k]))
            self._total -= len(self._per_eqn[drop_key])
            del self._per_eqn[drop_key]

    def sample(self, batch_size):
        if not self._per_eqn:
            return None, None
        eqns = list(self._per_eqn.keys())
        obs_list, act_list = [], []
        for _ in range(batch_size):
            # Pick an eqn uniformly
            key = random.choice(eqns)
            entries = self._per_eqn[key]
            if not entries:
                continue
            # Pick an entry, optionally weighted by 1/length
            if self.length_weighted and len(entries) > 1:
                # Weight by 1/L; shorter traces preferred
                weights = np.array([1.0 / max(1, e[2]) for e in entries])
                weights = weights / weights.sum()
                idx = int(np.random.choice(len(entries), p=weights))
            else:
                idx = random.randrange(len(entries))
            o, a, _L = entries[idx]
            obs_list.append(o)
            act_list.append(a)
        if not obs_list:
            return None, None
        return obs_list, np.array(act_list, dtype=np.int64)

    def __len__(self):
        return self._total

    def maintenance(self):
        pass


class FreshSuccessBuffer(SuccessBuffer):
    """Periodic refresh: drop oldest half every `refresh_every` calls to
    maintenance(). Combats staleness — entries the policy has already memorized
    stop wasting BC compute. Otherwise identical to SuccessBuffer.
    """
    def __init__(self, capacity=20000, refresh_every=20):
        super().__init__(capacity=capacity)
        self.refresh_every = int(refresh_every)
        self._calls = 0

    def maintenance(self):
        self._calls += 1
        if self._calls % self.refresh_every == 0 and len(self.obs) > 2:
            # Drop oldest half
            n = len(self.obs)
            drop = n // 2
            for _ in range(drop):
                self.obs.popleft()
                self.act.popleft()


def make_success_buffer(kind: str, capacity: int = 20000):
    """Factory for success-buffer variants."""
    kind = (kind or "flat").lower()
    if kind == "flat":
        return SuccessBuffer(capacity=capacity)
    elif kind == "balanced":
        return BalancedSuccessBuffer(capacity=capacity, length_weighted=False)
    elif kind == "balanced_lw":
        return BalancedSuccessBuffer(capacity=capacity, length_weighted=True)
    elif kind == "fresh":
        return FreshSuccessBuffer(capacity=capacity, refresh_every=20)
    else:
        raise ValueError(f"Unknown buffer kind: {kind}")

class SuccessReplayCallback(BaseCallback):
    """
    Harvests solved trajectories and runs small behavior-cloning (BC) updates
    between PPO rollouts. Adds to buffer only UNIQUE traces per
    (equation, action sequence).
    """
    def __init__(
        self,
        mix_ratio=0.5,
        batch_size=256,
        iters_per_rollout=10,
        capacity=20000,
        buffer_kind="flat",
        verbose=0
    ):
        super().__init__(verbose)
        self.mix_ratio = float(mix_ratio)
        self.batch_size = int(batch_size)
        self.iters_per_rollout = int(iters_per_rollout)
        self.buffer_kind = str(buffer_kind)
        self.buf = make_success_buffer(self.buffer_kind, capacity=capacity)
        self.seen_eqns = set()
        self.seen_traces = set()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            if info.get("is_solved"):
                main_eqn = info.get("main_eqn")
                if main_eqn not in self.seen_eqns:
                    self.seen_eqns.add(main_eqn)
                    cov = info.get("coverage", None)
                traj_obs = info.get("traj_obs", None)
                traj_act = info.get("traj_act", None)
                if traj_obs is not None and traj_act is not None and len(traj_obs) == len(traj_act):
                    act_tuple = tuple(int(a) for a in np.asarray(traj_act).tolist())
                    key = (str(main_eqn), act_tuple)
                    if key not in self.seen_traces:
                        self.seen_traces.add(key)
                        self.buf.add_episode(traj_obs, traj_act, eqn_key=str(main_eqn))
        return True

    # @torch.no_grad()
    # def _obs_list_to_tensor(self, obs_list):
    #     obs_tensor, _ = self.model.policy.obs_to_tensor(obs_list)
    #     return obs_tensor

    @torch.no_grad()
    def _obs_list_to_tensor(self, obs_list):
        # Handle empty
        if not obs_list:
            # Create a tiny dummy batch to keep shapes consistent
            dummy_obs = self.training_env.reset()[0]
            if isinstance(dummy_obs, dict):
                dummy_obs = {k: np.asarray(v)[None, ...] for k, v in dummy_obs.items()}
            else:
                dummy_obs = np.asarray(dummy_obs)[None, ...]
            obs_t, _ = self.model.policy.obs_to_tensor(dummy_obs)
            return obs_t

        sample = obs_list[0]
        if isinstance(sample, dict):
            # Collate list[dict] -> dict[np.array] with leading batch dim
            batch = {k: np.stack([np.asarray(o[k]) for o in obs_list], axis=0) for k in sample.keys()}
        else:
            # Collate list[np.array] -> np.array with leading batch dim
            batch = np.stack([np.asarray(o) for o in obs_list], axis=0)

        obs_t, _ = self.model.policy.obs_to_tensor(batch)
        return obs_t


    def _supervised_step(self, obs_batch, act_batch):
        device = self.model.policy.device
        obs_t = self._obs_list_to_tensor(obs_batch)
        act_t = torch.as_tensor(act_batch, device=device)
        self.model.policy.optimizer.zero_grad(set_to_none=True)
        dist = self.model.policy.get_distribution(obs_t)
        logp = dist.log_prob(act_t)
        loss_bc = -logp.mean()
        ent = dist.entropy().mean()
        loss = loss_bc - 1e-3 * ent
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 0.5)
        self.model.policy.optimizer.step()
        return float(loss_bc.item()), float(ent.item())

    def _on_rollout_end(self) -> None:
        # Run buffer maintenance (e.g., FreshSuccessBuffer drops oldest half).
        self.buf.maintenance()
        if len(self.buf) == 0:
            return
        n_envs = getattr(self.training_env, "num_envs", 1)
        n_steps = getattr(self.model, "n_steps", 2048)
        target_minibatches = max(1, int(self.mix_ratio * (n_envs * n_steps) / self.batch_size))
        total_iters = max(self.iters_per_rollout, target_minibatches)
        bc_losses, ents = [], []
        for _ in range(total_iters):
            obs_b, act_b = self.buf.sample(self.batch_size)
            if obs_b is None:
                break
            lbc, ent = self._supervised_step(obs_b, act_b)
            bc_losses.append(lbc); ents.append(ent)
        if self.verbose and bc_losses:
            print(f"[SuccessReplay/{self.buffer_kind}] size={len(self.buf)} bc_steps={len(bc_losses)} "
                  f"bc_loss={np.mean(bc_losses):.4f} ent={np.mean(ents):.3f}")


class MemProfileCallback(BaseCallback):
    """Opt-in memory profiler (set env MEM_PROFILE=1). Every `every` env steps:
    gc.collect(), then log RSS and a gc object-type histogram (counts, plus
    total bytes held in ndarrays / torch Tensors). Writes to
    logs/memprofile_<pid>.log. Stops training if RSS exceeds rss_cap_gb.
    Deliberately avoids tracemalloc -- its per-allocation tracing is a ~15x
    slowdown on this SymPy-heavy workload and its own buffers confound RSS."""
    def __init__(self, every=4096, rss_cap_gb=9.0, verbose=0):
        super().__init__(verbose)
        self.every = int(every)
        self.rss_cap_gb = float(rss_cap_gb)
        self._path = os.path.join("logs", f"memprofile_{os.getpid()}.log")

    def _rss_gb(self):
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)

    def _log(self, msg):
        with open(self._path, "a") as f:
            f.write(msg + "\n")

    def _on_training_start(self) -> None:
        self._log(f"=== mem profile start, pid {os.getpid()}, baseline RSS {self._rss_gb():.2f} GB ===")

    def _on_step(self) -> bool:
        if self.every <= 0 or self.num_timesteps % self.every != 0:
            return True
        import gc
        from collections import Counter
        gc.collect()
        rss = self._rss_gb()
        objs = gc.get_objects()
        hist = Counter(type(o).__name__ for o in objs)
        arr_bytes = arr_n = ten_bytes = ten_n = 0
        for o in objs:
            tn = type(o).__name__
            if tn == "ndarray":
                try:
                    arr_bytes += int(o.nbytes); arr_n += 1
                except Exception:
                    pass
            elif tn == "Tensor":
                try:
                    ten_bytes += o.element_size() * o.nelement(); ten_n += 1
                except Exception:
                    pass
        self._log(f"\n--- step {self.num_timesteps}  RSS {rss:.2f} GB  objs={len(objs)}  "
                  f"ndarray={arr_n}/{arr_bytes/1e6:.0f}MB  Tensor={ten_n}/{ten_bytes/1e6:.0f}MB ---")
        self._log("  types: " + ", ".join(f"{k}={v}" for k, v in hist.most_common(18)))
        if rss > self.rss_cap_gb:
            self._log(f"=== RSS {rss:.2f} GB > cap {self.rss_cap_gb}, stopping training ===")
            return False
        return True


# ==========================
# Env / agent factories
# ==========================

def _get_action_mask(env):
    return env.get_valid_action_mask()

def make_env(
    env_name: str,
    agent: str,
    gen: str,
    seed: int = 0,
    *,
    action_space: str = 'fixed',
    sparse_rewards: bool = False,
    use_relabel_constants: bool = False,
    use_curriculum: bool = False,
    use_action_mask: bool = False,
    use_success_replay: bool = False,
    use_cov: bool = False,
    anti_loop_penalty: float = 0.0,
    use_cbrt: bool = True,
):
    #state_rep = 'graph_integer_1d' if 'tree' in agent else 'integer_1d'
    use_graph = any(tok in agent.split('-') for tok in ('tree','gnn','sage'))
    state_rep = 'graph_integer_1d' if use_graph else 'integer_1d'

    if env_name == 'single_eqn':
        env = singleEqn(main_eqn='a*x+b', state_rep=state_rep)
    elif env_name == 'multi_eqn':
        if action_space == 'dynamic':
            EnvCls = multiEqnDynamic
            env = EnvCls(
                gen=gen,
                use_relabel_constants=use_relabel_constants,
                state_rep=state_rep,
                sparse_rewards=sparse_rewards,
                use_curriculum=use_curriculum,
                use_success_replay=use_success_replay,
                use_cov=use_cov,
                anti_loop_penalty=anti_loop_penalty,
                use_cbrt=use_cbrt,
            )
        else:
            EnvCls = multiEqn
            env = EnvCls(
                gen=gen,
                use_relabel_constants=use_relabel_constants,
                state_rep=state_rep,
                sparse_rewards=sparse_rewards,
                use_curriculum=use_curriculum,
                use_cov=use_cov,
            )
    else:
        raise ValueError(f"Unknown env_name: {env_name}")
    try:
        env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
    if action_space == 'dynamic' and use_action_mask:
        env = ActionMasker(env, _get_action_mask)
    return env

def make_train_vec_env(
    n_envs: int,
    env_name: str,
    agent: str,
    gen: str,
    seed: int,
    *,
    action_space: str,
    sparse_rewards: bool,
    use_relabel_constants: bool,
    use_curriculum: bool,
    use_action_mask: bool,
    use_success_replay: bool,
    use_cov: bool = False,
    anti_loop_penalty: float = 0.0,
    use_cbrt: bool = True,
):
    """Create a vectorized environment for training."""
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    return make_vec_env(
        lambda: make_env(
            env_name, agent, gen, seed=seed,
            action_space=action_space,
            sparse_rewards=sparse_rewards,
            use_relabel_constants=use_relabel_constants,
            use_curriculum=use_curriculum,
            use_action_mask=use_action_mask,
            use_success_replay=use_success_replay,
            use_cov=use_cov,
            anti_loop_penalty=anti_loop_penalty,
            use_cbrt=use_cbrt,
        ),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_cls
    )

def _infer_tree_limits_from_space(obs_space: spaces.Space) -> Tuple[int, int]:
    """
    Infer (max_nodes, max_edges) from observation_space for TreeMLPExtractor.
    Expects Dict with 'node_features' -> Box((N_nodes, F)).
    """
    if isinstance(obs_space, spaces.Dict) and "node_features" in obs_space.spaces:
        node_box = obs_space["node_features"]
        if isinstance(node_box, spaces.Box) and len(node_box.shape) >= 2:
            max_nodes = int(node_box.shape[0])
            max_edges = 2 * max_nodes
            return max_nodes, max_edges
    return 128, 256


def make_agent(
    agent: str,
    env,
    hidden_dim: int,
    seed: int = 0,
    load_path: Optional[str] = None,
    tree_kwargs: Optional[Dict[str, Any]] = None,
    action_space: str = 'fixed',
    ent_coef = 0.01
):
    """Create or load a PPO/MaskablePPO agent."""
    if load_path and os.path.isfile(load_path):
        timed_print(f"[{agent}] Loading model from: {load_path}")
        if action_space == 'dynamic':
            return MaskablePPO.load(load_path, env=env, device="auto", print_system_info=False, ent_coef=ent_coef)
        return PPO.load(load_path, env=env, device="auto", print_system_info=False)

    # ---- Decide extractor from agent string (backward compatible) ----
    tokens = set(agent.split('-'))
    use_tree = 'tree' in tokens                      # e.g. ppo-tree, ppo-tree-ICM
    use_gnn  = 'gnn' in tokens                       # e.g. ppo-gnn, ppo-gnn-RND
    use_sage = ('sage' in tokens) or ('graphsage' in tokens)

    # If none of the above → original MLP branch
    if not (use_tree or use_gnn or use_sage):
        policy_kwargs = dict(net_arch=[hidden_dim, hidden_dim])
        if action_space == 'dynamic':
            return MaskablePPO('MlpPolicy', env=env, policy_kwargs=policy_kwargs, seed=seed, verbose=0, n_steps=2048, ent_coef=ent_coef)
        return PPO('MlpPolicy', env=env, policy_kwargs=policy_kwargs, seed=seed, verbose=0, n_steps=2048, ent_coef=ent_coef)

    # ---- Graph extractors (TreeMLP / GCN / GraphSAGE) ----
    # Defaults (can be overridden via tree_kwargs)
    tkw = dict(
        embed_dim      = 32,
        hidden_dim     = 64,
        K              = 2,
        pooling        = "mean",
        vocab_min_id   = -10,
        pad_id         = 99,
        pi_sizes       = [128],
        vf_sizes       = [128],
        # optional extras understood by the GNN extractors:
        # dropout        = 0.0,
        # add_self_loops = True,
        # residual       = True,
    )
    if tree_kwargs:
        tkw.update({k: v for k, v in tree_kwargs.items() if v is not None})

    # Import extractors (TreeMLP already in your project; GNNs in utils.gnn_extractors)
    # from utils.utils_env import TreeMLPExtractor,
    # try:
    #     from utils.utils_env import GCNExtractor, GraphSAGEExtractor
    # except Exception:
    #     # If file not present, you can keep TreeMLP only
    #     GCNExtractor = GraphSAGEExtractor = None

    # Per-extractor class & kwargs
    if use_tree:
        fx_class = TreeMLPExtractor
        # TreeMLP expects max_nodes/max_edges derived from the obs space
        max_nodes, max_edges = _infer_tree_limits_from_space(env.observation_space)
        fx_kwargs = dict(
            max_nodes=max_nodes,
            max_edges=max_edges,
            vocab_min_id=tkw["vocab_min_id"],
            pad_id=tkw["pad_id"],
            embed_dim=tkw["embed_dim"],
            hidden_dim=tkw["hidden_dim"],
            K=tkw["K"],
            pooling=tkw["pooling"],
        )
    elif use_gnn:
        assert GCNExtractor is not None, "GCNExtractor not found. Ensure utils/gnn_extractors.py is available."
        fx_class = GCNExtractor
        # GCNExtractor does NOT need max_nodes/max_edges in __init__
        fx_kwargs = dict(
            vocab_min_id=tkw["vocab_min_id"],
            pad_id=tkw["pad_id"],
            embed_dim=tkw["embed_dim"],
            hidden_dim=tkw["hidden_dim"],
            K=tkw["K"],
            pooling=tkw["pooling"],
            # optional:
            # dropout=tkw.get("dropout", 0.0),
            # add_self_loops=tkw.get("add_self_loops", True),
            # residual=tkw.get("residual", True),
        )
    else:  # use_sage
        assert GraphSAGEExtractor is not None, "GraphSAGEExtractor not found. Ensure utils/gnn_extractors.py is available."
        fx_class = GraphSAGEExtractor
        fx_kwargs = dict(
            vocab_min_id=tkw["vocab_min_id"],
            pad_id=tkw["pad_id"],
            embed_dim=tkw["embed_dim"],
            hidden_dim=tkw["hidden_dim"],
            K=tkw["K"],
            pooling=tkw["pooling"],
            # optional:
            # dropout=tkw.get("dropout", 0.0),
            # add_self_loops=tkw.get("add_self_loops", True),
            # residual=tkw.get("residual", True),
        )

    kwargs = dict(
        features_extractor_class=fx_class,
        features_extractor_kwargs=fx_kwargs,
        net_arch=dict(pi=tkw["pi_sizes"], vf=tkw["vf_sizes"]),
    )

    if action_space == 'dynamic':
        return MaskablePPO(policy="MultiInputPolicy", env=env, policy_kwargs=kwargs, verbose=0, seed=seed, n_steps=2048, ent_coef=ent_coef)

    return PPO(policy="MultiInputPolicy", env=env, policy_kwargs=kwargs, verbose=0, seed=seed, n_steps=2048, ent_coef=ent_coef)



def make_agent_old(
    agent: str,
    env,
    hidden_dim: int,
    seed: int = 0,
    load_path: Optional[str] = None,
    tree_kwargs: Optional[Dict[str, Any]] = None,
    action_space: str = 'fixed',
    ent_coef = 0.01
):
    """Create or load a PPO/MaskablePPO agent."""
    if load_path and os.path.isfile(load_path):
        timed_print(f"[{agent}] Loading model from: {load_path}")
        if action_space == 'dynamic':
            return MaskablePPO.load(load_path, env=env, device="auto", print_system_info=False,ent_coef=ent_coef)
        return PPO.load(load_path, env=env, device="auto", print_system_info=False)

    if 'tree' not in agent:
        policy_kwargs = dict(net_arch=[hidden_dim, hidden_dim])
        if action_space == 'dynamic':
            return MaskablePPO('MlpPolicy', env=env, policy_kwargs=policy_kwargs, seed=seed, verbose=0, n_steps=2048,ent_coef=ent_coef)
        return PPO('MlpPolicy', env=env, policy_kwargs=policy_kwargs, seed=seed, verbose=0, n_steps=2048,ent_coef=ent_coef)
        
    tkw = dict(
        embed_dim      = 32,
        hidden_dim     = 64,
        K              = 2,
        pooling        = "mean",
        vocab_min_id   = -10,
        pad_id         = 99,
        pi_sizes       = [128],
        vf_sizes       = [128],
    )
    if tree_kwargs:
        tkw.update({k: v for k, v in tree_kwargs.items() if v is not None})

    max_nodes, max_edges = _infer_tree_limits_from_space(env.observation_space)

    kwargs = dict(
        features_extractor_class=TreeMLPExtractor,
        features_extractor_kwargs=dict(
            max_nodes=max_nodes,
            max_edges=max_edges,
            vocab_min_id=tkw["vocab_min_id"],
            pad_id=tkw["pad_id"],
            embed_dim=tkw["embed_dim"],
            hidden_dim=tkw["hidden_dim"],
            K=tkw["K"],
            pooling=tkw["pooling"],
        ),
        net_arch=dict(pi=tkw["pi_sizes"], vf=tkw["vf_sizes"]),
    )

    if action_space == 'dynamic':
        return MaskablePPO(policy="MultiInputPolicy",env=env,policy_kwargs=kwargs,verbose=0,seed=seed,n_steps=2048,ent_coef=ent_coef)

    return PPO(policy="MultiInputPolicy", env=env, policy_kwargs=kwargs, verbose=0, seed=seed, n_steps=2048, ent_coef=ent_coef)

# ==========================
# Eval helpers (no double reset)
# ==========================
def _set_equation_if_supported(env, eqn):
    seteq = getattr(env, "set_equation", None)
    if callable(seteq):
        seteq(eqn)
    setup = getattr(env, "setup", None)
    if callable(setup):
        setup()

def _start_on_eqn(env, eqn):
    """Reset once, then (if supported) pin the env to `eqn` and return the obs."""
    reset_obs, _ = env.reset()
    if callable(getattr(env, "set_equation", None)):
        _set_equation_if_supported(env, eqn)
        obs = getattr(env, "state", None)
        if obs is None:
            obs, _ = env.to_vec(env.lhs, env.rhs)
        return obs
    else:
        return reset_obs

def greedy_solve_one(model, env, eqn, max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_DET) -> bool:
    try:
        with time_limit(per_eqn_seconds):
            obs = _start_on_eqn(env, eqn)
            for _ in range(max_steps):
                action, _ = model.predict(obs, deterministic=True, action_masks=env.get_valid_action_mask() if hasattr(env, 'get_valid_action_mask') else None)
                obs, _, terminated, truncated, info = env.step(action)
                if info.get("is_solved", False):
                    return True
                if terminated or truncated:
                    break
            return False
    except EvalTimeout:
        return False
    except Exception:
        return False

def greedy_accuracy(model, env, equations, max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_DET) -> Optional[float]:
    if not equations:
        return None
    solved = sum(greedy_solve_one(model, env, eqn, max_steps, per_eqn_seconds) for eqn in equations)
    return solved / len(equations)

def success_at_n(model, env, equations, n_trials=10, max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_STOCH) -> Optional[float]:
    if not equations:
        return None
    solved_any = 0
    for eqn in equations:
        try:
            with time_limit(per_eqn_seconds):
                solved_this = False
                for _ in range(n_trials):
                    obs = _start_on_eqn(env, eqn)
                    for _ in range(max_steps):
                        action, _ = model.predict(obs, deterministic=False, action_masks=env.get_valid_action_mask() if hasattr(env, 'get_valid_action_mask') else None)
                        obs, _, terminated, truncated, info = env.step(action)
                        if info.get("is_solved", False):
                            solved_this = True
                            break
                        if terminated or truncated:
                            break
                    if solved_this:
                        break
                if solved_this:
                    solved_any += 1
        except (EvalTimeout, Exception):
            pass
    return solved_any / len(equations)

def _policy_action_probs(model, obs, action_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Return action probs for a single observation (Discrete action space)."""
    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_tensor)
        # Raw probs (batch size 1)
        probs = dist.distribution.probs.squeeze(0).detach().cpu().numpy()
    if action_mask is not None:
        action_mask = np.asarray(action_mask, dtype=probs.dtype)
        if action_mask.shape != probs.shape:
            # Fallback: don't mask if shapes mismatch (prevents accidental zeroing)
            pass
        else:
            probs = probs * action_mask
    s = probs.sum()
    if s <= 1e-12:
        # All actions masked (or numerically zero) → return zeros to signal no expansion
        return np.zeros_like(probs)
    return probs / s


def _policy_value(model, obs) -> float:
    """Return the PPO critic's value estimate V(s) for a single observation."""
    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        values = model.policy.predict_values(obs_tensor)
    return float(values.squeeze().cpu().item())


def _state_complexity(env) -> int:
    """Quick complexity proxy for the env's current (lhs, rhs)."""
    u = getattr(env, "unwrapped", env)
    try:
        from utils.utils_env import get_complexity_expression
        return int(get_complexity_expression(u.lhs) + get_complexity_expression(u.rhs))
    except Exception:
        return 0


def _snapshot_env(env) -> Dict[str, Any]:
    u = getattr(env, "unwrapped", env)
    return dict(
        lhs=u.lhs, rhs=u.rhs, state=u.state,
        current_steps=u.current_steps,
        main_eqn=u.main_eqn,
        map_constants=getattr(u, "map_constants", None),
        map_constants_history=list(getattr(u, "map_constants_history", [])),
        # CoV state (only present if use_cov=True; ignored otherwise)
        cov_inv=list(getattr(u, "cov_inv", [])),
        main_eqn_original_cov=getattr(u, "main_eqn_original_cov", None),
        solve_var=getattr(u, "solve_var", None),
        _timeout_count=getattr(u, "_timeout_count", 0),
        # optional: if your env tracks done flags internally
        _terminated=getattr(u, "_terminated", False),
        _truncated=getattr(u, "_truncated", False),
    )

def _restore_env(env, snap: Dict[str, Any]) -> None:
    u = getattr(env, "unwrapped", env)
    u.lhs = snap["lhs"]
    u.rhs = snap["rhs"]
    u.state = snap["state"]
    u.current_steps = snap["current_steps"]
    u.main_eqn = snap["main_eqn"]
    u.map_constants = snap["map_constants"]
    u.map_constants_history = list(snap["map_constants_history"])
    # Restore CoV state too (no-op for envs without it)
    if hasattr(u, "cov_inv"):
        u.cov_inv = list(snap.get("cov_inv", []))
    if hasattr(u, "main_eqn_original_cov"):
        u.main_eqn_original_cov = snap.get("main_eqn_original_cov", None)
    if hasattr(u, "solve_var") and snap.get("solve_var") is not None:
        u.solve_var = snap["solve_var"]
    u._timeout_count = snap["_timeout_count"]
    # optional:
    if "_terminated" in snap: u._terminated = snap["_terminated"]
    if "_truncated"  in snap: u._truncated  = snap["_truncated"]


def beam_solve_one(model, env, eqn, *, beam_width=5, topk_per_node=5,
                   max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_DET,
                   beam_lambda: float = 0.0, beam_alpha: float = 0.0,
                   beam_beta: float = 0.0) -> bool:
    """Beam search with optional value-guidance.

    Score(path) = sum_t log pi(a_t | s_t)
                + beam_lambda * V(s_t_final)        # value head bonus on current node
                - beam_alpha  * path_length         # prefer shorter paths
                - beam_beta   * complexity(s_final) # prefer simpler states

    With beam_lambda=alpha=beta=0 this is the original pure-policy beam.
    """
    try:
        with time_limit(per_eqn_seconds):
            obs0 = _start_on_eqn(env, eqn)
            base = _snapshot_env(env)
            # Beam entries: (score, snap, obs, solved, depth)
            beam = [(0.0, base, obs0, False, 0)]
            for _ in range(max_steps):
                new_beam = []
                for score, snap, obs, solved, depth in beam:
                    if solved:
                        new_beam.append((score, snap, obs, True, depth))
                        continue
                    _restore_env(env, snap)
                    # keep obs in sync with restored internal state
                    obs = getattr(getattr(env, "unwrapped", env), "state", obs)
                    mask = env.get_valid_action_mask() if hasattr(env, 'get_valid_action_mask') else None
                    probs = _policy_action_probs(model, obs, action_mask=mask)

                    if probs.ndim != 1:
                        probs = probs.reshape(-1)
                    if probs.sum() <= 1e-12:
                        # No valid actions from this node; skip expanding it
                        continue
                    if topk_per_node < len(probs):
                        top_idx = np.argpartition(probs, -topk_per_node)[-topk_per_node:]
                        top_idx = top_idx[np.argsort(-probs[top_idx])]
                    else:
                        top_idx = np.argsort(-probs)
                    for a in top_idx:
                        if probs[a] <= 0.0:
                            continue
                        _restore_env(env, snap)
                        obs_next, _, terminated, truncated, info = env.step(int(a))
                        solved_next = bool(info.get("is_solved", False))
                        # Cumulative log-policy term
                        log_pi = float(np.log(probs[a] + 1e-12))
                        score_next = score + log_pi
                        # Value-head bonus on the resulting state (skip if solved
                        # — V is meaningless after terminal and may be 0 anyway).
                        if beam_lambda != 0.0 and not solved_next:
                            try:
                                v = _policy_value(model, obs_next)
                                score_next += beam_lambda * v
                            except Exception:
                                pass
                        # Complexity penalty on the resulting state
                        if beam_beta != 0.0 and not solved_next:
                            score_next -= beam_beta * _state_complexity(env)
                        # Length penalty
                        if beam_alpha != 0.0:
                            score_next -= beam_alpha
                        new_beam.append((score_next, _snapshot_env(env), obs_next, solved_next, depth + 1))
                if not new_beam:
                    return False
                # Canonical-state dedup: collapse beam entries that reach the same
                # equation under the same CoV/relabel context, keeping the highest
                # score. The CoV-stack depth + relabel-history-depth are included
                # because two paths sharing (lhs, rhs) but different unwinds have
                # different futures.
                seen: Dict[Any, int] = {}
                for i, (sc, snp, ob, sol, dp) in enumerate(new_beam):
                    if sol:
                        key = (id(snp), True)  # never collapse solved entries
                    else:
                        try:
                            import sympy as _sp
                            lhs_s = str(_sp.expand(snp.get("lhs")))
                            rhs_s = str(_sp.expand(snp.get("rhs")))
                        except Exception:
                            lhs_s = str(snp.get("lhs"))
                            rhs_s = str(snp.get("rhs"))
                        cov_depth = len(snp.get("cov_inv") or [])
                        rel_depth = len(snp.get("map_constants_history") or [])
                        key = (lhs_s, rhs_s, cov_depth, rel_depth)
                    if key not in seen or new_beam[seen[key]][0] < sc:
                        seen[key] = i
                deduped = [new_beam[i] for i in seen.values()]
                beam = nlargest(beam_width, deduped, key=lambda x: x[0])
                if any(s for _, _, _, s, _ in beam):
                    return True
            return any(s for _, _, _, s, _ in beam)
    except EvalTimeout:
        return False
    except Exception:
        return False


def beam_accuracy(model, env, equations, *, beam_width=5, topk_per_node=5,
                  max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_DET,
                  beam_lambda: float = 0.0, beam_alpha: float = 0.0,
                  beam_beta: float = 0.0) -> Optional[float]:
    if not equations:
        return None
    solved = 0
    for eqn in equations:
        ok = beam_solve_one(model, env, eqn,
                            beam_width=beam_width,
                            topk_per_node=topk_per_node,
                            max_steps=max_steps,
                            per_eqn_seconds=per_eqn_seconds,
                            beam_lambda=beam_lambda,
                            beam_alpha=beam_alpha,
                            beam_beta=beam_beta)
        solved += int(ok)
    return solved / len(equations)


# ==========================
# Logging callback
# ==========================
class TrainingLogger(BaseCallback):
    def __init__(self, algo_name: str, train_env, eval_env, eval_interval: int, log_interval: int, save_dir: str, verbose=1, early_stop_patience: int = 0, eval_subsample: int = 0, eval_lite: bool = False):
        super().__init__(verbose)
        self.algo_name     = algo_name
        self.train_env     = train_env
        self.eval_env      = eval_env
        self.eval_interval = eval_interval
        self.log_interval  = log_interval
        self.save_dir      = save_dir
        self.ckpt_dir = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.curves_path   = os.path.join(self.save_dir, "learning_curves.csv")
        #self.train_eqns = getattr(eval_env, "train_eqns", None) or getattr(train_env, "train_eqns", None)
        #self.test_eqns  = getattr(eval_env, "test_eqns", None)  or getattr(train_env, "test_eqns", None)
        self.train_eqns = _unwrap_attr(eval_env, "train_eqns") or _unwrap_attr(train_env, "train_eqns")
        self.test_eqns  = _unwrap_attr(eval_env, "test_eqns")  or _unwrap_attr(train_env, "test_eqns")
        self.num_eqns   = len(self.train_eqns) if self.train_eqns is not None else 1
        # Pre-sample a deterministic test subset for in-training evals.
        # 0 means "use the full set". The end-of-training eval always uses the full set.
        self.eval_subsample = int(eval_subsample)
        if self.eval_subsample > 0 and self.test_eqns and len(self.test_eqns) > self.eval_subsample:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(self.test_eqns), size=self.eval_subsample, replace=False)
            self._eval_subset = [self.test_eqns[i] for i in sorted(idx.tolist())]
        else:
            self._eval_subset = self.test_eqns
        # When eval_lite is on we skip beam + at10 during training (greedy only).
        # Early-stop falls back to test_greedy in lite mode.
        self.eval_lite = bool(eval_lite)
        self.Tsolves: Dict[Any, int] = {}
        self.Tsolve: Optional[float]  = None
        self.Tconverge: Optional[int] = None
        self.log_steps: List[int] = []
        self.coverage: List[float] = []
        self.test_acc: List[float] = []
        self.test_beam: List[float] = []
        self.test_at10: List[float] = []
        # Early-stopping + best checkpoint
        self.early_stop_patience = int(early_stop_patience)
        self.best_test_beam: float = -1.0
        self.best_test_greedy: float = -1.0
        self.best_step: Optional[int] = None
        self.no_improve_count: int = 0
        self.stopped_early: bool = False
        os.makedirs(self.save_dir, exist_ok=True)

    def _log_eval(self, step: int):
        solved = min(len(self.Tsolves), self.num_eqns)
        cov = solved / self.num_eqns if self.num_eqns else 0.0
        eqns = self._eval_subset
        # Greedy always; beam + at10 only when not in lite mode.
        tst = greedy_accuracy(self.model, self.eval_env, eqns,
                              max_steps=5, per_eqn_seconds=EVAL_TIMEOUT_DET) if eqns else None
        if self.eval_lite:
            tbeam = None
            t10 = None
        else:
            tbeam = beam_accuracy(self.model, self.eval_env, eqns,
                                  beam_width=5, topk_per_node=5, max_steps=5,
                                  per_eqn_seconds=EVAL_TIMEOUT_DET) if eqns else None
            t10 = success_at_n(self.model, self.eval_env, eqns,
                               n_trials=10, max_steps=5, per_eqn_seconds=EVAL_TIMEOUT_STOCH) if eqns else None
        self.log_steps.append(step)
        self.coverage.append(cov)
        self.test_acc.append(tst if tst is not None else np.nan)
        self.test_beam.append(tbeam if tbeam is not None else np.nan)
        self.test_at10.append(t10 if t10 is not None else np.nan)
        # Track best for early-stopping. In lite mode use greedy; otherwise beam.
        if self.eval_lite:
            metric = tst
            best_attr = "best_test_greedy"
        else:
            metric = tbeam
            best_attr = "best_test_beam"
        current_best = getattr(self, best_attr)
        if metric is not None and metric > current_best:
            setattr(self, best_attr, float(metric))
            self.best_step = step
            self.no_improve_count = 0
            self.model.save(os.path.join(self.ckpt_dir, "best.zip"))
        else:
            self.no_improve_count += 1
        tst_s   = f"{tst:.2f}"   if tst   is not None else "NA"
        tbeam_s = f"{tbeam:.2f}" if tbeam is not None else "NA"
        t10_s   = f"{t10:.2f}"   if t10   is not None else "NA"
        best_label = "best_greedy" if self.eval_lite else "best_beam"
        best_val = getattr(self, best_attr)
        best_s  = f"{best_val:.2f}@{self.best_step}" if self.best_step is not None else "NA"
        timed_print(f"[{self.algo_name}] t={step}: coverage={cov:.3f} | test_(greedy,beam,@10) =({tst_s}, {tbeam_s}, {t10_s}) | {best_label}={best_s} | patience={self.no_improve_count}/{self.early_stop_patience}")
        pd.DataFrame({
            "step": self.log_steps,
            "coverage": self.coverage,
            "test_greedy": self.test_acc,
            "test_beam": self.test_beam,
            "test_at10": self.test_at10
        }).to_csv(self.curves_path + ".tmp", index=False)
        os.replace(self.curves_path + ".tmp", self.curves_path)

    def _on_training_start(self) -> None:
        timed_print(f"[{self.algo_name}] Training started (train_eqns={self.num_eqns}, test_eqns={len(self.test_eqns) if self.test_eqns else 0})")
        #self._log_eval(step=0)

    def _save_ckpt(self, step: int):
        path_step = os.path.join(self.ckpt_dir, f"model_step{step:07d}.zip")
        self.model.save(path_step)
        # keep a rolling "latest" for convenience
        self.model.save(os.path.join(self.ckpt_dir, "latest.zip"))

    def _on_step(self) -> bool:
        step = self.num_timesteps
        for info in self.locals.get("infos", []):
            if info.get("is_solved"):
                eqn  = info.get("main_eqn", "eqn")
                lhs  = info.get("lhs")
                rhs  = info.get("rhs")
                if eqn not in self.Tsolves:
                    self.Tsolves[eqn] = step
                    print(f"\033[33m[{self.algo_name}] Solved {eqn} ==> {lhs} = {rhs} at step {step}\033[0m")
                if self.Tconverge is None and len(self.Tsolves) >= self.num_eqns:
                    self.Tconverge = step
                    timed_print(f"[{self.algo_name}] Coverage 100% at step {step}")
        if self.eval_interval and step % self.eval_interval == 0:
            self._log_eval(step)
            self._save_ckpt(step)   # <-- add this line
            if self.early_stop_patience > 0 and self.no_improve_count >= self.early_stop_patience:
                self.stopped_early = True
                timed_print(f"[{self.algo_name}] Early stop at step {step}: test_beam stopped improving for {self.no_improve_count} evals (best={self.best_test_beam:.3f}@{self.best_step})")
                return False
        return True

    def _on_training_end(self) -> None:
        self.Tsolve = float(np.mean(list(self.Tsolves.values()))) if self.Tsolves else float('inf')
        timed_print(f"[{self.algo_name}] Training finished | Tsolve={self.Tsolve} | Tconverge={self.Tconverge}")
        pd.DataFrame({
            "step": self.log_steps,
            "coverage": self.coverage,
            "test_greedy": self.test_acc,
            "test_beam": self.test_beam,
            "test_at10": self.test_at10
        }).to_csv(self.curves_path, index=False)
        timed_print(f"[{self.algo_name}] Saved curves → {self.curves_path}")

# ==========================
# Worker: run one trial
# ==========================
def run_trial(
    agent: str,
    env_name: str,
    gen: str,
    Ntrain: int,
    eval_interval: int,
    log_interval: int,
    seed: int,
    save_dir: str,
    curiosity: Optional[str],
    hidden_dim: int,
    load_model_path: Optional[str],
    sparse_rewards: bool,
    use_relabel_constants: bool,
    use_curriculum: bool,
    tree_kwargs: Optional[Dict[str, Any]],
    n_envs: int,
    action_space: str,
    use_success_replay: bool = False,
    sr_mix_ratio: float = 0.5,
    sr_batch_size: int = 256,
    sr_iters_per_rollout: int = 10,
    sr_capacity: int = 20000,
    ent_coef = 0.01,
    use_cov: bool = False,
    early_stop_patience: int = 0,
    anti_loop_penalty: float = 0.0,
    eval_subsample: int = 0,
    eval_lite: bool = False,
    use_cbrt: bool = True,
    sr_buffer_kind: str = "flat",
):
    train_env = make_train_vec_env(
        n_envs=n_envs,
        env_name=env_name,
        agent=agent,
        gen=gen,
        seed=seed,
        action_space=action_space,
        sparse_rewards=sparse_rewards,
        use_relabel_constants=use_relabel_constants,
        use_curriculum=use_curriculum,
        use_action_mask=action_space == 'dynamic',
        use_success_replay=use_success_replay,
        use_cov=use_cov,
        anti_loop_penalty=anti_loop_penalty,
        use_cbrt=use_cbrt,
    )
    eval_env  = make_env(
        env_name, agent, gen, seed=seed + 777,
        action_space=action_space,
        sparse_rewards=sparse_rewards,
        use_relabel_constants=use_relabel_constants,
        use_curriculum=use_curriculum,
        use_action_mask=action_space == 'dynamic',
        use_success_replay=use_success_replay,
        use_cov=use_cov,
        use_cbrt=use_cbrt,
    )
    model = make_agent(agent, train_env, hidden_dim, ent_coef=ent_coef,  seed=seed, load_path=load_model_path, tree_kwargs=tree_kwargs, action_space=action_space)
    tag = f"seed{seed}"
    run_dir = os.path.join(save_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    cb_main = TrainingLogger(
        algo_name=agent,
        train_env=train_env,
        eval_env=eval_env,
        eval_interval=eval_interval,
        log_interval=log_interval,
        save_dir=run_dir,
        early_stop_patience=early_stop_patience,
        eval_subsample=eval_subsample,
        eval_lite=eval_lite,
    )
    #cb_list: List[BaseCallback] = [cb_main, ProgressBarCallback()]
    cb_list: List[BaseCallback] = [cb_main]
    if os.environ.get("MEM_PROFILE"):
        cb_list.append(MemProfileCallback(every=int(os.environ.get("MEM_PROFILE_EVERY", "4096"))))
    if use_success_replay:
        cb_list.append(
            SuccessReplayCallback(
                mix_ratio=sr_mix_ratio,
                batch_size=sr_batch_size,
                iters_per_rollout=sr_iters_per_rollout,
                capacity=sr_capacity,
                buffer_kind=sr_buffer_kind,
                verbose=0
            )
        )
    if curiosity is not None:
        wrapper_cls = NodeFeatureView if 'tree' in agent else None
        curiosity_vec = make_vec_env(
            lambda: make_env(
                env_name, agent, gen, seed=seed,
                action_space=action_space,
                sparse_rewards=sparse_rewards,
                use_relabel_constants=use_relabel_constants,
                use_curriculum=use_curriculum,
                use_action_mask=action_space == 'dynamic',
                use_success_replay=use_success_replay
            ),
            n_envs=1,
            seed=seed,
            vec_env_cls=DummyVecEnv,
            wrapper_class=wrapper_cls
        )
        irs = get_intrinsic_reward(curiosity, curiosity_vec)
        if irs:
            cb_list.append(IntrinsicReward(irs))
    model.learn(total_timesteps=Ntrain, callback=cb_list, progress_bar=True)
    model_path = os.path.join(run_dir, f"{tag}.zip")
    model.save(model_path)
    final_model_path = os.path.join(run_dir, "final_model.zip")
    model.save(final_model_path)
    timed_print(f"[{agent}] Saved model → {model_path}")
    timed_print(f"[{agent}] Saved model → {final_model_path}")
    test_eqns = getattr(eval_env, "test_eqns", [])
    final_test_acc   = greedy_accuracy(model, eval_env, test_eqns, max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_DET) or 0.0
    final_test_beam  = beam_accuracy(model, eval_env, test_eqns, beam_width=5, topk_per_node=5, max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_DET) or 0.0
    final_test_at10  = success_at_n(model, eval_env, test_eqns, n_trials=10, max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_STOCH) or 0.0
    cb = cb_main
    coverage_final = len(cb.Tsolves)
    num_eqns = cb.num_eqns
    coverage_final_rate = (coverage_final / num_eqns) if num_eqns else 0.0
    metrics = {
        "agent": agent,
        "env": env_name,
        "seed": seed,
        "coverage_final_rate": coverage_final_rate,
        "final_test_acc":  final_test_acc,
        "final_test_beam": final_test_beam,
        "final_test_at10": final_test_at10,
        "Tsolve": cb.Tsolve,
        "Tconverge": cb.Tconverge,
        "num_eqns": num_eqns,
        "model_path": model_path,
        "final_model_path": final_model_path
    }
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    timed_print(f"[{agent}] Saved metrics → {metrics_path}")
    try:
        train_env.close()
        eval_env.close()
    except Exception:
        pass
    return metrics, run_dir

def run_trial_wrapper(args):
    import traceback
    try:
        return run_trial(*args)
    except Exception:
        # Print the full traceback from inside the worker so the parent
        # process actually sees where it died.
        print("\n[worker traceback]\n" + traceback.format_exc(), flush=True)
        raise

def run_parallel(jobs, n_workers=4, timeout_per_job=None):
    """Submit all jobs; stream results in completion order."""
    rows, run_dirs = [], []
    ctx = mp.get_context("spawn")
    total, done = len(jobs), 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = [ex.submit(run_trial_wrapper, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                metrics, run_dir = fut.result(timeout=timeout_per_job)
                rows.append(metrics); run_dirs.append(run_dir)
                done += 1
                timed_print(f"✓ [{done}/{total}] {metrics['agent']} seed={metrics['seed']} | coverage_final={metrics['coverage_final_rate']:.2f} | test_(greedy,beam,@10)=({metrics['final_test_acc']:.2f}, {metrics['final_test_beam']:.2f}, {metrics['final_test_at10']:.2f})")
            except Exception as e:
                done += 1
                import traceback
                timed_print(f"✗ [{done}/{total}] Job crashed or timed out: {e}")
                timed_print(traceback.format_exc())
    return rows, run_dirs

# ==========================
# Main
# ==========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RL Sweep Script (VecEnv-ready)")

    env_group = parser.add_argument_group("Environment")
    env_group.add_argument('--env_name', type=str, default='multi_eqn', help='Environment name')
    env_group.add_argument('--action_space', type=str, default='dynamic', help='Action space: fixed or dynamic')
    env_group.add_argument('--gen', type=str, default='abel_level4', help='Equation generator')
    env_group.add_argument('--sparse_rewards', action='store_true', help='Use sparse rewards instead of shaping')
    env_group.add_argument('--use_curriculum', action='store_true', help='Use inverse sampling curriculum')
    env_group.add_argument('--use_relabel_constants', action='store_true', help='Enable relabel-constants macroaction')
    env_group.add_argument('--use_success_replay', action='store_true', help='Enable success replay BC updates')
    env_group.add_argument('--use_cov', action='store_true', help='enable change of variables')
    env_group.add_argument('--use_cbrt', action='store_true', default=True, help='Include cube-root in fixed action set. Required for the cubic class to be solvable. Default: True. Pass --no-use-cbrt (via argparse BooleanOptionalAction-style or by setting explicitly) to recover the pre-cbrt action ordering for legacy checkpoints.')
    env_group.add_argument('--use_action_mask', action='store_true', help='Enable action masking for dynamic action space')

    train_group = parser.add_argument_group("Training")
    train_group.add_argument('--Ntrain', type=int, default=10**7, help='Total training timesteps')
    train_group.add_argument('--n_trials', type=int, default=1, help='Number of trials per agent')
    train_group.add_argument('--n_workers', type=int, default=1, help='Number of parallel workers')
    train_group.add_argument('--base_seed', type=int, default=16, help='Base seed')
    train_group.add_argument('--n_envs', type=int, default=1, help='Number of parallel envs for training (VecEnv)')
    train_group.add_argument('--ent_coef', type=float, default=0.01)
    train_group.add_argument('--early_stop_patience', type=int, default=0, help='Stop trial after N evals with no test_beam improvement (0 = disabled). Also enables best.zip checkpointing on every improvement.')
    train_group.add_argument('--anti_loop_penalty', type=float, default=0.0, help='Reward penalty alpha when current action == previous action (targets REL-loop failure mode). 0 = disabled.')
    train_group.add_argument('--eval_subsample', type=int, default=0, help='Sample N test eqns for in-training evals (0 = use full set). Deterministic seed so curves are comparable across runs.')
    train_group.add_argument('--eval_lite', action='store_true', help='Skip beam + at10 during in-training evals (greedy only). Final eval at end-of-training still runs all three. Switches early-stop metric to test_greedy.')

    eval_group = parser.add_argument_group("Eval / Logging")
    eval_group.add_argument('--eval_interval', type=int, default=0, help='Evaluation interval (steps)')
    eval_group.add_argument('--log_interval', type=int, default=0, help='Log interval (steps)')
    eval_group.add_argument('--save_root', type=str, default=None, help='Save root directory')

    model_group = parser.add_argument_group("Model / Architecture")
    model_group.add_argument('--agents', nargs='+', default=['ppo'], help='List of agents')
    model_group.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for MLP policy (non-tree)')
    model_group.add_argument('--load_model_path', type=str, default=None, help='Path to load model from')

    tree_group = parser.add_argument_group("TreeMLP Hyperparameters")
    tree_group.add_argument('--tree_embed_dim', type=int, default=64, help='Embedding dim for TreeMLPExtractor')
    tree_group.add_argument('--tree_hidden_dim', type=int, default=128, help='Hidden dim inside TreeMLPExtractor')
    tree_group.add_argument('--tree_K', type=int, default=3, help='Neighborhood / hop K used by TreeMLPExtractor')
    tree_group.add_argument('--tree_pooling', type=str, choices=['mean', 'max'], default='mean', help='Pooling for TreeMLPExtractor')
    tree_group.add_argument('--tree_vocab_min_id', type=int, default=-10, help='Min vocab id (ops/features) expected by extractor')
    tree_group.add_argument('--tree_vocab_max_id', type=int, default=600, help='Max vocab id; must exceed max feature_dict id for the gen (mixed_v2_easy:163, mixed_v2_large:511, so 600 is safe for both)')
    tree_group.add_argument('--tree_pad_id', type=int, default=99, help='Padding token id')
    tree_group.add_argument('--tree_pi_sizes', type=int, nargs='+', default=[128], help='PI MLP head sizes')
    tree_group.add_argument('--tree_vf_sizes', type=int, nargs='+', default=[128], help='VF MLP head sizes')

    replay_group = parser.add_argument_group("Success Replay (optional)")
    replay_group.add_argument('--sr_mix_ratio', type=float, default=0.5, help='Fraction of PPO batch worth of BC steps')
    replay_group.add_argument('--sr_batch_size', type=int, default=256, help='BC minibatch size')
    replay_group.add_argument('--sr_iters_per_rollout', type=int, default=5, help='BC iters per PPO rollout (was 10; lowered for speed)')
    replay_group.add_argument('--sr_capacity', type=int, default=20000, help='Replay capacity')
    replay_group.add_argument('--sr_buffer_kind', type=str, default='flat',
                              choices=['flat', 'balanced', 'balanced_lw', 'fresh'],
                              help='Buffer type. flat: original; balanced: per-equation uniform sampling; balanced_lw: balanced + length-weighted (prefer shorter traces); fresh: drop oldest half every 20 rollouts.')

    args = parser.parse_args()
    args.use_curriculum = True
    if args.action_space == 'dynamic': args.use_action_mask = True

    print_parameters(vars(args))

    env_name   = args.env_name
    agents     = args.agents
    Ntrain     = args.Ntrain
    eval_int   = args.eval_interval if args.eval_interval > 0 else max(1, Ntrain // 20)
    log_int    = args.log_interval  if args.log_interval  > 0 else max(1, Ntrain // 20)
    n_trials   = args.n_trials
    base_seed  = args.base_seed
    n_workers  = args.n_workers
    gen        = args.gen
    hidden_dim = args.hidden_dim
    n_envs     = args.n_envs
    save_root  = args.save_root or f"{gen}_hidden_dim{hidden_dim}_nenvs{args.n_envs}"
    load_model_path = args.load_model_path

    # DIR_save
    base = 'data/'
    if args.action_space == 'dynamic': base += 'dynamic_actions/'
    if args.use_relabel_constants: base += 'use_relabel_constants/'
    if args.use_success_replay: base += 'use_buffer/'
    if not args.use_curriculum: base += 'no_curriculum/'
    if args.sparse_rewards: base += 'sparse_rewards/'
    save_root = base + save_root

    tree_kwargs = dict(
        embed_dim    = args.tree_embed_dim,
        hidden_dim   = args.tree_hidden_dim,
        K            = args.tree_K,
        pooling      = args.tree_pooling,
        vocab_min_id = args.tree_vocab_min_id,
        vocab_max_id = args.tree_vocab_max_id,
        pad_id       = args.tree_pad_id,
        pi_sizes     = args.tree_pi_sizes,
        vf_sizes     = args.tree_vf_sizes,
    )

    jobs: List[Tuple] = []
    for agent in agents:
        save_root_agent = os.path.join(save_root, agent)
        if agent == 'ppo':
            bump, curiosity_local = 0, None
        else:
            curiosity_type = agent.split('-')[-1]
            bump = {'ICM': 1000, 'E3B': 2000, 'RIDE': 3000, 'RND': 4000, 'RE3': 5000, 'NGU': 6000, 'tree': 7000, 'gnn':8000, 'sage':9000}.get(curiosity_type, 0)
            curiosity_local = curiosity_type if curiosity_type in {'ICM','E3B','RIDE','RND','RE3','NGU'} else None
        for t in range(n_trials):
            seed = base_seed + 1000 * t + bump
            jobs.append((
                agent, env_name, gen, Ntrain, eval_int, log_int, seed, save_root_agent,
                curiosity_local, hidden_dim, load_model_path,
                args.sparse_rewards, args.use_relabel_constants, args.use_curriculum, tree_kwargs,
                n_envs, args.action_space, args.use_success_replay, args.sr_mix_ratio,
                args.sr_batch_size, args.sr_iters_per_rollout, args.sr_capacity, args.ent_coef,
                args.use_cov, args.early_stop_patience, args.anti_loop_penalty,
                args.eval_subsample, args.eval_lite, args.use_cbrt, args.sr_buffer_kind,
            ))

    rows, run_dirs = run_parallel(jobs, n_workers=n_workers, timeout_per_job=TRIAL_WALLCLOCK_LIMIT)
    if not rows:
        timed_print("No results gathered — all trials failed/timeouts?")
        raise SystemExit(1)

    df = pd.DataFrame(rows)
    summary = df.groupby('agent').agg(
        coverage_mean          = ('coverage_final_rate', 'mean'),
        coverage_std           = ('coverage_final_rate', 'std'),
        final_test_greedy_mean = ('final_test_acc', 'mean'),
        final_test_greedy_std  = ('final_test_acc', 'std'),
        final_test_beam_mean   = ('final_test_beam', 'mean'),
        final_test_beam_std    = ('final_test_beam', 'std'),
        final_test_at10_mean   = ('final_test_at10', 'mean'),
        final_test_at10_std    = ('final_test_at10', 'std'),
    ).reset_index()

    def pm(mean, std):
        if pd.isna(std):
            return f"{mean:.3f}"
        return f"{mean:.3f} ± {std:.3f}"

    timed_print("\n=== Summary over trials ===")
    for _, r in summary.iterrows():
        line = (
            f"{r['agent']}: "
            f"cov={pm(r['coverage_mean'], r['coverage_std'])}, "
            f"greedy={pm(r['final_test_greedy_mean'], r['final_test_greedy_std'])}, "
            f"beam={pm(r['final_test_beam_mean'], r['final_test_beam_std'])}, "
            f"acc@10={pm(r['final_test_at10_mean'], r['final_test_at10_std'])}"
        )
        timed_print(line)
    os.makedirs(save_root, exist_ok=True)
    out_csv = os.path.join(save_root, "summary.csv")
    summary.to_csv(out_csv, index=False)
    timed_print(f"\nSaved summary → {out_csv}")
