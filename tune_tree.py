#!/usr/bin/env python3
"""
tune_tree.py

Sweep TreeMLPExtractor hyperparameters for ppo-tree on abel_level4.
PPO hyperparameters stay fixed (SB3 defaults), only extractor changes.

Outputs:
- data/<gen>_tree_sweep/ppo-tree/<cfg-hash>/seed<seed>/{final_model.zip, learning_curves.csv, metrics.json}
- data/<gen>_tree_sweep/summary.csv (aggregated over seeds/configs)
"""

import os
import json
import time
import signal
import hashlib
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import datetime

import gymnasium as gym
from gymnasium import ObservationWrapper
import torch

# --- Your envs ---
from envs.env_single_eqn_fixed import singleEqn
from envs.env_multi_eqn_fixed import multiEqn

# --- SB3 ---
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# --- Custom extractor ---
from utils.utils_env import TreeMLPExtractor

# ==========================
# Global knobs / defaults
# ==========================
EVAL_TIMEOUT_DET   = 0.75  # greedy per-eqn seconds
EVAL_TIMEOUT_STOCH = 1.00  # s@N per-eqn seconds

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))


# ---------------------------
# Utilities
# ---------------------------
def timed_print(msg: str):
    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}: {msg}")


class EvalTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
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


# ---------------------------
# Env / agent factories
# ---------------------------
def make_env(env_name: str, gen: str, seed: int):
    """Force graph obs for ppo-tree."""
    state_rep = 'graph_integer_1d'
    sparse_rewards = False
    use_relabel_constants = False

    if env_name == 'single_eqn':
        env = singleEqn(main_eqn='a*x+b', state_rep=state_rep)
    elif env_name == 'multi_eqn':
        env = multiEqn(gen=gen, use_relabel_constants=use_relabel_constants,
                       state_rep=state_rep, sparse_rewards=sparse_rewards)
    else:
        raise ValueError(f"Unknown env_name: {env_name}")
    try:
        env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
    return env


def make_tree_policy(env, embed_dim: int, hidden_dim: int, K: int, pooling: str):
    """Create PPO with MultiInputPolicy + TreeMLPExtractor configured."""
    # Conservative net sizes for the actor/critic heads; keep PPO stable
    kwargs = dict(
        features_extractor_class=TreeMLPExtractor,
        features_extractor_kwargs=dict(
            max_nodes=env.observation_dim // 2,
            max_edges=2 * (env.observation_dim // 2),
            vocab_min_id=-10,
            pad_id=99,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            K=K,
            pooling=pooling,  # "mean" or "max"
        ),
        net_arch=dict(pi=[128], vf=[128]),
    )
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=kwargs,
        verbose=0,
        seed=0,          # we reseed per trial in learn() via env seed; PPO seed here is fine
        n_steps=2048,    # keep PPO defaults; extractor is what we tune
    )
    return model


# ---------------------------
# Eval helpers
# ---------------------------
def _set_equation_if_supported(env, eqn):
    seteq = getattr(env, "set_equation", None)
    if callable(seteq):
        seteq(eqn)
    setup = getattr(env, "setup", None)
    if callable(setup):
        setup()


def greedy_solve_one(model, env, eqn, max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_DET):
    try:
        with time_limit(per_eqn_seconds):
            _, _ = env.reset()
            _set_equation_if_supported(env, eqn)
            obs, _ = env.reset()
            for _ in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                if info.get("is_solved", False):
                    return True
                if terminated or truncated:
                    break
        return False
    except (EvalTimeout, Exception):
        return False


def greedy_accuracy(model, env, equations, max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_DET):
    if not equations:
        return None
    solved = sum(greedy_solve_one(model, env, eq, max_steps, per_eqn_seconds) for eq in equations)
    return solved / len(equations)


def success_at_n(model, env, equations, n_trials=10, max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_STOCH):
    if not equations:
        return None
    solved_any = 0
    for eqn in equations:
        try:
            with time_limit(per_eqn_seconds):
                s = False
                for _ in range(n_trials):
                    _, _ = env.reset()
                    _set_equation_if_supported(env, eqn)
                    obs, _ = env.reset()
                    for _ in range(max_steps):
                        action, _ = model.predict(obs, deterministic=False)
                        obs, _, terminated, truncated, info = env.step(action)
                        if info.get("is_solved", False):
                            s = True
                            break
                        if terminated or truncated:
                            break
                    if s:
                        break
                if s:
                    solved_any += 1
        except (EvalTimeout, Exception):
            pass
    return solved_any / len(equations)


# ---------------------------
# Callback with periodic eval
# ---------------------------
class TrainingLogger(BaseCallback):
    def __init__(self, algo_name: str, train_env, eval_env, eval_interval: int, save_dir: str, verbose=1):
        super().__init__(verbose)
        self.algo_name     = algo_name
        self.train_env     = train_env
        self.eval_env      = eval_env
        self.eval_interval = eval_interval
        self.save_dir      = save_dir
        self.curves_path   = os.path.join(self.save_dir, "learning_curves.csv")

        self.train_eqns = getattr(train_env, "train_eqns", None)
        self.test_eqns  = getattr(train_env, "test_eqns", None)
        self.num_eqns   = len(self.train_eqns) if self.train_eqns is not None else 1

        self.Tsolves    = {}
        self.Tsolve     = None
        self.Tconverge  = None

        self.log_steps, self.coverage, self.test_acc, self.test_at10 = [], [], [], []
        os.makedirs(self.save_dir, exist_ok=True)

    def _log_eval(self, step):
        solved = min(len(self.Tsolves), self.num_eqns)
        cov = solved / self.num_eqns if self.num_eqns else 0.0
        tst = greedy_accuracy(self.model, self.eval_env, self.test_eqns,  max_steps=5) if self.test_eqns else None
        t10 = success_at_n(   self.model, self.eval_env, self.test_eqns, n_trials=10, max_steps=5) if self.test_eqns else None

        self.log_steps.append(step)
        self.coverage.append(cov)
        self.test_acc.append(tst if tst is not None else np.nan)
        self.test_at10.append(t10 if t10 is not None else np.nan)

        curves = pd.DataFrame({
            "step": self.log_steps,
            "coverage": self.coverage,
            "test_acc": self.test_acc,
            "test_at10": self.test_at10
        })
        tmp = self.curves_path + ".tmp"
        curves.to_csv(tmp, index=False)
        os.replace(tmp, self.curves_path)

    def _on_training_start(self) -> None:
        self._log_eval(step=0)

    def _on_step(self) -> bool:
        step = self.num_timesteps
        for info in self.locals.get("infos", []):
            if info.get("is_solved"):
                eqn  = info.get("main_eqn", "eqn")
                if eqn not in self.Tsolves:
                    self.Tsolves[eqn] = step
                if self.Tconverge is None and len(self.Tsolves) >= self.num_eqns:
                    self.Tconverge = step
        if self.eval_interval and step % self.eval_interval == 0:
            self._log_eval(step)
        return True

    def _on_training_end(self) -> None:
        self.Tsolve = float(np.mean(list(self.Tsolves.values()))) if self.Tsolves else float('inf')
        curves = pd.DataFrame({
            "step": self.log_steps,
            "coverage": self.coverage,
            "test_acc": self.test_acc,
            "test_at10": self.test_at10
        })
        curves.to_csv(self.curves_path, index=False)


# ---------------------------
# One trial (one config × one seed)
# ---------------------------
def cfg_hash(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:8]


def run_trial(env_name: str,
              gen: str,
              Ntrain: int,
              eval_interval: int,
              seed: int,
              save_root: str,
              extractor_cfg: dict):
    """Train & evaluate one ppo-tree run with a given TreeMLP config."""
    # Build envs
    train_env = make_env(env_name, gen, seed)
    eval_env  = make_env(env_name, gen, seed + 777)

    # Build model
    model = make_tree_policy(
        env=train_env,
        embed_dim=extractor_cfg["embed_dim"],
        hidden_dim=extractor_cfg["hidden_dim"],
        K=extractor_cfg["K"],
        pooling=extractor_cfg["pooling"],
    )

    # Save dir
    tag_cfg = cfg_hash(extractor_cfg)
    run_dir = os.path.join(save_root, "ppo-tree", f"cfg{tag_cfg}", f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    # Callbacks
    cb_main = TrainingLogger("ppo-tree", train_env, eval_env, eval_interval, run_dir)
    cb_prog = ProgressBarCallback()
    callbacks = [cb_main, cb_prog]

    # Train
    model.set_random_seed(seed)
    model.learn(total_timesteps=Ntrain, callback=callbacks)

    # Save model
    model_path = os.path.join(run_dir, "final_model.zip")
    model.save(model_path)
    timed_print(f"[ppo-tree] Saved model → {model_path}")

    # Final eval
    final_test_acc  = greedy_accuracy(model, eval_env, getattr(train_env, "test_eqns", []),  max_steps=60) or 0.0
    final_test_at10 = success_at_n(   model, eval_env, getattr(train_env, "test_eqns", []), n_trials=10, max_steps=60) or 0.0

    coverage_final = len(cb_main.Tsolves)
    num_eqns = cb_main.num_eqns
    coverage_final_rate = (coverage_final / num_eqns) if num_eqns else 0.0

    metrics = {
        "agent": "ppo-tree",
        "seed": seed,
        "extractor": extractor_cfg,
        "coverage_final_rate": coverage_final_rate,
        "final_test_acc":  final_test_acc,
        "final_test_at10": final_test_at10,
        "Tsolve": cb_main.Tsolve,
        "Tconverge": cb_main.Tconverge,
        "num_eqns": num_eqns,
        "run_dir": run_dir,
        "model_path": model_path,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Cleanup
    try:
        train_env.close(); eval_env.close()
    except Exception:
        pass

    return metrics, run_dir


# ---------------------------
# Parallel runner
# ---------------------------
def run_parallel(jobs, n_workers=4):
    rows, run_dirs = [], []
    ctx = mp.get_context("spawn")
    total, done = len(jobs), 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = [ex.submit(run_trial, *job) for job in jobs]
        for fut in as_completed(futures):
            try:
                metrics, run_dir = fut.result()
                rows.append(metrics); run_dirs.append(run_dir)
                done += 1
                timed_print(
                    f"✓ [{done}/{total}] cfg={cfg_hash(metrics['extractor'])} seed={metrics['seed']} "
                    f"| cov={metrics['coverage_final_rate']:.2f} acc={metrics['final_test_acc']:.2f} at10={metrics['final_test_at10']:.2f}"
                )
            except Exception as e:
                done += 1
                timed_print(f"✗ [{done}/{total}] Job crashed: {e}")
    return rows, run_dirs


# ---------------------------
# CLI
# ---------------------------
def parse_csv_list(s, cast):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def main():
    import argparse
    p = argparse.ArgumentParser("Tune TreeMLP extractor for ppo-tree")
    p.add_argument("--env_name", type=str, default="multi_eqn")
    p.add_argument("--gen", type=str, default="abel_level4")  # depth-4 as requested
    p.add_argument("--Ntrain", type=int, default=3*10**6)
    p.add_argument("--eval_interval", type=int, default=None)  # default → Ntrain//20
    p.add_argument("--n_workers", type=int, default=3)
    p.add_argument("--seeds", type=str, default="31,32,33")

    # Extractor sweeps (ONLY these are tuned)
    p.add_argument("--embed_dims", type=str, default="16,32,64")
    p.add_argument("--hidden_dims", type=str, default="64,128")
    p.add_argument("--Ks", type=str, default="2,4")
    p.add_argument("--poolings", type=str, default="mean")

    p.add_argument("--save_root", type=str, default=None)

    args = p.parse_args()

    seeds      = parse_csv_list(args.seeds, int)
    embed_dims = parse_csv_list(args.embed_dims, int)
    hidden_dims= parse_csv_list(args.hidden_dims, int)
    Ks         = parse_csv_list(args.Ks, int)
    poolings   = parse_csv_list(args.poolings, str)

    save_root = args.save_root or f"data/{args.gen}_tree_sweep"
    eval_interval = args.eval_interval or (args.Ntrain // 20)

    timed_print("--------------------------------------------------")
    timed_print(f"Tuning TreeMLP on {args.gen} | seeds={seeds}")
    timed_print(f"Grid: embed={embed_dims} hidden={hidden_dims} K={Ks} pooling={poolings}")
    timed_print("--------------------------------------------------")

    # Build jobs
    jobs = []
    for ed in embed_dims:
        for hd in hidden_dims:
            for k in Ks:
                for pool in poolings:
                    cfg = dict(embed_dim=ed, hidden_dim=hd, K=k, pooling=pool)
                    for s in seeds:
                        jobs.append((
                            args.env_name, args.gen, args.Ntrain, eval_interval,
                            s, save_root, cfg
                        ))

    rows, _ = run_parallel(jobs, n_workers=args.n_workers)
    if not rows:
        timed_print("No results gathered — all trials failed?")
        raise SystemExit(1)

    # Aggregate
    df = pd.DataFrame(rows)
    # explode extractor dict into columns
    ext = pd.json_normalize(df["extractor"])
    df = pd.concat([df.drop(columns=["extractor"]), ext], axis=1)

    group_cols = ["embed_dim", "hidden_dim", "K", "pooling"]
    summary = df.groupby(group_cols, as_index=False).agg(
        coverage_mean=("coverage_final_rate", "mean"),
        coverage_std =("coverage_final_rate", "std"),
        test_acc_mean=("final_test_acc", "mean"),
        test_acc_std =("final_test_acc", "std"),
        test_at10_mean=("final_test_at10", "mean"),
        test_at10_std =("final_test_at10", "std"),
    )

    os.makedirs(save_root, exist_ok=True)
    out_csv = os.path.join(save_root, "summary.csv")
    summary.to_csv(out_csv, index=False)
    timed_print("\n=== Summary (by extractor config) ===")
    timed_print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    timed_print(f"\nSaved summary → {out_csv}")


if __name__ == "__main__":
    main()
