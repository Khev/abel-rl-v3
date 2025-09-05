#!/usr/bin/env python3
"""
Tune script for train_abel.py

- Randomly samples N hyperparameter combinations (default: 50)
- Runs T trials for each (default: 3)
- Uses train_abel.run_parallel to execute all jobs
- Saves a master results CSV + per-config summary
- Generates pretty plots (leaderboard & scatter)
"""

import os
import re
import json
import math
import time
import uuid
import argparse
from datetime import datetime
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IMPORTANT: your existing sweep utilities live here
import train_abel as TA  # must be in PYTHONPATH / same folder

# ------------------------------
# Pretty printing / helpers
# ------------------------------
def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def pm(mean, std):
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"

# ------------------------------
# Search space
# ------------------------------
def sample_config(rng: np.random.Generator, args) -> dict:
    """
    Return a single random configuration dict.
    You can tweak ranges here freely.
    """
    # Agents: stick with tree policy by default; you can add 'ppo' too.
    agent = rng.choice(["ppo-tree"])

    # Model size & env parallelism
    hidden_dim = int(rng.choice([64, 128, 256]))
    n_envs     = int(rng.choice([1, 2]))  # keep small for speed/stability

    # Rewards / extras
    sparse_rewards        = bool(rng.choice([False, True]))
    use_relabel_constants = bool(rng.choice([False, True]))
    use_curriculum        = True  # default to on (matches your train file)

    # Success replay (off by default; try it sometimes)
    use_success_replay = bool(rng.choice([False, True, False]))  # ~33% on
    if use_success_replay:
        sr_mix_ratio         = float(rng.choice([0.25, 0.5, 0.75]))
        sr_batch_size        = int(rng.choice([128, 256, 512]))
        sr_iters_per_rollout = int(rng.choice([5, 10, 20, 40]))
        sr_capacity          = int(rng.choice([20_000, 50_000, 100_000]))
    else:
        sr_mix_ratio, sr_batch_size, sr_iters_per_rollout, sr_capacity = 0.5, 256, 10, 20_000

    # Tree extractor sizes (safe defaults; can be tuned too)
    tree_kwargs = dict(
        embed_dim    = int(rng.choice([32, 64])),
        hidden_dim   = int(rng.choice([64, 128])),
        K            = int(rng.choice([2, 3])),
        pooling      = rng.choice(["mean", "max"]),
        vocab_min_id = -10,
        pad_id       = 99,
        pi_sizes     = [128],
        vf_sizes     = [128],
    )

    # Eval cadence: log/eval ~ every 5% of training
    eval_interval = max(10_000, args.Ntrain // 20)
    log_interval  = eval_interval

    return dict(
        # env / meta
        env_name="multi_eqn",
        gen=args.gen,
        # algo / model
        agent=agent,
        hidden_dim=hidden_dim,
        n_envs=n_envs,
        # training
        Ntrain=args.Ntrain,
        eval_interval=eval_interval,
        log_interval=log_interval,
        # flags
        sparse_rewards=sparse_rewards,
        use_relabel_constants=use_relabel_constants,
        use_curriculum=use_curriculum,
        # success replay
        use_success_replay=use_success_replay,
        sr_mix_ratio=sr_mix_ratio,
        sr_batch_size=sr_batch_size,
        sr_iters_per_rollout=sr_iters_per_rollout,
        sr_capacity=sr_capacity,
        # extractor
        tree_kwargs=tree_kwargs,
        # curiosity / loading
        curiosity=None,
        load_model_path=None,
    )

def config_to_tag(cfg: dict) -> str:
    """
    Human-friendly short tag for folder names & plots.
    """
    bits = [
        cfg["agent"],
        f"hd{cfg['hidden_dim']}",
        f"envs{cfg['n_envs']}",
        "sr1" if cfg["use_success_replay"] else "sr0",
        f"mr{str(cfg['sr_mix_ratio']).replace('.','p')}" if cfg["use_success_replay"] else "",
        f"cap{cfg['sr_capacity']}" if cfg["use_success_replay"] else "",
        cfg["tree_kwargs"]["pooling"],
        f"K{cfg['tree_kwargs']['K']}",
        f"emb{cfg['tree_kwargs']['embed_dim']}",
    ]
    tag = "-".join([b for b in bits if b])
    return slugify(tag)

# ------------------------------
# Job builder
# ------------------------------
def build_jobs_for_config(cfg: dict, cfg_id: int, n_trials: int, base_seed: int, save_root_base: Path):
    """
    Build the job tuples that TA.run_parallel expects, for a single config across n_trials.
    """
    tag = f"cfg{cfg_id:03d}-{config_to_tag(cfg)}"
    save_root = save_root_base / tag
    ensure_dir(save_root)

    # Same intervals per config
    eval_int = cfg["eval_interval"]
    log_int  = cfg["log_interval"]

    jobs = []
    for t in range(n_trials):
        # mimic your previous seeding scheme
        seed = base_seed + 1000 * t + (7000 if "tree" in cfg["agent"] else 0)

        jobs.append((
            cfg["agent"],
            cfg["env_name"],
            cfg["gen"],
            cfg["Ntrain"],
            eval_int,
            log_int,
            seed,
            str(save_root / cfg["agent"]),  # per-agent subdir
            cfg["curiosity"],
            cfg["hidden_dim"],
            cfg["load_model_path"],
            cfg["sparse_rewards"],
            cfg["use_relabel_constants"],
            cfg["use_curriculum"],
            cfg["tree_kwargs"],
            cfg["n_envs"],
            # success replay knobs
            cfg["use_success_replay"],
            cfg["sr_mix_ratio"],
            cfg["sr_batch_size"],
            cfg["sr_iters_per_rollout"],
            cfg["sr_capacity"],
        ))
    return jobs, tag, save_root

# ------------------------------
# Plotting
# ------------------------------
def make_plots(master_df: pd.DataFrame, out_dir: Path, top_k: int = 10):
    ensure_dir(out_dir)

    # Leaderboard: top by coverage_final_rate then final_test_beam
    ranked = master_df.sort_values(
        ["coverage_final_rate", "final_test_beam", "final_test_acc"],
        ascending=[False, False, False]
    )

    # Aggregate per config (mean ± std)
    agg = (master_df
           .groupby("config_tag", as_index=False)
           .agg(
               coverage_mean=("coverage_final_rate", "mean"),
               coverage_std =("coverage_final_rate", "std"),
               test_beam_mean=("final_test_beam", "mean"),
               test_beam_std =("final_test_beam", "std"),
               test_greedy_mean=("final_test_acc", "mean"),
               test_greedy_std =("final_test_acc", "std"),
               trials=("config_tag", "count"),
           )
           .sort_values(["coverage_mean","test_beam_mean"], ascending=[False, False])
          )

    agg.to_csv(out_dir / "leaderboard.csv", index=False)

    # Pretty bar plot of top_k coverage
    top = agg.head(top_k)
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top)), top["coverage_mean"].values, xerr=top["coverage_std"].values, alpha=0.8)
    plt.gca().invert_yaxis()
    plt.yticks(range(len(top)), [t for t in top["config_tag"].values])
    plt.xlabel("Coverage (mean ± std)")
    plt.title(f"Top {top_k} configs by coverage")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "top_coverage.png", dpi=200)
    plt.close()

    # Scatter: coverage vs beam, color by success replay
    # infer sr flag from tag
    sr_flag = master_df["config_tag"].str.contains(r"-sr1").astype(int)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        master_df["coverage_final_rate"],
        master_df["final_test_beam"],
        c=sr_flag,
        cmap="coolwarm",
        alpha=0.7
    )
    plt.xlabel("Coverage (final)")
    plt.ylabel("Beam accuracy (final)")
    cbar = plt.colorbar(sc, ticks=[0, 1])
    cbar.ax.set_yticklabels(["SR=off", "SR=on"])
    plt.title("Final metrics per trial")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_cov_vs_beam.png", dpi=200)
    plt.close()

    # Save a short markdown report
    with open(out_dir / "REPORT.md", "w") as f:
        f.write("# Tuning Report\n\n")
        f.write(f"- Total trials: **{len(master_df)}**\n")
        f.write(f"- Unique configs: **{master_df['config_tag'].nunique()}**\n")
        if not top.empty:
            f.write("\n## Top configs (coverage)\n")
            for _, r in top.iterrows():
                f.write(f"- **{r['config_tag']}** — "
                        f"cov={pm(r['coverage_mean'], r['coverage_std'])}, "
                        f"beam={pm(r['test_beam_mean'], r['test_beam_std'])}, "
                        f"greedy={pm(r['test_greedy_mean'], r['test_greedy_std'])} "
                        f"(trials={int(r['trials'])})\n")

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuner for train_abel.py")
    parser.add_argument("--gen", type=str, default="abel_level3", help="Equation generator (e.g., abel_level2/3/4)")
    parser.add_argument("--Ntrain", type=int, default=1_000_000, help="Total timesteps per trial")
    parser.add_argument("--n_combinations", type=int, default=50, help="Number of random configs to try")
    parser.add_argument("--n_trials", type=int, default=3, help="Trials per config")
    parser.add_argument("--base_seed", type=int, default=1, help="Base seed")
    parser.add_argument("--max_workers", type=int, default=4, help="Parallel workers for jobs")
    parser.add_argument("--out_dir", type=str, default=None, help="Root directory for all tuning artifacts")
    args = parser.parse_args()

    rng = np.random.default_rng(args.base_seed)

    # Output layout
    run_tag = f"{args.gen}_{stamp()}"
    out_root = Path(args.out_dir) if args.out_dir else Path("data") / "tune" / run_tag
    ensure_dir(out_root)

    print(f"�� Tuning output dir: {out_root}")

    # Sample configurations
    configs = [sample_config(rng, args) for _ in range(args.n_combinations)]

    # Build ALL jobs across all configs & trials
    all_jobs = []
    job_config_tags = []  # parallel list aligned to returned rows by decoding save_dir later
    per_config_dirs = {}

    for i, cfg in enumerate(configs):
        cfg_jobs, cfg_tag, cfg_dir = build_jobs_for_config(
            cfg=cfg,
            cfg_id=i,
            n_trials=args.n_trials,
            base_seed=args.base_seed,
            save_root_base=out_root,
        )
        all_jobs.extend(cfg_jobs)
        job_config_tags.extend([cfg_tag] * len(cfg_jobs))
        per_config_dirs[cfg_tag] = cfg_dir

    print(f"🔧 Launching {len(all_jobs)} jobs "
          f"({args.n_combinations} configs × {args.n_trials} trials)...")

    # Fire!
    rows, run_dirs = TA.run_parallel(all_jobs, n_workers=args.max_workers, timeout_per_job=TA.TRIAL_WALLCLOCK_LIMIT)

    # rows are metrics dicts; enrich with config_tag by parsing model_path OR using our parallel list
    master = []
    for metrics, cfg_tag in zip(rows, job_config_tags):
        md = dict(metrics)  # copy
        md["config_tag"] = cfg_tag
        master.append(md)
    master_df = pd.DataFrame(master)

    # Save master results
    master_csv = out_root / "master_results.csv"
    master_df.to_csv(master_csv, index=False)
    print(f"✅ Saved results → {master_csv}")

    # Write a per-config summary (aggregating its trials)
    cfg_summary_rows = []
    for cfg_tag, sub in master_df.groupby("config_tag"):
        out = dict(
            config_tag=cfg_tag,
            trials=len(sub),
            coverage_mean=sub["coverage_final_rate"].mean(),
            coverage_std =sub["coverage_final_rate"].std(),
            test_beam_mean=sub["final_test_beam"].mean(),
            test_beam_std =sub["final_test_beam"].std(),
            test_greedy_mean=sub["final_test_acc"].mean(),
            test_greedy_std =sub["final_test_acc"].std(),
        )
        cfg_summary_rows.append(out)
    cfg_summary = pd.DataFrame(cfg_summary_rows).sort_values(["coverage_mean", "test_beam_mean"], ascending=[False, False])
    cfg_summary.to_csv(out_root / "config_summary.csv", index=False)
    print(f"✅ Saved config summary → {out_root / 'config_summary.csv'}")

    # Plots
    make_plots(master_df, out_root, top_k=10)
    print(f"🖼️ Plots saved under → {out_root}")

    # Optional: print a short leaderboard to console
    print("\n=== Top 5 (coverage) ===")
    for _, r in cfg_summary.head(5).iterrows():
        print(f"{r['config_tag']}: "
              f"cov={pm(r['coverage_mean'], r['coverage_std'])}, "
              f"beam={pm(r['test_beam_mean'], r['test_beam_std'])}, "
              f"greedy={pm(r['test_greedy_mean'], r['test_greedy_std'])} "
              f"(trials={int(r['trials'])})")

if __name__ == "__main__":
    main()

