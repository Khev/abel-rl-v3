#!/usr/bin/env python3
"""Visualize TreeMLP learned representations as a 2-D t-SNE plot.

For each test equation, encode it through the trained TreeMLP, take the pooled
graph embedding, and t-SNE to 2 dimensions. Color-code points by equation type.

Usage:
    python tsne_embeddings.py --run_dir data/dynamic_actions/.../seed7001 \
                              --gen abel_level3 --out figures/tsne_embeds.png
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import sympy as sp
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.env_multi_eqn import multiEqn as multiEqnDynamic
from envs.env_multi_eqn_fixed import multiEqn as multiEqnFixed

from eval_per_type import classify  # reuse the type classifier


def _get_action_mask(env):
    return env.get_valid_action_mask()


def extract_embeddings(env, model, eqns, max_steps_setup=0):
    """For each equation, set it as main_eqn, encode obs, and run it through
    the policy's features_extractor. Returns (embeddings, labels)."""
    underlying = env.unwrapped if hasattr(env, "unwrapped") else env
    embs = []
    labels = []
    fx = model.policy.features_extractor
    device = next(fx.parameters()).device

    for eqn in eqns:
        try:
            env.reset()
            setter = getattr(env, "set_equation", None) or getattr(underlying, "set_equation", None)
            setter(sp.sympify(eqn))
            obs = getattr(underlying, "state", None)
            if obs is None:
                obs, _ = underlying.to_vec(underlying.lhs, underlying.rhs)
            # Build batch dim, move to tensor
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                emb = fx(obs_tensor)
            emb = emb.cpu().numpy().squeeze()
            embs.append(emb)
            labels.append(classify(eqn))
        except Exception as e:
            print(f"[warn] skipped {eqn}: {e}", file=sys.stderr)
    return np.stack(embs, axis=0), np.array(labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--gen", required=True)
    p.add_argument("--action_space", default="dynamic", choices=["dynamic", "fixed"])
    p.add_argument("--use_cov", action="store_true")
    p.add_argument("--use_relabel_constants", action="store_true")
    p.add_argument("--state_rep", default=None)
    p.add_argument("--out", default="figures/tsne_embeds.png")
    p.add_argument("--perplexity", type=float, default=15.0)
    p.add_argument("--n_iter", type=int, default=1000)
    p.add_argument("--max_n", type=int, default=400,
                   help="cap eqns to t-SNE for runtime (sampled if larger).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.state_rep is None:
        run_name = os.path.basename(os.path.dirname(args.run_dir.rstrip("/")))
        args.state_rep = "graph_integer_1d" if any(t in run_name for t in ("tree","gnn","sage")) else "integer_1d"

    EnvCls = multiEqnDynamic if args.action_space == "dynamic" else multiEqnFixed
    env = EnvCls(
        gen=args.gen,
        state_rep=args.state_rep,
        use_cov=args.use_cov,
        use_relabel_constants=args.use_relabel_constants,
        sparse_rewards=False,
        use_curriculum=False,
    )
    if args.action_space == "dynamic":
        env = ActionMasker(env, _get_action_mask)

    rd = Path(args.run_dir)
    model_path = None
    for cand in ["final_model.zip", "model.zip"]:
        if (rd / cand).exists():
            model_path = rd / cand; break
    if model_path is None:
        zips = list(rd.glob("*.zip"))
        model_path = zips[0] if zips else None
    if model_path is None or not model_path.exists():
        print(f"no model.zip in {rd}", file=sys.stderr); sys.exit(2)

    try:
        model = MaskablePPO.load(str(model_path), env=env, device="cpu")
    except Exception:
        model = PPO.load(str(model_path), env=env, device="cpu")

    underlying = env.unwrapped if hasattr(env, "unwrapped") else env
    train_eqns = list(getattr(underlying, "train_eqns", []))
    test_eqns = list(getattr(underlying, "test_eqns", []))

    rng = np.random.default_rng(args.seed)
    eqns = train_eqns + test_eqns
    if len(eqns) > args.max_n:
        idx = rng.choice(len(eqns), size=args.max_n, replace=False)
        eqns = [eqns[i] for i in idx]
    print(f"Encoding {len(eqns)} equations...")

    embs, labels = extract_embeddings(env, model, eqns)
    print(f"Embedding shape: {embs.shape}")

    from sklearn.manifold import TSNE
    perp = min(args.perplexity, max(5.0, embs.shape[0] / 4 - 1))
    tsne = TSNE(n_components=2, perplexity=perp,
                n_iter=args.n_iter, random_state=args.seed, init="pca")
    coords = tsne.fit_transform(embs)

    # plot
    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(i % 10)], s=22, alpha=0.7, label=f"{lbl} (n={mask.sum()})",
                   edgecolors="none")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"t-SNE of TreeMLP pooled embeddings — {args.gen}\n"
                 f"(model: {rd.name}, perplexity={perp:.1f})", fontsize=11)
    ax.legend(loc="best", fontsize=8, markerscale=1.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
