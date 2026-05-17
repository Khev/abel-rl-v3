#!/usr/bin/env python3
"""Visualize TreeMLP embeddings with PCA + t-SNE in 2D and 3D.

Loads a trained model, encodes equations through TreeMLP, and produces a
2x2 figure: {PCA, t-SNE} x {2D, 3D}, all color-coded by equation type.

Usage:
    python plot_embeddings.py --run_dir data/.../seed7001 --gen abel_level3 \
                              --use_relabel_constants --out figures/embed_l3.png
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.env_multi_eqn import multiEqn as multiEqnDynamic
from envs.env_multi_eqn_fixed import multiEqn as multiEqnFixed
from eval_per_type import classify
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def _get_action_mask(env):
    return env.get_valid_action_mask()


def extract_embeddings(env, model, eqns):
    underlying = env.unwrapped if hasattr(env, "unwrapped") else env
    fx = model.policy.features_extractor
    embs, labels = [], []
    for eqn in eqns:
        try:
            env.reset()
            setter = getattr(env, "set_equation", None) or getattr(underlying, "set_equation", None)
            setter(sp.sympify(eqn))
            obs = getattr(underlying, "state", None)
            if obs is None:
                obs, _ = underlying.to_vec(underlying.lhs, underlying.rhs)
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                emb = fx(obs_tensor)
            embs.append(emb.cpu().numpy().squeeze())
            labels.append(classify(eqn))
        except Exception as e:
            print(f"[warn] skipped {eqn}: {e}", file=sys.stderr)
    return np.stack(embs, 0), np.array(labels)


def project(X, method, dim, seed):
    if method == "pca":
        return PCA(n_components=dim, random_state=seed).fit_transform(X)
    elif method == "tsne":
        perp = min(15.0, max(5.0, X.shape[0] / 4 - 1))
        return TSNE(n_components=dim, perplexity=perp,
                    max_iter=1000, random_state=seed, init="pca").fit_transform(X)
    raise ValueError(method)


def scatter_2d(ax, coords, labels, title):
    unique = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab10")
    for i, lbl in enumerate(unique):
        m = labels == lbl
        ax.scatter(coords[m, 0], coords[m, 1], c=[cmap(i % 10)],
                   s=18, alpha=0.7, edgecolors="none",
                   label=f"{lbl} (n={m.sum()})")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def scatter_3d(ax, coords, labels, title):
    unique = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab10")
    for i, lbl in enumerate(unique):
        m = labels == lbl
        ax.scatter(coords[m, 0], coords[m, 1], coords[m, 2],
                   c=[cmap(i % 10)], s=14, alpha=0.7, edgecolors="none",
                   label=f"{lbl} (n={m.sum()})")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=20, azim=-60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--gen", required=True)
    p.add_argument("--action_space", default="dynamic", choices=["dynamic", "fixed"])
    p.add_argument("--use_cov", action="store_true")
    p.add_argument("--use_relabel_constants", action="store_true")
    p.add_argument("--state_rep", default=None)
    p.add_argument("--out", default="figures/embed_panel.png")
    p.add_argument("--max_n", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.state_rep is None:
        run_name = os.path.basename(os.path.dirname(args.run_dir.rstrip("/")))
        args.state_rep = "graph_integer_1d" if any(t in run_name for t in ("tree","gnn","sage")) else "integer_1d"

    EnvCls = multiEqnDynamic if args.action_space == "dynamic" else multiEqnFixed
    env = EnvCls(
        gen=args.gen, state_rep=args.state_rep,
        use_cov=args.use_cov, use_relabel_constants=args.use_relabel_constants,
        sparse_rewards=False, use_curriculum=False,
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
    if model_path is None:
        print("no model"); sys.exit(2)
    try:
        model = MaskablePPO.load(str(model_path), env=env, device="cpu")
    except Exception:
        model = PPO.load(str(model_path), env=env, device="cpu")

    underlying = env.unwrapped if hasattr(env, "unwrapped") else env
    eqns = list(getattr(underlying, "train_eqns", [])) + list(getattr(underlying, "test_eqns", []))
    rng = np.random.default_rng(args.seed)
    if len(eqns) > args.max_n:
        idx = rng.choice(len(eqns), size=args.max_n, replace=False)
        eqns = [eqns[i] for i in idx]
    print(f"Encoding {len(eqns)} equations...")
    X, y = extract_embeddings(env, model, eqns)
    print(f"Embeddings: {X.shape}")

    # 4-panel figure: rows = method (PCA / t-SNE), cols = dim (2D / 3D)
    fig = plt.figure(figsize=(12, 10))
    coords_pca2 = project(X, "pca", 2, args.seed)
    coords_pca3 = project(X, "pca", 3, args.seed)
    coords_tsne2 = project(X, "tsne", 2, args.seed)
    coords_tsne3 = project(X, "tsne", 3, args.seed)

    ax1 = fig.add_subplot(2, 2, 1);             scatter_2d(ax1, coords_pca2,  y, "PCA (2D)")
    ax2 = fig.add_subplot(2, 2, 2, projection='3d'); scatter_3d(ax2, coords_pca3,  y, "PCA (3D)")
    ax3 = fig.add_subplot(2, 2, 3);             scatter_2d(ax3, coords_tsne2, y, "t-SNE (2D)")
    ax4 = fig.add_subplot(2, 2, 4, projection='3d'); scatter_3d(ax4, coords_tsne3, y, "t-SNE (3D)")

    # one legend at the top for the whole figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)),
               bbox_to_anchor=(0.5, 1.01), fontsize=9, frameon=False, markerscale=1.4)

    fig.suptitle(f"TreeMLP pooled embeddings — {args.gen} ({len(eqns)} eqns)",
                 fontsize=12, y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
