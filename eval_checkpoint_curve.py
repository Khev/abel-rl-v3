#!/usr/bin/env python3
"""Evaluate every model_step*.zip checkpoint of a run to build a test_beam
learning curve. Needed because the headline runs used --eval_lite (test_beam
not logged during training).

Outputs a CSV: step,test_greedy,test_beam_plain,test_beam_value

Usage:
    python eval_checkpoint_curve.py --run_dir <.../ppo-tree/seedXXXX> --gen mixed_v2_easy
"""
import argparse
import sys
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass


def eval_one(args):
    """Worker: eval a single checkpoint. Returns (step, greedy, plain, value)."""
    ckpt_path, gen = args
    import warnings
    warnings.filterwarnings("ignore")
    from sb3_contrib import MaskablePPO
    from envs.env_multi_eqn import multiEqn
    from train_abel import greedy_accuracy, beam_accuracy

    m = re.search(r"model_step0*(\d+)\.zip", str(ckpt_path))
    step = int(m.group(1)) if m else -1

    env = multiEqn(gen=gen, state_rep="graph_integer_1d",
                   use_cov=True, use_relabel_constants=True, use_success_replay=True)
    test = env.test_eqns
    try:
        model = MaskablePPO.load(str(ckpt_path), env=env, device="cpu")
    except Exception as e:
        return (step, None, None, None)

    g = greedy_accuracy(model, env, test, max_steps=10, per_eqn_seconds=0.75) or 0.0
    pb = beam_accuracy(model, env, test, beam_width=5, topk_per_node=5,
                       max_steps=10, per_eqn_seconds=0.75, beam_lambda=0.0) or 0.0
    vb = beam_accuracy(model, env, test, beam_width=5, topk_per_node=5,
                       max_steps=10, per_eqn_seconds=0.75, beam_lambda=1.0) or 0.0
    return (step, g, pb, vb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Path to a seedXXXX dir")
    p.add_argument("--gen", default="mixed_v2_easy")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ckpt_dir = Path(args.run_dir) / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("model_step*.zip"),
                   key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
    print(f"Found {len(ckpts)} checkpoints in {ckpt_dir}")

    out_path = args.out or (Path(args.run_dir) / "test_beam_curve.csv")
    jobs = [(str(c), args.gen) for c in ckpts]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(eval_one, jobs):
            results.append(r)
            print(f"  step={r[0]:>8}  greedy={r[1]}  plain={r[2]}  value={r[3]}", flush=True)

    results.sort(key=lambda r: r[0])
    with open(out_path, "w") as f:
        f.write("step,test_greedy,test_beam_plain,test_beam_value\n")
        for step, g, pb, vb in results:
            f.write(f"{step},{g},{pb},{vb}\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
