#!/usr/bin/env python3
"""Re-evaluate the closed abel_level4 ppo-tree-rc-buf checkpoints with the
value-guided beam decoder.

Motivation: the abel_level4 test_beam in learning_curves.csv was computed
with PLAIN beam during training, and the 10M-step runs drifted DOWN
(0.49-0.56) vs the 4M run (0.63). This isolates how much the value-beam
decoder + best-checkpoint selection recover, with NO retraining.

NOTE on max_steps: beam at max_steps=60 (the training-eval setting) costs
~25s/equation on abel_level4 -- infeasible across 70 checkpoints. We use
max_steps=10, the known-good operating point. The value-beam vs plain-beam
delta and the best-vs-final-checkpoint delta are both fully valid at a
fixed max_steps; only the absolute number is not comparable to the legacy
max_steps=60 figure (~0.55).

The env is built once per worker process (it costs ~45s) and reused across
all checkpoints. Action set matches the legacy closed training
(use_cov=False); the env action dim is fixed at 50 so checkpoints load
cleanly. Each checkpoint is scored on a fixed equation subsample.
"""
import argparse
import re
import sys
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass

BASE = Path("data/dynamic_actions/use_relabel_constants/use_buffer/"
            "abel_level4_hidden_dim256_nenvs1/ppo-tree")
SUBSAMPLE_SEED = 12345

# Per-worker globals (built once by the pool initializer, reused per job).
_ENV = None
_EQNS = None
_MAX_STEPS = 10


def _init_worker(k, max_steps):
    global _ENV, _EQNS, _MAX_STEPS
    import warnings
    warnings.filterwarnings("ignore")
    from envs.env_multi_eqn import multiEqn
    _ENV = multiEqn(gen="abel_level4", state_rep="graph_integer_1d",
                    use_cov=False, use_relabel_constants=True,
                    use_success_replay=True)
    test = list(_ENV.test_eqns)
    if 0 < k < len(test):
        idx = sorted(random.Random(SUBSAMPLE_SEED).sample(range(len(test)), k))
        _EQNS = [test[i] for i in idx]
    else:
        _EQNS = test
    _MAX_STEPS = max_steps


def eval_one(job):
    """Worker: greedy + plain beam + value beam for one checkpoint."""
    seed, ckpt_path = job
    from sb3_contrib import MaskablePPO
    from train_abel import greedy_accuracy, beam_accuracy

    m = re.search(r"model_step0*(\d+)\.zip", str(ckpt_path))
    step = int(m.group(1)) if m else -1
    try:
        model = MaskablePPO.load(str(ckpt_path), env=_ENV, device="cpu")
    except Exception as e:
        return (seed, step, None, None, None, f"load failed: {e}")

    g = greedy_accuracy(model, _ENV, _EQNS, max_steps=_MAX_STEPS,
                        per_eqn_seconds=0.75) or 0.0
    pb = beam_accuracy(model, _ENV, _EQNS, beam_width=5, topk_per_node=5,
                       max_steps=_MAX_STEPS, per_eqn_seconds=0.75,
                       beam_lambda=0.0) or 0.0
    vb = beam_accuracy(model, _ENV, _EQNS, beam_width=5, topk_per_node=5,
                       max_steps=_MAX_STEPS, per_eqn_seconds=0.75,
                       beam_lambda=1.0) or 0.0
    return (seed, step, g, pb, vb, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="7006,7007,7014,7015")
    p.add_argument("--subsample", type=int, default=150)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--smoke", action="store_true",
                   help="1 checkpoint per seed, 15 eqns -- sanity check")
    args = p.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    k = 15 if args.smoke else args.subsample

    jobs = []
    for seed in seeds:
        ckpts = sorted((BASE / f"seed{seed}" / "checkpoints").glob("model_step*.zip"),
                       key=lambda c: int(re.search(r"(\d+)", c.name).group(1)))
        if args.smoke:
            ckpts = ckpts[:1]
        for c in ckpts:
            jobs.append((seed, str(c)))
        print(f"seed{seed}: {len(ckpts)} checkpoint(s)", flush=True)
    print(f"{len(jobs)} jobs, {k} eqns each, max_steps={args.max_steps}, "
          f"{args.workers} workers", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(k, args.max_steps)) as ex:
        for r in ex.map(eval_one, jobs):
            results.append(r)
            seed, step, g, pb, vb, err = r
            if err:
                print(f"  seed{seed} step={step:>9}  ERROR: {err}", flush=True)
            else:
                print(f"  seed{seed} step={step:>9}  greedy={g:.3f}  "
                      f"plain={pb:.3f}  value={vb:.3f}", flush=True)

    # Per-seed curves + summary
    summary = {}
    for seed in seeds:
        ok = sorted([r for r in results if r[0] == seed and r[5] is None],
                    key=lambda r: r[1])
        if not ok:
            continue
        if not args.smoke:
            out = BASE / f"seed{seed}" / "abel4_decoder_curve.csv"
            with open(out, "w") as f:
                f.write("step,greedy,plain_beam,value_beam\n")
                for _, step, g, pb, vb, _e in ok:
                    f.write(f"{step},{g},{pb},{vb}\n")
            print(f"wrote {out}", flush=True)
        summary[seed] = (max(r[3] for r in ok), max(r[4] for r in ok))

    if summary:
        print(f"\n=== abel_level4 decoder re-eval ({k}-eqn subsample, "
              f"max_steps={args.max_steps}) ===")
        print(f"{'seed':8s} {'best plain beam':>16s} {'best value beam':>16s}")
        for seed, (pb, vb) in summary.items():
            print(f"{seed:8s} {pb:16.3f} {vb:16.3f}")
        n = len(summary)
        print(f"{'mean':8s} {sum(v[0] for v in summary.values())/n:16.3f} "
              f"{sum(v[1] for v in summary.values())/n:16.3f}")
        print("\n(legacy abel_level4 test_beam ~0.55: plain beam, final "
              "checkpoint, max_steps=60 -- not directly comparable)")


if __name__ == "__main__":
    main()
