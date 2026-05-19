#!/usr/bin/env python3
"""Focused dataset: just cubic + exponential, to test the "policy didn't learn"
vs "action set can't represent" hypothesis from the per-class breakdown.

If the agent trained on this gets > 0 on cubic, that contradicts the action-
set-limit hypothesis. If exponential reaches > 0, the gap was training-data:
mixed_v2_easy just didn't expose the policy enough.
"""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = [
    ("cov_v2_small/cubic",        500, 50),
    ("cov_v2_small/exponential",  500, 50),
]
SEED = 0


def read_eqns(path):
    with open(path) as f:
        return [ln.strip() for ln in f
                if ln.strip() and not ln.startswith("#")]


def main():
    rng = random.Random(SEED)
    train_all, test_all = [], []
    base = ROOT.parent
    for src, n_tr, n_te in SPEC:
        d = base / src
        tr_pool = read_eqns(d / "train_eqns.txt")
        te_pool = read_eqns(d / "test_eqns.txt")
        tr = rng.sample(tr_pool, k=min(n_tr, len(tr_pool)))
        te = rng.sample(te_pool, k=min(n_te, len(te_pool)))
        print(f"{src}: train={len(tr)}/{len(tr_pool)}, "
              f"test={len(te)}/{len(te_pool)}")
        train_all.extend(tr)
        test_all.extend(te)
    rng.shuffle(train_all); rng.shuffle(test_all)
    print(f"\nTotal: train={len(train_all)}, test={len(test_all)}")

    for split, data in [("train", train_all), ("test", test_all)]:
        out = ROOT / f"{split}_eqns.txt"
        with open(out, "w") as f:
            f.write(f"# {len(data)} {split} equations (cubic+exponential only)\n")
            for e in data:
                f.write(e + "\n")
        print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
