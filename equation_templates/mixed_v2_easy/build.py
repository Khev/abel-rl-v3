#!/usr/bin/env python3
"""Build a bootstrap-friendly mixed dataset.

Like mixed_v2_small but replaces some of the depth-3 closed equations
with depth-1 and depth-2 ones (abel_level1/2), so the agent has
trivial-to-easy targets to bootstrap success-replay.

Composition (1000 train / 100 test):
  - cov_v2_small/quadratic    150 train / 15 test  (open)
  - cov_v2_small/cubic        150 train / 15 test  (open)
  - cov_v2_small/quartic      150 train / 15 test  (open)
  - cov_v2_small/exponential  150 train / 15 test  (open)
  - abel_level1               100 train / 10 test  (trivial closed)
  - abel_level2               100 train / 10 test  (easy closed)
  - abel_level3               200 train / 20 test  (varied closed)
"""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = [
    ("cov_v2_small/quadratic",   150, 15),
    ("cov_v2_small/cubic",       150, 15),
    ("cov_v2_small/quartic",     150, 15),
    ("cov_v2_small/exponential", 150, 15),
    ("abel_level1",              100, 10),
    ("abel_level2",              100, 10),
    ("abel_level3",              200, 20),
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
            f.write(f"# {len(data)} {split} equations (mixed_v2_easy: "
                    f"open + abel_level{{1,2,3}})\n")
            for e in data:
                f.write(e + "\n")
        print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
