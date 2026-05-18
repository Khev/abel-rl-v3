#!/usr/bin/env python3
"""Build the mixed (open + closed) equation dataset, paper-sized.

Stratified sampling: 200 train + 20 test from each of 5 sources, giving
the closed-equation paper's large split: 10000 train + 1000 test.

Sources:
  - cov_v2_large/{quadratic, cubic, quartic, exponential}: 200/20 each
  - abel_level3:                                            200/20

Total: 1000 train + 100 test.
"""
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = [
    "cov_v2_large/quadratic",
    "cov_v2_large/cubic",
    "cov_v2_large/quartic",
    "cov_v2_large/exponential",
    "abel_level3",
]
TRAIN_PER_SOURCE = 2000
TEST_PER_SOURCE = 200
SEED = 0


def read_eqns(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def main():
    rng = random.Random(SEED)
    train_all = []
    test_all = []
    base = ROOT.parent  # equation_templates/
    for src in SOURCES:
        d = base / src
        tr_pool = read_eqns(d / "train_eqns.txt")
        te_pool = read_eqns(d / "test_eqns.txt")
        tr = rng.sample(tr_pool, k=min(TRAIN_PER_SOURCE, len(tr_pool)))
        te = rng.sample(te_pool, k=min(TEST_PER_SOURCE, len(te_pool)))
        print(f"{src}: sampled train={len(tr)}/{len(tr_pool)}, "
              f"test={len(te)}/{len(te_pool)}")
        train_all.extend(tr)
        test_all.extend(te)
    rng.shuffle(train_all)
    rng.shuffle(test_all)
    print(f"\nTotal: train={len(train_all)}, test={len(test_all)}")

    out_train = ROOT / "train_eqns.txt"
    out_test = ROOT / "test_eqns.txt"
    with open(out_train, "w") as f:
        f.write(f"# {len(train_all)} train equations (mixed: open + closed, "
                f"{TRAIN_PER_SOURCE}/source x {len(SOURCES)} sources)\n")
        for e in train_all:
            f.write(e + "\n")
    with open(out_test, "w") as f:
        f.write(f"# {len(test_all)} test equations (mixed, "
                f"{TEST_PER_SOURCE}/source x {len(SOURCES)} sources)\n")
        for e in test_all:
            f.write(e + "\n")
    print(f"\nWrote: {out_train}, {out_test}")


if __name__ == "__main__":
    main()
