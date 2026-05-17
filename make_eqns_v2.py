#!/usr/bin/env python3
"""V2 dataset generator: per-class train/test splits with validation + dedup.

Usage:
    python make_eqns_v2.py                                  # all classes, defaults
    python make_eqns_v2.py --classes quadratic cubic        # subset
    python make_eqns_v2.py --n_train 1000 --n_test 200      # custom sizes

Writes to: equation_templates/cov_v2/{class}/{train,test}_eqns.txt
"""
import argparse
import os
import sys
import numpy as np

from eqn_gen.registry import REGISTRY


def generate(cls, n, rng, seen, max_attempts_per_eq=50):
    """Generate up to n distinct, valid equations not already in `seen`."""
    out = []
    max_attempts = n * max_attempts_per_eq
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        eqn = cls.sample_form(rng)
        if not cls.is_valid(eqn):
            continue
        canon = cls.canonical_form(eqn)
        if canon in seen:
            continue
        seen.add(canon)
        out.append(eqn)
    if len(out) < n:
        print(f"[{cls.name}] WARN: produced {len(out)}/{n} after {attempts} attempts",
              file=sys.stderr)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classes", nargs="+", default=list(REGISTRY),
                   choices=list(REGISTRY))
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_root", default="equation_templates/cov_v2")
    args = p.parse_args()

    for name in args.classes:
        cls = REGISTRY[name]()
        seen = set()
        rng_train = np.random.default_rng(args.seed)
        rng_test = np.random.default_rng(args.seed + 10_000)

        train = generate(cls, args.n_train, rng_train, seen)
        # train canonical forms now in `seen`, so test won't collide with train
        test = generate(cls, args.n_test, rng_test, seen)

        out_dir = os.path.join(args.out_root, name)
        os.makedirs(out_dir, exist_ok=True)
        for split, data in [("train", train), ("test", test)]:
            path = os.path.join(out_dir, f"{split}_eqns.txt")
            with open(path, "w") as f:
                f.write(f"# {len(data)} {split} equations for class={name}\n")
                for eqn in data:
                    f.write(f"{eqn}\n")
        print(f"[{name}] train={len(train)}/{args.n_train}, "
              f"test={len(test)}/{args.n_test}  ->  {out_dir}")


if __name__ == "__main__":
    main()
