# `make_eqns.py` v2 — Per-class Generator Sketch

## Design goals

1. **One module per equation class.** Adding a class = drop a file.
2. **Strict no-leakage train/test splits**, with canonical-form dedup.
3. **Per-instance validation:** every emitted equation is SymPy-solvable AND solvable via the intended CoV within a step budget.
4. **Symbol diversity:** coefficients drawn from `{a, b, c, d, e, f, g}` with non-trivial permutations, so the agent can't memorize symbol positions.
5. **Deterministic given a seed.** Re-running yields the same dataset.

## Scope decision (2026-05-17)

**Cubics and quartics are restricted to "fully-depressable" special forms.** This is a hard constraint of the current formulation, not just a scope preference: special forms like `a*x**3 + b*x**2 + (b²/3a)*x + d` depress to a *pure* `x**n + const`, which the closed-equation agent can then solve. Generic cubics depress to `x³ + p·x + q`, requiring Cardano-style reasoning that the current closed-equation action set cannot perform. So the pipeline is:

```
open equation  → CoV agent →  depressed pure form  → closed-equation agent → solved
```

If we ever want generic cubics/quartics, the closed-equation agent would need to gain Cardano-like macroactions, which is a separate research thread.

## Directory layout

```
eqn_gen/
  __init__.py
  base.py              # CoVClass ABC + shared helpers
  quadratic.py
  cubic.py
  quartic.py
  exponential.py
  reciprocal_sym.py
  radical.py
  trig.py              # stretch
  logarithmic.py       # stretch
  registry.py          # name → class lookup
make_eqns.py           # CLI driver
```

## Base interface

```python
# eqn_gen/base.py
from abc import ABC, abstractmethod
import sympy as sp
import numpy as np
from typing import Optional

x = sp.symbols('x')
SYMBOL_POOL = list(sp.symbols('a b c d e f g'))


class CoVClass(ABC):
    """One change-of-variables equation class."""

    name: str                    # e.g. "quadratic"
    n_coeffs: int                # how many free symbols the form needs
    step_budget: int = 10        # solver step budget after applying CoV

    @abstractmethod
    def sample_form(self, rng: np.random.Generator) -> sp.Expr:
        """Return one random symbolic equation (in `x` and free coeffs)."""

    @abstractmethod
    def intended_cov(self, eqn: sp.Expr) -> Optional[sp.Expr]:
        """The substitution sub-expression `f(x_new)` that solves this class.
        Return None if the instance doesn't match the class pattern."""

    # --- defaults (override only when needed) ---

    def canonical_form(self, eqn: sp.Expr) -> str:
        """Relabel free symbols (excluding x) in canonical order, return str.
        Used for dedup."""
        free = sorted([s for s in eqn.free_symbols if s != x],
                      key=lambda s: s.name)
        mapping = {old: SYMBOL_POOL[i] for i, old in enumerate(free)}
        return str(sp.simplify(eqn.xreplace(mapping)))

    def is_valid(self, eqn: sp.Expr) -> bool:
        """All checks: SymPy-solvable, CoV-solvable, non-degenerate."""
        try:
            cov = self.intended_cov(eqn)
            if cov is None:
                return False
            transformed = sp.simplify(eqn.subs(x, cov))
            # post-CoV must be solvable by the closed-equation env
            # (cheap proxy: SymPy can solve it for x)
            sols = sp.solve(transformed, x)
            if not sols:
                return False
            # non-degenerate: equation actually depends on x
            if x not in eqn.free_symbols:
                return False
            return True
        except Exception:
            return False


def _random_coeffs(rng, k, pool=SYMBOL_POOL):
    """Pick k distinct symbols from the pool, in a random order."""
    idx = rng.choice(len(pool), size=k, replace=False)
    return [pool[i] for i in idx]
```

## Example class implementations

```python
# eqn_gen/quadratic.py
import sympy as sp
from .base import CoVClass, x, _random_coeffs


class Quadratic(CoVClass):
    name = "quadratic"
    n_coeffs = 3   # leading + linear + constant

    def sample_form(self, rng):
        A, B, C = _random_coeffs(rng, 3)
        # vary structural form so it's not always A*x**2 + B*x + C
        # e.g. drop B*x (pure square shifted), or drop C, etc.
        variant = rng.integers(0, 4)
        if variant == 0:
            return A*x**2 + B*x + C
        elif variant == 1:
            return A*x**2 + B*x        # C = 0
        elif variant == 2:
            return x**2 + B*x + C      # A = 1
        else:
            return A*(x**2) + 2*B*x + C  # already half-completed

    def intended_cov(self, eqn):
        poly = eqn.as_poly(x)
        if poly is None or poly.degree() != 2:
            return None
        A, B, _ = poly.all_coeffs()
        if A == 0:
            return None
        return x - B / (2*A)
```

```python
# eqn_gen/cubic.py
class Cubic(CoVClass):
    name = "cubic"
    n_coeffs = 4

    def sample_form(self, rng):
        A, B, C, D = _random_coeffs(rng, 4)
        # variants: full, missing-linear, etc.
        variant = rng.integers(0, 3)
        if variant == 0:
            return A*x**3 + B*x**2 + C*x + D
        elif variant == 1:
            return A*x**3 + B*x**2 + D       # C = 0
        else:
            return x**3 + B*x**2 + C*x + D   # A = 1

    def intended_cov(self, eqn):
        poly = eqn.as_poly(x)
        if poly is None or poly.degree() != 3:
            return None
        A, B = poly.all_coeffs()[0], poly.all_coeffs()[1]
        if A == 0:
            return None
        return x - B / (3*A)
```

```python
# eqn_gen/exponential.py
class Exponential(CoVClass):
    name = "exponential"
    n_coeffs = 3   # a, b, c (plus integer k)

    def sample_form(self, rng):
        A, B, C = _random_coeffs(rng, 3)
        k = int(rng.choice([1, 2, 3, 5]))
        return A * sp.exp(k*x) + B * sp.exp(-k*x) + C

    def intended_cov(self, eqn):
        # use Wild matching like in pi_cov_general
        aW, bW, cW, kW = sp.symbols('aW bW cW kW', cls=sp.Wild, exclude=[x])
        m = sp.simplify(eqn).match(aW*sp.exp(kW*x) + bW*sp.exp(-kW*x) + cW)
        if m and kW in m and m[kW] != 0:
            return sp.log(x) / m[kW]
        return None
```

```python
# eqn_gen/reciprocal_sym.py
class ReciprocalSymmetric(CoVClass):
    """Palindromic polynomials: a*x^4 + b*x^3 + c*x^2 + b*x + a.
    CoV: y = x + 1/x reduces to quadratic in y."""
    name = "reciprocal_sym"
    n_coeffs = 3

    def sample_form(self, rng):
        A, B, C = _random_coeffs(rng, 3)
        return A*x**4 + B*x**3 + C*x**2 + B*x + A

    def intended_cov(self, eqn):
        # the substitution lives in env logic (x + 1/x), this returns marker
        return x + 1/x   # env interprets as multiplicative shift
```

```python
# eqn_gen/radical.py
class Radical(CoVClass):
    """Equations in sqrt(x): a*sqrt(x) + b*x + c.
    CoV: y = sqrt(x), i.e. x := y**2."""
    name = "radical"
    n_coeffs = 3

    def sample_form(self, rng):
        A, B, C = _random_coeffs(rng, 3)
        return A * sp.sqrt(x) + B * x + C

    def intended_cov(self, eqn):
        if not eqn.has(sp.sqrt(x)):
            return None
        return x**2
```

## Driver

```python
# make_eqns.py
import argparse, os
import numpy as np
from eqn_gen.registry import REGISTRY


def generate(cls, n: int, rng: np.random.Generator, seen: set):
    """Generate n distinct, valid equations not already in `seen`."""
    out = []
    attempts = 0
    max_attempts = n * 50   # generous safety budget
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
        print(f"[{cls.name}] WARN: only generated {len(out)}/{n} after "
              f"{attempts} attempts")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classes", nargs="+", default=list(REGISTRY))
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_root", default="equation_templates/cov_v2")
    args = p.parse_args()

    for name in args.classes:
        cls = REGISTRY[name]()
        seen = set()
        rng_train = np.random.default_rng(args.seed)
        rng_test  = np.random.default_rng(args.seed + 10_000)
        train = generate(cls, args.n_train, rng_train, seen)
        test  = generate(cls, args.n_test,  rng_test,  seen)

        out_dir = os.path.join(args.out_root, name)
        os.makedirs(out_dir, exist_ok=True)
        for split, data in [("train", train), ("test", test)]:
            with open(os.path.join(out_dir, f"{split}.txt"), "w") as f:
                f.write(f"# {len(data)} {split} equations for {name}\n")
                for eqn in data:
                    f.write(f"{eqn}\n")
        print(f"[{name}] train={len(train)}, test={len(test)} -> {out_dir}")


if __name__ == "__main__":
    main()
```

## Open questions / design choices

1. **Coefficient values: symbolic only, or also concrete integers?** Existing templates use pure symbols (`a*x**2 + b*x + c`). Sticking with symbolic keeps the action set tight. If we want robustness to concrete coefficients, generate two variants per class.

2. **How much within-class variant diversity?** Too little = agent memorizes a pattern; too much = each class is really N classes. Suggested: 2–4 structural variants per class (full form, leading-coeff-=1, missing middle term, partially-completed form).

3. **Canonical form for dedup.** Simple symbol-renaming works for most classes but might miss equivalences like `2*b*x + c + x**2` ≡ `x**2 + 2*b*x + c` (just ordering). `sp.simplify` after renaming should handle that, but worth spot-checking.

4. **What counts as "trivially solvable"?** Probably: solvable by the closed-equation env (no-CoV PPO) in ≤3 steps. We can filter those out in a second pass once we have the trained no-CoV baseline.

5. **Per-class step budget.** Quadratics need 1 CoV step; quartics maybe 2 (depress + reciprocal-symmetric). Worth making per-class.
