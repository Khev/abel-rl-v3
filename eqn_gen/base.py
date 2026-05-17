"""Base interface for CoV equation-class generators."""
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import sympy as sp
import numpy as np


x = sp.symbols("x")
SYMBOL_POOL = list(sp.symbols("a b c d e f g"))


class CoVClass(ABC):
    """One change-of-variables equation class.

    Subclass contract:
      - `name`: short string identifier
      - `sample_form(rng) -> Expr`: random equation in `x` + free symbol coeffs
      - `intended_cov(eqn) -> Expr | None`: substitution `f(x_new)` such that
        `eqn.subs(x, f(x_new))` simplifies to a closed-equation-solvable form
        (target: pure `x**n + const` for polynomial classes).

    Validity = the intended CoV exists AND the post-CoV equation collapses to
    a "pure form" (≤2 polynomial terms) so the closed-equation agent can finish.
    """

    name: str
    step_budget: int = 3  # actions for CoV agent to build f(x); per-class override

    @abstractmethod
    def sample_form(self, rng: np.random.Generator) -> sp.Expr:
        ...

    @abstractmethod
    def intended_cov(self, eqn: sp.Expr) -> Optional[sp.Expr]:
        ...

    # ----- shared utilities (override if needed) -----

    def canonical_form(self, eqn: sp.Expr) -> str:
        """Relabel free symbols (excluding x) in alphabetical order. For dedup."""
        free = sorted(
            [s for s in eqn.free_symbols if s != x and s.is_Symbol],
            key=lambda s: s.name,
        )
        mapping = {old: SYMBOL_POOL[i] for i, old in enumerate(free)
                   if i < len(SYMBOL_POOL)}
        return str(sp.simplify(eqn.xreplace(mapping)))

    def is_post_cov_pure(self, eqn_after: sp.Expr, max_terms: int = 2) -> bool:
        """Post-CoV equation should be 'pure' for the closed-agent to solve.

        Default test: as polynomial in x, has at most `max_terms` non-zero terms
        and uses no transcendentals on x. Override for non-polynomial classes.
        """
        try:
            poly = eqn_after.as_poly(x)
            if poly is None:
                return False
            return len([c for c in poly.all_coeffs() if c != 0]) <= max_terms
        except Exception:
            return False

    def is_valid(self, eqn: sp.Expr) -> bool:
        """Cheap validity: intended CoV exists AND post-CoV is 'pure'.

        We do NOT call sp.solve() — for our special forms, pure means
        `a*x^n + const` which is trivially solvable by `(-const/a)^(1/n)`.
        sp.solve on quartics/cubics is far too slow.
        """
        if x not in eqn.free_symbols:
            return False
        try:
            cov = self.intended_cov(eqn)
            if cov is None:
                return False
            after = sp.simplify(eqn.subs(x, cov))
            return self.is_post_cov_pure(after)
        except Exception:
            return False


def random_coeffs(rng: np.random.Generator, k: int,
                  pool=SYMBOL_POOL) -> Tuple[sp.Symbol, ...]:
    """Pick k distinct symbols from the pool in a random order."""
    idx = rng.choice(len(pool), size=k, replace=False)
    return tuple(pool[i] for i in idx)


def random_int_scale(rng: np.random.Generator,
                     pool=(-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)) -> int:
    return int(rng.choice(pool))
