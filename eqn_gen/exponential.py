"""Exponential CoV: a*exp(k*x) + b*exp(-k*x) + c, sub `x -> log(x)/k`."""
import sympy as sp
from .base import CoVClass, x, random_coeffs, random_int_scale


class Exponential(CoVClass):
    name = "exponential"
    step_budget = 4   # log(x) then DIV k  (or via op composition)

    def sample_form(self, rng):
        A_sym, B_sym, C_sym = random_coeffs(rng, 3)
        sA = random_int_scale(rng)
        sB = random_int_scale(rng)
        sC = random_int_scale(rng)
        k = int(rng.choice([1, 2, 3, 5]))
        return (sA * A_sym) * sp.exp(k * x) + (sB * B_sym) * sp.exp(-k * x) + (sC * C_sym)

    def intended_cov(self, eqn):
        aW, bW, cW, kW = sp.symbols("aW bW cW kW", cls=sp.Wild, exclude=[x])
        m = sp.simplify(eqn).match(aW * sp.exp(kW * x) + bW * sp.exp(-kW * x) + cW)
        if m and kW in m and m[kW] != 0:
            return sp.log(x) / m[kW]
        return None

    def is_post_cov_pure(self, eqn_after, max_terms=3):
        """Post-CoV: a*x + b/x + c, a rational equation. We allow 3 'terms' here."""
        try:
            # multiply through by x to clear the 1/x: a*x^2 + c*x + b  (quadratic in x)
            simplified = sp.together(eqn_after)
            num, _ = simplified.as_numer_denom()
            poly = sp.expand(num).as_poly(x)
            if poly is None:
                return False
            return len([c for c in poly.all_coeffs() if c != 0]) <= max_terms
        except Exception:
            return False
