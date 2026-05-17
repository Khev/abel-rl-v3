"""Cubic CoV: depress to pure x^3 form.

Restricted to the SPECIAL form where depression `x -> x - b/(3a)` kills
both the x^2 AND the linear term, leaving `a*x^3 + const`.
Template: a*x^3 + b*x^2 + (b^2/(3*a))*x + d.

Coefficient scaling: we scale the symbolic leading coefficient A and the
quadratic-coefficient B by independent integer factors, then BUILD the linear
term as B^2/(3A) from those scaled values. This preserves the depression
property (since the linear coeff is derived from A,B by construction).
"""
import sympy as sp
from .base import CoVClass, x, random_coeffs, random_int_scale


class Cubic(CoVClass):
    name = "cubic"
    step_budget = 3   # SUB b, DIV 3, DIV a

    def sample_form(self, rng):
        A_sym, B_sym, D_sym = random_coeffs(rng, 3)
        sA = random_int_scale(rng)
        sB = random_int_scale(rng)
        sD = random_int_scale(rng)
        A = sA * A_sym
        B = sB * B_sym
        D = sD * D_sym
        variant = rng.integers(0, 2)
        if variant == 0:
            return A * x**3 + B * x**2 + (B**2 / (3 * A)) * x + D
        else:
            # unit leading variant: x^3 + 3*b*x^2 + (3*b^2)*x + d
            # depression: x -> x - b → leaves x^3 + (d - 2b^3)
            return x**3 + 3 * B * x**2 + 3 * B**2 * x + D

    def intended_cov(self, eqn):
        poly = eqn.as_poly(x)
        if poly is None or poly.degree() != 3:
            return None
        coeffs = poly.all_coeffs()
        A, B = coeffs[0], coeffs[1]
        if A == 0:
            return None
        return x - B / (3 * A)
