"""Quartic CoV: depress to pure x^4 form.

Restricted to the SPECIAL form where depression `x -> x - b/(4a)` kills
the x^3, x^2, AND linear terms, leaving `a*x^4 + const`.
Template: a*x^4 + b*x^3 + (3*b^2/(8*a))*x^2 + (b^3/(16*a^2))*x + e.

Scaling preserves the property because all lower-order coefficients are
derived from A,B by construction.
"""
import sympy as sp
from .base import CoVClass, x, random_coeffs, random_int_scale


class Quartic(CoVClass):
    name = "quartic"
    step_budget = 3   # SUB b, DIV 4, DIV a

    def sample_form(self, rng):
        A_sym, B_sym, E_sym = random_coeffs(rng, 3)
        sA = random_int_scale(rng)
        sB = random_int_scale(rng)
        sE = random_int_scale(rng)
        A = sA * A_sym
        B = sB * B_sym
        E = sE * E_sym
        variant = rng.integers(0, 2)
        if variant == 0:
            return (A * x**4
                    + B * x**3
                    + (3 * B**2 / (8 * A)) * x**2
                    + (B**3 / (16 * A**2)) * x
                    + E)
        else:
            # unit-leading: x^4 + 4*B*x^3 + 6*B^2*x^2 + 4*B^3*x + E
            # depression x -> x - B → leaves x^4 + (E - 3*B^4)
            return x**4 + 4*B*x**3 + 6*B**2*x**2 + 4*B**3*x + E

    def intended_cov(self, eqn):
        poly = eqn.as_poly(x)
        if poly is None or poly.degree() != 4:
            return None
        coeffs = poly.all_coeffs()
        A, B = coeffs[0], coeffs[1]
        if A == 0:
            return None
        return x - B / (4 * A)
