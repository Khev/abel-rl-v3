"""Quadratic CoV: complete the square."""
import sympy as sp
from .base import CoVClass, x, random_coeffs, random_int_scale


class Quadratic(CoVClass):
    name = "quadratic"
    step_budget = 3   # SUB b, DIV 2, DIV a  → cov = b/(2a) → f(x) = x - b/(2a)

    def sample_form(self, rng):
        A_sym, B_sym, C_sym = random_coeffs(rng, 3)
        sA = random_int_scale(rng)
        sB = random_int_scale(rng)
        sC = random_int_scale(rng)
        A = sA * A_sym
        B = sB * B_sym
        C = sC * C_sym
        variant = rng.integers(0, 4)
        if variant == 0:
            return A * x**2 + B * x + C
        elif variant == 1:
            # already half-completed form
            return A * x**2 + 2 * B * x + C
        elif variant == 2:
            # unit leading
            return x**2 + 2 * B * x + C
        else:
            # missing constant: a*x**2 + b*x
            return A * x**2 + B * x

    def intended_cov(self, eqn):
        poly = eqn.as_poly(x)
        if poly is None or poly.degree() != 2:
            return None
        coeffs = poly.all_coeffs()  # [A, B, C] for ax^2 + bx + c
        A, B = coeffs[0], coeffs[1]
        if A == 0:
            return None
        return x - B / (2 * A)
