import datetime
from sympy import symbols, sympify, simplify, expand, Poly, count_ops

x =  symbols('x')

# ==========================
# Utilities
# ==========================
def timed_print(msg: str) -> None:
    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}: {msg}")

# Define helper functions
def custom_identity(expr, term):
    """Return the expression unchanged."""
    return expr

def subs_x_with_f(lhs_rhs, f, xsym=x):
    """Substitute x -> f(x) into (lhs, rhs) or a single expr."""
    if isinstance(lhs_rhs, tuple):
        L, R = lhs_rhs
        L = sympify(L)
        R = sympify(R)
        return (simplify(L.subs(xsym, f)), simplify(R.subs(xsym, f)))
    return simplify(sympify(lhs_rhs).subs(xsym, f))

def C(expr_or_pair):
    """Calculate complexity of an expression or pair."""
    if isinstance(expr_or_pair, tuple):
        L, R = expr_or_pair
        expr = expand(L - R)
    else:
        expr = expand(expr_or_pair)
    try:
        P = Poly(expr, x)
    except sp.PolynomialError:
        return int(count_ops(expr))
    return len(P.terms())
