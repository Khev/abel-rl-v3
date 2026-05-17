import sympy as sp
import random

# Symbols
a,b,c,d,e,x = sp.symbols('a b c d e x')

ALL_COEFFS = [a,b,c,d,e]

# Base templates (exactly as you specified)
TEMPLATES = [
    a*x**2 + b*x + c,                                                # quadratic
    x**2 + 2*b*x + c,                                                # unit-leading quadratic
    a*x**3 + b*x**2 + (b**2/(3*a))*x + d,                            # special cubic (no y-term after shift)
    x**3 + 3*b*x**2 + (b**2/(3*a))*x + d,                            # unit-leading cubic (note 'a' in lower term)
    a*x**4 + b*x**3 + (3*b**2/(8*a))*x**2 + (b**3/(16*a**2))*x + e,  # special quartic (biquadratic after shift)
    x**4 + 4*b*x**3 + (3*b**2/(8*a))*x**2 + (b**3/(16*a**2))*x + e,   # unit-leading quartic (a appears below)
    a*sp.exp(x) + b*sp.exp(-x) + c                                   # exponential
]



def permute_coeffs(expr, perm):
    """Apply a renaming (permutation) of {a,b,c,d,e} to expr."""
    sub_map = {OLD: NEW for OLD, NEW in zip(ALL_COEFFS, perm)}
    return sp.simplify(expr.subs(sub_map))

def scale_coeffs(expr, scales):
    """
    Multiply each symbolic coefficient by a small integer factor.
    e.g. a -> s_a * a, b -> s_b * b, etc. (no zeros!)
    """
    sub_map = {sym: scales.get(sym, 1)*sym for sym in ALL_COEFFS}
    return sp.simplify(expr.subs(sub_map))

def maybe_inject_k_for_exp(expr, k):
    """
    If the expression has exp(x) or exp(-x), turn them into exp(k*x) and exp(-k*x).
    Leaves other terms untouched.
    """
    return expr.xreplace({
        sp.exp(x): sp.exp(k*x),
        sp.exp(-x): sp.exp(-k*x),
    })

def sample_equations(n_total, seed=0,
                     scale_pool=(-3,-2,-1,1,2,3),
                     k_pool=(1,2,3,4,5),
                     p_monic=0.5,
                     max_attempts=300000):
    rng = random.Random(seed)
    eqns, seen = [], set()
    attempts = 0

    while len(eqns) < n_total and attempts < max_attempts:
        attempts += 1
        base = rng.choice(TEMPLATES)

        # 1) random permutation of {a,b,c,d,e}
        perm = ALL_COEFFS[:]
        rng.shuffle(perm)
        expr = permute_coeffs(base, perm)

        # 1.5) optionally make it monic: a -> 1
        monic_applied = (rng.random() < p_monic)
        if monic_applied:
            expr = sp.simplify(expr.subs({a: 1}))

        # 2) optional small integer scaling of remaining symbols
        scales = {sym: rng.choice(scale_pool) for sym in ALL_COEFFS}
        if monic_applied:
            # keep a=1 if we monicized
            scales[a] = 1
        expr = scale_coeffs(expr, scales)

        # 3) optional exp(kx) (when exponential template present)
        k = rng.choice(k_pool)
        expr = maybe_inject_k_for_exp(expr, k)

        # 4) dedup
        key = sp.srepr(sp.together(sp.simplify(expr)))
        if key not in seen:
            seen.add(key)
            eqns.append(sp.simplify(expr))

    if len(eqns) < n_total:
        print(f"[warn] Only generated {len(eqns)} unique equations after {attempts} attempts.")
    return eqns

def make_train_test(n_train=1000, n_test=100, seed=0):
    """Build train/test splits with dedup and no overlap."""
    total_needed = n_train + n_test
    eqns = sample_equations(total_needed, seed=seed)
    rng = random.Random(seed)
    rng.shuffle(eqns)
    train = eqns[:n_train]
    test  = eqns[n_train:n_train+n_test]
    return train, test

# Example usage:
if __name__ == "__main__":
    train, test = make_train_test(n_train=1000, n_test=100, seed=42)
    
    import os
    os.makedirs("gemini", exist_ok=True)
    
    with open("gemini/train_equations.txt", "w") as f:
        for eq in train:
            f.write(str(eq) + "\n")
    with open("gemini/test_equations.txt", "w") as f:
        for eq in test:
            f.write(str(eq) + "\n")
    print(f"Done. Wrote {len(train)} train and {len(test)} test equations to gemini/ directory.")
