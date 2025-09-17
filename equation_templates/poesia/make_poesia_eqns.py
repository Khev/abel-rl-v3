import requests
import random
import re
from sympy import sympify, symbols  # For validating/parsing equations

# Fetch templates from local file
TEMPLATE_PATH = "equation_templates/poesia/equations-ct.txt"

def fetch_templates():
    with open(TEMPLATE_PATH, 'r') as f:
        templates = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('!')
        ]
    print(f"Fetched {len(templates)} templates from local file.")
    return templates

# Map integers to constants (letters): 1='a', 2='b', ..., 26='z', -1='-a', etc.
# For >26, use constN (though here we only generate -10..10)
def int_to_constant(n):
    if n == 0:
        return 'k'  # 0 → k (instead of "0")
    sign = '-' if n < 0 else ''
    abs_n = abs(n)
    if abs_n <= 26:
        letter = chr(ord('a') + abs_n - 1)
    else:
        letter = f'const{abs_n}'
    return sign + letter

# Replace numbers in a template with random ints (-10 to 10), map to constants,
# then clean up for sympify: convert "lhs = rhs" → "lhs - (rhs)" and "ax" → "a*x".
def generate_equation(template, seed):
    random.seed(seed)
    # Find all integer literals in template (e.g., -1, 2, 3)
    numbers = re.findall(r'-?\d+', template)
    eq = template
    for num_str in set(numbers):
        rand_int = random.randint(-10, 10)
        const = int_to_constant(rand_int)
        eq = eq.replace(num_str, const)

    # 1) turn “lhs = rhs” into “lhs - (rhs)”
    if '=' in eq:
        lhs_str, rhs_str = eq.split('=', 1)
        eq = f"{lhs_str.strip()} - ({rhs_str.strip()})"

    # 2) insert “*” between letter coefficient and x (e.g., “ax” → “a*x”)
    eq = re.sub(r'([A-Za-z])x\b', r'\1*x', eq)

    # Validate with sympy (optional, to ensure valid eq)
    try:
        sympify(eq)
    except Exception as e:
        print(f"Warning: Invalid eq {eq}: {e}")

    return eq

def generate_sets(
    num_test=10,
    test_seed_start=0,
    train_sample_size=100,
    train_seed_start=1000000
):
    templates = fetch_templates()

    test_set = []
    for seed in range(test_seed_start, test_seed_start + num_test):
        idx = seed % len(templates)
        template = templates[idx]
        eq = generate_equation(template, seed)
        test_set.append(eq)

    train_set = []
    for i in range(train_sample_size):
        seed = train_seed_start + i
        idx = seed % len(templates)
        template = templates[idx]
        eq = generate_equation(template, seed)
        train_set.append(eq)

    return test_set, train_set

if __name__ == "__main__":
    test, train_sample = generate_sets()

    print("Sample Test Set (first 5):", test[:5])
    print("Sample Train Set (first 5):", train_sample[:5])

    # Save to files
    with open('poesia_equations_test.txt', 'w') as f:
        f.write('\n'.join(test))
    with open('poesia_equations_train_sample.txt', 'w') as f:
        f.write('\n'.join(train_sample))
