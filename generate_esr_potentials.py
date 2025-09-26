import numpy as np
import sympy
import matplotlib.pyplot as plt
import esr.generation.duplicate_checker as duplicate_checker
import os
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# # Set random seed for reproducible comparison
# np.random.seed(42)
# random.seed(42)

# 1. Set up parameters for function generation
complexity = 8
# Use a predefined runname from the ESR library that includes sine functions
runname = "core_maths"

print(f"Generating functions with complexity {complexity} for run '{runname}'...")

# 2. Generate and process equations using the main entry point from the library

dirname = './CambESRPotential/esrfunctions'
try:
    duplicate_checker.main(runname, complexity, dirname=dirname)
except SystemExit:
    # The duplicate_checker.main function calls quit(), so we catch the SystemExit
    pass

print("\nFunction generation and processing complete.")



library_dir = dirname
eq_filename = os.path.join(library_dir, runname, f"compl_{complexity}", f"unique_equations_{complexity}.txt")
try:
    with open(eq_filename, "r") as f:
        all_functions = [line.strip() for line in f.readlines() if line.strip()]
except FileNotFoundError:
    print(f"Could not find file with generated equations: {eq_filename}")
    exit()

if not all_functions:
    print("No functions were generated. Exiting.")
    exit()

print(f"Found {len(all_functions)} unique functions.")


def create_objective_function(expr_template, param_symbols, x_vals):
    """Create an objective function for parameter optimization"""
    def objective(params):
        # Substitute parameters into expression
        substitutions = {param: params[i] for i, param in enumerate(param_symbols)}
        expr_with_params = expr_template.subs(substitutions)
        
        # Convert to callable and evaluate
        callable_func = sympy.lambdify([sympy.Symbol('x')], expr_with_params, modules=['numpy'])
        y_pred = callable_func(x_vals)
            
        # Check for invalid values
        invalid_mask = np.logical_or(np.isinf(y_pred), np.isneginf(y_pred))

        y_pred = np.where(invalid_mask, 1e10, y_pred)
                
        return y_pred
    return objective

nfuncs_to_test = min(10, len(all_functions))
test_func_idxs = np.random.choice(len(all_functions), size=nfuncs_to_test, replace=False)


x_eval = np.linspace(-1, 1, 100)

for idx in test_func_idxs:
    func_string = all_functions[idx]
    print(f"\nTesting function {idx+1}/{len(all_functions)}: {func_string}")
    # 5.1. Convert the string to a sympy expression
    x = sympy.symbols('x', real=True)
    # Create parameter symbols a0, a1, a2, etc.
    a_symbols = sympy.symbols([f'a{i}' for i in range(10)], real=True)
    
    # Define locals for sympy to understand the function string
    locs = {'x': x, 'sin': sympy.sin, 'cos': sympy.cos, 'inv': lambda x: 1/x, 
            'Abs': sympy.Abs, 'pow': sympy.Pow, 'exp': sympy.exp, 'log': sympy.log}
    
    # Add parameter symbols to locs
    for j, a_sym in enumerate(a_symbols):
        locs[f'a{j}'] = a_sym

    # Parse the function string
    expr_template = sympy.sympify(func_string, locals=locs)

    # Check if the expression has any parameter symbols
    param_symbols = [sym for sym in expr_template.free_symbols if str(sym).startswith('a')]

    param_vals = np.random.uniform(-1, 1, size=len(param_symbols))

    objective = create_objective_function(expr_template, param_symbols, x_eval)

    y_pred = objective(param_vals)


    param_vals_dict = {str(param_symbols[i]): f"{param_vals[i]:.4f}" for i in range(len(param_symbols))}

    plt.figure(figsize=(8, 5))
    plt.plot(x_eval, y_pred, label='Generated Function')
    plt.title(f"Function {idx+1}: {func_string}, params: {param_vals_dict}")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()


