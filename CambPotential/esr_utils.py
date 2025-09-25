import sympy
import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

def load_esr_function_string(file_path, idx=0):
    try:
        with open(file_path, "r") as f:
            all_functions = [line.strip() for line in f.readlines() if line.strip()]
            func_string = all_functions[idx]
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

            print(f"Loaded ESR function: {func_string} with parameters: {[str(p) for p in param_symbols]}")

            return func_string, expr_template, param_symbols

    except FileNotFoundError:
        print(f"Could not find file with generated equations: {file_path}")
        exit()

def create_callable_function(expr_template, param_symbols, x_vals):
    """Create a callable function for parameter evaluation"""

    def objective(params):
        # Substitute parameters into expression
        substitutions = {param: params[i] for i, param in enumerate(param_symbols)}
        expr_with_params = expr_template.subs(substitutions)
        
        # Convert to callable and evaluate
        callable_func = sympy.lambdify([sympy.Symbol('x')], expr_with_params, modules=['numpy'])
        y_pred = callable_func(x_vals)
        # print(f"Evaluated function with params {params}: {y_pred} at x={x_vals}")
        return y_pred
    return objective

def create_potential_table(expr_template, param_symbols, param_vals, phi_vals):
    """Create the potential table from the ESR /sympy expression and parameter values"""

    phi_padding = 1e-2 
    padded_phi_vals = np.concatenate((
        np.array([phi_vals[0] - phi_padding]), 
        phi_vals, 
        np.array([phi_vals[-1] + phi_padding])
    ))
    function = create_callable_function(expr_template, param_symbols, padded_phi_vals)
    log_V_vals = function(param_vals)
    # print(f"Evaluated log_V_vals: {log_V_vals}")
    V_vals = np.exp(log_V_vals)  # Ensure V(phi) > 0
    # Check for invalid values
    invalid_mask = np.logical_or(np.isinf(V_vals), np.isnan(V_vals))
    if np.any(invalid_mask):
        success = False
        return {'success': success, 'phi_train': None, 'V_train': None, 'dV_train': None, 'ddV_train': None}
    else:
        success = True
        # print(f"Shapes of phi and V: {padded_phi_vals.shape}, {V_vals.shape}")
        logV_interpolator = InterpolatedUnivariateSpline(padded_phi_vals, log_V_vals)
        dlogV_dphi = logV_interpolator.derivative(n=1)(phi_vals)
        dV_dphi = dlogV_dphi * V_vals[1:-1]
        ddV_dphi = V_vals[1:-1] * (logV_interpolator.derivative(n=2)(phi_vals) + dlogV_dphi**2)
        return {'success': success, 'phi_train': phi_vals, 'V_train': V_vals[1:-1], 'dV_train': dV_dphi, 'ddV_train': ddV_dphi}
