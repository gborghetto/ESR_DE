import sympy
import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

def is_function_valid(func_string: str) -> bool:
    """
    More robustly analyzes a function string to determine if it's valid.
    """
    # 1. Check for obviously invalid substrings
    invalid_substrings = ['nan', 'oo', '<class', 'i'] # 'I' is sympy for imaginary unit
    if any(sub in func_string.lower() for sub in invalid_substrings):
        print(f"Validation failed: Function '{func_string}' contains invalid substring.")
        return False

    try:
        # 2. Parse and check for dependency on 'x'
        x = sympy.symbols('x')
        # We don't need the 'a' symbols for this check
        locs = {'x': x, 'sin': sympy.sin, 'cos': sympy.cos, 'inv': lambda v: 1/v,
                'Abs': sympy.Abs, 'pow': sympy.Pow, 'exp': sympy.exp, 'log': sympy.log}
        expr = sympy.sympify(func_string, locals=locs)

        # 3. If 'x' is not a free symbol, the function is constant and thus invalid
        if x not in expr.free_symbols:
            print(f"Validation failed: Function '{func_string}' is constant with respect to x.")
            return False
            
    except Exception as e:
        print(f"Validation failed: Could not parse function '{func_string}'. Error: {e}")
        return False
        
    return True

def identify_fixed_and_variable_parameters(expr_template, param_symbols):
    """
    Analyzes a sympy expression to separate its parameters into 'fixed' 
    (purely additive constants) and 'variable' (all others).

    A parameter 'a_i' is classified as 'fixed' if and only if the partial 
    derivative of the expression with respect to 'a_i' is exactly 1.

    If any part of the symbolic analysis fails, it safely defaults to 
    classifying ALL parameters as variable.

    Args:
        expr_template (sympy.Expr): The symbolic expression for the potential.
        param_symbols (list): A list of the sympy symbols for the parameters (a0, a1...).

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
                                     1. The names of parameters to be fixed.
                                     2. The names of parameters to be treated as variable.
    """
    # Convert all symbols to a set of strings for easy processing
    all_param_names = {str(p) for p in param_symbols}
    
    try:
        params_to_fix = set()
        # Loop to identify the fixed parameters
        for param in param_symbols:
            derivative = sympy.diff(expr_template, param)
            # The parameter is a purely additive constant ONLY if the derivative is 1
            if derivative == 1:
                params_to_fix.add(str(param))
        
        # The variable parameters are all parameters MINUS the fixed ones
        params_to_sample = all_param_names.difference(params_to_fix)

        #convert to lists for return
        params_to_fix = list(params_to_fix)
        params_to_sample = list(params_to_sample)

        print(f"Identified fixed parameters: {params_to_fix}, variable parameters: {params_to_sample}")
        
        # Return sorted lists for a deterministic order
        return sorted(list(params_to_fix)), sorted(list(params_to_sample))

    except Exception as e:
        # If ANY part of the symbolic analysis fails, default to sampling everything
        print(f"Warning: Symbolic analysis failed for expression '{expr_template}'. Error: {e}")
        print("Defaulting to sampling all parameters for this function.")
        
        # Return an empty list for fixed params, and all params as variable
        return [], sorted(list(all_param_names))

def load_esr_function_string(file_path, idx=0,verbose=False)-> dict:
    try:
        with open(file_path, "r") as f:
            all_functions = [line.strip() for line in f.readlines() if line.strip()]
            func_string = all_functions[idx]

            is_valid = is_function_valid(func_string)

            function_dict = {}

            if not is_valid:
                function_dict['valid'] = False
                return function_dict


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

            fixed_params, variable_params = identify_fixed_and_variable_parameters(expr_template, param_symbols)

            if verbose:
                print(f"Loaded ESR function: {func_string} with parameters: {[str(p) for p in param_symbols]}")

            function_dict['valid'] = True
            function_dict['func_string'] = func_string
            function_dict['expr_template'] = expr_template
            function_dict['param_symbols'] = param_symbols
            function_dict['fixed_params'] = fixed_params
            function_dict['variable_params'] = variable_params

            return function_dict

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


import sympy
import numpy as np
# InterpolatedUnivariateSpline is no longer needed for derivatives

# def create_potential_table(expr_template, param_symbols, param_vals, phi_vals):
#     """
#     Create the potential table using symbolic differentiation from sympy for accuracy.
#     """
#     x = sympy.symbols('x')
    
#     # 1. Substitute the numerical parameter values into the symbolic expression
#     substitutions = {param: val for param, val in zip(param_symbols, param_vals)}
#     expr_with_params = expr_template.subs(substitutions)

#     # 2. Use sympy to calculate the derivatives symbolically
#     try:
#         V_expr = sympy.exp(expr_with_params) # The potential V is exp(logV)
#         Vp_expr = sympy.diff(V_expr, x)       # V' = dV/dphi
#         Vpp_expr = sympy.diff(Vp_expr, x)     # V'' = d^2V/dphi^2
#     except Exception as e:
#         print(f"ERROR: Symbolic differentiation failed: {e}")
#         return {'success': False, 'phi_train': None, 'V_train': None, 'dV_train': None, 'ddV_train': None}

#     # 3. Convert the symbolic expressions into fast, callable numpy functions
#     #    We use 'numpy' as the module for lambdify to handle array inputs.
#     V_func = sympy.lambdify(x, V_expr, modules='numpy')
#     Vp_func = sympy.lambdify(x, Vp_expr, modules='numpy')
#     Vpp_func = sympy.lambdify(x, Vpp_expr, modules='numpy')
    
#     # 4. Evaluate the functions on the dense phi grid to create the table
#     try:
#         with np.errstate(all='raise'): # Temporarily treat numpy warnings as errors
#             V_train = V_func(phi_vals)
#             dV_train = Vp_func(phi_vals)
#             ddV_train = Vpp_func(phi_vals)
#     except FloatingPointError as e:
#         print(f"ERROR: Numerical evaluation failed (e.g., division by zero) for expression '{expr_template}' with params {param_vals}. Error: {e}")
#         return {'success': False, 'phi_train': None, 'V_train': None, 'dV_train': None, 'ddV_train': None}

#     # 5. Check for any remaining NaN or Inf values
#     if np.any(np.isnan(V_train)) or np.any(np.isinf(V_train)):
#         return {'success': False, 'phi_train': None, 'V_train': None, 'dV_train': None, 'ddV_train': None}

#     return {'success': True, 
#             'phi_train': phi_vals, 
#             'V_train': V_train, 
#             'dV_train': dV_train, 
#             'ddV_train': ddV_train}

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
