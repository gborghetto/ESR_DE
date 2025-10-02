# Test for valid function strings and find additive constant parameters

import sympy
from CambESRPotential.esr_utils import load_esr_function_string

def is_function_valid(func_string: str) -> bool:
    """
    More robustly analyzes a function string to determine if it's valid.
    """
    # 1. Check for obviously invalid substrings
    invalid_substrings = ['nan', 'oo', '<class', 'i']
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

def find_parameters_to_fix(expr_template, param_symbols):
    """
    Definitive version to find purely additive constant parameters.

    A parameter 'a_i' is a purely additive constant if and only if the
    partial derivative of the expression with respect to 'a_i' is exactly 1.
    """
    params_to_fix = []

    for param in param_symbols:
        try:
            # Calculate the partial derivative of the expression w.r.t. the parameter
            derivative = sympy.diff(expr_template, param)

            # The parameter is a purely additive constant ONLY if the derivative is 1
            if derivative == 1:
                params_to_fix.append(str(param))

        except Exception as e:
            # If differentiation fails for any reason, better to sample it
            print(f"Warning: Could not differentiate w.r.t {param} for expression '{expr_template}'. Error: {e}")
            continue

    return sorted(params_to_fix)

# def find_parameters_to_fix(expr_template, param_symbols):
#     """
#     Analyzes a sympy expression to find purely additive parameters that can be fixed.
#     This version correctly handles complex cases.
#     """
#     x = sympy.symbols('x')
#     all_params_set = set(param_symbols)

#     # 1. Isolate the "variable part" of the expression (all terms containing 'x')
#     variable_part = sympy.S(0)
#     for term in sympy.Add.make_args(expr_template):
#         if term.has(x):
#             variable_part += term

#     # 2. Find all parameters that appear in this variable part
#     params_in_variable_part = {p for p in param_symbols if variable_part.has(p)}
    
#     # 3. A parameter is purely constant if it's in the full list but NOT in the variable part
#     purely_constant_params = all_params_set.difference(params_in_variable_part)
    
#     if not purely_constant_params:
#         return []

#     # 4. Return the full, sorted list of parameters to be fixed
#     return sorted([str(p) for p in purely_constant_params])


def main():
    # 1. Set up parameters for function generation
    complexity = 6
    # Use a predefined runname from the ESR library that includes sine functions
    runname = "core_maths"
    esr_functions_file = f'./CambESRPotential/esrfunctions/{runname}/compl_{complexity}/unique_equations_{complexity}.txt'
    # potential_function_index = 48  # Index of the ESR function to use
    num_esr_functions = 335

    for potential_function_index in range(num_esr_functions):
        print(f"\n\nRunning potential function index: {potential_function_index}\n")
        esr_function_string, esr_function_template, esr_param_symbols = load_esr_function_string(esr_functions_file, potential_function_index)
        valid_function = is_function_valid(esr_function_string)
        if not valid_function:
            print(f"Skipping invalid function at index {potential_function_index}: {esr_function_string}")
        else:
            esr_param_names = [str(p) for p in esr_param_symbols]
            print(f"Using ESR potential function: {esr_function_string}, with parameters: {esr_param_names}")
            params_to_fix = find_parameters_to_fix(esr_function_template, esr_param_symbols)
            if params_to_fix:
                print(f"Parameters to fix for function index {potential_function_index}: {params_to_fix}")
            else:
                print(f"No parameters to fix for function index {potential_function_index}.")


if __name__ == "__main__":
    main()