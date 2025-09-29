import os
import pandas as pd
from pathlib import Path
import sympy  # Import sympy for the helper function

# ==========================================================================
# 1. HELPER FUNCTION (Provided by you)
# ==========================================================================
def load_esr_function_string(file_path, idx=0, verbose=False):
    """Loads a specific function string from a file."""
    try:
        with open(file_path, "r") as f:
            all_functions = [line.strip() for line in f.readlines() if line.strip()]
            if idx >= len(all_functions):
                return f"Index {idx} out of bounds for file {file_path}", None, None
            func_string = all_functions[idx]
            # The rest of the function is for sympy conversion, we only need the string
            return func_string, None, None
    except FileNotFoundError:
        return f"Function file not found at {file_path}", None, None
    except Exception as e:
        return f"Error reading function file: {e}", None, None
    
def load_benchmark_chi2(model_path):
    """
    Loads the chi2_total from a .minimum.txt file for a benchmark model.
    """
    try:
        min_file_path = Path(model_path + ".minimum.txt")
        if not min_file_path.exists():
            print(f"Warning: Benchmark file not found at {min_file_path}")
            return None
            
        data = pd.read_csv(
            min_file_path, sep=r'\s+', comment='#',
            header=None, engine='python'
        )
        with open(min_file_path, 'r') as f:
            header_line = f.readline().strip()
            column_names = header_line.replace('#', '').strip().split()
        data.columns = column_names
        
        return data['chi2_total'].iloc[0]
    except Exception as e:
        print(f"Warning: Could not load benchmark model from {model_path}. Error: {e}")
        return None

# ==========================================================================
# 2. MAIN ANALYSIS FUNCTION (Modified)
# ==========================================================================
def analyze_cobaya_runs(base_directory="chains/camb_esr",chi2_lcdm=None,chi2_cpl=None):
    """
    Analyzes all results.minimum.txt files, extracts the potential function index
    and string, and finds the top 5 runs based on chi2_total.
    """
    all_results = []
    
    if not os.path.isdir(base_directory):
        print(f"Error: Base directory not found at '{os.path.abspath(base_directory)}'")
        return

    print(f"Searching for run outputs in: {os.path.abspath(base_directory)}")

    for root, dirs, files in os.walk(base_directory):
        target_filename = "results.minimum.txt"
        if target_filename in files:
            min_file_path = Path(root) / target_filename
            
            try:
                # --- Path Parsing Logic ---
                p = min_file_path
                potential_index_str = p.parent.name
                if potential_index_str == 'results':
                    potential_index_str = p.parent.parent.name
                
                potential_index = int(potential_index_str)
                
                complexity_dir = p.parent.parent if potential_index_str == p.parent.name else p.parent.parent.parent
                complexity = int(complexity_dir.name.split('_')[-1])
                run_name = complexity_dir.parent.name
                
                source_id = f"{run_name}/{complexity_dir.name}/{potential_index}"
                
                # --- NEW: Load the corresponding function string ---
                # Assumption: The function file is named 'functions.txt' and lives
                # in the complexity directory (e.g., .../compl_6/functions.txt)
                function_file_path = f'./CambESRPotential/esrfunctions/{run_name}/compl_{complexity}/unique_equations_{complexity}.txt'
                func_string, _, _ = load_esr_function_string(function_file_path, idx=potential_index)
                
                # --- Load the numerical results ---
                data = pd.read_csv(
                    min_file_path, sep=r'\s+', comment='#',
                    header=None, engine='python'
                )
                
                if data.empty:
                    print(f"Skipping empty file for run: {source_id}")
                    continue

                with open(min_file_path, 'r') as f:
                    header_line = f.readline().strip()
                    column_names = header_line.replace('#', '').strip().split()
                
                data.columns = column_names
                
                # Add all extracted information as new columns
                data['source_run'] = source_id
                data['potential_index'] = potential_index
                data['function_string'] = func_string

                # --- NEW: Calculate Delta chi2 values ---
                if chi2_lcdm is not None:
                    data['d_chi2_lcdm'] = data['chi2_total'] - chi2_lcdm
                if chi2_cpl is not None:
                    data['d_chi2_cpl'] = data['chi2_total'] - chi2_cpl                

                all_results.append(data)

            except Exception as e:
                print(f"Could not process file {min_file_path}. Error: {e}")

    if not all_results:
        print("\nNo valid results found. Exiting.")
        return
    
    full_results_df = pd.concat(all_results, ignore_index=True)
    top_10_results = full_results_df.sort_values(by='chi2_total').head(10)

    # --- Display the results ---
    print("\n" + "="*120)
    print("                                     TOP 10 COBAYA MINIMIZATION RESULTS")
    print("="*120)
    
    # Define the base columns we always want to see
    display_columns = [
        'source_run', 'potential_index', 'chi2_total', 'd_chi2_lcdm', 'd_chi2_cpl', 'function_string'
    ]
    
    # --- NEW: Dynamically find and add parameter columns ---
    # Find all columns in the results that look like 'a0', 'a1', etc.
    param_columns = sorted([
        col for col in top_10_results.columns if col.startswith('a') and col[1:].isdigit()
    ])
    
    # Add the found parameter columns to our list for display
    display_columns.extend(param_columns)
    # --- END OF NEW LOGIC ---

    existing_display_columns = [col for col in display_columns if col in top_10_results.columns]
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', None)
    
    # Format floating point columns for better readability
    float_formatters = {col: '{:.4g}'.format for col in top_10_results.select_dtypes(include='float').columns}

    print(top_10_results[existing_display_columns].to_string(index=False, formatters=float_formatters))
    print("="*120)


if __name__ == "__main__":

    # --- NEW: Load the chi-squared values for the benchmark models ---
    lcdm_path = "./chains/LCDM_CMB_lite"
    cpl_path = "./chains/CPL_CMB_lite"
    chi2_lcdm = load_benchmark_chi2(lcdm_path)
    chi2_cpl = load_benchmark_chi2(cpl_path)

    analyze_cobaya_runs(chi2_lcdm=chi2_lcdm, chi2_cpl=chi2_cpl)
