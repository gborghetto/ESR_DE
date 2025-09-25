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

# ==========================================================================
# 2. MAIN ANALYSIS FUNCTION (Modified)
# ==========================================================================
def analyze_cobaya_runs(base_directory="chains/camb_esr"):
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
                run_name = complexity_dir.parent.name
                
                source_id = f"{run_name}/{complexity_dir.name}/{potential_index}"
                
                # --- NEW: Load the corresponding function string ---
                # Assumption: The function file is named 'functions.txt' and lives
                # in the complexity directory (e.g., .../compl_6/functions.txt)
                function_file_path = complexity_dir / "unique_equations_6.txt"
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
                
                all_results.append(data)

            except Exception as e:
                print(f"Could not process file {min_file_path}. Error: {e}")

    if not all_results:
        print("\nNo valid results found. Exiting.")
        return
    
    full_results_df = pd.concat(all_results, ignore_index=True)
    top_5_results = full_results_df.sort_values(by='chi2_total').head(5)

    # --- Display the results ---
    print("\n" + "="*120)
    print("                                     TOP 5 COBAYA MINIMIZATION RESULTS")
    print("="*120)
    
    # Define the base columns we always want to see
    display_columns = [
        'source_run', 'potential_index', 'chi2_total', 'function_string'
    ]
    
    # --- NEW: Dynamically find and add parameter columns ---
    # Find all columns in the results that look like 'a0', 'a1', etc.
    param_columns = sorted([
        col for col in top_5_results.columns if col.startswith('a') and col[1:].isdigit()
    ])
    
    # Add the found parameter columns to our list for display
    display_columns.extend(param_columns)
    # --- END OF NEW LOGIC ---
    
    existing_display_columns = [col for col in display_columns if col in top_5_results.columns]
    
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', None)
    
    # Format floating point columns for better readability
    float_formatters = {col: '{:.4g}'.format for col in top_5_results.select_dtypes(include='float').columns}

    print(top_5_results[existing_display_columns].to_string(index=False, formatters=float_formatters))
    print("="*120)


if __name__ == "__main__":
    analyze_cobaya_runs()
