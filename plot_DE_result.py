import camb
from CambESRPotential.esr_utils import load_esr_function_string, create_potential_table
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys

def main():
    # 1. Set up parameters for function generation
    complexity = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    # Use a predefined runname from the ESR library
    runname = "custom_DE"
    esr_functions_file = f'./CambESRPotential/esrfunctions/{runname}/compl_{complexity}/unique_equations_{complexity}.txt'
    potential_idx = int(sys.argv[2]) if len(sys.argv) > 1 else 42
    function_dict = load_esr_function_string(esr_functions_file, potential_idx)
    is_valid = function_dict['valid']
    if not is_valid:
        print(f"Skipping invalid function at index {potential_idx}")
        exit(1)




    esr_function_string = function_dict['func_string']
    print(f"Using ESR potential function: {esr_function_string}")
    esr_param_symbols = function_dict['param_symbols']
    print(f"ESR potential parameters: {esr_param_symbols}")
    esr_param_names = [str(p) for p in function_dict['param_symbols']]
    params_to_fix = function_dict['fixed_params']
    variable_params = function_dict['variable_params']
    esr_function_template = function_dict['expr_template']
    phi_vals = np.linspace(-2,2,200)

    # load params from bestfit file
    chains_dir = f"chains/camb_esr/{runname}/compl_{complexity}/{potential_idx}/results"
    min_file_path = Path(chains_dir + ".minimum.txt")
    data = pd.read_csv(
            min_file_path, sep=r'\s+', comment='#',
            header=None, engine='python'
    )
    with open(min_file_path, 'r') as f:
        header_line = f.readline().strip()
        column_names = header_line.replace('#', '').strip().split()
    
    data.columns = column_names
    print(data)

    H0 = data['H0'].values[0]
    ombh2 = data['ombh2'].values[0]
    omch2 = data['omch2'].values[0]
    theta_i = data['theta_i'].values[0] or 0.0

    esr_params = []
    print(f"ESR param names: {esr_param_names}, fixed: {params_to_fix}")
    for param in esr_param_names:
        print(f"Processing parameter: {param}")
        if param in data.columns:
            print(f"Parameter {param} is obtained from minimization.")
            esr_params.append(data[param].values[0])
        if param in params_to_fix:
            print(f"Parameter {param} is fixed to zero.")
            esr_params.append(0.0) # fixed to zero
    print(f"ESR params: {esr_params}")
    potential_dict = create_potential_table(esr_function_template, 
                                            esr_param_symbols,
                                            esr_params, phi_vals)
    success = potential_dict['success']
    
    phi_train = potential_dict['phi_train']
    V_train = potential_dict['V_train']
    dV_train = potential_dict['dV_train']
    ddV_train = potential_dict['ddV_train']    
    pars = camb.set_params(
        H0=H0, ombh2=ombh2, omch2=omch2,
        dark_energy_model='QuintessenceInterp', V_train=V_train, theta_i=theta_i,
        dV_train=dV_train, ddV_train=ddV_train, phi_train=phi_train)

    # 3. Perform the CAMB calculation
    results = camb.get_background(pars)

    # plot DE eos 
    z_arr = np.linspace(0, 2.5, 500)
    _, wde = np.array(results.get_dark_energy_rho_w(1/(1+z_arr)))

    phi, phidot = results.get_dark_energy_phi_phidot(1/(1+z_arr))

    phi_f = phi[0]
    phi_i = theta_i

    fig = plt.figure(figsize=(12, 8))
    ax_dict = fig.subplot_mosaic([['w_de', 'phi_plot'],
                                  ['w_de', 'phidot_plot'],
                                  ['potential', 'potential']],
                                  height_ratios=[1, 1, 1.5]) # Give more vertical space to the potential plot
    
    fig.suptitle(f"ESR Potential: {'exp('+esr_function_string+')'}, cosmological params: $H_0={H0:.2f}, \Omega_c h^2={omch2:.4f}, \Omega_b h^2={ombh2:.4f}, $", fontsize=12)

    # --- Top Row ---
    # Plot w_DE vs z
    ax_dict['w_de'].plot(z_arr, wde, label=f'ESR, $\\theta_i={theta_i:.2f}$')
    ax_dict['w_de'].axhline(-1, ls='--', color='k', label='LCDM',lw=1.5)
    ax_dict['w_de'].set_xlim(z_arr.min(), z_arr.max())
    ax_dict['w_de'].set_xlabel(r'$z$')
    ax_dict['w_de'].set_ylabel(r'$w_{\mathrm{DE}}$')

    # --- Right Column (phi and phidot plots stacked) ---
    # Upper right: phi evolution
    ax_dict['phi_plot'].plot(z_arr, phi, label=r'$\phi$')
    ax_dict['phi_plot'].set_ylabel(r'$\phi$')
    ax_dict['phi_plot'].legend()
    # Remove x-axis labels to avoid overlapping with the plot below
    ax_dict['phi_plot'].tick_params(axis='x', labelbottom=False)

    # Lower right: phidot evolution
    ax_dict['phidot_plot'].plot(z_arr, phidot, label=r'$\dot{\phi}$', color='C1')
    ax_dict['phidot_plot'].set_xlabel(r'$z$')
    ax_dict['phidot_plot'].set_ylabel(r'$\dot{\phi}$')
    ax_dict['phidot_plot'].legend()


    # --- Bottom Row (Potential Plot) ---
    ax_dict['potential'].plot(phi_train, V_train, '-.', label='ESR Potential')
    ax_dict['potential'].axvspan(phi_f, phi_i, alpha=0.3, color='orange', label='Field Evolution')
    ax_dict['potential'].set_xlabel(r'$\phi$')
    ax_dict['potential'].set_ylabel(r'$V(\phi)$')
    ax_dict['potential'].legend(loc='upper right')

    # --- Add Inset Plot to the Potential Subplot ---
    ax_inset = ax_dict['potential'].inset_axes([0.075, 0.5, 0.33, 0.4])
    
    ax_inset.plot(phi_train, V_train, '-.', color='C0')
    
    phi_min_zoom = min(phi_i, phi_f)
    phi_max_zoom = max(phi_i, phi_f)
    ax_inset.set_xlim(phi_min_zoom, phi_max_zoom)
    
    mask = (phi_train >= phi_min_zoom) & (phi_train <= phi_max_zoom)
    if np.any(mask):
        V_in_range = V_train[mask]
        V_min_zoom, V_max_zoom = V_in_range.min(), V_in_range.max()
        padding = (V_max_zoom - V_min_zoom) * 0.1
        ax_inset.set_ylim(V_min_zoom - padding, V_max_zoom + padding)

    ax_inset.set_xlabel(r'$\phi$ (zoom)', fontsize=10)
    ax_inset.set_ylabel(r'$V(\phi)$', fontsize=10)
    ax_inset.tick_params(axis='x', labelsize=8)
    ax_inset.tick_params(axis='y', labelsize=8)
    
    ax_dict['potential'].indicate_inset_zoom(ax_inset, edgecolor="black")
    # --- End of Inset Code ---

    # Add CPL results for comparison to the w_de plot
    min_file_path_cpl = Path("chains/CPL_CMB_lite.minimum.txt")
    cpl_data = pd.read_csv(
            min_file_path_cpl, sep=r'\s+', comment='#',
            header=None, engine='python'
    )
    with open(min_file_path_cpl, 'r') as f:
        header_line = f.readline().strip()
        cpl_column_names = header_line.replace('#', '').strip().split()
    
    cpl_data.columns = cpl_column_names
    w0 = cpl_data['w'].values[0]
    wa = cpl_data['wa'].values[0]
    wde_cpl = w0 + wa * z_arr / (1 + z_arr)
    ax_dict['w_de'].plot(z_arr, wde_cpl, ls='--', label='CPL')
    ax_dict['w_de'].legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()