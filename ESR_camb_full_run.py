#!/usr/bin/env python3

"""
Python script to run a Cobaya MCMC analysis for a Quintessence model.

This script defines the model, parameters, likelihoods, and sampler settings
for a cosmological analysis using a custom Quintessence theory code,
and a combination of CMB, BAO, and Supernova likelihoods.
"""

from cobaya.run import run
from CambESRPotential.CobayaInterfaceESR import load_esr_function_string
import argparse

def create_cobaya_info_dict(esr_functions_file, potential_function_index, esr_param_names):
    """
    Create the Cobaya info dictionary with the necessary settings.
    This includes the theory, likelihoods, parameters, and sampler settings.
    """
    # Define the dictionary with all the settings for the run

    # Define the dictionary with all the settings for the run
    info = {
        # Theory
        "theory": {
            "CambESRPotential.CobayaInterfaceESR.CambQuintessenceESR": 
            {
                "python_path": '.',
                "stop_at_error": False,
                "path_ESR_data": esr_functions_file,
                "potential_function_idx": potential_function_index,  # Index of the ESR function to use
            }
        },

        # Likelihoods
        "likelihood": {
            "cmb_lite_3d": {
                "python_path": "/Users/amk/Library/CloudStorage/OneDrive-SwanseaUniversity/Codes/VPhi_Q/cmb_lite"
            },
            "bao.desi_dr2.desi_bao_all": None,
            "sn.union3": None
        },

        # Parameters
        "params": {
            # Fixed parameters
            "omk": 0.,
            'mnu': 0.06,  # Commented out as in the original request

            # Sampled parameters
            "omch2": {
                "latex": r"\Omega_\mathrm{c} h^2",
                "prior": {
                    "min": 0.05,
                    "max": 0.2
                },
                "ref": {
                    "dist": "norm",
                    "loc": 0.11985,
                    "scale": 0.001
                },
                "proposal": 0.0001
            },
            "ombh2": {
                "latex": r"\Omega_\mathrm{b} h^2",
                "prior": {
                    "min": 0.01,
                    "max": 0.04
                },
                "ref": {
                    "dist": "norm",
                    "loc": 0.02223,
                    "scale": 0.0001
                },
                "proposal": 0.0001
            },
            "H0": {
                "latex": r"H_0",
                "prior": {
                    "min": 60,
                    "max": 72
                },
                "ref": {
                    "dist": "norm",
                    "loc": 67.0,
                    "scale": 0.1
                },
                "proposal": 0.05
            },

            # Derived parameters
            "chi2__BAO": {
                "latex": r"\chi^2_\mathrm{BAO}",
                "derived": True
            },
            "chi2__CMB": {
                "latex": r"\chi^2_\mathrm{CMB}",
                "derived": True
            },
            "chi2__SN": {
                "latex": r"\chi^2_\mathrm{SN}",
                "derived": True
            },
            "chi2_total": {
                "latex": r"\chi^2_\mathrm{total}",
                "derived": "lambda chi2__BAO, chi2__SN, chi2__CMB: chi2__BAO + chi2__SN + chi2__CMB"
            },
            "omegam": {
                "latex": r"\Omega_\mathrm{m}",
                "derived": "lambda omch2, ombh2, H0: (omch2 + ombh2) / (H0 / 100.0)**2"
            },
            "rdrag": {
                "latex": r"r_\mathrm{drag}",
                "derived": True
            },
            "hrd": {
                "latex": r"hr_d",
                "derived": "lambda H0, rdrag: H0/100 * rdrag"
            },
            "thetastar": {
                "latex": r"\theta_*",
                "derived": True
            }
        },
        
        # Output settings
        # "output": "chains/spline_4_free_pchord", # Commented out
        "output": f"chains/camb_esr/{runname}/compl_{complexity}/{potential_function_index}/results",
    }

    # Update info with ESR potential parameters
    if esr_param_names:
        for param in esr_param_names:
            info['params'][param] = {
                "latex": param,
                "prior": {
                    "min": -5.0,
                    "max": 5.0
                },
                "ref": {
                    "dist": "norm",
                    "loc": 0.0,
                    "scale": 0.2
                },
                "proposal": 0.1
            }

    return info


def run_single_potential(esr_functions_file, potential_function_index, resume, test, force, debug):
    """
    Run a single potential function with the given parameters.
    """
    esr_function_string, esr_function_template, esr_param_symbols = load_esr_function_string(esr_functions_file, potential_function_index)
    esr_param_names = [str(p) for p in esr_param_symbols]

    print(f"Using ESR potential function: {esr_function_string}, with parameters: {esr_param_names}")

    print(f"Resume: {resume}, Test: {test}, Force: {force}, Debug: {debug}")

    info = create_cobaya_info_dict(esr_functions_file, potential_function_index, esr_param_names)

    mcmc_info = {"sampler": {
            "mcmc": {
                "drag": False,
                "oversample_power": 0.4,
                "proposal_scale": 1.9,
                "Rminus1_stop": 0.1,
                "Rminus1_cl_stop": 0.2,
                "max_tries": 100,
                "max_samples": 10000,
            }
        }
    }
    info.update(mcmc_info)

    # Run the sampler with the defined settings
    try:
        updated_info, sampler = run(info, resume=resume, test=test, force=force, debug=debug)
    except Exception as e:
        print(f"Error during Cobaya mcmc run: {e}")

    minimize_info = {"sampler": {
        "minimize": {
            "method": "scipy",
            "best_of": 80,
            }
        }
    }
    try:
        info.update(minimize_info)
        updated_info, sampler = run(info, debug=debug,force=True, resume=resume)
    except Exception as e:
        print(f"Error during minimization run: {e}")

# Main execution block
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous run", default=False
    )
    parser.add_argument(
        "--test", action="store_true", help="Test run", default=False
    )
    # default True for force per user request; using store_true keeps typical flag behavior
    parser.add_argument(
        "--force", action="store_true", help="Force overwrite of existing chains", default=False
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode (prints more info)", default=False
    )

    args = parser.parse_args()
    resume = bool(args.resume)
    test = bool(args.test)
    force = bool(args.force)
    debug = bool(args.debug)

    # 1. Set up parameters for function generation
    complexity = 6
    # Use a predefined runname from the ESR library that includes sine functions
    runname = "core_maths"
    esr_functions_file = f'./CambPotential/esrfunctions/{runname}/compl_{complexity}/unique_equations_{complexity}.txt'
    # potential_function_index = 48  # Index of the ESR function to use
    num_esr_functions = 335

    for potential_function_index in range(num_esr_functions):
        print(f"\n\nRunning potential function index: {potential_function_index}\n")
        run_single_potential(esr_functions_file, potential_function_index, resume, test, force, debug)