#!/usr/bin/env python3

"""
Python script to run a Cobaya MCMC analysis for a Quintessence model.

This script defines the model, parameters, likelihoods, and sampler settings
for a cosmological analysis using a custom Quintessence theory code,
and a combination of CMB, BAO, and Supernova likelihoods.
"""

from cobaya.run import run
import argparse

# Define the dictionary with all the settings for the run
info = {
    # Theory
    "theory": {
        "camb": {
            "path": "global",
            "stop_at_error": False,
            "extra_args": {
                'bbn_predictor': 'PArthENoPE_880.2_standard.dat',
                'lens_potential_accuracy': 1,
                'num_massive_neutrinos': 1,
                'AccuracyBoost': 2,
                'nnu': 3.046,
                'dark_energy_model': 'QuintessenceModel', 
                'model_idx': 1, 
                'frac_lambda0': 0.,
                'use_zc': False
            }
        }
    },

    # Likelihoods
    "likelihood": {
        "cmb_lite_3d": {
            "python_path": "/Users/amk/Library/CloudStorage/OneDrive-SwanseaUniversity/Codes/VPhi_Q/cmb_lite"
        },
        # BBN prior on ombh2 (commented out as in the original request)
        # "like_ombh2_BBN": {
        #     "external": "lambda _self: stats.norm.logpdf(_self.provider.get_param('ombh2'), loc=0.02218, scale=0.0006)",
        #     "requires": ["ombh2"]
        # },
        "bao.desi_dr2.desi_bao_all": None,
        "sn.union3": None
    },

    # Parameters
    "params": {
        # Fixed parameters
        "omk": 0.,
        # 'mnu': 0.06,  # Commented out as in the original request

        # Quintessence initial conditions (fixed)
        "theta_i": 0.,

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
        "n": {
            "latex": r"\lambda_\phi",
            "prior": {
                "min": 0.,
                "max": 2.
            },
            "ref": {
                "dist": "norm",
                "loc": 0.2,
                "scale": 0.05
            },
            "proposal": 0.05
        },
        
        # Derived parameters
        "chi2__BAO": {
            "latex": r"\chi^2_\mathrm{BAO}",
            "derived": True
        },
        "chi2__SN": {
            "latex": r"\chi^2_\mathrm{SN}",
            "derived": True
        },
        "chi2_total": {
            "latex": r"\chi^2_\mathrm{total}",
            "derived": "lambda chi2__BAO, chi2__SN: chi2__BAO + chi2__SN"
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
        }
    },

    # Sampler settings
    "sampler": {
        "mcmc": {
            "drag": False,
            "oversample_power": 0.4,
            "proposal_scale": 1.9,
            "Rminus1_stop": 0.01,
            "Rminus1_cl_stop": 0.1,
            "max_tries": 100,
            "max_samples": 10000,
            # "covmat": "chains/test_solver_exp.covmat"
        }
        # Polychord settings (commented out as in original request)
        # "polychord": {
        #     "nlive": "25d",
        #     "precision_criterion": 0.1,
        #     "maximise": True
        # }
    },
    
    # Output settings
    # "output": "chains/spline_4_free_pchord", # Commented out
    "output": "chains/camb_exp_mcmc"
}


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

    print(f"Resume: {resume}, Test: {test}, Force: {force}, Debug: {debug}")

    # Run the sampler with the defined settings
    updated_info, sampler = run(info, resume=resume, test=test, force=force, debug=debug)