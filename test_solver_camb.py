"""
A direct comparison script for the custom QuintessenceSolver and a modified pycamb.

This script is based on the provided CAMB example using the 'EarlyQuintessenceAS' model.
"""
import numpy as np
import camb
from scipy.integrate import quad

# IMPORTANT: Make sure this import path correctly points to your solver class
from bkgd_solver.solver import QuintessenceSolver

# ==========================================================================
# 1. DEFINE THE FIDUCIAL COSMOLOGY
# ==========================================================================
fiducial_params = {
    'H0': 67.2,
    'ombh2': 0.022,
    'omch2': 0.122,
    'omk': 0.0,
    'mnu': 0.06,
    'phi_init': 0.0,
    'phidot_init': 0.0, # Start from rest, as is common
    'lambda_phi': 1.0,  # The parameter for your exponential potential
    'As': 2.1e-9,
    'ns': 0.965,
}

# ==========================================================================
# 2. RUN CAMB (using the syntax from your example)
# ==========================================================================
print("\n--- Running CAMB with 'QuintessenceModel' model ---")
camb.set_feedback_level(level=2)

# Use the convenient camb.set_params function
pars_camb = camb.set_params(
    ombh2=fiducial_params['ombh2'],
    omch2=fiducial_params['omch2'],
    omk=fiducial_params['omk'],
    H0=fiducial_params['H0'],
    mnu=fiducial_params['mnu'],
    As=fiducial_params['As'],
    ns=fiducial_params['ns'],
    # Set the custom dark energy model and its parameter
    dark_energy_model='QuintessenceModel',
    n=fiducial_params['lambda_phi'] # Pass lambda_phi using the 'n' keyword
)

# Get results from CAMB
results_camb = camb.get_results(pars_camb)

# Extract the values we want to compare
zdrag_camb = results_camb.get_derived_params()['zdrag']
rdrag_camb = results_camb.get_derived_params()['rdrag']
zstar_camb = results_camb.get_derived_params()['zstar']
print(f"CAMB zstar = {zstar_camb}, previous fiducial was 1089.8, zdrag = {zdrag_camb}")
# Use the direct method for the scaled theta from your example
theta_star_camb = results_camb.cosmomc_theta() 
# Get w(z=0) by requesting w at scale factor a=1.0
# The result is [[rho_de, w_de]], so we take [0][1]
_, w0_camb =  np.array(results_camb.get_dark_energy_rho_w(1.)).T
w0_camb = float(w0_camb)


# ==========================================================================
# 3. RUN YOUR CUSTOM SOLVER
# ==========================================================================
print("\n--- Running Custom QuintessenceSolver ---")

T_CMB = 2.7255  # Kelvin
Omega_photons = 3 * (T_CMB/2.7)**4 / (4 * 31500 * (fiducial_params['H0'] / 100.0)**2)  # From T_CMB=2.7255K
print(f"Calculated Omega_photons = {Omega_photons:.5e}")

# Instantiate your solver class
solver = QuintessenceSolver(
    H0=fiducial_params['H0'],
    Omega_m=(fiducial_params['omch2'] + fiducial_params['ombh2']) / (fiducial_params['H0'] / 100.0)**2,
    # A more accurate Omega_r based on T_CMB=2.7255K
    # Omega_r=4.15e-5 / (fiducial_params['H0'] / 100.0)**2,
    Omega_r=Omega_photons,
    Omega_k=fiducial_params['omk'],
    phi_init=fiducial_params['phi_init'],
    phidot_init=fiducial_params['phidot_init'],
    # Define the potential and its derivative, matching the exponential form
    V_base=lambda phi, **kwargs: np.exp(-kwargs.get('lambda_phi', 1.0) * phi),
    dV_base_dphi=lambda phi, **kwargs: -kwargs.get('lambda_phi', 1.0) * np.exp(-kwargs.get('lambda_phi', 1.0) * phi),
    V_kwargs={'lambda_phi': fiducial_params['lambda_phi']},
    verbose=False,
    z_init=1e8,
)

# --- Calculate rdrag using the custom solver's H(z) ---
ombh2 = fiducial_params['ombh2']
omch2 = fiducial_params['omch2']
#rdrag_custom = 147.05* (ombh2/0.02236)**(-0.13) * ((ombh2+omch2)/0.1432)**(-0.23)  # in Mpc, for Neff = 3.04 # this needs to be corrected for D.E. evolution


zstar = 1048*(1+0.00124*ombh2**(-0.738))*(1+ (0.0783*ombh2**(-0.238)/(1+39.5*ombh2**0.763)) * (omch2+ombh2)**(0.560/(1+21.1*ombh2**1.81)))
print(f"zstar = {zstar}, previous fiducial was 1089.8")

# zstar = 1089.8
theta_star_custom =  solver.compute_theta_star(ombh2, z_star=zstar)

# # This logic should be part of your cobaya wrapper's calculate() method
_c_km_s = 299792.458
def rdrag_integrand(z_prime):
    H_z_prime_kmsMpc = solver.H_of_z(z_prime)
    R_baryon_photon = 31500 * fiducial_params['ombh2']
    cs_squared = (_c_km_s**2) / (3.0 * (1 + R_baryon_photon / (1 + z_prime)))
    return np.sqrt(cs_squared) / H_z_prime_kmsMpc

rdrag_custom, _ = quad(rdrag_integrand, 1060, 1e8, epsabs=1e-8, epsrel=1e-8)
# rdrag_custom = 147.05* (ombh2/0.02236)**(-0.13) * ((ombh2+omch2)/0.1432)**(-0.23)  # in Mpc, for Neff = 3.04 # this needs to be corrected for D.E. evolution
print(f"Custom solver rdrag = {rdrag_custom}")

# # --- Calculate 100*theta_star using the custom solver ---
# z_star = 1089.8
# # Use rdrag as a close approximation for r_star (sound horizon at decoupling)
# r_star_custom = rdrag_custom 
# DA_star_custom = solver.angular_diameter_distance(z_star)
# DM_star_custom = (1 + z_star) * DA_star_custom
# theta_star_custom = 100 * r_star_custom / DM_star_custom

# --- Get w(z=0) from the custom solver ---
w0_custom = solver.w_de(z=0.0)


# ==========================================================================
# 4. PRINT THE COMPARISON
# ==========================================================================
print("\n" + "="*65)
print("      COMPARISON OF SOLVER RESULTS")
print("="*65)
print(f"{'Parameter':<18} | {'CAMB':<15} | {'Custom Solver':<15} | {'% Difference':<15}")
print("-"*65)

def percent_diff(v_camb, v_custom):
    if v_camb == 0: return float('inf')
    return 100 * (v_custom - v_camb) / v_camb

print(f"{'100*theta_star':<18} | {theta_star_camb:<15.6f} | {theta_star_custom:<15.6f} | {percent_diff(theta_star_camb, theta_star_custom):<15.2f}%")
print(f"{'r_drag (Mpc)':<18} | {rdrag_camb:<15.4f} | {rdrag_custom:<15.4f} | {percent_diff(rdrag_camb, rdrag_custom):<15.2f}%")
print(f"{'w(z=0)':<18} | {w0_camb:<15.4f} | {w0_custom:<15.4f} | {percent_diff(w0_camb, w0_custom):<15.2f}%")
print("="*65)