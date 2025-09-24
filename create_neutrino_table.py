import numpy as np
import camb
import os

# --- Define your fixed fiducial cosmology ---
# These parameters (especially mnu) should match the fixed values in your MCMC.
FIDUCIAL_MNU = 0.06  # The fixed sum of neutrino masses in eV
FIDUCIAL_OMCH2 = 0.120
FIDUCIAL_OMBH2 = 0.0224
FIDUCIAL_H0 = 67.2

# --- Run CAMB to get the high-precision evolution ---
print(f"Running CAMB for mnu = {FIDUCIAL_MNU} eV...")

# Using the convenient camb.set_params function
pars = camb.set_params(
    H0=FIDUCIAL_H0,
    ombh2=FIDUCIAL_OMBH2,
    omch2=FIDUCIAL_OMCH2,
    mnu=FIDUCIAL_MNU,
    # Standard N_eff for 3 neutrino species
    num_massive_neutrinos=3,
    nnu=3.046
)

# Set up a redshift grid for the calculation
z_grid = np.logspace(-4, 8, 1000)

# Get background evolution results from CAMB
results = camb.get_background(pars)

# --- CORRECTED SYNTAX IS HERE ---
# Get Omega_nu(z) using the correct 'z' keyword argument
h = FIDUCIAL_H0 / 100.0
Omega_nu_z = results.get_Omega('nu', z=z_grid)
onuh2_z = Omega_nu_z * h**2
# ------------------------------

print('Omega nu today =', onuh2_z[0] / h**2)

a_grid = 1.0 / (1.0 + z_grid)

# --- Save the results to a file ---
output_data = np.vstack([a_grid, onuh2_z]).T
output_file = "neutrino_density_table.txt"
np.savetxt(
    output_file,
    output_data,
    header="1:Scale Factor (a)    2:Neutrino Density (Omega_nu * h^2)"
)

print(f"Successfully saved neutrino density table to '{output_file}'")