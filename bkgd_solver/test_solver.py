import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from solver import QuintessenceSolver, QuintessenceSpline
from scipy.integrate import quad

# Speed of light [km/s]
_c_km_s = 299792.458

# Define potential and derivative (matches cobaya_interface_claude)
def Vphi(phi, **kwargs):
    lambda_phi = kwargs.get('lambda_phi', 1.0)
    return np.exp(-lambda_phi * phi)

def dVphi_dphi(phi, **kwargs):
    lambda_phi = kwargs.get('lambda_phi', 1.0)
    return -lambda_phi * Vphi(phi, **kwargs)


def compute_rdrag(solver, ombh2, H0):
    """Compute an r_drag-like integral (comoving sound horizon) in Mpc.

    This follows the same structure as the cobaya interface: integrate
    cs(z)/H(z) from z_drag (~1060) up to the solver's z_init, then add an
    analytic tail assuming radiation domination to approximate the remainder.
    """
    R_baryon_photon = 31500 * ombh2

    def cs_squared(z_prime):
        return (_c_km_s ** 2) / (3.0 * (1 + R_baryon_photon / (1 + z_prime)))

    def integrand(z_prime):
        H_z_prime_Gyr = float(np.atleast_1d(solver.H_of_z(z_prime))[0])
        # convert Gyr^-1 to km/s/Mpc
        H_z_prime_kmsMpc = H_z_prime_Gyr * (3.0857e19 / 3.1536e16)
        cs_z_prime = np.sqrt(cs_squared(z_prime))
        return cs_z_prime / H_z_prime_kmsMpc

    z_drag = 1060.0

    # integrate from z_drag to z_max
    result, err = quad(integrand, z_drag, 1e8, epsabs=1e-8, epsrel=1e-8)

    # # analytic tail assuming radiation domination for z>z_max
    # try:
    #     H0_kmsMpc = H0
    #     Omega_r_solver = float(getattr(solver, 'Omega_r', 9e-5))
    #     cs_inf = _c_km_s / np.sqrt(3.0)
    #     if Omega_r_solver > 0:
    #         tail = (cs_inf / (H0_kmsMpc * np.sqrt(Omega_r_solver))) * (1.0 / (1.0 + z_max))
    #     else:
    #         tail = 0.0
    # except Exception:
    #     tail = 0.0

    rdrag = float(result) #+ float(tail)
    return rdrag


def main():
    # Cosmology and solver parameters
    H0 = 67.66  # km/s/Mpc
    ombh2 = 0.022383
    omch2 = 0.12011
    H0_conv = H0
    omegam = (ombh2 + omch2) / (H0_conv / 100.0)**2
    omega_r = 9e-5
    omega_k = 0.0

    phi_init = 0.
    phidot_init = 0.0
    lambda_phi = 1.5

    # Initialize solver
    solver = QuintessenceSolver(
        H0=H0,
        Omega_m=omegam,
        Omega_r=omega_r,
        Omega_k=omega_k,
        phi_init=phi_init,
        phidot_init=phidot_init,
        V_base=Vphi,
        dV_base_dphi=dVphi_dphi,
        V_kwargs={'lambda_phi': lambda_phi},
        z_init=1e5,
        verbose=True,
    )

    # Prepare redshift array for plotting
    z_vals = np.linspace(0, 10, 400)

    # Compute quantities
    phi_vals = np.array([solver.phi(z) for z in z_vals])
    phidot_vals = np.array([solver.phidot(z) for z in z_vals])
    w_vals = np.array([solver.w_de(z) for z in z_vals])
    H_vals = np.array([solver.H_of_z(z) for z in z_vals])

    # Potential sampled over phi range
    phi_grid = np.linspace(min(phi_vals), max(phi_vals), 100)
    V_grid = Vphi(phi_grid, lambda_phi=lambda_phi) * solver.amplitude

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0,0]
    ax.plot(phi_grid, V_grid)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$V(\phi)$")
    ax.set_title("Potential")

    ax = axes[0,1]
    ax.plot(z_vals, phi_vals)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\phi(z)$")
    ax.set_title("Field evolution")
    ax.invert_xaxis()

    ax = axes[1,0]
    ax.plot(z_vals, w_vals)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$w_{DE}(z)$")
    ax.set_title("Dark energy equation of state")
    ax.invert_xaxis()

    ax = axes[1,1]
    ax.plot(z_vals, H_vals)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$H(z)$ [Gyr$^{-1}$]")
    ax.set_title("Hubble parameter")
    ax.invert_xaxis()

    plt.tight_layout()
    outname = "quintessence_results.png"
    plt.savefig(outname, dpi=150)
    print(f"Saved figure to {outname}")

    # Compute and print rdrad-like quantity used in BAO calculations
    rdrad_val = compute_rdrag(solver, ombh2=ombh2, H0=H0)
    print(f"Computed rdrad-like value: {rdrad_val:.6f} Mpc")


if __name__ == "__main__":
    main()
