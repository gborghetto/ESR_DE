import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from solver import QuintessenceSolver, QuintessenceSpline

# Define potential and derivative (matches cobaya_interface_claude)
def Vphi(phi, **kwargs):
    lambda_phi = kwargs.get('lambda_phi', 1.0)
    return np.exp(-lambda_phi * phi)

def dVphi_dphi(phi, **kwargs):
    lambda_phi = kwargs.get('lambda_phi', 1.0)
    return -lambda_phi * Vphi(phi, **kwargs)


def main():
    # Cosmology and solver parameters
    H0 = 67.66  # km/s/Mpc
    ombh2 = 0.022383
    omch2 = 0.12011
    H0_conv = H0
    omegam = (ombh2 + omch2) / (H0_conv / 100.0)**2
    omega_r = 9e-5
    omega_k = 0.0

    phi_init = 0.5
    phidot_init = 0.0
    lambda_phi = 1.0

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
        z_init=1e3,
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
    phi_grid = np.linspace(np.min(phi_vals)-0.5, np.max(phi_vals)+0.5, 400)
    V_grid = np.array([Vphi(p, lambda_phi=lambda_phi) * solver.amplitude for p in phi_grid])

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


if __name__ == "__main__":
    main()
