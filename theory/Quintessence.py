# my_quintessence_theory.py

import numpy as np
from scipy.integrate import quad
from cobaya.theory import Theory
from .solver import QuintessenceSolver

# Speed of light in km/s
_c_km_s = 299792.458

def Vphi(phi, params):
    """
    Example potential function V(phi).
    This can be modified to implement different quintessence potentials.
    """
    # Example: simple quadratic potential
    m = params.get('m', 1e-33)  # mass scale in eV
    return 0.5 * m**2 * phi**2

def dVphi_dphi(phi, params):
    """
    Derivative of the potential function V(phi).
    """
    m = params.get('m', 1e-33)  # mass scale in eV
    return m**2 * phi

class QuintessenceTheory(Theory):
    """
    Cobaya Theory module wrapping a Quintessence solver to provide:
      - H(z)
      - comoving_AngularDistance(z)
      - luminosity_distance(z)
    """

    omega_r = 9e-5  # Default radiation density if not provided

    def initialize(self):
        """Nothing to do until provider is available"""
        pass

    def initialize_with_provider(self, provider):
        """
        Called after all other components are initialized.
        We build the quintessence solver using sampled parameters.
        """
        self.provider = provider
        # helper to read parameters with alternative common names
        def P(name, default=None):
            # try provider.get_param for a single name; allow common aliases
            aliases = {
                'H0': ['H0', 'h', 'H_0','Hubble'],
                'Omega_m': ['Omega_m', 'omegam', 'Omega_m0'],
                'Omega_r': ['Omega_r', 'omegar', 'Omega_r0'],
                'Omega_k': ['Omega_k', 'omk', 'Omega_k0'],
                'phi_init': ['phi_init', 'theta_i', 'phi0'],
                'phidot_init': ['phidot_init', 'phidot0'],
                'm': ['m'],
            }
            if name in aliases:
                for a in aliases[name]:
                    try:
                        return provider.get_param(a)
                    except Exception as e:
                        print(f"DEBUG: Failed to get param '{a}'. Reason: {e}")
                        pass
                return default
            try:
                return provider.get_param(name)
            except Exception:
                return default

        # redshifts sent by likelihoods will be handled dynamically
        H0_val = P('H0')
        print("H0_val in QuintessenceTheory:", H0_val)
        if H0_val is None:
            # try to build H0 from h and H0=100*h
            hval = P('h')
            if hval is not None:
                H0_val = 100.0 * hval

        self.solver = QuintessenceSolver(
            H0=float(H0_val),
            Omega_m=float(P('Omega_m', 0.3)),
            Omega_r=float(P('Omega_r', 0.0)),
            Omega_k=float(P('Omega_k', 0.0)),
            phi_init=float(P('phi_init', 1.0)),
            phidot_init=float(P('phidot_init', 0.0)),
            V_base=Vphi,
            dV_base_dphi=dVphi_dphi,
            V_params={'m': P('m', 1e-33)},
            z_init=float(P('z_init', 1e2)),
            A_min=float(P('A_min', 1e-10)),
            A_max=float(P('A_max', 1e10)),
            atol=float(P('atol', 1e-10)),
            rtol=float(P('rtol', 1e-10)),
            verbose=False,
        )

    def get_requirements(self):
        """
        Quantities always needed by this theory...
        """
        return {
            'H0': None,
            'omk': None,
            'phi_init': None,
            'phidot_init': None,
            'm': None,
            'omch2': None,   # <-- ADD THIS
            'ombh2': None,   # <-- AND ADD THIS
        }

    def must_provide(self, **requirements):
        """
        Conditionally declare further requirements if specific observables are requested.
        Here, no extra conditional inputs beyond z.
        """
        provides = {}
        if 'H' in requirements:
            provides['z'] = None
        if 'comoving_AngularDistance' in requirements or 'luminosity_distance' in requirements:
            provides['z'] = None
        return provides

    def get_can_provide(self):
        """List of quantities this theory can compute"""
        return ['Hubble', 'angular_diameter_distance', 'luminosity_distance', 'rdrag']

# In QuintessenceTheory class

    def calculate(self, state, want_derived=False, **params_values_dict):
        """
        Compute raw observables and store in state.
        """
        z = np.atleast_1d(state['z'])
        
        # Get ombh2 from the provider
        ombh2 = self.provider.get_param('ombh2')

        # --- Calculate H(z) and Distances (your existing code) ---
        sec_per_Gyr = 3.1536e16
        km_per_Mpc = 3.0857e19
        H_Gyr = self.solver.H_of_z(z)
        H_km_s_Mpc = H_Gyr * (1.0 / sec_per_Gyr) * km_per_Mpc

        def inv_H_Mpc(zp):
            Hp_Gyr = float(np.atleast_1d(self.solver.H_of_z(zp))[0])
            Hp = Hp_Gyr * (1.0 / sec_per_Gyr) * km_per_Mpc
            return _c_km_s / Hp
        
        chi = np.array([quad(inv_H_Mpc, 0.0, float(zi))[0] for zi in z])
        DA = chi / (1.0 + z)
        DL = chi * (1.0 + z)
        
        # --- NEW: Calculate r_drag ---
        z_drag = 1060  # Approximate drag epoch redshift

        # Sound speed squared c_s^2(z) in (km/s)^2, using Eq. 3.5 & 3.6
        # The factor 31500 approximates 3*rho_b0 / (4*rho_g0 * ombh2)
        R_baryon_photon = 31500 * ombh2 
        def cs_squared(z_prime):
            return (_c_km_s**2) / (3.0 * (1 + R_baryon_photon / (1 + z_prime)))

        # Integrand for the sound horizon: c_s(z) / H(z)
        def rdrag_integrand(z_prime):
            # We need H(z') in km/s/Mpc, calculated from the solver
            H_z_prime_Gyr = float(np.atleast_1d(self.solver.H_of_z(z_prime))[0])
            H_z_prime = H_z_prime_Gyr * (1.0 / sec_per_Gyr) * km_per_Mpc
            cs_z_prime = np.sqrt(cs_squared(z_prime))
            return cs_z_prime / H_z_prime

        # Perform the integration from z_drag to infinity (Eq. 3.4)
        rdrag_val, _ = quad(rdrag_integrand, z_drag, np.inf)

        # --- Store all results in the state dictionary ---
        state['Hubble'] = H_km_s_Mpc
        state['angular_diameter_distance'] = DA
        state['luminosity_distance'] = DL
        state['rdrag'] = rdrag_val # <-- STORE THE RESULT

        if want_derived:
            state['derived'] = {}


    def get_Hubble(self):
        """Return H(z) computed in calculate"""
        return self.current_state['Hubble']

    def get_angular_diameter_distance(self):
        """Return angular diameter distance D_A(z)"""
        return self.current_state['angular_diameter_distance']   

    def get_luminosity_distance(self):
        """Return luminosity distance D_L(z)"""
        return self.current_state['luminosity_distance']
    
    def get_rdrag(self):
        """Return the comoving sound horizon at the drag epoch."""
        return self.current_state['rdrag']
