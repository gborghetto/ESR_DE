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

    # Declare any sampler parameters (none internal here)
    params = {}

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
                'H0': ['H0', 'h', 'H_0'],
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
                    except Exception:
                        pass
                return default
            try:
                return provider.get_param(name)
            except Exception:
                return default

        # redshifts sent by likelihoods will be handled dynamically
        H0_val = P('H0')
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
        Quantities always needed by this theory: the redshift array 'z' supplied by likelihoods.
        Plus any cosmological params and spline inputs.
        """
        return {
            'z': None,
            'H0': None,
            'Omega_m': None,
            'Omega_r': None,
            'Omega_k': None,
            'phi_init': None,
            'phidot_init': None,
            'nodes': None,
            'vals': None,
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
        return ['H', 'angular_diameter_distance', 'luminosity_distance']

    def calculate(self, state, want_derived=False, **params_values_dict):
        """
        Compute raw observables and store in state.
        Must fill state[...] and optionally state['derived'].
        """
        z = np.atleast_1d(state['z'])

        # solver returns H in Gyr^-1 (H_of_z) — convert to km/s/Mpc for Cobaya
        # H [km/s/Mpc] = H_Gyr * (1/Gyr -> s^-1) * (km/s to km/Mpc)
        # But simpler: use H0 provided in km/s/Mpc in solver.H0 and solver.H0_Gyr conversion
        # We'll get H(z) from solver in Gyr^-1 and convert using:
        # 1 Gyr = 3.1536e16 s, 1 Mpc = 3.0857e19 km
        sec_per_Gyr = 3.1536e16
        km_per_Mpc = 3.0857e19
        H_Gyr = self.solver.H_of_z(z)
        H_km_s_Mpc = H_Gyr * (1.0 / sec_per_Gyr) * km_per_Mpc

        # Comoving distance chi(z) = integral_0^z c / H(z') dz'
        # Use c in km/s and H in km/s/Mpc so chi in Mpc
        def inv_H_Mpc(zp):
            Hp = float(np.atleast_1d(self.solver.H_of_z(zp))[0] * (1.0 / sec_per_Gyr) * km_per_Mpc)
            return _c_km_s / Hp

        chi = np.array([quad(inv_H_Mpc, 0.0, float(zi))[0] for zi in z])
        DA = chi / (1.0 + z)
        DL = chi * (1.0 + z)

        state['H'] = H_km_s_Mpc
        state['angular_diameter_distance'] = DA 
        state['luminosity_distance'] = DL

        if want_derived:
            state['derived'] = {}

    def get_H(self):
        """Return H(z) computed in calculate"""
        return self.current_state['H']

    def get_angular_diameter_distance(self):
        """Return angular diameter distance D_A(z)"""
        return self.current_state['angular_diameter_distance']   

    def get_luminosity_distance(self):
        """Return luminosity distance D_L(z)"""
        return self.current_state['luminosity_distance']