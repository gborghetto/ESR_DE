# my_quintessence_theory.py

import numpy as np
from scipy.integrate import quad
from cobaya.theory import Theory
import cobaya
from .solver import QuintessenceSolver

# Speed of light in km/s
_c_km_s = 299792.458

def Vphi(phi, **kwargs):
    """
    Example potential function V(phi).
    This can be modified to implement different quintessence potentials.
    """
    # Example: simple quadratic potential
    m = kwargs.get('m', 1e-33)  # mass scale in eV
    return 0.5 * m**2 * phi**2

def dVphi_dphi(phi, **kwargs):
    """
    Derivative of the potential function V(phi).
    """
    m = kwargs.get('m', 1e-33)  # mass scale in eV
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
        """Create a pool to collect redshifts from all likelihoods."""
        self.z_pool = set()

    def initialize_with_provider(self, provider):
        """
        Called after all other components are initialized.
        We build the quintessence solver using sampled parameters.
        """
        print("provider in initialize_with_provider:", provider)
        super().initialize_with_provider(provider)
        # self.provider = provider

        # # Get the other parameters directly. Cobaya ensures they exist if they are in get_requirements.
        # ombh2 = self.provider.get_param('ombh2')
        # omch2 = self.provider.get_param('omch2')
        # omk_val = self.provider.get_param('omk')
        # phi_init_val = self.provider.get_param('phi_init')
        # phidot_init_val = self.provider.get_param('phidot_init')
        # m_val = self.provider.get_param('m')
        # H0_val = self.provider.get_param('H0')
        # omegam_val = (ombh2 + omch2) / (H0_val / 100.0)**2

         # Initialize the solver


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
        Collect all redshift arrays 'z' requested by likelihoods.
        """
        # It's still good practice to call the parent method first.
        super().must_provide(**requirements)

        # Look inside each requirement for a 'z' key.
        for req_name, req_options in requirements.items():
            if isinstance(req_options, dict) and 'z' in req_options:
                # Add the requested redshifts to our pool.
                self.z_pool.update(req_options['z'])

        # This method returns further requirements for OTHER components.
        # Since we have none, we return an empty dictionary.
        return {}
    
    # def must_provide(self, **requirements):

    #     super().must_provide(**requirements)

    #     # for k

    #     # ====================================================================
    #     # DEBUGGING BLOCK: This will print what likelihoods are asking for.
    #     print("\n" + "="*50)
    #     print(f"DEBUG: Inside must_provide for '{self.get_name()}'")
    #     print(f"DEBUG: Received requirement keys: {list(requirements.keys())}")
    #     # ====================================================================

    #     needs = {}
    #     # Define the quantities that require a redshift array 'z'.
    #     # We will check against the requirement keys printed above.
    #     z_dependent_quantities = {'Hubble', 'angular_diameter_distance', 'luminosity_distance', 'rdrag'}

    #     # Check if any of the requested quantities need 'z'.
    #     if any(q in requirements for q in z_dependent_quantities):
    #         print("DEBUG: Found a z-dependent quantity! Requesting 'z'.")
    #         needs['z'] = None
    #     else:
    #         print("DEBUG: NO z-dependent quantity found. Not requesting 'z'.")
        
    #     print(f"DEBUG: Returning needs dictionary: {needs}")
    #     print("="*50 + "\n")
        
    #     return needs

    # def must_provide(self, **requirements):
    #     """
    #     Declare that if a z-dependent quantity is requested, we need 'z'.
    #     """
    #     # This dictionary will store our conditional requirements.
    #     needs = {}
    #     # Define all the quantities that depend on redshift 'z'.
    #     z_dependent_quantities = {'H', 'angular_diameter_distance', 'luminosity_distance'}

    #     # Check if any of the requested quantities are in our z-dependent list.
    #     if any(q in requirements for q in z_dependent_quantities):
    #         needs['z'] = None
            
    #     return needs

    # def must_provide(self, **requirements):
    #     """
    #     Conditionally declare further requirements if specific observables are requested.
    #     Here, no extra conditional inputs beyond z.
    #     """
    #     provides = {}
    #     if 'Hubble' in requirements:
    #         provides['z'] = None
    #     if 'angular_diameter_distance' in requirements or 'luminosity_distance' in requirements:
    #         provides['z'] = None
    #     return provides

    def get_can_provide(self):
        """List of quantities this theory can compute"""
        return ['Hubble', 'angular_diameter_distance', 'luminosity_distance', 'rdrag']


    def calculate(self, state, want_derived=False, **params_values_dict):
        """
        Compute observables using the collected pool of redshifts.
        """
        # 1. Get the unique, sorted redshifts from the pool.
        z = np.sort(list(self.z_pool))

        # 2. IMPORTANT: Put the redshift array back into the state for the
        #    likelihoods to use.
        state['z'] = z
        
        # 3. Proceed with your calculations as before, using this 'z' array.
        ombh2 = self.provider.get_param('ombh2')
        # Get ombh2 from the provider
        ombh2 = self.provider.get_param('ombh2')
        H0 = self.provider.get_param('H0')
        omch2 = self.provider.get_param('omch2')
        omegam= (ombh2 + omch2) / (H0 / 100.0)**2
        omk = self.provider.get_param('omk')
        phi_init = self.provider.get_param('phi_init')
        phidot_init = self.provider.get_param('phidot_init')
        m = self.provider.get_param('m')

        self.solver = QuintessenceSolver(
            H0=H0,
            Omega_m=omegam,
            Omega_r=0.0,  # Using a default fixed value for radiation
            Omega_k=omk,
            phi_init=phi_init,
            phidot_init=phidot_init,
            V_base=Vphi,
            dV_base_dphi=dVphi_dphi,
            V_kwargs={'m': m},
            # Add other solver params like z_init, atol, etc. if needed
        )

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
