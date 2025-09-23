# my_quintessence_theory.py
from typing import Any, NamedTuple
from collections.abc import Callable
import numpy as np
from scipy.integrate import quad
from cobaya.tools import (
    Pool1D,
    Pool2D,
    PoolND,
    VersionCheckError,
    check_module_version,
    get_class_methods,
    get_properties,
    getfullargspec,
    str_to_list,
)
from cobaya.typing import InfoDict, empty_dict
from cobaya.log import LoggedError, abstract, get_logger
from cobaya.theory import Theory
from cobaya.tools import check_2d, combine_1d, combine_2d, deepcopy_where_possible
import cobaya
from .solver import QuintessenceSolver


# Result collector
class Collector(NamedTuple):
    method: Callable
    args: list = []
    kwargs: dict = {}
    z_pool: PoolND | None = None
    post: Callable | None = None

# Speed of light in km/s
_c_km_s = 299792.458

def Vphi(phi, **kwargs):
    """
    Example potential function V(phi).
    This can be modified to implement different quintessence potentials.
    """
    # Example: simple quadratic potential
    lambda_phi = kwargs.get('lambda_phi', 1.)
    return np.exp(-lambda_phi * phi)

def dVphi_dphi(phi, **kwargs):
    """
    Derivative of the potential function V(phi).
    """
    lambda_phi = kwargs.get('lambda_phi', 1.)
    return -lambda_phi * Vphi(phi, **kwargs)

class QuintessenceTheory(Theory):
    """
    Cobaya Theory module wrapping a Quintessence solver to provide:
      - H(z)
      - comoving_AngularDistance(z)
      - luminosity_distance(z)
    """
    _must_provide: dict
    path: str
    omega_r = 9e-5  # Default radiation density if not provided
    solver: QuintessenceSolver

    def initialize(self):
        # Dict of named tuples to collect requirements and computation methods
        self.collectors = {}
        # Additional input parameters (e.g. to pass to setter function, to set as attr...)
        self._must_provide = {}

    def set_collector_with_z_pool(self, k, zs, method, args=(), kwargs=empty_dict, d=1):
        """
        Creates a collector for a z-dependent quantity, keeping track of the pool of z's.
        """
        print(f"Setting collector for {k} with zs={zs}")
        if k in self.collectors:
            z_pool = self.collectors[k].z_pool
            z_pool.update(zs)
        else:
            Pool = {1: Pool1D, 2: Pool2D}[d]
            z_pool = Pool(zs)
        if d == 1:
            kwargs_with_z = {"z": z_pool.values}
        else:
            kwargs_with_z = {
                "z1": np.array(z_pool.values[:, 0]),
                "z2": np.array(z_pool.values[:, 1]),
            }
        kwargs_with_z.update(kwargs)
        self.collectors[k] = Collector(
            method=method, z_pool=z_pool, kwargs=kwargs_with_z, args=args
        )

    def _get_z_dependent(self, quantity, z, pool=None):
        if pool is None:
            pool = self.collectors[quantity].z_pool
        try:
            i_kwarg_z = pool.find_indices(z)
        except ValueError:
            raise LoggedError(
                self.log,
                f"{quantity} not computed for all z requested. "
                f"Requested z are {z}, but computed ones are "
                f"{pool.values}.",
            )
        return np.array(self.current_state[quantity], copy=True)[i_kwarg_z]

    def _get_z_pair_dependent(self, quantity, z_pairs, inv_value=0):
        """
        ``inv_value`` (default=0) is assigned to pairs for which ``z1 > z2``.
        """
        try:
            check_2d(z_pairs, allow_1d=False)
        except ValueError:
            raise LoggedError(
                self.log,
                f"z_pairs={z_pairs} not correctly formatted for "
                f"{quantity}. It should be a list of pairs.",
            )
        # Only recover for correctly sorted pairs
        z_pairs_arr = np.array(z_pairs)
        i_right = z_pairs_arr[:, 0] <= z_pairs_arr[:, 1]
        pool = self.collectors[quantity].z_pool
        try:
            i_z_pair = pool.find_indices(z_pairs_arr[i_right])
        except ValueError:
            raise LoggedError(
                self.log,
                f"{quantity} not computed for all z pairs requested. "
                f"Requested z are {z_pairs}, but computed ones are "
                f"{pool.values}.",
            )
        result = np.full(len(z_pairs), inv_value, dtype=float)
        result[i_right] = np.array(self.current_state[quantity], copy=True)[i_z_pair]
        return result

    def initialize_with_provider(self, provider):
        """
        Called after all other components are initialized.
        We build the quintessence solver using sampled parameters.
        """
        self.provider = provider

        # Get all necessary parameters directly from the provider.
        # This will now work because the super() call is removed.
        ombh2 = self.provider.get_param('ombh2')
        omch2 = self.provider.get_param('omch2')
        H0 = self.provider.get_param('H0')
        omk = self.provider.get_param('omk')
        phi_init = self.provider.get_param('phi_init')
        phidot_init = self.provider.get_param('phidot_init')
        lambda_phi = self.provider.get_param('lambda_phi')

        # Calculate derived Omega_m
        omegam = (ombh2 + omch2) / (H0 / 100.0)**2

        # Create the solver using the retrieved parameters
        self.solver = QuintessenceSolver(
            H0=H0,
            Omega_m=omegam,
            Omega_r=0.0,
            Omega_k=omk,
            phi_init=phi_init,
            phidot_init=phidot_init,
            V_base=Vphi,
            dV_base_dphi=dVphi_dphi,
            V_kwargs={'lambda_phi': lambda_phi},
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
            'lambda_phi': None,
            'omch2': None,  
            'ombh2': None, 
        }
    
    def must_provide(self, **requirements):
        super().must_provide(**requirements)

        # Since self.solver now exists, this loop will work correctly
        for k, v in self._must_provide.items():
            if k == "Hubble":
                # Note: Your solver must have a method named H_of_z
                self.set_collector_with_z_pool(k, v["z"], self.solver.H_of_z)
            elif k == "angular_diameter_distance":
                # Note: Your solver must have a method named angular_diameter_distance
                self.set_collector_with_z_pool(k, v["z"], self.solver.angular_diameter_distance)
            elif k == "luminosity_distance":
                # Note: Your solver must have a method named luminosity_distance
                 self.set_collector_with_z_pool(k, v["z"], self.solver.luminosity_distance)
            # You don't need to handle rdrag here as it's not z-dependent
    
    def get_can_provide(self):
        """List of quantities this theory can compute"""
        return ['Hubble', 'angular_diameter_distance', 'luminosity_distance', 'rdrag']


    def calculate(self, state, want_derived=False, **params_values_dict):
        # Execute all the calculation methods stored in the collectors
        for k, collector in self.collectors.items():
            state[k] = collector.method(**collector.kwargs)
        
        # Calculate rdrag (which is z-independent)
        ombh2 = self.provider.get_param('ombh2')
        R_baryon_photon = 31500 * ombh2
        def cs_squared(z_prime):
            return (_c_km_s**2) / (3.0 * (1 + R_baryon_photon / (1 + z_prime)))
        def rdrag_integrand(z_prime):
            # We need H(z') from the already-initialized solver
            H_z_prime_Gyr = float(np.atleast_1d(self.solver.H_of_z(z_prime))[0])
            H_z_prime_kmsMpc = H_z_prime_Gyr * (3.0857e19 / 3.1536e16)
            cs_z_prime = np.sqrt(cs_squared(z_prime))
            return cs_z_prime / H_z_prime_kmsMpc
        
        rdrag_val, _ = quad(rdrag_integrand, 1060, np.inf)
        state['rdrag'] = rdrag_val


    def get_Hubble(self, z, **kwargs):
        return self._get_z_dependent("Hubble", z)

    def get_angular_diameter_distance(self, z, **kwargs):
        return self._get_z_dependent("angular_diameter_distance", z)

    def get_luminosity_distance(self, z, **kwargs):
        return self._get_z_dependent("luminosity_distance", z)
    
    def get_rdrag(self):
        return self.current_state['rdrag']

    # def calculate(self, state, want_derived=False, **params_values_dict):
    #     """
    #     Compute observables using the collected pool of redshifts.
    #     """
    #     print("CALCULATE method called with state keys:", state.keys())

    #     # # 1. Get the unique, sorted redshifts from the pool.
    #     # z = np.sort(list(self.z_pool))

    #     for key in state:
    #         print(f"State key: {key}, Value: {state[key]}")
        
    #     # 3. Proceed with your calculations as before, using this 'z' array.
    #     ombh2 = self.provider.get_param('ombh2')
    #     # Get ombh2 from the provider
    #     ombh2 = self.provider.get_param('ombh2')
    #     H0 = self.provider.get_param('H0')
    #     omch2 = self.provider.get_param('omch2')
    #     omegam= (ombh2 + omch2) / (H0 / 100.0)**2
    #     omk = self.provider.get_param('omk')
    #     phi_init = self.provider.get_param('phi_init')
    #     phidot_init = self.provider.get_param('phidot_init')
    #     lambda_phi = self.provider.get_param('lambda_phi')

    #     self.solver = QuintessenceSolver(
    #         H0=H0,
    #         Omega_m=omegam,
    #         Omega_r=0.0,  # Using a default fixed value for radiation
    #         Omega_k=omk,
    #         phi_init=phi_init,
    #         phidot_init=phidot_init,
    #         V_base=Vphi,
    #         dV_base_dphi=dVphi_dphi,
    #         V_kwargs={'lambda_phi': lambda_phi},
    #         # Add other solver params like z_init, atol, etc. if needed
    #     )


    #     z_Hubble = self.collectors.get('Hubble').z_pool.values
    #     z_DA = self.collectors.get('angular_diameter_distance').z_pool.values
    #     z_DL = self.collectors.get('luminosity_distance').z_pool.values


    #     # --- Calculate H(z) and Distances (your existing code) ---
    #     sec_per_Gyr = 3.1536e16
    #     km_per_Mpc = 3.0857e19
    #     H_Gyr = self.solver.H_of_z(z_Hubble)
    #     H_km_s_Mpc = H_Gyr * (1.0 / sec_per_Gyr) * km_per_Mpc

    #     def inv_H_Mpc(zp):
    #         Hp_Gyr = float(np.atleast_1d(self.solver.H_of_z(zp))[0])
    #         Hp = Hp_Gyr * (1.0 / sec_per_Gyr) * km_per_Mpc
    #         return _c_km_s / Hp

    #     chi_DA = np.array([quad(inv_H_Mpc, 0.0, float(zi))[0] for zi in z_DA])
    #     DA = chi_DA / (1.0 + z_DA)
    #     chi_DL = np.array([quad(inv_H_Mpc, 0.0, float(zi))[0] for zi in z_DL])
    #     DL = chi_DL * (1.0 + z_DL)

    #     # --- NEW: Calculate r_drag ---
    #     z_drag = 1060  # Approximate drag epoch redshift

    #     # Sound speed squared c_s^2(z) in (km/s)^2, using Eq. 3.5 & 3.6
    #     # The factor 31500 approximates 3*rho_b0 / (4*rho_g0 * ombh2)
    #     R_baryon_photon = 31500 * ombh2 
    #     def cs_squared(z_prime):
    #         return (_c_km_s**2) / (3.0 * (1 + R_baryon_photon / (1 + z_prime)))

    #     # Integrand for the sound horizon: c_s(z) / H(z)
    #     def rdrag_integrand(z_prime):
    #         # We need H(z') in km/s/Mpc, calculated from the solver
    #         H_z_prime_Gyr = float(np.atleast_1d(self.solver.H_of_z(z_prime))[0])
    #         H_z_prime = H_z_prime_Gyr * (1.0 / sec_per_Gyr) * km_per_Mpc
    #         cs_z_prime = np.sqrt(cs_squared(z_prime))
    #         return cs_z_prime / H_z_prime

    #     # Perform the integration from z_drag to infinity (Eq. 3.4)
    #     rdrag_val, _ = quad(rdrag_integrand, z_drag, np.inf)

    #     # --- Store all results in the state dictionary ---
    #     state['Hubble'] = H_km_s_Mpc
    #     state['angular_diameter_distance'] = DA
    #     state['luminosity_distance'] = DL
    #     state['derived']['rdrag'] = rdrag_val # <-- STORE THE RESULT

    #     # if want_derived:
    #     #     state['derived'] = {}

    # def get_Hubble(self,z,units="km/s/Mpc"):
    #     """Return H(z) computed in calculate"""
    #     return self.current_state['Hubble']

    # def get_angular_diameter_distance(self,z):
    #     """Return angular diameter distance D_A(z)"""
    #     # print("Requested z in get_angular_diameter_distance:", z)
    #     res = self.current_state['angular_diameter_distance']
    #     # print(f"result in get_angular_diameter_distance: {res}")
    #     return res

    # def get_luminosity_distance(self):
    #     """Return luminosity distance D_L(z)"""
    #     return self.current_state['luminosity_distance']
    
    # def get_rdrag(self):
    #     """Return the comoving sound horizon at the drag epoch."""
    #     # print("Requested rdrag in get_rdrag")
    #     return self.current_state['derived']['rdrag']
