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
# from .numba_solver import NumbaOptimizedQuintessenceSolver as QuintessenceSolver
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.integrate.quad")

# Speed of light in km/s
_c_km_s = 299792.458
_T_CMB = 2.73 # CMB temperature in K

# Conversion factor from Gyr^-1 to km/s/Mpc
_Gyr_inv_to_kms_per_Mpc = 978.46

# Result collector
class Collector(NamedTuple):
    method: Callable
    args: list = []
    kwargs: dict = {}
    z_pool: PoolND | None = None
    post: Callable | None = None

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
    path: str
    omega_r = 9e-5  # Default radiation density if not provided
    solver: QuintessenceSolver

    def initialize(self):
        # Dict of named tuples to collect requirements and computation methods
        self.collectors = {}
        # Additional input parameters (e.g. to pass to setter function, to set as attr...)
        self._must_provide = {}
        # Flag to track if solver has been initialized
        self._solver_initialized = False

    def set_collector_with_z_pool(self, k, zs, method, args=(), kwargs=empty_dict, d=1):
        """
        Creates a collector for a z-dependent quantity, keeping track of the pool of z's.
        """
        # print(f"Setting collector for {k} with {len(zs)} z-values from {zs.min():.3f} to {zs.max():.3f}")
        
        # Convert to numpy array for easier manipulation
        zs = np.atleast_1d(zs)
        
        # Always create a fresh pool for this quantity with the provided z-values
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
        Store the provider for later use in calculate method.
        """
        # print("Provider in initialize_with_provider called")
        super().initialize_with_provider(provider)
        # Don't initialize solver here - do it in calculate method when parameters are available

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
        """
        Store requirements for later processing once solver is initialized
        """
        super().must_provide(**requirements)
        
        # Store requirements to process later when solver is available
        self._must_provide.update(requirements)

    def _setup_collectors(self):
        """
        Setup collectors once solver is initialized
        """
        # Collect ALL z-values needed by ANY quantity
        all_z_values = []
        for k, v in self._must_provide.items():
            if k in ["Hubble", "angular_diameter_distance", "luminosity_distance"]:
                z_requested = np.atleast_1d(v["z"])
                all_z_values.extend(z_requested)
        
        # Create a unified z-pool that includes ALL requested z-values
        if all_z_values:
            all_z_unique = np.unique(np.array(all_z_values))
            # print(f"Creating unified z-pool with z-values: {all_z_unique}")
            
            # Set up collectors using the unified z-pool
            for k, v in self._must_provide.items():
                if k == "Hubble":
                    self.set_collector_with_z_pool(k, all_z_unique, self._interpolated_H_of_z)
                elif k == "angular_diameter_distance":
                    self.set_collector_with_z_pool(k, all_z_unique, self._interpolated_angular_diameter_distance)
                elif k == "luminosity_distance":
                    self.set_collector_with_z_pool(k, all_z_unique, self._interpolated_luminosity_distance)

    def _interpolated_H_of_z(self, z):
        """Wrapper that handles interpolation for H(z)"""
        z = np.atleast_1d(z)
        result = []
        for z_i in z:
            result.append(self.solver.H_of_z(z_i))
        return np.array(result)

    def _interpolated_angular_diameter_distance(self, z):
        """Wrapper that handles interpolation for angular diameter distance"""
        z = np.atleast_1d(z)
        result = []
        for z_i in z:
            result.append(self.solver.angular_diameter_distance(z_i))
        return np.array(result)

    def _interpolated_luminosity_distance(self, z):
        """Wrapper that handles interpolation for luminosity distance"""
        z = np.atleast_1d(z)
        result = []
        for z_i in z:
            result.append(self.solver.luminosity_distance(z_i))
        return np.array(result)
    
    def get_can_provide(self):
        """List of quantities this theory can compute"""
        return ['Hubble', 'angular_diameter_distance', 'luminosity_distance']

    def get_can_provide_params(self):
        """List of derived parameters this theory can compute"""
        return ['rdrag','thetastar']

    def _initialize_solver(self, **params_values_dict):
        """
        Initialize the solver with current parameter values
        """
        # Get parameters from the current parameter dict
        ombh2 = params_values_dict['ombh2']
        omch2 = params_values_dict['omch2']
        H0 = params_values_dict['H0']
        omk = params_values_dict['omk']
        phi_init = params_values_dict['phi_init']
        phidot_init = params_values_dict['phidot_init']
        lambda_phi = params_values_dict['lambda_phi']

        # Calculate derived Omega_m
        omegam = (ombh2 + omch2) / (H0 / 100.0)**2


        # omegar*h^2 = 3 / (4 * 31500) 
        # omrh2 = (3 / (4 * 31500)) * (_T_CMB / 2.73)**4  # Radiation density parameter times h^2
        # omegar = omrh2 / (H0 / 100.0)**2 
        omegar = 9e-5  # Fixed radiation density parameter

        # CREATE THE SOLVER HERE
        self.solver = QuintessenceSolver(
            H0=H0,
            Omega_m=omegam,
            Omega_r=omegar,
            Omega_k=omk,
            phi_init=phi_init,
            phidot_init=phidot_init,
            V_base=Vphi,
            dV_base_dphi=dVphi_dphi,
            V_kwargs={'lambda_phi': lambda_phi},
            verbose=False
        )

        # Now setup collectors
        self._setup_collectors()
        self._solver_initialized = True
        return self.solver.success
        # print("Quintessence solver initialized in calculate method.")

    def calculate(self, state, want_derived=False, **params_values_dict):
        # Initialize solver on first call
        # if not self._solver_initialized:
        success = self._initialize_solver(**params_values_dict)
        
        if success:
            # Execute all the calculation methods stored in the collectors
            for k, collector in self.collectors.items():
                # print(f"Calculating {k} using {collector.method} with args={collector.args} and kwargs={collector.kwargs}")
                state[k] = collector.method(**collector.kwargs)
                # print("State key:", k, "Value:", state[k])
            
            # Calculate rdrag (which is z-independent)
            ombh2 = params_values_dict['ombh2']
            omch2 = params_values_dict['omch2']

            # R_baryon_photon = 31500 * ombh2
            # def cs_squared(z_prime):
            #     return (_c_km_s**2) / (3.0 * (1 + R_baryon_photon / (1 + z_prime)))
            # def rdrag_integrand(z_prime):
            #     # We need H(z') from the already-initialized solver
            #     H_z_prime_Gyr = float(np.atleast_1d(self.solver.H_of_z(z_prime))[0])
            #     H_z_prime_kmsMpc = H_z_prime_Gyr * (3.0857e19 / 3.1536e16)
            #     cs_z_prime = np.sqrt(cs_squared(z_prime))
            #     return cs_z_prime / H_z_prime_kmsMpc

            # rdrag_val, _ = quad(rdrag_integrand, 1060, 1e8, epsabs=1e-8, epsrel=1e-8)
            rdrag_val = 147.05* (ombh2/0.02236)**(-0.13) * ((ombh2+omch2)/0.1432)**(-0.23)  # in Mpc, for Neff = 3.04 # this needs to be corrected for D.E. evolution
            theta_star_rad = self.solver.compute_theta_star(ombh2, z_star=1089.8)
            state['derived']['thetastar'] = 100 * theta_star_rad
            # print(f"100thetastar: {100 * theta_star_rad}, thetastar = {theta_star_rad}")

            state['derived']['rdrag'] = rdrag_val
            # print('Calculated rdrag:', rdrag_val)
            return True
        else:
            return False

    def get_Hubble(self, z, units="km/s/Mpc", **kwargs):
        H_kms_mpc = self._get_z_dependent("Hubble", z)
        if units == "1/Mpc":
            return H_kms_mpc / _c_km_s
        elif units == "km/s/Mpc":
            return H_kms_mpc
        else:
            raise ValueError(f"Unknown units for Hubble: {units}")

    def get_angular_diameter_distance(self, z, **kwargs):
        return self._get_z_dependent("angular_diameter_distance", z)

    def get_luminosity_distance(self, z, **kwargs):
        return self._get_z_dependent("luminosity_distance", z)
    
    def get_rdrag(self):
        return self.current_state['rdrag']
    
    def get_thetastar(self):
        return self.current_state['thetastar']