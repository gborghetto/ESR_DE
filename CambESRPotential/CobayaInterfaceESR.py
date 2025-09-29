import numpy as np
from cobaya.theory import Theory
from cobaya.log import LoggedError
import camb

# Imports to handle z-dependent quantities correctly, as in the old code
from typing import NamedTuple, Callable
from cobaya.tools import Pool1D
from cobaya.typing import empty_dict
from scipy.interpolate import InterpolatedUnivariateSpline

from .esr_utils import load_esr_function_string, create_potential_table

# Speed of light in km/s
_c_km_s = 299792.458
_T_CMB = 2.73 # CMB temperature in K


# Collector class to manage calculation requirements, based on cobaya
class Collector(NamedTuple):
    method: Callable
    args: list = []
    kwargs: dict = {}
    z_pool: Pool1D | None = None
    post: Callable | None = None

class CambQuintessenceESR(Theory):
    """
    A standalone Cobaya Theory class that uses CAMB as an internal engine
    to solve for a quintessence model with a tabulated potential.

    This version includes robust handling of z-dependent quantities by
    pre-calculating observables over the full range of redshifts requested
    by all likelihoods, mirroring the logic of the old theory interface.
    """
    # Parameters for generating the potential tables. Can be overridden in the YAML.
    phi_min: float = -2.0
    phi_max: float = 2.0
    phi_steps: int = 200
    path_ESR_data: str = './esrfunctions/core_maths/compl_4/unique_equations_4.txt'  # Path to the ESR functions data file
    potential_function_idx: int = 0  # Index of the potential function in the ESR data file
    esr_param_symbols = []

    def initialize(self):
        """Sets up the collectors for theory requirements."""
        self._must_provide = {}
        self.collectors = {}
        function_dict = load_esr_function_string(self.path_ESR_data, self.potential_function_idx)
        if not function_dict['valid']:
            raise LoggedError(self.log, f"ESR function at index {self.potential_function_idx} is invalid.")
        self.esr_param_symbols = function_dict['param_symbols']
        self.esr_param_names = [str(p) for p in function_dict['param_symbols']]
        self.esr_function_string = function_dict['func_string']
        self.esr_function_template = function_dict['expr_template']
        self.phi_vals = np.linspace(self.phi_min, self.phi_max, self.phi_steps)
        print(f"Using ESR potential function: {self.esr_function_string}, with parameters: {self.esr_param_names}")

    def must_provide(self, **requirements):
        """
        Collects the requirements from all likelihoods (e.g., z values).
        This method is called by Cobaya for each component that needs something
        from this theory.
        """
        for k, v in requirements.items():
            if k not in self._must_provide:
                self._must_provide[k] = v
            # For z-dependent quantities, we merge the z arrays to create a unified pool
            elif isinstance(v, dict) and 'z' in v:
                current_z = np.atleast_1d(self._must_provide[k].get('z', []))
                new_z = np.atleast_1d(v['z'])
                self._must_provide[k]['z'] = np.union1d(current_z, new_z)
            else:
                 self._must_provide[k] = v

    def get_requirements(self):
        """Specifies the cosmological parameters this theory needs from the sampler + additional potential params required by the ESR potential."""
        reqs = {'omch2': None, 'ombh2': None, 'H0': None, 'omk': None, 'mnu': None,}
        if self.esr_param_names:
            potential_params = dict(zip(self.esr_param_names, [None]*len(self.esr_param_names)))
            reqs.update(potential_params)
        return reqs

    def get_can_provide(self):
        """Specifies the observables this theory can provide to likelihoods."""
        return ['Hubble', 'angular_diameter_distance', 'luminosity_distance']

    def get_can_provide_params(self):
        """Specifies the derived parameters this theory can provide."""
        return ['rdrag', 'thetastar']

    def _setup_collectors(self, camb_results):
        """
        Setup collectors once the camb_results object is available.
        This defines *what* to calculate based on the likelihood requirements.
        """
        all_z_values = []
        for k, v in self._must_provide.items():
            if k in ["Hubble", "angular_diameter_distance", "luminosity_distance"]:
                if isinstance(v, dict) and 'z' in v:
                    z_requested = np.atleast_1d(v["z"])
                    all_z_values.extend(z_requested)
        
        if all_z_values:
            all_z_unique = np.unique(np.array(all_z_values))
            
            # Map required quantities to their calculation methods from the CAMB results object
            method_map = {
                "Hubble": camb_results.hubble_parameter,
                "angular_diameter_distance": camb_results.angular_diameter_distance,
                "luminosity_distance": camb_results.luminosity_distance
            }
            
            for k, method in method_map.items():
                if k in self._must_provide:
                    self.set_collector_with_z_pool(k, all_z_unique, method)

    def set_collector_with_z_pool(self, k, zs, method, args=(), kwargs=empty_dict):
        """
        Helper function from the old code to create a collector for a z-dependent quantity.
        """
        zs = np.atleast_1d(zs)
        z_pool = Pool1D(zs)
        kwargs_with_z = {"z": z_pool.values}
        kwargs_with_z.update(kwargs)
        self.collectors[k] = Collector(method=method, z_pool=z_pool, kwargs=kwargs_with_z, args=args)

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        The main calculation method. This is called for each point in the parameter space.
        """
        try:
            #extract esr potential parameters from params_values_dict
            esr_params = []
            # print(f"Parameters values dict: {params_values_dict}")
            for p in self.esr_param_names:
                if p not in params_values_dict:
                    raise LoggedError(f"Missing potential parameter '{p}' required by ESR potential.")
                esr_params.append(params_values_dict[p])
            potential_dict = create_potential_table(self.esr_function_template, 
                                                    self.esr_param_symbols,
                                                    esr_params, self.phi_vals)
            success = potential_dict['success']
            if not success:
                self.log.error("Failed to create a valid potential table from ESR function for the given esr parameters")
                return False
            
            phi_train = potential_dict['phi_train']
            V_train = potential_dict['V_train']
            dV_train = potential_dict['dV_train']
            ddV_train = potential_dict['ddV_train']

            # print("V_train:", V_train)
            # print("dV_train:", dV_train)
            # print("ddV_train:", ddV_train)
                                                                               
            pars = camb.set_params(
                H0=params_values_dict['H0'], ombh2=params_values_dict['ombh2'], omch2=params_values_dict['omch2'],
                mnu=params_values_dict.get('mnu', 0.06), omk=params_values_dict.get('omk', 0.0),
                dark_energy_model='QuintessenceInterp', V_train=V_train,
                dV_train=dV_train, ddV_train=ddV_train, phi_train=phi_train)

            # 3. Perform the CAMB calculation
            results = camb.get_background(pars)
            
            # 4. Setup collectors based on requirements and the CAMB results object
            self._setup_collectors(results)

            # 5. Execute collectors to pre-calculate all z-dependent quantities
            for k, collector in self.collectors.items():
                state[k] = collector.method(*collector.args, **collector.kwargs)
                if collector.z_pool:
                    state[k + '_z_pool'] = collector.z_pool

            # 6. Store derived parameters
            if want_derived:
                derived_params = results.get_derived_params()
                state['derived'] = {
                    'rdrag': derived_params.get('rdrag'),
                    'thetastar': 100 * results.cosmomc_theta()}
        except Exception as e:
            self.log.error(f"CAMB calculation failed with error: {e}")
            return False
        return True
    
    def _get_z_dependent(self, quantity, z):
        """
        Helper function to retrieve pre-computed results for specific redshifts.
        This uses the exact logic from the old theory interface.
        """
        pool = self.current_state.get(quantity + '_z_pool')
        if pool is None:
             raise LoggedError(
                self.log, f"Could not find z_pool for quantity '{quantity}'. "
                         "Was it requested by any likelihood?")
        try:
            i_kwarg_z = pool.find_indices(z)
        except ValueError:
            raise LoggedError(
                self.log, f"{quantity} not computed for all z requested. "
                f"Requested z are {z}, but computed ones are {pool.values}.")
        return np.array(self.current_state[quantity], copy=True)[i_kwarg_z]

    # Getter methods for likelihoods
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
        return self.current_state['derived']['rdrag']

    def get_thetastar(self):
        return self.current_state['derived']['thetastar']