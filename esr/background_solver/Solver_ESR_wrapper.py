import numpy as np
import sympy
import warnings
from scipy.interpolate import UnivariateSpline
from esr.fitting.sympy_symbols import (
    square, cube, sqrt, log, pow, x, a0, a1, a2, inv
)
import matplotlib.pyplot as plt


class QuintessenceESRBridge:
    """
    Bridge class to connect ESR-generated potentials with QuintessenceSolver
    and DESI likelihood calculations.
    """
    
    def __init__(self, 
                 H0=67.4,              # km/s/Mpc
                 Omega_m=0.315,        # Total matter density
                 Omega_r=9.1e-5,       # Radiation density  
                 Omega_k=0.0,          # Curvature density
                 phi_init=0.,         # Initial field value
                 phidot_init=0.,     # Initial field velocity
                 z_init=100,          # Starting redshift for integration
                 phi_range=(-2, 2),    # Range for potential evaluation
                 n_phi_points=50,     # Points for potential interpolation
                 z_max=3.0,            # Max redshift for H(z) output
                 n_z_points=50,      # Points for H(z) output
                 verbose=False):
        
        # Store cosmological parameters
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_r = Omega_r
        self.Omega_k = Omega_k
        
        # Initial conditions for quintessence field
        self.phi_init = phi_init
        self.phidot_init = phidot_init
        self.z_init = z_init
        
        # Potential evaluation parameters
        self.phi_range = phi_range
        self.n_phi_points = n_phi_points
        self.phi_nodes = np.linspace(phi_range[0], phi_range[1], n_phi_points)
        
        # Output parameters
        self.z_max = z_max
        self.n_z_points = n_z_points
        self.z_output = np.linspace(0, z_max, n_z_points)
        
        self.verbose = verbose
        
        # Import here to avoid circular imports
        try:
            from esr.background_solver.Quintessence import QuintessenceSolver
            self.QuintessenceSolver = QuintessenceSolver
        except ImportError:
            try:
                # Try different import path
                from .Quintessence import QuintessenceSolver
                self.QuintessenceSolver = QuintessenceSolver
            except ImportError:
                raise ImportError("Could not import QuintessenceSolver class")
    
    def esr_function_to_potential_functions(self, fcn_str, params):
        """
        Convert ESR function string and parameters to potential and derivative functions
        
        Args:
            fcn_str: String representation of potential V(φ) from ESR
            params: Array of parameter values
            
        Returns:
            tuple: (V_base_func, dV_base_dphi_func) - callable functions for potential and derivative
        """
        try:
            # Clean the function string
            fcn_str = fcn_str.replace('\n', '').replace('\'', '')
            
            # Create sympy expression
            eq = sympy.sympify(fcn_str, locals={
                "inv": inv, "square": square, "cube": cube, "sqrt": sqrt,
                "log": log, "pow": pow, "x": x, "a0": a0, "a1": a1, "a2": a2
            })
            
            # Create derivative
            dV_dphi_eq = sympy.diff(eq, x)
            
            # Create numerical functions
            if len(params) == 0:
                V_base_func = sympy.lambdify([x], eq, modules=["numpy"])
                dV_base_dphi_func = sympy.lambdify([x], dV_dphi_eq, modules=["numpy"])
            else:
                # Create parameter symbols
                param_symbols = [sympy.Symbol(f'a{i}') for i in range(len(params))]
                
                # Create wrapper functions that include the parameters
                def V_wrapper(phi):
                    V_func = sympy.lambdify([x] + param_symbols, eq, modules=["numpy"])
                    return V_func(phi, *params)
                
                def dV_wrapper(phi):
                    dV_func = sympy.lambdify([x] + param_symbols, dV_dphi_eq, modules=["numpy"])
                    return dV_func(phi, *params)
                    
                V_base_func = V_wrapper
                dV_base_dphi_func = dV_wrapper
            
            # Test the functions at a sample point to ensure they work
            test_phi = 1.0
            try:
                V_test = V_base_func(test_phi)
                dV_test = dV_base_dphi_func(test_phi)
                
                if not (np.isfinite(V_test) and np.isfinite(dV_test)):
                    if self.verbose:
                        print(f"Warning: Non-finite potential or derivative for function {fcn_str}")
                    return None, None
                    
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Error testing functions for {fcn_str}: {e}")
                return None, None
            
            return V_base_func, dV_base_dphi_func
            
        except Exception as e:
            if self.verbose:
                print(f"Error processing ESR function {fcn_str}: {e}")
            return None, None
    
    def solve_background(self, fcn_str, params):
        """
        Solve quintessence background evolution for given ESR function
        
        Args:
            fcn_str: String representation of potential V(φ)
            params: Array of parameter values
            
        Returns:
            tuple: (z_array, H_z_array) where H_z is in km/s/Mpc, or (None, None) if failed
        """
 
        try:
            
            # Convert ESR function to shape functions
            V_base_func, dV_base_dphi_func = self.esr_function_to_potential_functions(
                fcn_str, params
            )
            
            if V_base_func is None:
                return None, None
            
            # Create solver with manual amplitude (no auto-tuning)
            solver = self.QuintessenceSolver(
                H0=self.H0,
                Omega_m=self.Omega_m,
                Omega_r=self.Omega_r,
                Omega_k=self.Omega_k,
                phi_init=self.phi_init,
                phidot_init=self.phidot_init,
                V_base=V_base_func,
                dV_base_dphi=dV_base_dphi_func,
                z_init=self.z_init,
                fixed_amplitude=1e-8,  # Fixed amplitude
                verbose=self.verbose
            )
            
            # Get H(z) and current Ω_DE for constraint
            z_array = np.linspace(0, self.z_max, self.n_z_points)
            H_z_array = np.array([solver.H_of_z(z) * solver.H0 / solver.H0_Gyr for z in z_array])
            
            return z_array, H_z_array
            
        except Exception as e:
            if self.verbose:
                print(f"Error solving background: {e}")
            return None, None
    
    def create_background_solver_function(self):
        """
        Create a function compatible with DESILikelihood.background_solver
        
        Returns:
            function: Background solver function that takes parameters and returns (z, H)
        """
        def background_solver_func(params):
            # This function will be called by the likelihood with the current function string
            # stored in the likelihood object
            if hasattr(self, '_current_function_str'):
                return self.solve_background(self._current_function_str, params)
            else:
                return None, None
        
        return background_solver_func
    
    def set_current_function(self, fcn_str):
        """
        Set the current ESR function string for background solving
        
        Args:
            fcn_str: String representation of the potential
        """
        self._current_function_str = fcn_str


class QuintessenceDESILikelihood:
    """
    Combined likelihood class that integrates ESR, QuintessenceSolver, and DESI BAO data
    """
    
    def __init__(self, 
                 data_dir=None,
                 fn_set='core_maths',
                 cosmology_params=None,
                 quintessence_params=None,
                 verbose=False):
        
        # Set default cosmological parameters
        default_cosmo = {
            'H0': 67.4,
            'Omega_m': 0.315, 
            'Omega_r': 9.1e-5,
            'Omega_k': 0.0
        }
        
        # Set default quintessence parameters
        default_quint = {
            'phi_init': 0.1,
            'phidot_init': 0.01,
            'z_init': 1000,
            'phi_range': (-5, 5),
            'n_phi_points': 100
        }
        
        if cosmology_params:
            default_cosmo.update(cosmology_params)
        if quintessence_params:
            default_quint.update(quintessence_params)
        
        # Create the bridge
        self.bridge = QuintessenceESRBridge(
            verbose=verbose,
            **default_cosmo,
            **default_quint
        )
        
        # Import and create DESI likelihood
        try:
            from esr.fitting.likelihood import DESILikelihood
        except ImportError:
            raise ImportError("Could not import DESILikelihood class")
        
        # Create DESI likelihood with our background solver
        self.desi_likelihood = DESILikelihood(
            fn_set=fn_set,
            data_dir=data_dir,
            background_solver=self.bridge.create_background_solver_function()
        )
        
        # Copy essential attributes for ESR compatibility
        self.data_dir = self.desi_likelihood.data_dir
        self.fn_dir = self.desi_likelihood.fn_dir
        self.like_dir = self.desi_likelihood.like_dir
        self.base_out_dir = self.desi_likelihood.base_out_dir
        self.temp_dir = self.desi_likelihood.temp_dir
        self.out_dir = self.desi_likelihood.out_dir
        self.fig_dir = self.desi_likelihood.fig_dir
        
        # ESR-required attributes
        self.fnprior_prefix = self.desi_likelihood.fnprior_prefix
        self.combineDL_prefix = self.desi_likelihood.combineDL_prefix
        self.final_prefix = self.desi_likelihood.final_prefix
        self.is_mse = self.desi_likelihood.is_mse
        
        # Data attributes
        self.yvar = self.desi_likelihood.yvar
        self.xvar = self.desi_likelihood.z_eff
        self.xlabel = r'Redshift $z$'
        self.ylabel = self.desi_likelihood.ylabel
        self.inv_cov = self.desi_likelihood.inv_cov

        if hasattr(self.desi_likelihood, 'yerr'):
            self.yerr = self.desi_likelihood.yerr
        
        self.verbose = verbose
    
    def get_pred(self, zp1, a, eq_numpy, integrated=False, **kwargs):
        """ESR-compatible prediction method"""
        return self.desi_likelihood.get_pred(zp1, a, eq_numpy, integrated, **kwargs)
    
    def negloglike(self, a, eq_numpy, integrated=False, **kwargs):
        """ESR-compatible likelihood method"""
        return self.desi_likelihood.negloglike(a, eq_numpy, integrated, **kwargs)
    
    def run_sympify(self, fcn_i, tmax=5, try_integration=False):
        """
        Process ESR function and set up for background solving
        
        Args:
            fcn_i: String representation of potential V(φ)
            tmax: Not used
            try_integration: Not used
            
        Returns:
            tuple: (cleaned_string, sympy_expression, False)
        """
        # Clean the function string
        fcn_i = fcn_i.replace('\n', '').replace('\'', '')
        
        # Set the current function in the bridge
        self.bridge.set_current_function(fcn_i)
        
        # Create sympy expression for compatibility
        try:
            eq = sympy.sympify(fcn_i, locals={
                "inv": inv, "square": square, "cube": cube, "sqrt": sqrt,
                "log": log, "pow": pow, "x": x, "a0": a0, "a1": a1, "a2": a2
            })
        except Exception:
            eq = x  # Fallback to simple expression
        
        return fcn_i, eq, False
    
    def clear_data(self):
        """Clear cached data"""
        self.desi_likelihood.clear_data()
    
    def get_data_info(self):
        """Get information about loaded data"""
        return self.desi_likelihood.get_data_info()
    
    def test_function(self, fcn_str, params):
        """
        Test a specific function with given parameters
        
        Args:
            fcn_str: String representation of potential
            params: Array of parameter values
            
        Returns:
            dict: Test results including likelihood value
        """
        results = {
            'function': fcn_str,
            'parameters': params,
            'success': False,
            'likelihood': np.inf,
            'error': None
        }
        
        try:
            # Set up function
            self.bridge.set_current_function(fcn_str)
            
            # Compute likelihood
            likelihood_val = self.negloglike(params, None)
            
            results['likelihood'] = likelihood_val
            results['success'] = np.isfinite(likelihood_val)
            
            if self.verbose:
                print(f"Function: {fcn_str}")
                print(f"Parameters: {params}")
                print(f"Likelihood: {likelihood_val}")
                print(f"Success: {results['success']}")
            
        except Exception as e:
            results['error'] = str(e)
            if self.verbose:
                print(f"Error testing function {fcn_str}: {e}")
        
        return results
    
    def plot_best_function(self, fcn_str, params, save_path=None, show_plot=True):
        """
        Plot the best-fit quintessence model
        
        Args:
            fcn_str: Function string of best model
            params: Best-fit parameters
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            QuintessenceSolver instance and matplotlib figure
        """
        try:
            # Set up the function
            self.bridge.set_current_function(fcn_str)
            
            # Get the solver for this function
            if len(params) == 0:
                return None, None
                
            #amplitude = abs(params[0])
            #shape_params = params[1:] if len(params) > 1 else []
            
            # Get potential functions
            V_base_func, dV_base_dphi_func = self.bridge.esr_function_to_potential_functions(
                fcn_str, shape_params
            )
            
            if V_base_func is None:
                return None, None
            
            # Create solver
            solver = self.bridge.QuintessenceSolver(
                H0=self.bridge.H0,
                Omega_m=self.bridge.Omega_m,
                Omega_r=self.bridge.Omega_r,
                Omega_k=self.bridge.Omega_k,
                phi_init=self.bridge.phi_init,
                phidot_init=self.bridge.phidot_init,
                V_base=V_base_func,
                dV_base_dphi=dV_base_dphi_func,
                z_init=self.bridge.z_init,
                fixed_amplitude=1e-8,
                verbose=False
            )
            
            # Create plots
            fig, axes = solver.plot_all()
            
            # Add title with function info
            fig.suptitle(f'Best Quintessence Model: {fcn_str}\n'
                        f'Parameters: {params}\n'
                        f'Ω_DE(z=0) = {solver.compute_current_Omega_DE():.3f}', 
                        fontsize=14, y=0.98)
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            
            if show_plot:
                plt.show()
            
            return solver, fig
            
        except Exception as e:
            print(f"Error plotting function {fcn_str}: {e}")
            return None, None


# # Example usage functions
# def create_quintessence_likelihood(data_dir=None, verbose=False):
#     """
#     Convenience function to create a QuintessenceDESILikelihood instance
    
#     Args:
#         data_dir: Directory containing DESI data files
#         verbose: Whether to print debug information
        
#     Returns:
#         QuintessenceDESILikelihood instance
#     """
#     return QuintessenceDESILikelihood(data_dir=data_dir, verbose=verbose)


# def test_esr_function(fcn_str, params, data_dir=None, verbose=True):
#     """
#     Quick test function for ESR-generated potentials
    
#     Args:
#         fcn_str: String representation of potential V(φ)
#         params: Array of parameter values
#         data_dir: Directory containing DESI data
#         verbose: Whether to print results
        
#     Returns:
#         dict: Test results
#     """
#     likelihood = create_quintessence_likelihood(data_dir=data_dir, verbose=verbose)
#     return likelihood.test_function(fcn_str, params)


# # Example of how to use with ESR
# def integrate_with_esr():
#     """
#     Example showing how to integrate with ESR workflow
#     """
    
#     # Create the likelihood
#     likelihood = create_quintessence_likelihood(verbose=True)
    
#     # Example ESR functions to test
#     test_functions = [
#         ("x**2", [1.0]),
#         ("a0 * exp(-x/a1)", [1.0, 2.0]),
#         ("a0 * x**a1", [1.0, 2.0]),
#         ("a0 + a1*x**2", [1.0, 0.5])
#     ]
    
#     print("Testing ESR functions with QuintessenceDESILikelihood:")
#     print("="*60)
    
#     for fcn_str, params in test_functions:
#         result = likelihood.test_function(fcn_str, params)
#         print(f"Function: {fcn_str}")
#         print(f"Parameters: {params}")
#         print(f"Success: {result['success']}")
#         print(f"Likelihood: {result['likelihood']:.2f}")
#         print("-" * 40)
    
#     return likelihood