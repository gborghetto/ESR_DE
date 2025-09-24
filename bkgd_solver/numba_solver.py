from functools import partial
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from numba import jit, vectorize, float64, types
from numba.experimental import jitclass
import numba as nb

# Speed of light in km/s
_c_km_s = 299792.458

# Numba-optimized core functions
@jit(nopython=True, fastmath=True, cache=True)
def exp_potential_numba(phi, lambda_phi):
    """Optimized exponential potential V(phi) = exp(-lambda_phi * phi)"""
    return np.exp(-lambda_phi * phi)

@jit(nopython=True, fastmath=True, cache=True)
def exp_potential_deriv_numba(phi, lambda_phi):
    """Optimized derivative of exponential potential"""
    return -lambda_phi * np.exp(-lambda_phi * phi)

@jit(nopython=True, fastmath=True, cache=True)
def compute_rho_phi_numba(phi, phidot, amplitude, lambda_phi):
    """Compute quintessence energy density"""
    V = amplitude * exp_potential_numba(phi, lambda_phi)
    return 0.5 * phidot**2 + V

@jit(nopython=True, fastmath=True, cache=True)
def compute_H_numba(phi, phidot, z, H0_Gyr, Omega_r, Omega_m, Omega_k, 
                   density_crit0, amplitude, lambda_phi):
    """Optimized Hubble parameter calculation"""
    rho_phi = compute_rho_phi_numba(phi, phidot, amplitude, lambda_phi)
    Omega_phi = rho_phi / density_crit0
    E2 = (Omega_r * (1+z)**4 + 
          Omega_m * (1+z)**3 + 
          Omega_k * (1+z)**2 + 
          Omega_phi)
    return H0_Gyr * np.sqrt(E2)

@jit(nopython=True, fastmath=True, cache=True)
def N_to_z_numba(N):
    """Convert N-fold to redshift"""
    a = np.exp(N)
    return 1.0/a - 1.0

@jit(nopython=True, fastmath=True, cache=True)
def z_to_N_numba(z):
    """Convert redshift to N-fold"""
    a = 1.0/(1.0 + z)
    return np.log(a)

@jit(nopython=True, fastmath=True, cache=True)
def dydN_numba(N, y, H0_Gyr, Omega_r, Omega_m, Omega_k, density_crit0, amplitude, lambda_phi):
    """Optimized ODE system for phi and phidot evolution"""
    phi, phidot = y[0], y[1]
    z = N_to_z_numba(N)
    H = compute_H_numba(phi, phidot, z, H0_Gyr, Omega_r, Omega_m, Omega_k, 
                       density_crit0, amplitude, lambda_phi)
    
    # Potential derivative
    dV = amplitude * exp_potential_deriv_numba(phi, lambda_phi)
    
    dphi_dN = phidot / H
    dphidot_dN = -(3*H*phidot + dV) / H
    
    return np.array([dphi_dN, dphidot_dN])

@jit(nopython=True, fastmath=True, cache=True)
def compute_Omega_phi0_numba(logA, phi_init, phidot_init, N_init, 
                            H0_Gyr, Omega_r, Omega_m, Omega_k, 
                            density_crit0, lambda_phi, Omega_phi_target):
    """Optimized Omega_phi0 computation for amplitude tuning"""
    amplitude = 10.0**logA
    
    # Simple RK4 integration (faster than scipy for this case)
    N_span = np.linspace(N_init, 0.0, 1000)
    h = N_span[1] - N_span[0]
    
    y = np.array([phi_init, phidot_init])
    
    for i in range(len(N_span)-1):
        N_curr = N_span[i]
        k1 = dydN_numba(N_curr, y, H0_Gyr, Omega_r, Omega_m, Omega_k, 
                       density_crit0, amplitude, lambda_phi)
        k2 = dydN_numba(N_curr + h/2, y + h*k1/2, H0_Gyr, Omega_r, Omega_m, Omega_k, 
                       density_crit0, amplitude, lambda_phi)
        k3 = dydN_numba(N_curr + h/2, y + h*k2/2, H0_Gyr, Omega_r, Omega_m, Omega_k, 
                       density_crit0, amplitude, lambda_phi)
        k4 = dydN_numba(N_curr + h, y + h*k3, H0_Gyr, Omega_r, Omega_m, Omega_k, 
                       density_crit0, amplitude, lambda_phi)
        
        y = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    phi0, phidot0 = y[0], y[1]
    rho_phi0 = compute_rho_phi_numba(phi0, phidot0, amplitude, lambda_phi)
    return rho_phi0 / density_crit0 - Omega_phi_target

@jit(nopython=True, fastmath=True, cache=True)
def solve_evolution_numba(phi_init, phidot_init, N_init, N_final, num_points,
                         H0_Gyr, Omega_r, Omega_m, Omega_k, density_crit0, 
                         amplitude, lambda_phi):
    """Optimized evolution solver using RK4"""
    N_vals = np.linspace(N_init, N_final, num_points)
    h = N_vals[1] - N_vals[0]
    
    phi_vals = np.zeros(num_points)
    phidot_vals = np.zeros(num_points)
    
    phi_vals[0] = phi_init
    phidot_vals[0] = phidot_init
    
    y = np.array([phi_init, phidot_init])
    
    for i in range(num_points-1):
        N_curr = N_vals[i]
        
        k1 = dydN_numba(N_curr, y, H0_Gyr, Omega_r, Omega_m, Omega_k, 
                       density_crit0, amplitude, lambda_phi)
        k2 = dydN_numba(N_curr + h/2, y + h*k1/2, H0_Gyr, Omega_r, Omega_m, Omega_k, 
                       density_crit0, amplitude, lambda_phi)
        k3 = dydN_numba(N_curr + h/2, y + h*k2/2, H0_Gyr, Omega_r, Omega_m, Omega_k, 
                       density_crit0, amplitude, lambda_phi)
        k4 = dydN_numba(N_curr + h, y + h*k3, H0_Gyr, Omega_r, Omega_m, Omega_k, 
                       density_crit0, amplitude, lambda_phi)
        
        y = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        phi_vals[i+1] = y[0]
        phidot_vals[i+1] = y[1]
    
    return N_vals, phi_vals, phidot_vals

@vectorize([float64(float64, float64, float64, float64, float64, float64, float64, float64, float64)], 
           nopython=True, fastmath=True, cache=True)
def H_of_z_vectorized(z, phi, phidot, H0_Gyr, Omega_r, Omega_m, Omega_k, 
                     density_crit0, amplitude, lambda_phi):
    """Vectorized Hubble parameter computation"""
    return compute_H_numba(phi, phidot, z, H0_Gyr, Omega_r, Omega_m, Omega_k, 
                          density_crit0, amplitude, lambda_phi)

@jit(nopython=True, fastmath=True, cache=True)
def comoving_distance_integrand(z_vals, H_vals, c_km_s):
    """Compute comoving distance using trapezoidal integration"""
    # Convert H from Gyr^-1 to km/s/Mpc
    H_kmsMpc = H_vals * (3.0857e19 / 3.1536e16)
    integrand = c_km_s / H_kmsMpc
    return np.trapz(integrand, z_vals)

@jit(nopython=True, fastmath=True, cache=True)
def compute_angular_diameter_distance(dc, z, Omega_k, H0):
    """Compute angular diameter distance from comoving distance"""
    c_km_s = 299792.458
    
    if abs(Omega_k) < 1e-10:  # Flat universe
        dk = dc
    elif Omega_k > 0:  # Open universe
        sqrtOk = np.sqrt(Omega_k)
        dh = c_km_s / H0  # Hubble distance
        dk = dh / sqrtOk * np.sinh(sqrtOk * dc / dh)
    else:  # Closed universe
        sqrtOk = np.sqrt(-Omega_k)
        dh = c_km_s / H0  # Hubble distance
        dk = dh / sqrtOk * np.sin(sqrtOk * dc / dh)
    
    return dk / (1.0 + z)

class NumbaOptimizedQuintessenceSolver:
    def __init__(
        self,
        H0,                 # Hubble parameter today [km/s/Mpc]
        Omega_m,            # Matter density today
        Omega_r,            # Radiation density today
        Omega_k,            # Curvature density today
        phi_init,           # Initial _phi at early time
        phidot_init,        # Initial _phi̇ at early time
        z_init=1e2,         # Starting redshift
        A_min = 1e-10,
        A_max = 1e10,  
        atol = 1e-8,
        rtol = 1e-8,
        verbose = True, 
        tune_amplitude_flag=True,      
        V_kwargs = {},      # Additional kwargs for the potential function
        num_evolution_points=500,  # Number of points for evolution
    ):
        """
        Numba-optimized QuintessenceSolver for maximum speed
        """
        # Store cosmological parameters
        self.H0 = H0
        # Convert H0 → 1/Gyr
        H0_si = H0 * 1000 / 3.0857e22    # s⁻¹
        self.H0_Gyr = H0_si * 3.1536e16      # Gyr⁻¹

        self.Omega_m = Omega_m
        self.Omega_r = Omega_r
        self.Omega_k = Omega_k
        self.Omega_phi_target = 1.0 - Omega_m - Omega_r - Omega_k

        # Critical density today in natural units (8πG=1)
        self.density_crit0 = 3 * self.H0_Gyr**2

        # Initial conditions
        self.phi_init = phi_init
        self.phidot_init = phidot_init
        self.z_init = z_init
        self.N_init = z_to_N_numba(z_init)

        # Potential parameters (assuming exponential potential)
        self.lambda_phi = V_kwargs.get('lambda_phi', 1.0)

        # Store tolerances and options
        self.atol = atol
        self.rtol = rtol
        self.verbose = verbose
        self.num_evolution_points = num_evolution_points

        # Speed of light in km/s
        self.c_km_s = 299792.458

        # Pre-compile numba functions
        self._compile_numba_functions()

        # Initialize
        if tune_amplitude_flag:
            self._tune_amplitude_numba(A_min=A_min, A_max=A_max)
            self._solve_evolution_numba()
        else:
            self.amplitude = 1.0
            self.solution = None

    def _compile_numba_functions(self):
        """Pre-compile all numba functions for better first-run performance"""
        if self.verbose:
            print("Pre-compiling numba functions...")
        
        # Trigger compilation with dummy values
        exp_potential_numba(0.0, 1.0)
        exp_potential_deriv_numba(0.0, 1.0)
        compute_rho_phi_numba(0.0, 0.0, 1.0, 1.0)
        compute_H_numba(0.0, 0.0, 0.0, 1.0, 1e-5, 0.3, 0.0, 1.0, 1.0, 1.0)
        N_to_z_numba(0.0)
        z_to_N_numba(0.0)
        dydN_numba(0.0, np.array([0.0, 0.0]), 1.0, 1e-5, 0.3, 0.0, 1.0, 1.0, 1.0)
        compute_Omega_phi0_numba(0.0, 0.0, 0.0, -1.0, 1.0, 1e-5, 0.3, 0.0, 1.0, 1.0, 0.7)
        
        if self.verbose:
            print("Numba compilation complete.")

    def _tune_amplitude_numba(self, A_min=1e-10, A_max=1e10, max_attempts=100):
        """Tune amplitude using numba-optimized function"""
        A_min_log = np.log10(A_min)
        A_max_log = np.log10(A_max)

        # Create a wrapper for the numba function
        def objective(logA):
            return compute_Omega_phi0_numba(
                logA, self.phi_init, self.phidot_init, self.N_init,
                self.H0_Gyr, self.Omega_r, self.Omega_m, self.Omega_k,
                self.density_crit0, self.lambda_phi, self.Omega_phi_target
            )

        # Check bounds
        val_min = objective(A_min_log)
        if np.isnan(val_min):
            for i in range(max_attempts):
                A_min_log += 0.25
                val_min = objective(A_min_log)
                if not np.isnan(val_min):
                    break
            else:
                raise ValueError("Could not find valid A_min")

        val_max = objective(A_max_log)
        if np.isnan(val_max):
            for i in range(max_attempts):
                A_max_log -= 0.25
                val_max = objective(A_max_log)
                if not np.isnan(val_max):
                    break
            else:
                raise ValueError("Could not find valid A_max")

        # Root finding
        self.amplitude_log, res = brentq(
            objective, A_min_log, A_max_log, xtol=1e-6, full_output=True
        )
        
        self.amplitude = 10**self.amplitude_log
        
        if self.verbose:
            print(f"Tuned amplitude: {self.amplitude:.3e}, converged: {res.converged}")

    def _solve_evolution_numba(self):
        """Solve evolution using numba-optimized integrator"""
        N_vals, phi_vals, phidot_vals = solve_evolution_numba(
            self.phi_init, self.phidot_init, self.N_init, 0.0, 
            self.num_evolution_points, self.H0_Gyr, self.Omega_r, 
            self.Omega_m, self.Omega_k, self.density_crit0, 
            self.amplitude, self.lambda_phi
        )
        
        # Store solution
        self.solution = {
            'N': N_vals,
            'phi': phi_vals,
            'phidot': phidot_vals
        }

        # Create interpolators
        self.Phi_of_N = interp1d(N_vals, phi_vals, kind='linear', 
                                bounds_error=False, fill_value='extrapolate')
        self.Phidot_of_N = interp1d(N_vals, phidot_vals, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
        
        # Pre-compute H(z) values for fast interpolation
        z_vals = np.array([N_to_z_numba(N) for N in N_vals])
        H_vals = H_of_z_vectorized(z_vals, phi_vals, phidot_vals, 
                                  self.H0_Gyr, self.Omega_r, self.Omega_m, self.Omega_k,
                                  self.density_crit0, self.amplitude, self.lambda_phi)
        
        self.H_of_N = interp1d(N_vals, H_vals, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        
        # Pre-compute distances
        self._precompute_distances(z_vals, H_vals)

    def _precompute_distances(self, z_vals, H_vals):
        """Pre-compute distance measures for fast interpolation"""
        da_vals = np.zeros_like(z_vals)
        dl_vals = np.zeros_like(z_vals)
        
        # Sort by redshift for integration
        sorted_indices = np.argsort(z_vals)
        z_sorted = z_vals[sorted_indices]
        H_sorted = H_vals[sorted_indices]
        
        for i in range(1, len(z_sorted)):
            # Comoving distance using trapezoidal rule
            z_slice = z_sorted[:i+1]
            H_slice = H_sorted[:i+1]
            dc = comoving_distance_integrand(z_slice, H_slice, self.c_km_s)
            
            # Angular diameter distance
            da = compute_angular_diameter_distance(dc, z_sorted[i], self.Omega_k, self.H0)
            
            # Luminosity distance
            dl = da * (1 + z_sorted[i])**2
            
            # Map back to original indices
            orig_idx = sorted_indices[i]
            da_vals[orig_idx] = da
            dl_vals[orig_idx] = dl
        
        # Create interpolators
        self.da_interp = interp1d(z_vals, da_vals, kind='linear', 
                                 bounds_error=False, fill_value='extrapolate')
        self.dl_interp = interp1d(z_vals, dl_vals, kind='linear', 
                                 bounds_error=False, fill_value='extrapolate')

    # Public interface methods
    def phi(self, z):
        """Return phi at redshift z"""
        z = np.atleast_1d(z)
        N = np.array([z_to_N_numba(z_i) for z_i in z])
        result = self.Phi_of_N(N)
        return result[0] if len(result) == 1 else result

    def phidot(self, z):
        """Return phidot at redshift z"""
        z = np.atleast_1d(z)
        N = np.array([z_to_N_numba(z_i) for z_i in z])
        result = self.Phidot_of_N(N)
        return result[0] if len(result) == 1 else result

    def H_of_z(self, z):
        """Fast Hubble parameter at redshift z"""
        z = np.atleast_1d(z)
        N = np.array([z_to_N_numba(z_i) for z_i in z])
        result = self.H_of_N(N)
        return result[0] if len(result) == 1 else result

    def angular_diameter_distance(self, z):
        """Fast angular diameter distance at redshift z"""
        z = np.atleast_1d(z)
        result = self.da_interp(z)
        return result[0] if len(result) == 1 else result

    def luminosity_distance(self, z):
        """Fast luminosity distance at redshift z"""
        z = np.atleast_1d(z)
        result = self.dl_interp(z)
        return result[0] if len(result) == 1 else result

    def rho_de(self, z):
        """Dark energy density at redshift z"""
        phi_z = self.phi(z)
        phidot_z = self.phidot(z)
        z = np.atleast_1d(z)
        if len(z) == 1:
            return compute_rho_phi_numba(phi_z, phidot_z, self.amplitude, self.lambda_phi)
        else:
            return np.array([compute_rho_phi_numba(p, pd, self.amplitude, self.lambda_phi) 
                           for p, pd in zip(phi_z, phidot_z)])

    def w_de(self, z):
        """Dark energy equation of state at redshift z"""
        phi_z = self.phi(z)
        phidot_z = self.phidot(z)
        
        z = np.atleast_1d(z)
        if len(z) == 1:
            rho = compute_rho_phi_numba(phi_z, phidot_z, self.amplitude, self.lambda_phi)
            V = self.amplitude * exp_potential_numba(phi_z, self.lambda_phi)
            p = 0.5 * phidot_z**2 - V
            return p / rho
        else:
            w_vals = []
            for p, pd in zip(phi_z, phidot_z):
                rho = compute_rho_phi_numba(p, pd, self.amplitude, self.lambda_phi)
                V = self.amplitude * exp_potential_numba(p, self.lambda_phi)
                pressure = 0.5 * pd**2 - V
                w_vals.append(pressure / rho)
            return np.array(w_vals)

    # Compatibility methods for potential functions
    def Vphi(self, phi):
        """Potential function V(phi)"""
        return self.amplitude * exp_potential_numba(phi, self.lambda_phi)

    def dV_dphi(self, phi):
        """Derivative of potential function"""
        return self.amplitude * exp_potential_deriv_numba(phi, self.lambda_phi)