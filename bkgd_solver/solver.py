from functools import partial
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
from scipy.interpolate import interp1d, UnivariateSpline
from numba import jit, vectorize, float64, types
import os
from pathlib import Path

@jit(nopython=True)
def _dydN_jit(N, y, H_func, dV_dphi_func, N_to_z_func):
    """
    This is the JIT-compiled version of the ODE system.
    It takes functions as arguments to avoid using 'self'.
    """
    phi, phidot = y
    z = N_to_z_func(N)
    H = H_func(phi, phidot, z)

    # A safety check to prevent division by zero
    if H == 0:
        return [0.0, 0.0]

    dphi_dN = phidot / H
    dphidot_dN = -(3 * H * phidot + dV_dphi_func(phi)) / H
    return [dphi_dN, dphidot_dN]

@jit(nopython=True)
def Vphi_jit(phi, amplitude, lambda_phi):
    """JIT-compiled version of the exponential potential."""
    return amplitude * np.exp(-lambda_phi * phi)

@jit(nopython=True)
def dV_dphi_jit(phi, amplitude, lambda_phi):
    """JIT-compiled version of the potential's derivative."""
    # Re-use the fast Vphi_jit function inside
    return -lambda_phi * Vphi_jit(phi, amplitude, lambda_phi)

@jit(nopython=True)
def H_of_jit(phi, phidot, z, H0_Gyr, Omega_m, Omega_r, Omega_k, density_crit0, amplitude, lambda_phi, Omega_nu_z=0.0):
    """JIT-compiled version of the Hubble parameter calculation."""
    # Call the already-compiled potential function for efficiency
    rho_phi = 0.5 * phidot**2 + Vphi_jit(phi, amplitude, lambda_phi)
    Omega_phi_term = rho_phi / density_crit0

    E2 = (
        Omega_nu_z + # Neutrino density term from interpolation table
        Omega_r * (1 + z)**4 +
        Omega_m * (1 + z)**3 +
        Omega_k * (1 + z)**2 +
        Omega_phi_term
    )

    # Safety check to prevent numerical errors
    if E2 < 0:
        return 0.0
    
    return H0_Gyr * np.sqrt(E2)

@jit(nopython=True, fastmath=True, cache=True)
def sound_speed_squared(z, Omega_b_h2):
    """Compute sound speed squared in baryon-photon fluid"""
    c_km_s = 299792.458
    R_baryon_photon = 31500 * Omega_b_h2  # From CMB temperature relation
    return (c_km_s**2) / (3.0 * (1.0 + R_baryon_photon / (1.0 + z)))

@jit(nopython=True, fastmath=True, cache=True)
def sound_horizon_integrand(z_array, H_array, Omega_b_h2):
    """Compute integrand for sound horizon calculation"""
    c_km_s = 299792.458
    # Convert H from Gyr^-1 to km/s/Mpc
    H_kmsMpc = H_array * (3.0857e19 / 3.1536e16)
    
    integrand_vals = np.zeros_like(z_array)
    for i in range(len(z_array)):
        cs_squared = sound_speed_squared(z_array[i], Omega_b_h2)
        cs = np.sqrt(cs_squared)
        integrand_vals[i] = cs / H_kmsMpc[i]
    
    return integrand_vals

class QuintessenceSolver:
    def __init__(
        self,
        H0,                 # Hubble parameter today [km/s/Mpc]
        Omega_m,            # Matter density today
        Omega_r,            # Radiation density today
        Omega_k,            # Curvature density today
        phi_init,           # Initial _phi at early time
        phidot_init,        # Initial _phi̇ at early time
        V_base,             # Base potential function V_base(phi)
        dV_base_dphi,       # Derivative of base potential
        z_init=1e8,         # Starting redshift
        A_min = 1e-10,
        A_max = 1e10,  
        atol = 1e-8,
        rtol = 1e-8,
        verbose = True, 
        tune_amplitude_flag=True,      
        V_kwargs = {},      # Additional kwargs for the potential function
        neutrino_table_path='neutrino_density_table.txt'  # Path to neutrino density table
    ):
        """
        Class to solve the background cosmological equations of motion for a Quintessence potential.

        Arguments
        -----------------
        H0 : float
            Hubble parameter today in km/s/Mpc.
        Omega_m : float
            Matter density today (dimensionless).
        Omega_r : float
            Radiation density today (dimensionless).
        Omega_k : float
            Curvature density today (dimensionless).
        phi_init : float
            Initial value of _phi at early time.
        phidot_init : float
            Initial value of _phi̇ at early time.
        V_base : function
            Base potential function V_base(phi). After tuning, the potential is V(phi) = A * V_base(phi).
        dV_base_dphi : function
            Derivative of base potential function with respect to _phi.
        z_init : float
            Initial redshift (default: 1e2).
        A_min : float
            Minimum value for the amplitude of the potential (default: 1e-10).
        A_max : float
            Maximum value for the amplitude of the potential (default: 1e10).
        atol : float
            Absolute tolerance for the ODE solver (default: 1e-10).
        rtol : float
            Relative tolerance for the ODE solver (default: 1e-10).
        verbose : bool
            If True, print tuning information (default: True).
        """    
    
        # Store cosmological parameters
        self.H0 = H0
        # Convert H0 → 1/Gyr
        H0_si    = H0 * 1000 / 3.0857e22    # s⁻¹
        self.H0_Gyr = H0_si * 3.1536e16      # Gyr⁻¹

        self.Omega_m = Omega_m
        self.Omega_r = Omega_r
        self.Omega_k = Omega_k

        # --- NEW: Load and interpolate the neutrino density table ---
        try:
            solver_directory = Path(__file__).parent
            # Join this directory with the filename to get the full path.
            absolute_neutrino_path = solver_directory / "neutrino_density_table.txt"
            a_table, onuh2_table = np.loadtxt(absolute_neutrino_path, unpack=True)
            self.a_nu_tab_max = a_table.max()
            self.a_nu_tab_min = a_table.min()
            # Use log-space interpolation for better accuracy
            self._log_onuh2_of_log_a = interp1d(
                np.log(a_table), np.log(onuh2_table),
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
            omega_nu_today = self.onuh2_of_z(0) / (self.H0 / 100.0)**2
            print("Successfully loaded neutrino density table.")
        except IOError:
            raise IOError(f"Could not find neutrino data file: {neutrino_table_path}")

        print(f"Calculated Omega_nu today = {omega_nu_today:.5e}, Omega photons = {Omega_r:.5e}")

        self.Omega_phi_target = 1.0 - Omega_m - Omega_r - Omega_k - omega_nu_today


        # Critical density today in natural units (8πG=1)
        self.density_crit0 = 3 * self.H0_Gyr**2

        # Initial conditions
        self.phi_init = phi_init
        self.phidot_init = phidot_init
        self.z_init = z_init
        # We solve in efolds so N = log(a)
        self.N_init = self.z_to_N(z_init)

        # Potential functions
        self.V_base = V_base
        self.dV_base_dphi = dV_base_dphi
        self.V_kwargs = V_kwargs    

        # Placeholder for tuned amplitude and solution
        self.solution = None

        # Store tolerances
        self.atol = atol
        self.rtol = rtol

        self.verbose = verbose

        # Speed of light in km/s
        self.c_km_s = 299792.458

        # Automatically tune on init
        if tune_amplitude_flag:
        # if self.amplitude is None:
            try:
                self._tune_amplitude(A_min=A_min, A_max=A_max)
                self.success = True
            except Exception as e:
                print('Error in tuning', e)
                self.success = False
        # After tuning, solve evolution
            self._solve_evolution()
        else:
            self.amplitude  = 1.0
            self.solution = None

    def Vphi(self, phi):
        """Wrapper that calls the fast, JIT-compiled potential function."""
        # Get lambda_phi from the kwargs dictionary
        lambda_phi = self.V_kwargs.get('lambda_phi', 1.0)
        return Vphi_jit(phi, self.amplitude, lambda_phi)

    def dV_dphi(self, phi):
        """Wrapper that calls the fast, JIT-compiled derivative function."""
        lambda_phi = self.V_kwargs.get('lambda_phi', 1.0)
        return dV_dphi_jit(phi, self.amplitude, lambda_phi)

    def onuh2_of_z(self, z):
        """Fast interpolation of the neutrino density."""
        # Handle z=0 safely, as log(0) is -inf
        a = np.clip(1.0 / (1.0 + z), self.a_nu_tab_min, self.a_nu_tab_max)
        result = np.exp(self._log_onuh2_of_log_a(np.log(a)))
        return result

    def H_of(self, phi, phidot, z):
        """Wrapper that calls the fast, JIT-compiled H(z) function."""
        lambda_phi = self.V_kwargs.get('lambda_phi', 1.0)
        Omega_nu = self.onuh2_of_z(z) / (self.H0 / 100.0)**2
        return H_of_jit(
            phi, phidot, z,
            self.H0_Gyr, self.Omega_m, self.Omega_r, self.Omega_k,
            self.density_crit0, self.amplitude, lambda_phi, Omega_nu)

    # def Vphi(self, phi):
    #     return self.amplitude * self.V_base(phi, **self.V_kwargs)

    # def dV_dphi(self, phi):
    #     return self.amplitude * self.dV_base_dphi(phi,**self.V_kwargs)

    # def H_of(self, phi, phidot, z):
    #     rho_phi = 0.5 * phidot**2 + self.Vphi(phi)
    #     Omega_phi = rho_phi / self.density_crit0
    #     E2 = (
    #         self.Omega_r * (1+z)**4 +
    #         self.Omega_m * (1+z)**3 +
    #         self.Omega_k * (1+z)**2 +
    #         Omega_phi
    #     )
    #     return self.H0_Gyr * np.sqrt(E2)
    
    def N_to_z(self, N):
        # Convert N to z
        a = np.exp(N)
        z = 1/a - 1
        return z
    
    def z_to_N(self, z):
        # Convert z to N
        a = 1/(1 + z)
        N = np.log(a)
        return N

    def _dydN(self, N, y):
        """ System of equations to solve for _phi and _phi̇ """
        phi, phidot = y
        z  = self.N_to_z(N)
        H = self.H_of(phi, phidot, z)
        dphi_dN = phidot / H
        dphidot_dN = -(3*H*phidot + self.dV_dphi(phi)) / H
        return [dphi_dN, dphidot_dN]
    
    # def _dydN(self, N, y):
    #     """
    #     Wrapper method that calls the fast, Numba-compiled ODE function.
    #     """
    #     return _dydN_jit(N, y, self.H_of, self.dV_dphi, self.N_to_z)

    def _compute_Omega_phi0(self, logA):
        """ Compute Omega_phi0 for a given log amplitude """
        self.amplitude = 10**logA
        y0 = [self.phi_init, self.phidot_init]
        sol = solve_ivp(
            self._dydN,
            [self.N_init, 0.0],
            y0,
            atol=self.atol,
            rtol=self.rtol,
            dense_output=True,
        )
        phi0, phidot0 = sol.y[:, -1]
        rho_phi0 = 0.5 * phidot0**2 + self.Vphi(phi0)
        return rho_phi0 / self.density_crit0 - self.Omega_phi_target

    def _tune_amplitude(self,A_min = 1e-10, A_max = 1e10, max_attempts=100):
        """ Tune the amplitude of the potential to match Omega_phi0 """
        A_min = np.log10(A_min)
        A_max = np.log10(A_max)

        # Check if the values at min and max are not NaN
        val_A_min = self._compute_Omega_phi0(A_min)
        if np.isnan(val_A_min):
            # print(f"Value at A_min is NaN, trying to find a valid value...")
            for i in range(max_attempts):
                A_min += 0.25
                val_A_min = self._compute_Omega_phi0(A_min)
                if not np.isnan(val_A_min):
                    break
            else:
                raise ValueError("Could not find a valid value for A_min")
        val_A_max = self._compute_Omega_phi0(A_max)
        if np.isnan(val_A_max):
            # print(f"Value at A_max is NaN, trying to find a valid value...")
            for i in range(max_attempts):
                A_max -= 0.25
                val_A_max = self._compute_Omega_phi0(A_max)
                if not np.isnan(val_A_max):
                    break
            else:
                raise ValueError("Could not find a valid value for A_max")

        self.amplitude, res = brentq(
            self._compute_Omega_phi0,
            A_min,
            A_max,
            xtol=1e-8,
            full_output=True,
        )
        # Convert back to linear scale
        self.amplitude = 10**self.amplitude
        if self.verbose:
            print(f"Tuned amplitude: {self.amplitude:.3e}, converged: {res.converged}")

    def _solve_evolution(self, num_points=500):
        """ Solve the equations of motion for _phi and _phi̇ after tuning the amplitude """
        N_vals = np.linspace(self.N_init, 0.0, num_points)
        sol = solve_ivp(
            self._dydN,
            [self.N_init, 0.0],
            [self.phi_init, self.phidot_init],
            t_eval=N_vals,
            atol=self.atol,
            rtol=self.rtol,
            dense_output=True,
        )
        self.solution = {
            'N': sol.t,
            'phi': sol.y[0],
            'phidot': sol.y[1]
        }

        self.Phi_of_N = interp1d(
            self.solution['N'],
            self.solution['phi'],
            kind='linear',
            fill_value='extrapolate'
        )

        self.Phidot_of_N = interp1d(
            self.solution['N'],
            self.solution['phidot'],
            kind='linear',
            fill_value='extrapolate'
        )

        H_of_N = np.array([
            self.H_of(phi_i, phidot_i, self.N_to_z(N_i))
            for phi_i, phidot_i, N_i
            in zip(self.solution['phi'], self.solution['phidot'], self.solution['N'])
        ])
        self.H_of_N = interp1d(
            self.solution['N'],
            H_of_N,
            kind='linear',
            fill_value='extrapolate'
        )

    # Public methods to retrieve quantities
    def phi(self, z=None):
        """ Return _phi at redshift z """
        z = z if z is not None else self.solution['z']
        N = self.z_to_N(z)
        return self.Phi_of_N(N) #np.interp(z, self.solution['z'], self.solution['phi'])

    def phidot(self, z=None):
        """ Return _phi̇ at redshift z """
        z = z if z is not None else self.solution['z']
        N = self.z_to_N(z)
        return self.Phidot_of_N(N) #np.interp(z, self.solution['z'], self.solution['phi'])

    def rho_de(self, z=None):
        """ Return the energy density of dark energy at redshift z """
        z = z if z is not None else self.solution['z']
        phi = self.phi(z)
        phidot = self.phidot(z)
        return 0.5 * phidot**2 + self.Vphi(phi)

    def w_de(self, z=None):
        """ Return the equation of state parameter w(z) for dark energy """
        z = z if z is not None else self.solution['z']
        phi = self.phi(z)
        phidot = self.phidot(z)
        rho = 0.5 * phidot**2 + self.Vphi(phi)
        p = 0.5 * phidot**2 - self.Vphi(phi)
        return p / rho

    # def H_of_z(self, z=None):
    #     """ Return the Hubble parameter at redshift z """
    #     z_arr = z if z is not None else self.solution['z']
    #     return self.H_of_N(self.z_to_N(z_arr))
    
    def H_of_z(self, z=None):
        """ 
        Return the Hubble parameter at redshift z.
        Uses an analytical approximation for z > z_init.
        """
        z_arr = np.atleast_1d(z)
        result = np.empty_like(z_arr, dtype=float)

        result = self.H_of_N(self.z_to_N(z_arr))

        # # Create a boolean mask to separate high-z and low-z points
        # mask_high_z = z_arr > self.z_init

        # # Analytical part for z > z_init (Radiation-dominated era)
        # if np.any(mask_high_z):
        #     z_high = z_arr[mask_high_z]
        #     result[mask_high_z] = self.H0_Gyr * np.sqrt(self.Omega_r) * (1 + z_high)**2

        # # Numerical part for z <= z_init (using the ODE solution)
        # if np.any(~mask_high_z):
        #     z_low = z_arr[~mask_high_z]
        #     result[~mask_high_z] = self.H_of_N(self.z_to_N(z_low))

        # convert to km/s/Mpc
        result = result * (3.0857e19 / 3.1536e16)  # Gyr⁻¹ to km/s/Mpc
        
        return result[0] if len(result) == 1 else result

    def comoving_distance(self, z):
        """
        Calculate comoving distance to redshift z in Mpc.
        
        Parameters:
        z : float or array
            Redshift(s)
            
        Returns:
        float or array: Comoving distance in Mpc
        """
        z = np.atleast_1d(z)
        distances = []
        
        def integrand(z_prime):
            H_z_prime_kmsMpc = np.atleast_1d(self.H_of_z(z_prime))
            # # Convert Gyr^-1 to km/s/Mpc: 1 Gyr^-1 = (3.0857e19 km) / (3.1536e16 s) km/s/Mpc
            # H_z_prime_kmsMpc = H_z_prime_Gyr * (3.0857e19 / 3.1536e16)
            return self.c_km_s / H_z_prime_kmsMpc

        for z_i in z:
            # result, _ = quad(integrand, 0, z_i, epsabs=1e-8, epsrel=1e-8)
            z_arr = np.linspace(0, z_i, 1000)
            result = np.trapz(integrand(z_arr), z_arr)
            distances.append(result)
        
        distances = np.array(distances)
        return distances[0] if len(distances) == 1 else distances

    def angular_diameter_distance(self, z):
        """
        Calculate angular diameter distance to redshift z in Mpc.
        
        Parameters:
        z : float or array
            Redshift(s)
            
        Returns:
        float or array: Angular diameter distance in Mpc
        """
        z = np.atleast_1d(z)
        dc = self.comoving_distance(z)
        
        if abs(self.Omega_k) < 1e-10:  # Flat universe
            dk = dc
        elif self.Omega_k > 0:  # Open universe
            sqrtOk = np.sqrt(self.Omega_k)
            dh = self.c_km_s / self.H0  # Hubble distance in Mpc
            dk = dh / sqrtOk * np.sinh(sqrtOk * dc / dh)
        else:  # Closed universe
            sqrtOk = np.sqrt(-self.Omega_k)
            dh = self.c_km_s / self.H0  # Hubble distance in Mpc
            dk = dh / sqrtOk * np.sin(sqrtOk * dc / dh)
        
        # Angular diameter distance
        da = dk / (1 + z)
        
        return da[0] if len(da) == 1 else da

    def luminosity_distance(self, z):
        """
        Calculate luminosity distance to redshift z in Mpc.
        
        Parameters:
        z : float or array
            Redshift(s)
            
        Returns:
        float or array: Luminosity distance in Mpc
        """
        z = np.atleast_1d(z)
        da = self.angular_diameter_distance(z)
        
        # Luminosity distance
        dl = da * (1 + z)**2
        
        return dl[0] if len(dl) == 1 else dl
    
    def compute_sound_horizon(self, z_end, Omega_b_h2):
        """
        Compute comoving sound horizon from z_end to infinity
        
        Parameters:
        z_end : float
            Final redshift (e.g., drag epoch z_d ~ 1060)
        Omega_b_h2 : float
            Baryon density parameter Omega_b * h^2
            
        Returns:
        float: Sound horizon in Mpc
        """
        # Create integration grid from z_end to high redshift
        z_max_integration = 1e8  # Integrate to high redshift
        z_integration = np.logspace(np.log10(z_end), np.log10(z_max_integration), 1000)
        
        # Get H(z) values for the integration grid
        H_integration = np.array([self.H_of_z(z_integration)]).flatten()
        
        # Compute integrand
        integrand_vals = sound_horizon_integrand(z_integration, H_integration, Omega_b_h2)
        
        # Integrate using trapezoidal rule
        sound_horizon = np.trapz(integrand_vals, z_integration)
        
        return sound_horizon
    
    def compute_theta_star(self, Omega_b_h2, z_star=1089.8):
        """
        Compute angular size of sound horizon at recombination (theta_star)
        
        Parameters:
        Omega_b_h2 : float
            Baryon density parameter Omega_b * h^2
        z_star : float
            Redshift of recombination (default: 1089.8)
            
        Returns:
        float: theta_star in radians
        """
        # Compute sound horizon at recombination
        r_star = self.compute_sound_horizon(z_star, Omega_b_h2)
        
        # Compute angular diameter distance to recombination
        D_M_star = self.angular_diameter_distance(z_star)
        
        # theta_star = r_star / D_M_star
        theta_star = r_star / D_M_star
        
        return theta_star



class QuintessenceSpline(QuintessenceSolver):


    def __init__(
        self,
        nodes,
        vals,
        H0, Omega_m, Omega_r, Omega_k,
        phi_init, phidot_init,
        z_init=1e2,
        A_min = 1e-10,
        A_max = 1e10,
        atol = 1e-10,
        rtol = 1e-10,
        verbose = False,
    ):
        """
        Class to solve the background cosmological equations of motion for Qintessence. Instead of directly 
        giving the potential, we give a set of points (nodes,vals) that define the potential.
        The potential is defined as a cubic spline interpolation of the points (nodes,vals).

        Arguments:
        -----------------
        nodes : array-like
            Array of nodes for the potential.
        vals : array-like
            Array of values for the potential at the nodes.
        H0 : float
            Hubble parameter today in km/s/Mpc.
        Omega_m : float
            Matter density today (dimensionless).
        Omega_r : float
            Radiation density today (dimensionless).
        Omega_k : float
            Curvature density today (dimensionless).
        phi_init : float
            Initial value of _phi at early time.
        phidot_init : float
            Initial value of _phi̇ at early time.
        z_init : float
            Initial redshift (default: 1e2).
        A_min : float
            Minimum value for the amplitude of the potential (default: 1e-10).
        A_max : float
            Maximum value for the amplitude of the potential (default: 1e10).
        atol : float
            Absolute tolerance for the ODE solver (default: 1e-10).
        rtol : float
            Relative tolerance for the ODE solver (default: 1e-10).
        verbose : bool
            If True, print tuning information (default: True).
        """
        
        self.nodes = nodes
        self.vals = vals

        # Define the base potential and its derivative
        spl = UnivariateSpline(nodes, vals, s=0)
        V_base = spl
        dV_base_dphi = spl.derivative()

        super().__init__(
            H0=H0,
            Omega_m=Omega_m,
            Omega_r=Omega_r,
            Omega_k=Omega_k,
            phi_init=phi_init,
            phidot_init=phidot_init,
            V_base=V_base,
            dV_base_dphi=dV_base_dphi,
            z_init=z_init,
            A_min=A_min,
            A_max=A_max,
            atol=atol,
            rtol=rtol,
            verbose=verbose,
        )