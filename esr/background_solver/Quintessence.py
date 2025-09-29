import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import interp1d, UnivariateSpline
import jax
jax.config.update("jax_enable_x64", True)
from interpax import CubicSpline 
import matplotlib.pyplot as plt


class QuintessenceSolver:
    def __init__(
        self,
        H0,                 # Hubble parameter today [km/s/Mpc]
        Omega_m,            # Matter density today
        Omega_r,            # Radiation density today
        Omega_k,            # Curvature density today
        phi_init,           # Initial φ at early time
        phidot_init,        # Initial φ̇ at early time
        V_base,             # Base potential function V_base(phi)
        dV_base_dphi,       # Derivative of base potential
        z_init=1e2,         # Starting redshift

        verbose = True,  
        fixed_amplitude=1e-8,     
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
            Initial value of φ at early time.
        phidot_init : float
            Initial value of φ̇ at early time.
        V_base : function
            Base potential function V_base(phi). After tuning, the potential is V(phi) = A * V_base(phi).
        dV_base_dphi : function
            Derivative of base potential function with respect to φ.
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
        self.Omega_phi = 1 - Omega_m - Omega_r - Omega_k

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

        self.amplitude = fixed_amplitude

        self.verbose = verbose

        self._solve_evolution()
    

    def Vphi(self, phi):
        return self.amplitude * self.V_base(phi)

    def dV_dphi(self, phi):
        return self.amplitude * self.dV_base_dphi(phi)

    def H_of(self, phi, phidot, z):
        rho_phi = 0.5 * phidot**2 + self.Vphi(phi)
        Omega_phi = rho_phi / self.density_crit0
        E2 = (
            self.Omega_r * (1+z)**4 +
            self.Omega_m * (1+z)**3 +
            self.Omega_k * (1+z)**2 +
            Omega_phi
        )
        return self.H0_Gyr * np.sqrt(E2)
    
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

    # can be jitted
    def _dydN(self, N, y):
        """ System of equations to solve for φ and φ̇ """
        phi, phidot = y
        z  = self.N_to_z(N)
        H = self.H_of(phi, phidot, z)
        dphi_dN = phidot / H
        dphidot_dN = -(3*H*phidot + self.dV_dphi(phi)) / H
        return [dphi_dN, dphidot_dN]



    def _solve_evolution(self, num_points=500):
        """ Solve the equations of motion for φ and φ̇ after tuning the amplitude """
        N_vals = np.linspace(self.N_init, 0.0, num_points)
        sol = solve_ivp(
            self._dydN,
            [self.N_init, 0.0],
            [self.phi_init, self.phidot_init],
            t_eval=N_vals,
            #atol=self.atol,
            #rtol=self.rtol,
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
        """ Return φ at redshift z """
        z = z if z is not None else self.solution['z']
        N = self.z_to_N(z)
        return self.Phi_of_N(N) #np.interp(z, self.solution['z'], self.solution['phi'])

    def phidot(self, z=None):
        """ Return φ̇ at redshift z """
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

    def H_of_z(self, z=None):
        """ Return the Hubble parameter at redshift z """
        z_arr = z if z is not None else self.solution['z']
        return self.H_of_N(self.z_to_N(z_arr))
    
    


    def plot_potential(self, phi_range=(-3, 3), n_points=200, ax=None, **kwargs):
        """Plot the potential V(φ) vs φ"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        phi_vals = np.linspace(phi_range[0], phi_range[1], n_points)
        V_vals = [self.Vphi(phi) for phi in phi_vals]
        
        ax.plot(phi_vals, V_vals, **kwargs)
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$V(\phi)$')
        ax.set_title('Quintessence Potential')
        ax.grid(True, alpha=0.3)
        
        return ax

    def plot_field_evolution(self, z_max=None, n_points=200, ax=None, **kwargs):
        """Plot φ(z) and φ̇(z) vs redshift"""
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        else:
            ax1, ax2 = ax if isinstance(ax, (list, tuple)) else (ax, None)
        
        if z_max is None:
            z_max = self.N_to_z(self.solution['N'].min())
        
        z_vals = np.linspace(0, z_max, n_points)
        phi_vals = [self.phi(z) for z in z_vals]
        phidot_vals = [self.phidot(z) for z in z_vals]
        
        # Plot φ(z)
        ax1.plot(z_vals, phi_vals, **kwargs)
        ax1.set_xlabel('Redshift z')
        ax1.set_ylabel(r'$\phi(z)$')
        ax1.set_title('Scalar Field Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Plot φ̇(z)
        if ax2 is not None:
            ax2.plot(z_vals, phidot_vals, **kwargs)
            ax2.set_xlabel('Redshift z')
            ax2.set_ylabel(r'$\dot{\phi}(z)$')
            ax2.set_title('Field Velocity Evolution')
            ax2.grid(True, alpha=0.3)
        
        return (ax1, ax2) if ax2 is not None else ax1

    def plot_equation_of_state(self, z_max=3, n_points=200, ax=None, **kwargs):
        """Plot w(z) vs redshift"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        z_vals = np.linspace(0.01, z_max, n_points)  # Start slightly above 0
        w_vals = [self.w_de(z) for z in z_vals]
        
        ax.plot(z_vals, w_vals, **kwargs)
        ax.axhline(y=-1, color='red', linestyle='--', alpha=0.7, label='Cosmological constant')
        ax.axhline(y=-1/3, color='orange', linestyle='--', alpha=0.7, label='Phantom divide')
        ax.set_xlabel('Redshift z')
        ax.set_ylabel(r'$w_{DE}(z)$')
        ax.set_title('Dark Energy Equation of State')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax

    def plot_density_evolution(self, z_max=3, n_points=200, ax=None, **kwargs):
        """Plot Ω_DE(z) vs redshift"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        z_vals = np.linspace(0, z_max, n_points)
        omega_de_vals = []
        
        for z in z_vals:
            rho_de = self.rho_de(z)
            # Total critical density at redshift z
            rho_crit_z = self.density_crit0 * (
                self.Omega_r * (1+z)**4 +
                self.Omega_m * (1+z)**3 +
                self.Omega_k * (1+z)**2 +
                rho_de / self.density_crit0
            )
            omega_de_vals.append(rho_de / rho_crit_z)
        
        ax.plot(z_vals, omega_de_vals, label='Dark Energy', **kwargs)
        
        # Add matter and radiation for comparison
        omega_m_vals = [self.Omega_m * (1+z)**3 / (
            self.Omega_r * (1+z)**4 + self.Omega_m * (1+z)**3 + 
            self.Omega_k * (1+z)**2 + omega_de_vals[i]
        ) for i, z in enumerate(z_vals)]
        
        omega_r_vals = [self.Omega_r * (1+z)**4 / (
            self.Omega_r * (1+z)**4 + self.Omega_m * (1+z)**3 + 
            self.Omega_k * (1+z)**2 + omega_de_vals[i]
        ) for i, z in enumerate(z_vals)]
        
        ax.plot(z_vals, omega_m_vals, '--', alpha=0.7, label='Matter')
        ax.plot(z_vals, omega_r_vals, ':', alpha=0.7, label='Radiation')
        
        ax.set_xlabel('Redshift z')
        ax.set_ylabel(r'$\Omega_i(z)$')
        ax.set_title('Density Parameter Evolution')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
        
        return ax

    def plot_all(self, z_max=3, phi_range=(-3, 3), figsize=(16, 12)):
        """Create a comprehensive 4-panel plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot potential
        self.plot_potential(phi_range=phi_range, ax=ax1, color='blue', linewidth=2)
        
        # Plot field evolution (just φ(z) for space)
        z_vals = np.linspace(0, z_max, 200)
        phi_vals = [self.phi(z) for z in z_vals]
        ax2.plot(z_vals, phi_vals, color='green', linewidth=2)
        ax2.set_xlabel('Redshift z')
        ax2.set_ylabel(r'$\phi(z)$')
        ax2.set_title('Scalar Field Evolution')
        ax2.grid(True, alpha=0.3)
        
        # Plot equation of state
        self.plot_equation_of_state(z_max=z_max, ax=ax3, color='purple', linewidth=2)
        
        # Plot density evolution
        self.plot_density_evolution(z_max=z_max, ax=ax4, color='red', linewidth=2)
        
        plt.tight_layout()
        return fig, (ax1, ax2, ax3, ax4)

# ================== BAO OBSERVABLE CALCULATIONS ==================

    def comoving_distance(self, z_array):
        """
        Calculate comoving distance DM(z) in Mpc
        
        Args:
            z_array: Array of redshifts
            
        Returns:
            Array of comoving distances in Mpc
        """
        z_array = np.atleast_1d(z_array)
        
        # For each redshift, integrate c/H(z') from 0 to z
        distances = []
        
        for z_target in z_array:
            if z_target <= 0:
                distances.append(0.0)
                continue
                
            # Create integration grid
            z_int = np.linspace(0, z_target, 200)
            H_int = self.H_of_z_km_s_Mpc(z_int)
            
            # Check for valid H values
            if not np.all(np.isfinite(H_int)) or np.any(H_int <= 0):
                distances.append(np.inf)
                continue
                
            # Integrate c/H(z) dz
            integrand = self.c_km_s / H_int
            distance = np.trapz(integrand, z_int)
            distances.append(distance)
            
        return np.array(distances)

    def hubble_distance(self, z_array):
        """
        Calculate Hubble distance DH(z) = c/H(z) in Mpc
        
        Args:
            z_array: Array of redshifts
            
        Returns:
            Array of Hubble distances in Mpc
        """
        z_array = np.atleast_1d(z_array)
        H_z = self.H_of_z_km_s_Mpc(z_array)
        
        # Check for valid H values
        valid_mask = np.isfinite(H_z) & (H_z > 0)
        
        dh = np.full_like(z_array, np.inf)
        dh[valid_mask] = self.c_km_s / H_z[valid_mask]
        
        return dh

    def volume_averaged_distance(self, z_array):
        """
        Calculate volume-averaged distance DV(z) = [z*DM(z)^2*DH(z)]^(1/3) in Mpc
        
        Args:
            z_array: Array of redshifts
            
        Returns:
            Array of volume-averaged distances in Mpc
        """
        z_array = np.atleast_1d(z_array)
        
        dm = self.comoving_distance(z_array)
        dh = self.hubble_distance(z_array)
        
        # DV = [z * DM^2 * DH]^(1/3)
        valid_mask = np.isfinite(dm) & np.isfinite(dh) & (dm > 0) & (dh > 0) & (z_array > 0)
        
        dv = np.full_like(z_array, np.inf)
        dv[valid_mask] = (z_array[valid_mask] * dm[valid_mask]**2 * dh[valid_mask])**(1/3)
        
        return dv

    def get_bao_observables(self, z_array, rd=None):
        """
        Calculate all BAO observables: DM/rd, DH/rd, DV/rd
        
        Args:
            z_array: Array of redshifts
            rd: Sound horizon at drag epoch in Mpc (default: 147.09 Mpc from DESI)
            
        Returns:
            dict: Dictionary with keys 'DM_over_rd', 'DH_over_rd', 'DV_over_rd'
        """
        if rd is None:
            rd = self.rd_fid
            
        z_array = np.atleast_1d(z_array)
        
        dm = self.comoving_distance(z_array)
        dh = self.hubble_distance(z_array)
        dv = self.volume_averaged_distance(z_array)
        
        return {
            'z': z_array,
            'DM_over_rd': dm / rd,
            'DH_over_rd': dh / rd,
            'DV_over_rd': dv / rd,
            'DM': dm,
            'DH': dh,
            'DV': dv
        }

    def get_desi_predictions(self, desi_z_values):
        """
        Get BAO predictions for DESI measurement redshifts
        
        Args:
            desi_z_values: Array of DESI measurement redshifts
            
        Returns:
            dict: BAO observables at DESI redshifts
        """
        return self.get_bao_observables(desi_z_values)

    def compare_to_desi_data(self, desi_data):
        """
        Compare model predictions to DESI data
        
        Args:
            desi_data: Dictionary or DataFrame with DESI measurements
                      Should have keys/columns: 'z_eff', 'value', 'quantity'
                      
        Returns:
            dict: Comparison results with residuals and chi-squared
        """
        
        # Extract unique redshifts from DESI data
        if hasattr(desi_data, 'keys'):  # Dictionary-like
            z_values = desi_data['z_eff']
            measurements = desi_data['value']
            quantities = desi_data['quantity']
        else:  # Assume it's a structured array or similar
            z_values = desi_data[:, 0]
            measurements = desi_data[:, 1] 
            quantities = desi_data[:, 2] if desi_data.shape[1] > 2 else None

        unique_z = np.unique(z_values)
        predictions = self.get_bao_observables(unique_z)
        
        results = {
            'z_values': z_values,
            'measurements': measurements,
            'quantities': quantities,
            'predictions': {},
            'residuals': [],
            'chi2_contributions': []
        }
        
        # Create prediction lookup
        pred_lookup = {}
        for i, z in enumerate(unique_z):
            pred_lookup[z] = {
                'DM_over_rd': predictions['DM_over_rd'][i],
                'DH_over_rd': predictions['DH_over_rd'][i], 
                'DV_over_rd': predictions['DV_over_rd'][i]
            }
        
        # Compare each measurement
        for i in range(len(z_values)):
            z_val = z_values[i]
            meas_val = measurements[i]
            
            if quantities is not None:
                quant = quantities[i]
                if 'DM_over_r' in str(quant):
                    pred_val = pred_lookup[z_val]['DM_over_rd']
                elif 'DH_over_r' in str(quant):
                    pred_val = pred_lookup[z_val]['DH_over_rd']
                elif 'DV_over_r' in str(quant):
                    pred_val = pred_lookup[z_val]['DV_over_rd']
                else:
                    pred_val = np.nan
            else:
                # Default to DM if no quantity specified
                pred_val = pred_lookup[z_val]['DM_over_rd']
            
            residual = pred_val - meas_val
            results['predictions'][i] = pred_val
            results['residuals'].append(residual)
            results['chi2_contributions'].append(residual**2)
        
        results['residuals'] = np.array(results['residuals'])
        results['chi2_contributions'] = np.array(results['chi2_contributions'])
        results['total_chi2'] = np.sum(results['chi2_contributions'])
        
        return results
    
    @property
    def c_km_s(self):
        return 299792.458

    @property 
    def rd_fid(self):
        return 147.09

    def H_of_z_km_s_Mpc(self, z):
        return self.H_of_z(z) * self.H0 / self.H0_Gyr

    def get_H_z_for_desi_likelihood(self, z_max=3.0, n_points=200):
        z_array = np.linspace(0, z_max, n_points)
        H_z_array = self.H_of_z_km_s_Mpc(z_array)
        return z_array, H_z_array



