from functools import partial
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import interp1d, UnivariateSpline

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
        z_init=1e2,         # Starting redshift
        A_min = 1e-10,
        A_max = 1e10,  
        atol = 1e-10,
        rtol = 1e-10,
        verbose = True, 
        tune_amplitude_flag=True,      
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
        self.Omega_phi_target = 1.0 - Omega_m - Omega_r - Omega_k

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

        # Placeholder for tuned amplitude and solution
        self.solution = None

        # Store tolerances
        self.atol = atol
        self.rtol = rtol

        self.verbose = verbose

        # Automatically tune on init
        if tune_amplitude_flag:
        # if self.amplitude is None:
            self._tune_amplitude(A_min=A_min, A_max=A_max)
        # After tuning, solve evolution
            self._solve_evolution()
        else:
            self.amplitude  = 1.0
            self.solution = None
    

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
        """ System of equations to solve for _phi and _phi̇ """
        phi, phidot = y
        z  = self.N_to_z(N)
        H = self.H_of(phi, phidot, z)
        dphi_dN = phidot / H
        dphidot_dN = -(3*H*phidot + self.dV_dphi(phi)) / H
        return [dphi_dN, dphidot_dN]

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

    def H_of_z(self, z=None):
        """ Return the Hubble parameter at redshift z """
        z_arr = z if z is not None else self.solution['z']
        return self.H_of_N(self.z_to_N(z_arr))


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