import numpy as np
import camb

def Vphi(phi, lambda_phi):
    """Exponential potential V(phi) = V0 * exp(-lambda * phi)"""
    return np.exp(-lambda_phi * phi)

def dVphi_dphi(phi, lambda_phi):
    """Derivative of the exponential potential dV/dphi = -lambda * V0 * exp(-lambda * phi)"""
    return -lambda_phi * np.exp(-lambda_phi * phi)

def ddVphi_ddphi(phi, lambda_phi):
    """Second derivative of the exponential potential d^2V/dphi^2 = lambda^2 * V0 * exp(-lambda * phi)"""
    return lambda_phi**2 * np.exp(-lambda_phi * phi)

# ==========================================================================
# 1. DEFINE THE FIDUCIAL COSMOLOGY
# ==========================================================================
fiducial_params = {
    'H0': 67.2,
    'ombh2': 0.022,
    'omch2': 0.122,
    'omk': 0.0,
    'mnu': 0.06,
    'lambda_phi': 0.1,  # The parameter for your exponential potential
    'As': 2.1e-9,
    'ns': 0.965,
}

print("Comparing solvers with the following fiducial parameters:")
print(fiducial_params)


# ==========================================================================
# 2. RUN CAMB (using the syntax from your example)
# ==========================================================================
print("\n--- Running CAMB with 'QuintessenceModel' model ---")

phi_train = np.linspace(-1.,1.,100)
V_train = Vphi(phi_train, fiducial_params['lambda_phi'])
dV_train = dVphi_dphi(phi_train, fiducial_params['lambda_phi'])
ddV_train = ddVphi_ddphi(phi_train, fiducial_params['lambda_phi'])

# Use the convenient camb.set_params function
camb.set_feedback_level(level=1)
pars_camb = camb.set_params(
    ombh2=fiducial_params['ombh2'],
    omch2=fiducial_params['omch2'],
    omk=fiducial_params['omk'],
    H0=fiducial_params['H0'],
    mnu=fiducial_params['mnu'],
    As=fiducial_params['As'],
    ns=fiducial_params['ns'],
    # Set the custom dark energy model and its parameter
    dark_energy_model='QuintessenceInterp',
    phi_train=phi_train,
    V_train=V_train,
    dV_train=dV_train,
    ddV_train=ddV_train
)

# Get results from CAMB
results_camb = camb.get_results(pars_camb)

# Extract the values we want to compare
zdrag_camb = results_camb.get_derived_params()['zdrag']
rdrag_camb = results_camb.get_derived_params()['rdrag']
zstar_camb = results_camb.get_derived_params()['zstar']

print(f"results from camb: zstar = {zstar_camb}, previous fiducial was 1089.8, zdrag = {zdrag_camb}, rdrag = {rdrag_camb}")

_, w0_camb =  np.array(results_camb.get_dark_energy_rho_w(1.)).T
w0_camb = float(w0_camb)
print(f"CAMB w0 = {w0_camb}")

theta_star_camb = results_camb.cosmomc_theta() 
print(f"CAMB theta_star = {theta_star_camb}, previous fiducial was 1.0411e-2")