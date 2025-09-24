from cobaya.typing import InfoDict, empty_dict
from cobaya.log import LoggedError, abstract, get_logger
from cobaya.theory import Theory
import numpy as np

def Vphi(phi, lambda_phi):
    """Exponential potential V(phi) = V0 * exp(-lambda * phi)"""
    return np.exp(-lambda_phi * phi)

def dVphi_dphi(phi, lambda_phi):
    """Derivative of the exponential potential dV/dphi = -lambda * V0 * exp(-lambda * phi)"""
    return -lambda_phi * np.exp(-lambda_phi * phi)

def ddVphi_ddphi(phi, lambda_phi):
    """Second derivative of the exponential potential d^2V/dphi^2 = lambda^2 * V0 * exp(-lambda * phi)"""
    return lambda_phi**2 * np.exp(-lambda_phi * phi)

class PotentialRegressor(Theory):

    path: str
