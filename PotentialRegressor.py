import numpy as np
from cobaya.theories import camb
from cobaya.log import LoggedError

# Helper functions for the exponential potential
def Vphi(phi, lambda_phi):
    """Exponential potential V(phi) = V0 * exp(-lambda * phi)"""
    return np.exp(-lambda_phi * phi)

def dVphi_dphi(phi, lambda_phi):
    """Derivative of the exponential potential dV/dphi"""
    return -lambda_phi * np.exp(-lambda_phi * phi)

def ddVphi_ddphi(phi, lambda_phi):
    """Second derivative of the exponential potential d^2V/dphi^2"""
    return lambda_phi**2 * np.exp(-lambda_phi * phi)

class CambQuintessence(camb.CAMB):
    """
    An integrated CAMB theory class for Cobaya that handles a custom
    tabulated quintessence potential.

    This class inherits from the base CAMB theory and modifies its parameter
    dictionary to include the tables for the 'QuintessenceInterp' model
    before calling the CAMB library.
    """
    # Parameters for generating the potential tables. Can be overridden in the YAML.
    phi_min: float = -7.0
    phi_max: float = 7.0
    phi_steps: int = 200

    # Let Cobaya know this theory has a new parameter 'lambda_phi'
    # This makes it available to be sampled.
    params = {'lambda_phi': None}

    def initialize_with_params(self):
        """
        Overrides the parent method to ensure the 'camb.transfers' helper
        knows about our custom 'lambda_phi' parameter.
        """
        # First, call the parent's initialize method to set up the transfers helper
        super().initialize_with_params()
        # Now, explicitly add 'lambda_phi' to the helper's input parameters.
        # This ensures Cobaya passes it in the params_values_dict when calling
        # the helper's 'calculate' method, which in turn calls our 'set' method.
        if 'lambda_phi' not in self._camb_transfers.input_params:
            self._camb_transfers.input_params.append('lambda_phi')

    def set(self, params_values_dict, state):
        """
        Overrides the parent CAMB.set method.

        This is the correct injection point to add our dynamically-generated
        potential tables to the parameters that will be passed to the
        CAMB library.
        """
        args = {self.translate_param(p): v for p, v in params_values_dict.items()}
        # Generate and save
        self.log.debug("Setting parameters: %r and %r", args, self.extra_args)
        # Get the value of our quintessence parameter for this step
        lambda_phi = params_values_dict.get('lambda_phi')
        if lambda_phi is None:
            raise LoggedError(
                self.log, "Parameter 'lambda_phi' not found in input parameters.")

        # 1. INJECT: Generate tables and add them to self.extra_args temporarily
        phi_train = np.linspace(self.phi_min, self.phi_max, self.phi_steps)
        
        # Define the dictionary of custom parameters to inject
        custom_de_params = {
            'dark_energy_model': 'QuintessenceInterp',
            'phi_train': phi_train,
            'V_train': Vphi(phi_train, lambda_phi),
            'dV_train': dVphi_dphi(phi_train, lambda_phi),
            'ddV_train': ddVphi_ddphi(phi_train, lambda_phi)
        }
        
        # Add them to the `extra_args` dictionary that the parent `set` method uses
        self.extra_args.update(custom_de_params)

        # 2. EXECUTE: Call the original `set` method from the parent CAMB class.
        # It will now use our modified self.extra_args.
        camb_params = super().set(params_values_dict, state)

        # 3. CLEAN UP: Remove our temporary keys from extra_args so they don't
        # persist for the next parameter evaluation.
        for key in custom_de_params:
            del self.extra_args[key]

        return camb_params

