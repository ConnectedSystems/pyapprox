from typing import Callable, Optional, List, Dict

from types import MethodType

import pyapprox as pya
from pyapprox.model_interfaces.model import PyaModel

import numpy as np


def build(self, **kwargs):
    """Build method for `approximate` approaches.

    Parameters
    ----------
    kwargs : Dict,
        additional keyword arguments to pass on to the `approximate` method
        constructor
    """
    t_samps = self.training_samples
    v_res = self.validation_results
    
    arg_names = pya.get_arg_names(self._approach)
    if 'variable' in arg_names:
        kwargs['variable'] = self.variables


    self.model = self._approach(t_samps, v_res, **kwargs)

    def call(self, X: np.ndarray):
        return self.approx(X)

    # Make GP object callable
    # Have to attach to type rather than instance, otherwise __call__
    # will not work.
    mod_type = type(self.model)
    setattr(mod_type, '__call__', MethodType(call, self.model))

    return self


def define_approx(model: PyaModel, approach: Callable):
    """
    Define the `pya.approximate` method to use.

    Parameters
    ----------
    model : PyaModel,
        PyApprox Model wrapper
    approach : function,
        Approximate method constructor
    """
    model._attach_approach_methods(__name__)
    model.variables = pya.IndependentMultivariateRandomVariable(model.variables)

    model._approach = approach

    return model
