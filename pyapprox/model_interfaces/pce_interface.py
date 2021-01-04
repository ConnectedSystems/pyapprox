from typing import Callable, Optional, List, Dict

import warnings
from functools import partial
from dataclasses import dataclass
from types import MethodType, FunctionType

import pyapprox as pya
from pyapprox.models.wrappers import get_arg_names
from pyapprox.model_interfaces.model import PyaModel

import numpy as np


def sample(self, num_samples: int, sampler: Optional[Callable] = None):
    """Custom sampler method for PCE methods."""
    if not sampler:
        sampler = pya.generate_independent_random_samples

    return sampler(self.var_trans.variable, num_samples)


def set_admissibility(self, max_admiss_func: Callable,
                      variance_refinement_func: Callable,
                      max_num_samples,
                      error_tol,
                      growth_rule,
                      max_level: Optional[np.float],
                      max_level_1d: Optional[List[np.float]] = None):
    if not max_level_1d:
        max_level_1d = [max_level]*(self.nvars)

    num_samples = self.training_samples.shape[1]
    if max_num_samples >= num_samples:
        msg = f"""Max. number of samples >= available samples
        ({max_num_samples} >= {num_samples})! This can cause issues..."""
        warnings.warn(msg)

    admissibility_func = partial(max_admiss_func,
                                 max_level,
                                 max_level_1d,
                                 max_num_samples,
                                 error_tol)

    if not variance_refinement_func:
        variance_refinement_func = pya.variance_pce_refinement_indicator

    self.model.set_refinement_functions(variance_refinement_func,
                                      admissibility_func,
                                      growth_rule)

    return self


def build(self, admissibility: Dict,
          callback: Optional[Callable] = None, **kwargs):
    """Default PCE build method.

    Parameters
    ----------
    admissibility: Dict,
        admissibility_options
    callback: function, optional
        Additional function to call
    """
    approach = self._approach

    # Handle common args
    approach_args = get_arg_names(approach)
    if 'num_vars' in approach_args:
        kwargs['num_vars'] = self.nvars
    if 'nvars' in approach_args:
        kwargs['nvars'] = self.nvars
    if 'candidate_samples' in approach_args:
        kwargs['candidate_samples'] = self.training_samples

    poly = approach(**kwargs)
    poly.set_function(self.target_model, self.var_trans)
    self.model = poly

    self.set_admissibility(**admissibility)
    
    self.model.build(callback)

    return self


def plot_error_decay(self):
    """Display training results in terms of error decay."""
    import matplotlib.pyplot as plt

    plt.figure()
    plt.loglog(self.build_samples, self.build_errors, 'o-')
    plt.show()
    # Could return axes objects here...
    

def define_pce(model: PyaModel, approach: Callable,
               transform_approach: Callable,
               **kwargs):
    """
    Define the Polynomial Chaos Expansion method to use.

    Parameters
    ----------
    approach : function,
        PCE method constructor
    transform_approach: function,
        variable transform method
    kwargs : Dict,
        additional keyword arguments to pass on to the PCE method
        constructor
    """
    model._attach_approach_methods(__name__)
    model.variables = pya.IndependentMultivariateRandomVariable(model.variables)
    model.var_trans = transform_approach(model.variables)

    model._approach = approach

    return model
