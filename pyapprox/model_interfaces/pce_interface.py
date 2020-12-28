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
    if not sampler:
        sampler = pya.generate_independent_random_samples

    return sampler(self.var_trans.variable, num_samples)


def sample_training(self, num_samples: int,
                    sampler: Optional[Callable] = None):
    """Create training samples.

    Parameters
    ----------
    num_samples : int,
        Number of samples to take
    sampler : function, optional
        NumPy-provided sampler to use. Defaults to `np.random.uniform`
    """
    if not sampler:
        sampler = np.random.uniform

    var_ranges = np.array(self.variable_ranges)

    self.training_samples = sampler(low=var_ranges[:, 0],
                                    high=var_ranges[:, 1],
                                    size=(num_samples, self.nvars)).T

    return self


def sample_validation(self,
                      num_samples: int,
                      sampler: Optional[Callable] = None,
                      model_args: Optional[Dict] = None):
    """Create validation samples.

    Parameters
    ----------
    num_samples : int,
        Number of samples to take
    sampler : function, optional
        NumPy-provided sampler to use. Defaults to `np.random.uniform`
    model_args : Dict, optional
        Additional arguments to pass to target model
    """
    self.validation_samples = self.sample(num_samples, sampler)

    if not model_args:
        model_args = {}

    self.validation_results = self.run_model(
        self.validation_samples, **model_args)

    return self


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

    self.pce.set_refinement_functions(variance_refinement_func,
                                      admissibility_func,
                                      growth_rule)

    return self


def override_build_function(self, fun: Callable):
    """Overrides build method with provided function.

    Parameters
    ----------
    fun : function,
        Build function to use instead of the default approach.
    """
    self.build = MethodType(fun, self)

    return self


def build(self, callback: Callable):
    """Default build method."""
    self.pce.build(callback)

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
               num_samples: int,
               **kwargs):
    """
    Define the Polynomial Chaos Expansion method to use.

    Parameters
    ----------
    approach : function,
        PCE method constructor
    transform_approach: function,
        variable transform method
    num_samples : int,
        number of initial training/candidate samples
    kwargs : Dict,
        additional keyword arguments to pass on to the PCE method
        constructor
    """
    from inspect import getmembers, isfunction
    import importlib

    def is_function_local(object):
        return isinstance(object, FunctionType) and object.__module__ == __name__

    # Attach PCE interface methods
    mod = importlib.import_module(__name__)
    funcs = getmembers(mod, is_function_local)
    for n, f in funcs:
        setattr(model, n, MethodType(f, model))

    # Remove this constructor method
    del model.define_pce

    # Handle common args
    approach_args = get_arg_names(approach)
    if 'num_vars' in approach_args:
        kwargs['num_vars'] = model.nvars
    if 'nvars' in approach_args:
        kwargs['nvars'] = model.nvars
    if 'candidate_samples' in approach_args:
        if not hasattr(model, 'training_samples'):
            model.sample_training(num_samples)

        kwargs['candidate_samples'] = model.training_samples

    poly = approach(**kwargs)
    variables = pya.IndependentMultivariateRandomVariable(model.variables)
    model.var_trans = transform_approach(variables)

    try:
        poly.set_function(model.target_model, model.var_trans)
    except AttributeError:
        raise AttributeError(
            "Defining poly options failed: must define variables first.")

    model.pce = poly

    return model
