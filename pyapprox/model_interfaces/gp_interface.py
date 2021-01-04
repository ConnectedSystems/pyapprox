from typing import Callable, Optional, List, Dict

import copy
from types import MethodType

import numpy as np

import pyapprox as pya
from pyapprox.model_interfaces.model import PyaModel


def sample(self, num_samples: int, sampler: Callable):
    """Custom sampler for GP methods.
    """
    return sampler(num_samples)


def set_sampler(self, num_samples: int, kernel: Callable, sampler: Callable):
    """Set sampler for Gaussian Process.

    Parameters
    ----------
    num_samples: int,
        Number of candidate samples to generate, from which final samples are chosen
    kernel : function, optional
        PyApprox kernel
    sampler : function, optional
        PyApprox sampler
    """
    sampler = sampler(self.nvars, num_samples, None)
    sampler.set_kernel(copy.deepcopy(kernel))

    self.sampler = sampler

    return self


def sample_training(self,
                    num_samples: int):
    """Create training samples.

    Parameters
    ----------
    num_samples : int,
        Number of samples to take
    """
    self.training_samples, _ = self.sampler(num_samples)

    return self


def build(self, samples: int, **kwargs):
    """Build method for `approximate` approaches.

    Parameters
    ----------
    samples : int,
        Number of samples to refine the Gaussian Process with
    kwargs : Dict,
        additional keyword arguments to pass on to the `GaussianProcess` method
        constructor
    """
    if not hasattr(self, 'sampler'):
        raise AttributeError("Must create training samples first.")

    self.model = self._approach(self.sampler, self.target_model, **kwargs)
    self.model.refine(samples)

    return self


def define_gp(model: PyaModel, approach: Callable):
    """
    Define the `pya.gaussian_process` method to use.

    Parameters
    ----------
    approach : function,
        GP method constructor
    transform_approach: function,
        variable transform method
    num_samples : int,
        number of initial training/candidate samples
    """
    model._attach_approach_methods(__name__)
    model.variables = pya.IndependentMultivariateRandomVariable(model.variables)

    model._approach = approach

    return model
