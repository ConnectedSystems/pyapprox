from typing import Callable, Optional, List, Dict

import copy
from types import MethodType

import numpy as np

import pyapprox as pya
from pyapprox.model_interfaces.model import PyaModel


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
    try:
        self.training_samples, _ = self.sample(num_samples, self.sampler)
    except AttributeError:
        raise AttributeError("Sampler must be set first!")

    return self


def build(self, samples: int, **kwargs):
    """Generate emulator using Gaussian Process approaches.

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

    if self.is_sklearn_gp:
        # Handle methods which inherit from sklearn
        self.model = self._approach(kernel=self.sampler.kernel, **kwargs)

        try:
            self.model.setup(self.target_model, self.sampler)
        except AttributeError as e:
            if 'setup' in str(e):
                # Given method doesn't have a setup() defined
                pass
            else:
                raise AttributeError(e)
    else:
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

    # For compatibility with sklearn-based methods
    import inspect
    classes = inspect.getmro(model._approach)
    model.is_sklearn_gp = False
    for cls_i in classes:
        if 'GaussianProcessRegressor' in cls_i.__name__:
            model.is_sklearn_gp = True
            break

    return model
