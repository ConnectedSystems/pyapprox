from typing import Callable, Optional, List, Dict
from types import MethodType, FunctionType

import os
import importlib
import pkgutil
import warnings
from functools import partial
from dataclasses import dataclass
from inspect import getmembers, isfunction

import numpy as np

import pyapprox as pya


def is_pya_constructor(func):
    return isfunction(func) and func.__name__.startswith('define_')


def get_method_interfaces(pkg):
    """Create list of available method interfaces.

    Parameters
    ----------
    pkg : module
        module to inspect

    Returns
    ---------
    method : list
        A list of available submodules
    """
    methods = [modname for importer, modname, ispkg in
               pkgutil.walk_packages(path=pkg.__path__)
               if modname not in
               ['model'] and 'test' not in modname]

    return methods


@dataclass
class PyaModel(object):
    def __init__(self, variables: List, variable_ranges: List,
                 target_model: Callable, var_names: Optional[List[str]] = None):
        """
        PyApprox model interface.

        Parameters
        ----------
        variables: List,
            Model parameters defined as scipy rv objects
        variable_ranges: List,
            Parameter ranges
        target_model: Callable,
            Model to emulate
        var_names: Optional[List[str]]
            name of parameters
        """
        if var_names:
            assert len(var_names) == len(variables) == len(variable_ranges)
        else:
            assert len(variables) == len(variable_ranges)

        self.variables = variables
        self.variable_ranges = variable_ranges
        self.target_model = target_model
        self.var_names = var_names

        self.nvars = len(variables)
        self._approach: Optional[Callable] = None

        self._add_interfaces()

    def _attach_approach_methods(self, interface):
        """Dynamically attach methods for the given approach.
        """
        from inspect import getmembers, isfunction, ismethod
        import importlib

        mod = importlib.import_module(interface)

        def is_function_local(obj):
            return isinstance(obj, FunctionType) and obj.__module__ == interface

        # Attach interface methods
        funcs = getmembers(mod, is_function_local)
        for n, f in funcs:
            setattr(self, n, MethodType(f, self))

        methods = getmembers(self, predicate=ismethod)
        for mth, _ in methods:
            if mth.startswith('define_'):
                # Remove constructor methods
                delattr(self, mth)

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

    def _add_interfaces(self):
        """Dynamically add constructors for available methods.

        All interfaces must have a `define_*` method.
        """
        interfaces = get_method_interfaces(pya.model_interfaces)

        for interface in interfaces:
            mod = importlib.import_module(
                f'pyapprox.model_interfaces.{interface}')
            funcs = getmembers(mod, is_pya_constructor)
            for n, f in funcs:
                setattr(self, n, MethodType(f, self))

    def sample(self, num_samples: int, sampler: Optional[Callable] = None):
        if not sampler:
            sampler = pya.generate_independent_random_samples

        return sampler(self.variables, num_samples)

    def sample_training(self,
                        num_samples: int,
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

            # Adaptive Leja PCE does not work with this sampler...
            # sampler = pya.generate_independent_random_samples

        var_ranges = np.array(self.variable_ranges)
        self.training_samples = sampler(low=var_ranges[:, 0],
                                        high=var_ranges[:, 1],
                                        size=(num_samples, self.nvars)).T
        # self.training_samples = self.sample(num_samples, sampler)

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

    def generate_training_set(self, num_training: int,
                              num_validation: int,
                              num_test: Optional[int] = None,
                              sampler: Optional[Callable] = None,
                              model_args: Optional[Dict] = None):
        self.sample_training(num_training, sampler)
        self.sample_validation(num_training, sampler, model_args)

        if num_test:
            num_test = int(num_test)
            self.sample_test = self.sample(num_test, sampler)

        return self

    def run_model(self, *args, **kwargs):
        return self.target_model(*args, **kwargs)

    def plot_performance(self,
                         y: np.ndarray,
                         y_hat: np.ndarray,
                         log: Optional[bool] = False):
        """Display emulator performance.

        TODO: split into separate file.
        """
        import matplotlib.pyplot as plt

        plt.figure()
        plt.scatter(y, y_hat)

        ax = plt.gca()

        if log:
            ax.set_yscale('log')
            ax.set_xscale('log')

        ax.set_xlabel('$y$')
        ax.set_ylabel('$\hat{y}$')

        plt.show()
