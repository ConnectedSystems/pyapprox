from typing import Callable, Optional, List, Dict
from types import MethodType

import os
import importlib
import pkgutil
import warnings
from functools import partial
from dataclasses import dataclass
from inspect import getmembers, isfunction

import pyapprox as pya


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
        self.variables = variables
        self.variable_ranges = variable_ranges
        self.target_model = target_model
        self.var_names = var_names

        self.nvars = len(variables)

        self._add_interfaces()

    def _add_interfaces(self):
        """Dynamically add constructors for available methods.
        
        All interfaces must have a `define_*` method.
        """
        interfaces = get_method_interfaces(pya.model_interfaces)

        def is_pya_constructor(func):
            return isfunction(func) and func.__name__.startswith('define_')

        for interface in interfaces:
            mod = importlib.import_module(f'pyapprox.model_interfaces.{interface}')
            funcs = getmembers(mod, is_pya_constructor)
            for n, f in funcs:
                setattr(self, n, MethodType(f, self))

    def run_model(self, *args, **kwargs):
        return self.target_model(*args, **kwargs)
