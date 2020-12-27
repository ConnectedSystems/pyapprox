from typing import Dict, List
import warnings
import numpy as np
from scipy.linalg import solve_triangular

from pyapprox.variables import (IndependentMultivariateRandomVariable,
                                get_distribution_info,
                                float_rv_discrete,
                                define_iid_random_variables,
                                is_bounded_continuous_variable)

from pyapprox.rosenblatt_transformation import (rosenblatt_transformation,
                                                inverse_rosenblatt_transformation)

from pyapprox.nataf_transformation import (covariance_to_correlation,
                                           trans_x_to_u,
                                           trans_u_to_x,
                                           transform_correlations)

from pyapprox.univariate_quadrature import gauss_hermite_pts_wts_1D


def map_hypercube_samples(current_samples: np.ndarray, current_ranges: np.ndarray, new_ranges: np.ndarray,
                          active_vars: np.ndarray = None, tol: np.float = 2*np.finfo(float).eps):
    """
    Transform samples from one hypercube to another hypercube with different
    bounds.

    Parameters
    ----------
    current_samples : np.ndarray (num_vars, num_samples)
        The samples to be transformed

    current_ranges : np.ndarray (2*num_vars)
        The lower and upper bound of each variable of the current samples
        [lb_1,ub_1,...,lb_d,ub_d]

    new_ranges : np.ndarray (2*num_vars)
        The desired lower and upper bound of each variable
        [lb_1,ub_1,...,lb_d,ub_d]

    active_vars : np.ndarray (num_active_vars)
        The active vars to which the variable transformation should be applied
        The inactive vars have the identity applied, i.e. they remain 
        unchanged.

    tol : float
        Some functions such as optimizers will create points very close
         but outside current bounds. In this case allow small error 
         and move these points to boundary

    Returns
    -------
    new_samples : np.ndarray (num_vars, num_samples)
        The transformed samples
    """
    new_ranges = np.asarray(new_ranges)
    current_ranges = np.asarray(current_ranges)

    if np.allclose(new_ranges, current_ranges):
        return current_samples

    new_samples = current_samples.copy()
    num_vars = current_samples.shape[0]
    assert num_vars == new_ranges.shape[0]//2
    assert num_vars == current_ranges.shape[0]//2

    if active_vars is None:
        active_vars = np.arange(num_vars)

    for dd in active_vars:
        lb = current_ranges[2*dd]
        ub = current_ranges[2*dd+1]

        assert current_samples[dd, :].min() >= lb-tol
        assert current_samples[dd, :].max() <= ub+tol

        if tol > 0:
            I = np.where(current_samples[dd, :] < lb)[0]
            if I.shape[0] > 0:
                print(('c', current_samples[dd, I]))
            current_samples[dd, I] = lb
            J = np.where(current_samples[dd, :] > ub)[0]
            current_samples[dd, J] = ub

        assert ub-lb > tol

        new_samples[dd, :] = (current_samples[dd, :].copy()-lb)/(ub-lb)
        lb = new_ranges[2*dd]
        ub = new_ranges[2*dd+1]
        assert ub-lb > tol

        new_samples[dd, :] = new_samples[dd, :]*(ub-lb)+lb
        
    return new_samples


class VariableTransformInterface(object):
    """
    Interface for the various VariableTransformation classes.
    """
    def map_from_canonical_space(self, samples):
        pass

    def map_to_canonical_space(self, samples):
        pass

    def num_vars(self):
        return self.nvars



class IdentityTransformation(VariableTransformInterface):
    def __init__(self, num_vars):
        self.nvars = num_vars

    def map_from_canonical_space(self, samples):
        return samples

    def map_to_canonical_space(self, samples):
        return samples

    def map_derivatives_from_canonical_space(self, derivatives):
        return derivatives


class AffineBoundedVariableTransformation(VariableTransformInterface):
    def __init__(self, canonical_ranges, user_ranges):
        assert len(user_ranges) == len(canonical_ranges)
        self.canonical_ranges = np.asarray(canonical_ranges)
        self.user_ranges = np.asarray(user_ranges)
        self.nvars = int(len(self.user_ranges)/2)

    def map_from_canonical_space(self, canonical_samples):
        return map_hypercube_samples(
            canonical_samples, self.canonical_ranges, self.user_ranges)

    def map_to_canonical_space(self, user_samples):
        return map_hypercube_samples(
            user_samples, self.user_ranges, self.canonical_ranges)


class AffineRandomVariableTransformation(VariableTransformInterface):
    def __init__(self, variable, enforce_bounds=False):
        """
        Variable uniqueness depends on both the type of random variable
        (e.g. beta, gaussian, etc.) and the parameters of that distribution
        (e.g. loc and scale parameters) as well as any additional parameters.
        """
        if (type(variable) != IndependentMultivariateRandomVariable):
            variable = IndependentMultivariateRandomVariable(variable)
        self.variable = variable
        self.nvars = variable.num_vars()
        self.enforce_bounds = enforce_bounds

        self.scale_parameters = np.empty((self.variable.nunique_vars, 2))
        for ii in range(self.variable.nunique_vars):
            var = self.variable.unique_variables[ii]
            name, scale_dict, __ = get_distribution_info(var)

            # copy is essential here because code below modifies scale
            loc, scale = scale_dict['loc'].copy(), scale_dict['scale'].copy()
            if (is_bounded_continuous_variable(var) or
                (type(var.dist) == float_rv_discrete and
                 var.dist.name != 'discrete_chebyshev')):
                lb, ub = -1, 1
                scale /= (ub-lb)
                loc = loc-scale*lb

            self.scale_parameters[ii, :] = loc, scale

    def map_to_canonical_space(self, user_samples):
        canonical_samples = user_samples.copy()
        for ii in range(self.variable.nunique_vars):
            indices = self.variable.unique_variable_indices[ii]
            loc, scale = self.scale_parameters[ii, :]

            bounds = [loc-scale, loc+scale*2]
            var = self.variable.unique_variables[ii]
            if ((self.enforce_bounds is True) and
                (is_bounded_continuous_variable(var) is True) and
                ((np.any(user_samples[indices, :] < bounds[0])) or
                 (np.any(user_samples[indices, :] > bounds[1])))):
                I = np.where((user_samples[indices, :] < bounds[0]) |
                             (user_samples[indices, :] > bounds[1]))[1]
                # print(user_samples[indices, I], bounds)
                raise Exception(f'Sample outside the bounds {bounds}')

            canonical_samples[indices, :] = (
                user_samples[indices, :]-loc)/scale
            # if ((self.enforce_bounds is True) and
            #     (is_bounded_continuous_variable(var) is True) and
            #     ((np.any(canonical_samples[indices, :]<-1)) or
            #      (np.any(canonical_samples[indices, :]>1)))):
            #     I = np.where((canonical_samples[indices, :]<-1)|
            #                  (canonical_samples[indices, :]>1))[1]
            #     print(user_samples[indices, I], bounds)
            #     print(canonical_samples[indices, I])
            #     msg = 'After mapping, canonical samples of bounded variable '
            #     msg += 'are outside [-1,1]'
            #     raise Exception(msg)

        return canonical_samples

    def map_from_canonical_space(self, canonical_samples):
        user_samples = canonical_samples.copy()
        for ii in range(self.variable.nunique_vars):
            indices = self.variable.unique_variable_indices[ii]
            loc, scale = self.scale_parameters[ii, :]
            user_samples[indices, :] = canonical_samples[indices, :]*scale+loc

        return user_samples

    def map_derivatives_from_canonical_space(self, derivatives):
        assert derivatives.shape[0] % self.num_vars() == 0
        num_samples = int(derivatives.shape[0]/self.num_vars())
        mapped_derivatives = derivatives.copy()
        for ii in range(self.variable.nunique_vars):
            var_indices = self.variable.unique_variable_indices[ii]
            idx = np.tile(var_indices*num_samples, num_samples)+np.tile(
                np.arange(num_samples), var_indices.shape[0])
            loc, scale = self.scale_parameters[ii, :]
            mapped_derivatives[idx, :] /= scale
        return mapped_derivatives

    def samples_of_bounded_variables_inside_domain(self, samples):
        for ii in range(self.variable.nunique_vars):
            var = self.variable.unique_variables[ii]
            lb, ub = var.interval(1)
            indices = self.variable.unique_variable_indices[ii]
            if (samples[indices, :].max() > ub):
                print(samples[indices, :].max(), ub, 'ub violated')
                return False
            if samples[indices, :].min() < lb:
                print(samples[indices, :].min(), lb, 'lb violated')
                return False
        return True

    def get_ranges(self):
        ranges = np.empty((2*self.num_vars()), dtype=float)
        for ii in range(self.variable.nunique_vars):
            var = self.variable.unique_variables[ii]
            lb, ub = var.interval(1)
            indices = self.variable.unique_variable_indices[ii]
            ranges[2*indices], ranges[2*indices+1] = lb, ub
        return ranges


def define_iid_random_variable_transformation(variable_1d, num_vars):
    variable = define_iid_random_variables(variable_1d, num_vars)
    var_trans = AffineRandomVariableTransformation(variable)
    return var_trans


class RosenblattTransformation(VariableTransformInterface):
    def __init__(self, joint_density, num_vars, opts):
        self.joint_density = joint_density
        self.limits = opts['limits']
        self.num_quad_samples_1d = opts['num_quad_samples_1d']
        self.tol = opts.get('tol', 1e-12)
        self.num_bins = opts.get('num_bins', 101)
        self.nvars = num_vars
        self.canonical_variable_types = ['uniform']*self.num_vars()

    def map_from_canonical_space(self, canonical_samples):
        user_samples = inverse_rosenblatt_transformation(
            canonical_samples, self.joint_density, self.limits,
            self.num_quad_samples_1d, self.tol, self.num_bins)
        return user_samples

    def map_to_canonical_space(self, user_samples):
        canonical_samples = rosenblatt_transformation(
            user_samples, self.joint_density, self.limits,
            self.num_quad_samples_1d)
        return canonical_samples


class UniformMarginalTransformation(VariableTransformInterface):
    """
    Transform variables to have uniform marginals on [0,1]
    """

    def __init__(self, x_marginal_cdfs, x_marginal_inv_cdfs,
                 enforce_open_bounds=True):
        """
        enforce_open_bounds: boolean
            If True  - enforce that canonical samples are in (0,1)
            If False - enforce that canonical samples are in [0,1]
        """
        self.nvars = len(x_marginal_cdfs)
        self.x_marginal_cdfs = x_marginal_cdfs
        self.x_marginal_inv_cdfs = x_marginal_inv_cdfs
        self.enforce_open_bounds = enforce_open_bounds

    def map_from_canonical_space(self, canonical_samples):
        # there is a singularity at the boundary of the unit hypercube when
        # mapping to the (semi) unbounded distributions
        if self.enforce_open_bounds:
            assert canonical_samples.min() > 0 and canonical_samples.max() < 1
        user_samples = np.empty_like(canonical_samples)
        for ii in range(self.nvars):
            user_samples[ii, :] = self.x_marginal_inv_cdfs[ii](
                canonical_samples[ii, :])
        return user_samples

    def map_to_canonical_space(self, user_samples):
        canonical_samples = np.empty_like(user_samples)
        for ii in range(self.nvars):
            canonical_samples[ii, :] = self.x_marginal_cdfs[ii](
                user_samples[ii, :])
        return canonical_samples


class NatafTransformation(VariableTransformInterface):
    def __init__(self, x_marginal_cdfs, x_marginal_inv_cdfs,
                 x_marginal_pdfs, x_covariance, x_marginal_means,
                 bisection_opts=dict()):

        self.nvars = len(x_marginal_cdfs)
        self.x_marginal_cdfs = x_marginal_cdfs
        self.x_marginal_inv_cdfs = x_marginal_inv_cdfs
        self.x_marginal_pdfs = x_marginal_pdfs
        self.x_marginal_means = x_marginal_means

        self.x_correlation = covariance_to_correlation(x_covariance)
        self.x_marginal_stdevs = np.sqrt(np.diag(x_covariance))

        quad_rule = gauss_hermite_pts_wts_1D(11)
        self.z_correlation = transform_correlations(
            self.x_correlation, self.x_marginal_inv_cdfs, self.x_marginal_means,
            self.x_marginal_stdevs, quad_rule, bisection_opts)

        self.z_correlation_cholesky_factor = np.linalg.cholesky(
            self.z_correlation)

    def map_from_canonical_space(self, canonical_samples):
        return trans_u_to_x(
            canonical_samples, self.x_marginal_inv_cdfs,
            self.z_correlation_cholesky_factor)

    def map_to_canonical_space(self, user_samples):
        return trans_x_to_u(
            user_samples, self.x_marginal_cdfs,
            self.z_correlation_cholesky_factor)


class TransformationComposition(VariableTransformInterface):
    def __init__(self, transformations):
        """
        Parameters
        ----------
        transformations : list of transformation objects
            The transformations are applied first to last for 
            map_to_canonical_space and in reverse order for 
            map_from_canonical_space
        """
        self.transformations = transformations
        self.nvars = self.transformations[0].num_vars()

    def map_from_canonical_space(self, canonical_samples):
        user_samples = canonical_samples
        num_transforms = len(self.transformations)
        for ii in range(num_transforms-1, -1, -1):
            user_samples = \
                self.transformations[ii].map_from_canonical_space(user_samples)

        return user_samples

    def map_to_canonical_space(self, user_samples):
        canonical_samples = user_samples
        num_transforms = len(self.transformations)
        for ii in range(num_transforms):
            canonical_samples = \
                self.transformations[ii].map_to_canonical_space(
                    canonical_samples)

        return canonical_samples


class ConfigureVariableTransformation(VariableTransformInterface):
    """
    Class which maps one-to-one configure indices in [0, 1, 2, 3,...]
    to a set of configure values accepted by a function

    Parameters
    ---------
    nvars : integer
        The number of configure variables

    config_values : list
        The list of configure values for each configure variable. Each entry
        in the list is a 1D np.ndarray with potentiallly different sizes
    """

    def __init__(self, config_values):
        self.nvars = len(config_values)
        assert (type(config_values[0]) == list or
                type(config_values[0]) == np.ndarray)
        self.config_values = config_values

    def map_from_canonical_space(self, canonical_samples):
        """
        Map a configure multi-dimensional index to the corresponding 
        configure values
        """
        assert canonical_samples.shape[0] == self.nvars
        samples = np.empty_like(canonical_samples)
        for ii in range(samples.shape[1]):
            for jj in range(self.nvars):
                kk = canonical_samples[jj, ii]
                samples[jj, ii] = self.config_values[jj][int(kk)]
        return samples
