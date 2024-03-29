"""
A Theil-Sen Estimator for Multiple Linear Regression Model
"""

# Author: Florian Wilhelm <florian.wilhelm@gmail.com>
#
# License: BSD 3 clause

from __future__ import division, print_function, absolute_import

import warnings
from itertools import combinations

import numpy as np
from scipy import linalg
from scipy.special import binom
from scipy.linalg.lapack import get_lapack_funcs

from .base import LinearModel
from ..base import RegressorMixin
from ..utils import check_array, check_random_state, ConvergenceWarning
from ..utils import check_consistent_length, _get_n_jobs
from ..externals.joblib import Parallel, delayed
from ..externals.six.moves import xrange


def _modified_weiszfeld_step(X, y):
    """Modified Weiszfeld step.

    Parameters
    ----------
    X : array, shape = [n_samples, n_features]
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array, shape = [n_features]
        Current start vector.

    Returns
    -------
    new_y : array, shape = [n_features]
        New iteration step.

    Notes
    -----

    This function defines one iteration step in order to approximate the
    spatial median (L1 median). It is a form of an iteratively re-weighted
    least squares method.
    """
    X = X.T
    diff = X.T - y
    normdiff = np.sqrt(np.sum(diff ** 2, axis=1))
    mask = normdiff >= 1e-6
    if mask.sum() < X.shape[1]:
        eta = 1.
    else:
        eta = 0.
    diff = diff[mask, :]
    normdiff = normdiff[mask][:, np.newaxis]
    res = np.sum(diff / normdiff, axis=0)
    T_denom = np.sum(1 / normdiff, axis=0)
    T_nom = np.sum(X.T[mask, :] / normdiff, axis=0)
    if T_denom != 0.:
        T = T_nom / T_denom
    else:
        T = T_nom
    r = linalg.norm(res)
    if r < 1.e-6:
        r = 1.
    return max(0., 1. - eta / r) * T + min(1., eta / r) * y


def _spatial_median(X, n_iter=300, tol=1.e-3):
    """Spatial median (L1 median).

    Parameters
    ----------
    X : array, shape = [n_samples, n_features]
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    n_iter : int, optional
        Maximum number of iterations.  Default is 300.

    tol : float, optional
        Stop the algorithm if spmed has converged. Default is 1.e-3.

    Returns
    -------
    spmed : array, shape = [n_features]
        Spatial median.

    References
    ----------

    - On Computation of Spatial Median for Robust Data Mining, 2005
      T. Kärkkäinen and S. Äyrämö
      http://users.jyu.fi/~samiayr/pdf/ayramo_eurogen05.pdf
    """
    # We are computing the tol on the squared norm
    tol **= 2
    spmed_old = np.mean(X, axis=0)
    for _ in xrange(n_iter):
        spmed = _modified_weiszfeld_step(X, spmed_old)
        if np.sum((spmed_old - spmed) ** 2) < tol:
            return spmed
        else:
            spmed_old = spmed
    warnings.warn("Maximum number of iterations reached.", ConvergenceWarning)
    return spmed


def _breakdown_point(n_samples, n_subsamples):
    """Approximation of the breakdown point.

    Parameters
    ----------
    n_samples : int
        Number of samples.

    n_subsamples : int
        Number of subsamples to consider.

    Returns
    -------
    breakdown_point : float
        Approximation of breakdown point.
    """
    return 1 - (0.5 ** (1 / n_subsamples) * (n_samples - n_subsamples + 1) +
                n_subsamples - 1) / n_samples


def _lstsq(X, y, indices, intercept):
    """Least Squares Estimator for TheilSen class.

    Parameters
    ----------
    X : array, shape = [n_samples, n_features]
        Design matrix, where n_samples is the number of samples and
        n_features is the number of features.

    y : array, shape = [n_samples]
        Target vector, where n_samples is the number of samples.

    indices : array, shape = [n_subpopulation, n_subsamples]
        Indices of all subsamples with respect to the chosen subpopulation.

    intercept : bool
        Fit intercept or not.

    Notes
    -----
    This function calculates the least squares method on a subset of rows of X
    and y defined by the indices array. Optionally, an intercept column is
    added if intercept is set to true.
    """
    first_elem = 1 if intercept else 0
    n_dim = X.shape[1] + first_elem
    n_subsamples = indices.shape[1]
    weights = np.empty((indices.shape[0], n_dim))
    X_sub = np.ones((n_subsamples, n_dim))
    # gelss need to pad y to be of the max dim of X
    this_y = np.zeros((max(n_subsamples, n_dim)))
    lstsq, = get_lapack_funcs(('gelss',), (X_sub, this_y))
    for i, ix in enumerate(indices):
        ix = list(ix)
        X_sub[:, first_elem:] = X[ix, :]
        this_y[:n_subsamples] = y[ix]
        weights[i, :] = lstsq(X_sub, this_y)[1][:n_dim]
    return weights


class TheilSen(LinearModel, RegressorMixin):
    """Theil-Sen Estimator: robust multivariate regression model.

    Parameters
    ----------
    fit_intercept : boolean, optional, default True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations.

    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.

    max_subpopulation : int, optional, default 1e4
        Instead of computing with a set of cardinality 'n choose k', where n is
        the number of samples and k is the number of subsamples (at least
        number of features), consider only a stochastic subpopulation of a
        given maximal size if 'n choose k' is larger than max_subpopulation.
        For other than small problem sizes this parameter will determine
        memory usage and runtime if n_subsamples is not changed.

    n_subsamples : int, optional, default None
        Number of samples to calculate the parameters. This is at least the
        number of features (plus 1 if fit_intercept=True) and the number of
        samples as a maximum. A lower number leads to a higher breakdown
        point and a low efficiency while a high number leads to a low
        breakdown point and a high efficiency. If None, take the
        minimum number of subsamples leading to maximal robustness.
        If n_subsamples is set to n_samples, Theil-Sen is identical to least
        squares.

    n_iter : int, optional, default 300
        Maximum number of iterations for the calculation of spatial median.

    tol : float, optional, default 1.e-3
        Tolerance when calculating spatial median.

    random_state : RandomState or an int seed, optional, default None
        A random number generator instance to define the state of the
        random permutations generator.

    n_jobs : integer, optional, default 1
        Number of CPUs to use during the cross validation. If ``-1``, use
        all the CPUs.

    verbose : boolean, optional, default False
        Verbose mode when fitting the model.

    Attributes
    ----------
    `coef_` : array, shape = (n_features)
        Coefficients of the regression model (median of distribution).

    `intercept_` : float
        Estimated intercept of regression model.

    `breakdown_` : float
        Approximated breakdown point.

    `random_state_` : RandomState
        The current random state.

    References
    ----------

    - Theil-Sen Estimators in a Multiple Linear Regression Model, 2009
      Xin Dang, Hanxiang Peng, Xueqin Wang and Heping Zhang
      http://www.math.iupui.edu/~hpeng/MTSE_0908.pdf

    Notes
    -----
    The algorithm calculates least square solutions on subsets with size
    n_subsamples of the samples in X. Any value of n_subsamples between the
    number of features and samples leads to an estimator with a compromise
    between robustness and efficiency. Since the number of least square
    solutions is "n_samples choose n_subsamples", it can be extremely large
    and can therefore be limited with max_subpopulation. If this limit is
    reached, the subsets are chosen randomly. In a final step, the spatial
    median (or L1 median) is calculated of all least square solutions.
    """

    def __init__(self, fit_intercept=True, copy_X=True,
                 max_subpopulation=1e4, n_subsamples=None, n_iter=300,
                 tol=1.e-3, random_state=None, n_jobs=1, verbose=False):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_subpopulation = int(max_subpopulation)
        self.n_subsamples = n_subsamples
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _check_subparams(self, n_samples, n_features):
        if self.fit_intercept:
            n_dim = n_features + 1
        else:
            n_dim = n_features
        n_subsamples = self.n_subsamples
        if n_subsamples is not None:
            assert n_subsamples <= n_samples
            if n_samples >= n_features:
                assert n_dim <= n_subsamples
            else:  # if n_samples < n_features
                assert n_subsamples == n_samples
        else:
            n_subsamples = min(n_dim, n_samples)
        if self.max_subpopulation <= 0:
            raise ValueError("Subpopulation must be positive.")
        n_all = max(1, np.rint(binom(n_samples, n_subsamples)))
        n_subpop = int(min(self.max_subpopulation, n_all))
        return n_dim, n_subsamples, n_subpop

    def fit(self, X, y):
        self.random_state_ = check_random_state(self.random_state)
        X = check_array(X)
        y = check_array(y, ensure_2d=False)
        check_consistent_length(X, y)
        n_samples, n_features = X.shape
        n_dim, n_subsample, n_subpop = self._check_subparams(n_samples,
                                                             n_features)
        self.breakdown_ = _breakdown_point(n_samples, n_subsample)
        if self.verbose:
            print("Breakdown point: {0}".format(self.breakdown_))
            print("Number of samples: {0}".format(n_samples))
            tol_outliers = int(self.breakdown_ * n_samples)
            print("Tolerable outliers: {0}".format(tol_outliers))
            print("Number of subpopulations: {0}".format(n_subpop))
        # Determine indices of subpopulation
        if np.rint(binom(n_samples, n_subsample)) <= self.max_subpopulation:
            indices = list(combinations(xrange(n_samples), n_subsample))
        else:
            indices = [self.random_state_.randint(0, n_samples, n_subsample)
                       for _ in xrange(n_subpop)]
        n_jobs = _get_n_jobs(self.n_jobs)
        idx_list = np.array_split(indices, n_jobs)
        weights = Parallel(n_jobs=n_jobs,
                           backend="multiprocessing",
                           max_nbytes=10e6,
                           verbose=self.verbose)(
            delayed(_lstsq)(X, y, idx_list[job], self.fit_intercept)
            for job in xrange(n_jobs))
        weights = np.vstack(weights)
        coefs = _spatial_median(weights, n_iter=self.n_iter, tol=self.tol)

        if self.fit_intercept:
            self.intercept_ = coefs[0]
            self.coef_ = coefs[1:]
        else:
            self.intercept_ = 0.
            self.coef_ = coefs

        return self
