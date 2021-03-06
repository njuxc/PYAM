"""Nearest Neighbor related algorithms"""

# Author: Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Sparseness support by Lars Buitinck <L.J.Buitinck@uva.nl>
#
# License: BSD, (C) INRIA, University of Amsterdam

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, issparse

from .base import BaseEstimator, ClassifierMixin, RegressorMixin
from .ball_tree import BallTree
from .metrics import euclidean_distances
from .utils import safe_asanyarray, atleast2d_or_csr


class NeighborsClassifier(BaseEstimator, ClassifierMixin):
    """Classifier implementing the k-nearest neighbors (k-NN) algorithm.

    Parameters
    ----------
    n_neighbors : int, optional
        Default number of neighbors. Defaults to 5.

    window_size : int, optional
        Window size passed to BallTree

    algorithm : {'auto', 'ball_tree', 'brute'}, optional
       Algorithm used to compute the nearest neighbors. 'ball_tree' will
       construct a BallTree while 'brute' will perform brute-force
       search. 'auto' will guess the most appropriate based on current dataset.
       Fitting on sparse input will override the setting of this parameter.

    Examples
    --------
    >>> samples = [[0, 0, 1], [1, 0, 0]]
    >>> labels = [0, 1]
    >>> from scikits.learn.neighbors import NeighborsClassifier
    >>> neigh = NeighborsClassifier(n_neighbors=1)
    >>> neigh.fit(samples, labels)
    NeighborsClassifier(n_neighbors=1, window_size=1, algorithm='auto')
    >>> print neigh.predict([[0,0,0]])
    [1]

    See also
    --------
    BallTree

    References
    ----------
    http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
    """

    def __init__(self, n_neighbors=5, algorithm='auto', window_size=1):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.algorithm = algorithm

    def fit(self, X, y, **params):
        """Fit the model using X, y as training data

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data.

        y : {array-like, sparse matrix}, shape = [n_samples]
            Target values, array of integer values.

        params : list of keyword, optional
            Overwrite keywords from __init__
        """
        X = safe_asanyarray(X)
        if y is None:
            raise ValueError("y must not be None")
        self._y = np.asanyarray(y)
        self._set_params(**params)

        if issparse(X):
            self.algorithm = 'brute'
        if self.algorithm == 'ball_tree' or \
           (self.algorithm == 'auto' and X.shape[1] < 20):
            self.ball_tree = BallTree(X, self.window_size)
        else:
            self.ball_tree = None
            self._fit_X = X
        return self

    def kneighbors(self, X, return_distance=True, **params):
        """Finds the K-neighbors of a point.

        Returns distance

        Parameters
        ----------
        point : array-like
            The new point.

        n_neighbors : int
            Number of neighbors to get (default is the value
            passed to the constructor).

        return_distance : boolean, optional. Defaults to True.
           If False, distances will not be returned

        Returns
        -------
        dist : array
            Array representing the lengths to point, only present if
            return_distance=True

        ind : array
            Indices of the nearest points in the population matrix.

        Examples
        --------
        In the following example, we construnct a NeighborsClassifier
        class from an array representing our data set and ask who's
        the closest point to [1,1,1]

        >>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
        >>> labels = [0, 0, 1]
        >>> from scikits.learn.neighbors import NeighborsClassifier
        >>> neigh = NeighborsClassifier(n_neighbors=1)
        >>> neigh.fit(samples, labels)
        NeighborsClassifier(n_neighbors=1, window_size=1, algorithm='auto')
        >>> print neigh.kneighbors([1., 1., 1.]) # doctest: +ELLIPSIS
        (array([[ 0.5]]), array([[2]]...))

        As you can see, it returns [[0.5]], and [[2]], which means that the
        element is at distance 0.5 and is the third element of samples
        (indexes start at 0). You can also query for multiple points:

        >>> X = [[0., 1., 0.], [1., 0., 1.]]
        >>> neigh.kneighbors(X, return_distance=False) # doctest: +ELLIPSIS
        array([[1],
               [2]]...)

        """
        self._set_params(**params)
        X = atleast2d_or_csr(X)
        if self.ball_tree is None:
            dist = euclidean_distances(X, self._fit_X, squared=True)
            # XXX: should be implemented with a partial sort
            neigh_ind = dist.argsort(axis=1)[:, :self.n_neighbors]
            if not return_distance:
                return neigh_ind
            else:
                return dist.T[neigh_ind], neigh_ind
        else:
            return self.ball_tree.query(X, self.n_neighbors,
                                        return_distance=return_distance)

    def predict(self, X, **params):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X: array
            A 2-D array representing the test point.

        n_neighbors : int
            Number of neighbors to get (default is the value
            passed to the constructor).

        Returns
        -------
        labels: array
            List of class labels (one for each data sample).
        """
        X = atleast2d_or_csr(X)
        self._set_params(**params)

        # get neighbors
        neigh_ind = self.kneighbors(X, return_distance=False)

        # compute the most popular label
        pred_labels = self._y[neigh_ind]
        from scipy import stats
        mode, _ = stats.mode(pred_labels, axis=1)
        return mode.flatten().astype(np.int)


###############################################################################
# NeighborsRegressor class for regression problems

class NeighborsRegressor(NeighborsClassifier, RegressorMixin):
    """Regression based on k-Nearest Neighbor Algorithm

    The target is predicted by local interpolation of the targets
    associated of the k-Nearest Neighbors in the training set.

    Different modes for estimating the result can be set via parameter
    mode. 'barycenter' will apply the weights that best reconstruct
    the point from its neighbors while 'mean' will apply constant
    weights to each point.

    Parameters
    ----------
    n_neighbors : int, optional
        Default number of neighbors. Defaults to 5.

    window_size : int, optional
        Window size passed to BallTree

    mode : {'mean', 'barycenter'}, optional
        Weights to apply to labels.

    algorithm : {'auto', 'ball_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors. 'ball_tree' will
        construct a BallTree, while 'brute' will perform brute-force
        search. 'auto' will guess the most appropriate based on current
        dataset.

    Examples
    --------
    >>> X = [[0], [1], [2], [3]]
    >>> y = [0, 0, 1, 1]
    >>> from scikits.learn.neighbors import NeighborsRegressor
    >>> neigh = NeighborsRegressor(n_neighbors=2)
    >>> neigh.fit(X, y)
    NeighborsRegressor(n_neighbors=2, window_size=1, mode='mean',
              algorithm='auto')
    >>> print neigh.predict([[1.5]])
    [ 0.5]

    Notes
    -----
    http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
    """

    def __init__(self, n_neighbors=5, mode='mean', algorithm='auto',
                 window_size=1):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.mode = mode
        self.algorithm = algorithm

    def predict(self, X, **params):
        """Predict the target for the provided data

        Parameters
        ----------
        X : array
            A 2-D array representing the test data.

        n_neighbors : int, optional
            Number of neighbors to get (default is the value
            passed to the constructor).

        Returns
        -------
        y: array
            List of target values (one for each data sample).
        """
        X = np.atleast_2d(np.asanyarray(X))
        self._set_params(**params)

        # compute nearest neighbors
        neigh_ind = self.kneighbors(X, return_distance=False)
        if self.ball_tree is None:
            neigh = self._fit_X[neigh_ind]
        else:
            neigh = self.ball_tree.data[neigh_ind]

        # compute interpolation on y
        if self.mode == 'barycenter':
            W = barycenter_weights(X, neigh)
            return (W * self._y[neigh_ind]).sum(axis=1)

        elif self.mode == 'mean':
            return np.mean(self._y[neigh_ind], axis=1)

        else:
            raise ValueError(
                'Unsupported mode, must be one of "barycenter" or '
                '"mean" but got %s instead' % self.mode)


###############################################################################
# Utils k-NN based Functions

def barycenter_weights(X, Z, reg=1e-3):
    """Compute barycenter weights of X from Y along the first axis

    We estimate the weights to assign to each point in Y[i] to recover
    the point X[i]. The barycenter weights sum to 1.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_dim)

    Z : array-like, shape (n_samples, n_neighbors, n_dim)

    reg: float, optional
        amount of regularization to add for the problem to be
        well-posed in the case of n_neighbors > n_dim

    Returns
    -------
    B : array-like, shape (n_samples, n_neighbors)

    Notes
    -----
    See developers note for more information.
    """
    X, Z = map(np.asanyarray, (X, Z))
    n_samples, n_neighbors = X.shape[0], Z.shape[1]
    if X.dtype.kind == 'i':
        X = X.astype(np.float)
    if Z.dtype.kind == 'i':
        Z = Z.astype(np.float)
    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    # this might raise a LinalgError if G is singular and has trace
    # zero
    for i, A in enumerate(Z.transpose(0, 2, 1)):
        C = A.T - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[::Z.shape[1] + 1] += R
        w = linalg.solve(G, v, sym_pos=True)
        B[i, :] = w / np.sum(w)
    return B


def kneighbors_graph(X, n_neighbors, mode='connectivity', reg=1e-3):
    """Computes the (weighted) graph of k-Neighbors for points in X

    Parameters
    ----------
    X : array-like or BallTree, shape = [n_samples, n_features]
        Sample data, in the form of a numpy array or a precomputed
        :class:`BallTree`.

    n_neighbors : int
        Number of neighbors for each sample.

    mode : {'connectivity', 'distance', 'barycenter'}, optional
        Type of returned matrix: 'connectivity' will return the
        connectivity matrix with ones and zeros, in 'distance' the
        edges are euclidian distance between points. In 'barycenter'
        they are the weights that best reconstruncts the point from
        its nearest neighbors.

    reg : float, optional
        Amount of regularization when solving the least-squares
        problem. Only relevant if mode='barycenter'. If None, use the
        default.

    Returns
    -------
    A : sparse matrix in CSR format, shape = [n_samples, n_samples]
        A[i,j] is assigned the weight of edge that connects i to j.

    Examples
    --------
    >>> X = [[0], [3], [1]]
    >>> from scikits.learn.neighbors import kneighbors_graph
    >>> A = kneighbors_graph(X, 2)
    >>> A.todense()
    matrix([[ 1.,  0.,  1.],
            [ 0.,  1.,  1.],
            [ 1.,  0.,  1.]])
    """
    if isinstance(X, BallTree):
        ball_tree = X
        X = ball_tree.data
    else:
        X = np.asanyarray(X)
        ball_tree = BallTree(X)

    n_samples = X.shape[0]
    n_nonzero = n_neighbors * n_samples
    A_indptr = np.arange(0, n_nonzero + 1, n_neighbors)

    # construct CSR matrix representation of the k-NN graph
    if mode is 'connectivity':
        A_data = np.ones((n_samples, n_neighbors))
        A_ind = ball_tree.query(
            X, k=n_neighbors, return_distance=False)

    elif mode is 'distance':
        data, ind = ball_tree.query(X, k=n_neighbors + 1)
        A_data, A_ind = data[:, 1:], ind[:, 1:]

    elif mode is 'barycenter':
        ind = ball_tree.query(
            X, k=n_neighbors + 1, return_distance=False)
        A_ind = ind[:, 1:]
        A_data = barycenter_weights(X, X[A_ind], reg=reg)

    else:
        raise ValueError(
            'Unsupported mode, must be one of "connectivity", '
            '"distance" or "barycenter" but got %s instead' % mode)

    return csr_matrix((A_data.flatten(), A_ind.flatten(), A_indptr),
                      shape=(n_samples, n_samples))
