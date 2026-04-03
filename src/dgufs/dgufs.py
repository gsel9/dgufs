"""
The Dependence Guided Unsupervised Feature Selection algorithm by Jun Guo and
Wenwu Zhu (2018).
"""
# third party
import warnings

import numpy as np
import pandas as pd
from . import utils
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class DGUFS(BaseEstimator, TransformerMixin):
    """The Dependence Guided Unsupervised Feature Selection (DGUFS) algorithm
    developed by Jun Guo and Wenwu Zhu.

    Parameters
    ----------
    num_features : int, default=2
        The number of features to select.
    num_clusters : int, default=2
        The number of clusters (used for graph-rank constraint on L).
    alpha : float, default=0.5
        Regularization parameter controlling the rank of the graph matrix L.
        Must be in (0, 1).
    beta : float, default=0.9
        Regularization parameter weighting the similarity term. Must be > 0.
    tol : float, default=1e-6
        Convergence tolerance. Optimization stops when both constraint
        violations fall below this threshold.
    max_iter : int, default=100
        Maximum number of ADMM iterations before stopping.
    mu : float, default=1e-6
        Initial penalty parameter for the augmented Lagrangian.
    max_mu : float, default=1e10
        Upper bound on the penalty parameter ``mu``.
    rho : float, default=1.1
        Multiplicative factor by which ``mu`` is increased each iteration.
    """

    NAME = "DGUFSSelection"

    def __init__(
        self,
        num_features=2,
        num_clusters=2,
        alpha=0.5,
        beta=0.9,
        tol=1e-6,
        max_iter=100,
        mu=1e-6,
        max_mu=1e10,
        rho=1.1,
    ):
        self.num_features = num_features
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.beta = beta
        self.tol = tol
        self.max_iter = max_iter
        self.mu = mu
        self.max_mu = max_mu
        self.rho = rho

        # NOTE: Attributes set with instance.
        self.S = None
        self.H = None
        self.Y = None
        self.Z = None
        self.M = None
        self.L = None
        self.V = None
        self.Lamda1 = None
        self.Lamda2 = None

    def _setup_matrices(self, nrows, ncols):
        # Setup.
        self.Y = np.zeros((ncols, nrows), dtype=float)
        self.Z = np.zeros((ncols, nrows), dtype=float)

        self.M = np.zeros((nrows, nrows), dtype=float)
        self.L = np.zeros((nrows, nrows), dtype=float)

        self.Lamda1 = np.zeros((ncols, nrows), dtype=float)
        self.Lamda2 = np.zeros((nrows, nrows), dtype=float)

        return self

    def __name__(self):
        return self.NAME

    def _check_X(self, X):
        # Type checking and formatting of feature matrix.
        nrows, ncols = np.shape(X)
        if self.num_features > ncols:
            raise ValueError(
                "Number of features to select exceeds the number "
                "of columns in X ({})".format(ncols)
            )
        if nrows < 2:
            raise RuntimeError("Feature selection requires more than two " "samples")
        # NB: From nrows x ncols to ncols x nrows as algorithm given in the
        # paper.
        X_trans = np.transpose(X)

        return X_trans, nrows, ncols

    @property
    def support(self):
        """Returns the column indicators of selected features."""

        # Select features based on where the transformed feature matrix has
        # column sums != 0.
        selected_cols = np.atleast_1d(
            np.squeeze(np.where(np.sum(self.Y.T, axis=0) != 0))
        )
        # Sanity check.
        assert len(selected_cols) <= self.num_features

        return selected_cols

    @property
    def memberships(self):
        """Return the cluster indicator labels for each observation."""

        check_is_fitted(self, attributes=["L", "Y", "Z", "M"])

        # NOTE: Alternatively use scipy.sparse.linalg.eigs with
        # k=self.num_clusters.
        # eigh is faster and more numerically stable for symmetric matrices.
        eigD, eigV = linalg.eigh(np.maximum(self.L, np.transpose(self.L)))
        # eigh returns eigenvalues in ascending order; reverse to get largest first.
        eigD = eigD[::-1][: self.num_clusters]
        eigV = eigV[:, ::-1][:, : self.num_clusters]
        eigD_diag = np.diag(np.real(eigD))
        # Clip to avoid sqrt of negative values from floating-point noise.
        self.V = np.dot(np.real(eigV), np.sqrt(np.maximum(eigD_diag, 0)))
        # The final cluster labels can be obtained by determining the position
        # of the largest element at each cluster indicator in V.
        return np.argmax(self.V, axis=1)

    def fit(self, X, **kwargs):
        """Select features from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix.

        Returns
        -------
        self : DGUFS
            Fitted estimator.
        """
        X_trans, nrows, ncols = self._check_X(X)

        self.S = utils.similarity_matrix(X)
        # Experimental version where H := H / (n - 1).
        self.H = utils.centering_matrix(nrows)

        self._setup_matrices(nrows, ncols)

        converged = False
        for i in range(1, int(self.max_iter) + 1):
            # Alternate optimization of matrices.
            self._update_Z(X_trans, ncols)
            self._update_Y()
            self._update_L()
            self._update_M(nrows)

            # Check if stop criterion is satisfied.
            leq1 = self.Z - self.Y
            leq2 = self.L - self.M
            stopC1 = np.max(np.abs(leq1))
            stopC2 = np.max(np.abs(leq2))
            if (stopC1 < self.tol) and (stopC2 < self.tol):
                converged = True
                break
            # Update Lagrange multipliers.
            self.Lamda1 = self.Lamda1 + self.mu * leq1
            self.Lamda2 = self.Lamda2 + self.mu * leq2
            self.mu = min(self.max_mu, self.mu * self.rho)

        if not converged:
            warnings.warn(
                "DGUFS did not converge within {} iterations. "
                "Consider increasing max_iter or relaxing tol.".format(self.max_iter),
                RuntimeWarning,
                stacklevel=2,
            )

        return self

    def _update_Z(self, X, ncols):
        """Update the auxiliary variable Z via l_{2,0} projection.

        Parameters
        ----------
        X : ndarray of shape (n_features, n_samples)
            Transposed feature matrix.
        ncols : int
            Total number of features (used to compute the complement count).
        """
        YHLH = self.Y.dot(self.H).dot(self.L).dot(self.H)
        U = X - self.Y - (((1 - self.beta) * YHLH - self.Lamda1) / self.mu)
        self.Z = X - utils.solve_l20(U, (ncols - self.num_features))

        return self

    def _update_Y(self):
        """Update the feature selection matrix Y via l_{2,0} projection."""
        ZHLH = self.Z.dot(self.H).dot(self.L).dot(self.H)
        U = self.Z + (((1 - self.beta) * ZHLH + self.Lamda1) / self.mu)
        self.Y = utils.solve_l20(U, self.num_features)

        return self

    def _update_L(self):
        """Update the graph matrix L via rank-constrained Lagrangian step."""
        speed_up = utils.speed_up(
            self.H.dot(np.transpose(self.Y)).dot(self.Z).dot(self.H)
        )
        U = (1 - self.beta) * speed_up + self.beta * self.S - self.Lamda2
        self.L = utils.solve_rank_lagrange(
            utils.speed_up(U / self.mu + self.M), 2 * self.alpha / self.mu
        )
        return self

    def _update_M(self, nrows, gamma=5e-3):
        """Update the binary adjacency matrix M via l0-binary projection.

        Parameters
        ----------
        nrows : int
            Number of samples (sets the identity diagonal block size).
        gamma : float, default=5e-3
            Sparsity regularization parameter.
        """
        _M = self.L + self.Lamda2 / self.mu
        _M = utils.solve_l0_binary(_M, 2 * gamma / self.mu)

        self.M = _M - np.diag(np.diag(_M)) + np.eye(nrows)

        return self

    def transform(self, X, **kwargs):
        """Retain selected features from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix.

        Returns
        -------
        output : array-like of shape (n_samples, num_features)
            The feature matrix containing only the selected features.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before ``fit``.
        TypeError
            If ``X`` is neither a numpy array nor a pandas DataFrame.
        """
        check_is_fitted(self, attributes=["Y"])

        if isinstance(X, pd.DataFrame):
            data = X.values
            output = pd.DataFrame(
                data[:, self.support], columns=X.columns[self.support], index=X.index
            )
        elif isinstance(X, np.ndarray):
            output = X[:, self.support]
        else:
            raise TypeError("Cannot transform data of type {}".format(type(X)))

        return output
