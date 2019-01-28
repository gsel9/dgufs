# -*- coding: utf-8 -*_
#
# dgufs.py
#
# This module is part of dgufs
#

"""
The Dependence Guided Unsupervised Feature Selection algorithm by Jun Guo and
Wenwu Zhu (2018).

"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'langberg91@gmail.no'


import numpy as np
import pandas as pd

from dgufs import utils

from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin


class DGUFS(BaseEstimator, TransformerMixin):
    """The Dependence Guided Unsupervised Feature Selection (DGUFS) algorithm
    developed by Jun Guo and Wenwu Zhu.

    num_features (int): The number of features to select.
    num_clusters (int):
    alpha (): Regularization parameter
    beta (): Regularization parameter
    tol (float): Tolerance used to determine optimization convergance. Defaults
        to 5e-7.
    max_iter (): The maximum number of iterations of the
    mu ():
    max_mu ():
    rho ():

    """

    def __init__(
        self,
        num_features=2,
        num_clusters=2,
        alpha=0.5,
        beta=0.9,
        tol=5e-7,
        max_iter=1e2,
        mu=1e-6,
        max_mu=1e10,
        rho=1.1
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

    def _check_X(self, X):
        # Type checking and formatting of feature matrix.
        nrows, ncols = np.shape(X)
        if self.num_features > ncols:
            raise ValueError('Number of features to select exceeds the number '
                             'of columns in X ({})'.format(ncols))
        if nrows < 2:
            raise RuntimeError('Feature selection requires more than two '
                               'samples')
        # NB: From nrows x ncols to ncols x nrows as algorithm given in the
        # paper.
        X_trans = np.transpose(X)

        return X_trans, nrows, ncols

    @property
    def indicators(self):
        """Returns the column indicators of selected features."""

        # Select features based on where the transformed feature matrix has
        # column sums != 0.
        selected_cols = np.squeeze(np.where(np.sum(self.Y.T, axis=0) != 0))
        # Sanity check.
        assert len(selected_cols) <= self.num_features

        return selected_cols

    @property
    def cluster_labels(self):

        # NOTE: Alternatively use scipy.sparse.linalg.eigs with
        # k=self.num_clusters.
        eigD, eigV = linalg.eig(np.maximum(self.L, np.transpose(self.L)))
        # Discard imaginary parts and truncate assuming comps are sorted.
        eigD = np.real(np.diag(eigD)[:self.num_clusters, :self.num_clusters])
        eigV = np.real(eigV[:, :self.num_clusters])

        V = np.dot(eigV, np.sqrt(eigD))
        label = np.argmax(V, axis=0)

        return np.transpose(V)

    def fit(self, X, **kwargs):
        """Select features from X.

        Args:
            X (array-like): The feature matrix with shape
                (n samples x n features).

        """
        X_trans, nrows, ncols = self._check_X(X)

        self.S = utils.similarity_matrix(X)
        # Experimental version where H := H / (n - 1).
        self.H = utils.centering_matrix(nrows)

        self._setup_matrices(nrows, ncols)

        i = 1
        while i <= self.max_iter:

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
                i = self.max_iter
            else:
                # Update Lagrange multipliers.
                self.Lamda1 = self.Lamda1 + self.mu * leq1
                self.Lamda2 = self.Lamda2 + self.mu * leq2
                self.mu = min(self.max_mu, self.mu * self.rho);
                # Update counter.
                i = i + 1

        return self

    def _update_Z(self, X, ncols):
        # Updates the Z matrix.
        YHLH = self.Y.dot(self.H).dot(self.L).dot(self.H)
        U = X - self.Y - (((1 - self.beta) * YHLH - self.Lamda1) / self. mu)
        self.Z = X - utils.solve_l20(U, (ncols - self.num_features))

        return self

    def _update_Y(self):
        # Updates the Y matrix.
        ZHLH = self.Z.dot(self.H).dot(self.L).dot(self.H)
        U = self.Z + (((1 - self.beta) * ZHLH + self.Lamda1) / self.mu)
        self.Y = utils.solve_l20(U, self.num_features)

        return self

    def _update_L(self):
        # Updates the L matrix.
        speed_up = utils.speed_up(
            self.H.dot(np.transpose(self.Y)).dot(self.Z).dot(self.H)
        )
        U = ((1 - self.beta) * speed_up + self.beta * self.S - self.Lamda2)
        # Solve
        L = utils.solve_rank_lagrange(
            utils.speed_up(U / self.mu + self.M), 2 * self.alpha / self.mu
        )
        return self

    def _update_M(self, nrows, gamma=5e-3):
        # Updates the M matrix.
        M = self.L + self.Lamda2 / self.mu
        M = utils.solve_l0_binary(M, 2 * gamma / self.mu)

        self.M = M - np.diag(np.diag(M)) + np.eye(nrows)

        return self

    def transform(self, X, **kwargs):
        """Retain selected features from X.

        Args:
            X (array-like): The feature matrix with shape
                (n samples x n features).

        Returns:
            (array-like): The feature matrix containing only the selected
                features.

        """

        if isinstance(X, pd.DataFrame):
            data = X.values
            output = pd.DataFrame(
                data[:, self.indicators],
                columns=X.columns[self.indicators],
                index=X.index
            )
        elif isinstance(X, np.ndarray):
            output = X[:, self.indicators]
        else:
            raise TypeError('Cannot transform data of type {}'.format(type(X)))

        return output


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler

    iris = load_iris(return_X_y=False)
    X, y = iris.data, iris.target

    dgufs = DGUFS(num_features=2)
    dgufs.fit(X)
