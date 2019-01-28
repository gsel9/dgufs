# -*- coding: utf-8 -*_
#
# dgufs.py
#
# MATLAB basis for implementation:
# https://github.com/eeGuoJun/AAAI2018_DGUFS/blob/master/JunGuo_AAAI_2018_DGUFS_code/
#
import numpy as np

import utils

from scipy import linalg
from scipy.spatial import distance

from sklearn.base import BaseEstimator, TransformerMixin

"""
The Dependence Guided Unsupervised Feature Selection algorithm by Jun Guo and
Wenwu Zhu (2018).

"""


class DGUFS(BaseEstimator, TransformerMixin):
    """The Dependence Guided Unsupervised Feature Selection (DGUFS) algorithm
    developed by Jun Guo and Wenwu Zhu.

    alpha (): Regularization term for the Lagrange multipliers.

    """

    def __init__(
        self,
        num_features=2,
        num_clusters=2,
        num_neighbors=2,
        alpha=0.5,
        beta=0.9,
        tol=5e-7,
        max_iter=1, #1e2,
        mu=1e-6,
        max_mu=1e10,
        rho=1.1
    ):

        self.num_features = num_features
        self.num_clusters = num_clusters
        self.num_neighbors = num_neighbors
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

    def _construct_matrices(self, nrows, ncols):
        # Setup:

        self.Y = np.zeros((ncols, nrows), dtype=float)
        self.Z = np.zeros((ncols, nrows), dtype=float)

        self.M = np.zeros((nrows, nrows), dtype=float)
        self.L = np.zeros((nrows, nrows), dtype=float)

        self.Lamda1 = np.zeros((ncols, nrows), dtype=float)
        self.Lamda2 = np.zeros((nrows, nrows), dtype=float)

        return self

    def _check_X(self, X):

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

    def fit(self, X, y=None, **kwargs):

        # NOTE: Returns transposed of X.
        X, nrows, ncols = self._check_X(X)

        self._construct_matrices(nrows, ncols)

        self.S = utils.similarity_matrix(X)

        scaled = (np.ones((1, nrows)) / nrows)
        self.H = np.eye(nrows) - np.ones((nrows, 1)) * scaled
        self.H = self.H / (nrows - 1)

        i = 1
        while i <= self.max_iter:

            # Alternate optimization.
            self._update_Z(X, ncols)
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

        # Obtain labels.
        #eigD, eigV = linalg.eigs(np.max(L, np.transpose(L)), self.num_clusters, 'la')
        #V = eigV * np.sqrt(eigD)

        # Sanity check.
        #assert (nrows, self.num_clusters) == np.shape(V)

        #[~, Label] = max(abs(V),[],2);
        #V = V'; % final: [nClass,nSmp]=size(V)

        # returns Y,L,V,Label
        # Y are the selected features. Each column is a sample (return transposed).

        return self

    def _update_Z(self, X, ncols):

        YHLH = self.Y.dot(self.H).dot(self.L).dot(self.H)
        U = X - self.Y - ((1 - self.beta) * YHLH - self.Lamda1) / self. mu
        self.Z = X - utils.solve_l20(U, (ncols - self.num_features))

        return self

    def _update_Y(self):

        ZLH = self.Z.dot(self.H).dot(self.L).dot(self.H)
        U = self.Z + ((1 - self.beta) * ZLH + self.Lamda1) / self.mu
        self.Y = utils.solve_l20(U, self.num_features)

        return self

    def _update_L(self):

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

        M = self.L + self.Lamda2 / self.mu
        M = utils.solve_l0_binary(M, 2 * gamma / self.mu)
        self.M = M - np.diag(np.diag(M)) + np.eye(nrows)

        return self

    def transform(self, X, y=None, **kwargs):
        pass



if __name__ == '__main__':

    #X = np.array(
    #    [[ 1, -4, 22], [12,  4,  0], [12,  0, -2], [12,  15, -2], [9,  3, 0]]
    #).T

    import pandas as pd
    X = pd.read_csv('./../../ms/data_source/to_analysis/sqroot_concat.csv', index_col=0)

    dgufs = DGUFS()
    dgufs.fit(X.values)
    #df_Y = pd.DataFrame(Y, columns=X.columns, index=X.index)
    #print(df_Y)
    #print(df_Y.columns[df_Y.sum() != 0])
