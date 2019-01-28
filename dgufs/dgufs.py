# From: Dependence Guided Unsupervised Feature Selection, Guo & Zhu.

import numpy as np

import utils

from scipy.spatial.distance import pdist, squareform

from sklearn.base import BaseEstimator, TransformerMixin

"""
The implementation is based on the MATLAB code:
https://github.com/eeGuoJun/AAAI2018_DGUFS/blob/master/JunGuo_AAAI_2018_DGUFS_code/files/speedUp.m

"""


def feature_screening():
    # PArallelizable work func for feature screening.
    pass


# Checkout: https://github.com/eeGuoJun/AAAI2018_DGUFS/tree/master/JunGuo_AAAI_2018_DGUFS_code
class DGUFS(BaseEstimator, TransformerMixin):
    """

    reg_alpha (): Regularizationterm for the Lagrange multiplier

    """

    def __init__(
        self,
        num_features=2,
        num_clusters=2,
        num_neighbors=4,
        reg_alpha=0.5,
        reg_beta=0.5,
        tol=5e-7,
        max_iter=1e2,
        mu=1e-6,
        max_mu=1e10,
        rho=1.1
    ):

        self.num_features = num_features
        self.num_clusters = num_clusters
        self.num_neighbors = num_neighbors
        self.reg_alpha = reg_alpha
        self.reg_beta = reg_beta
        self.tol = tol
        self.max_iter = max_iter
        self.mu = mu
        self.max_mu = max_mu
        self.rho = rho

        self._sel_features = None
        self._cluster_labels = None

    def fit(self, X, y=None, **kwargs):

        nrows, ncols = np.shape(X)

        if self.num_features > ncols:
            raise ValueError('Number of features to select exceeds the number '
                             'of columns in X ({})'.format(ncols))

        if nrows < 2:
            raise RuntimeError('Feature selection requires more than two '
                               'samples')

        H = np.eye(nrows) - np.ones((nrows, 1)) * (np.ones((1, nrows)) / nrows)
        H = H / (nrows - 1)


        Y = np.zeros((ncols, nrows), dtype=float)
        Z = np.zeros((ncols, nrows), dtype=float)

        M = np.zeros((nrows, nrows), dtype=float)
        L = np.zeros((nrows, nrows), dtype=float)

        Lamda1 = np.zeros((ncols, nrows), dtype=float)
        Lamda2 = np.zeros((nrows, nrows), dtype=float)

        # TEMP:
        rho = 1.1
        max_mu = 1e10
        mu = 1e-6
        # maximum number of iterations
        max_Iter = 1e2
        # the other stop criterion for iteration
        tol = 5e-7



        print(H)
        # As in MATLAB implementation.
        #X = np.transpose(X)
        #X_sim = utils.similarity_matrix(X)

        # Update Y:
        # Let U = Z + 1/mu((1 - beta)ZHLH + Lambda1): Update Y by algorithm 1.

        # Update Z:
        # Optimal (X - Z) with algorithm 1. Update Z by using the definition of U.


    def transform(self, X, y=None, **kwargs):
        pass



if __name__ == '__main__':

    X = np.array(
        [[ 1, -4, 22], [12,  4,  0], [12,  0, -2], [12,  15, -2], [9,  3, 0]]
    )
    dgufs = DGUFS()
    dgufs.fit(X)
