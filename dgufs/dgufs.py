# From: Dependence Guided Unsupervised Feature Selection, Guo & Zhu.

import numpy as np

import utils

from scipy import linalg
from scipy.spatial import distance

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
        num_neighbors=2,
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
        self.num_neighbors = num_neighbors
        self.alpha = alpha
        self.beta = beta
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

        # From nrows x ncols to ncols x nrows.
        X = np.transpose(X)

        S = self.similarity_matrix(X)

        H = np.eye(nrows) - np.ones((nrows, 1)) * (np.ones((1, nrows)) / nrows)
        H = H / (nrows - 1)

        # Setup:
        Y = np.zeros((ncols, nrows), dtype=float)
        Z = np.zeros((ncols, nrows), dtype=float)

        M = np.zeros((nrows, nrows), dtype=float)
        L = np.zeros((nrows, nrows), dtype=float)

        Lamda1 = np.zeros((ncols, nrows), dtype=float)
        Lamda2 = np.zeros((nrows, nrows), dtype=float)


        # TEMP:
        max_iter = 1#1e2

        i = 1
        while i <= max_iter:

            # Update Z.
            U1 = X - Y - ((1 - self.beta) * Y.dot(H).dot(L).dot(H) - Lamda1) / self. mu
            Z = X - self.solve_l20(U1, (ncols - self.num_features))

            # Update Y.
            U1 = Z + ((1 - self.beta) * Z.dot(H).dot(L).dot(H) + Lamda1) / self.mu
            Y = self.solve_l20(U1, self.num_features)

            # Update L.
            speed_up = self.speed_up(H.dot(np.transpose(Y)).dot(Z).dot(H))
            U2 = ((1 - self.beta) * speed_up + self.beta * S - Lamda2) / self.mu + M
            L = self.solve_rank_lagrange(self.speed_up(U2), 2 * self.alpha / self.mu)

            # Update M.
            M = L + Lamda2 / self.mu
            gamma = 5e-3
            M = self.solve_l0_binary(M, 2 * gamma / self.mu)
            M = M - np.diag(np.diag(M)) + np.eye(nrows)

            # Check if stop criterion is satisfied.
            leq1 = Z - Y
            leq2 = L - M
            # Infinite norm.
            stopC1 = np.max(np.abs(leq1))
            stopC2 = np.max(np.abs(leq2))

            if (stopC1 < self.tol) and (stopC2 < self.tol):
                i = self.max_iter
            else:
                # Update Lagrange multipliers.
                Lamda1 = Lamda1 + self.mu * leq1
                Lamda2 = Lamda2 + self.mu * leq2
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

        return Y.T

    @staticmethod
    def similarity_matrix(X):

        S = distance.squareform(distance.pdist(np.transpose(X)))

        return -S / np.max(S)

    def solve_l20(self, Q, nfeats):

        # b(i) is the (l2-norm)^2 of the i-th row of Q.
        b = np.sum(Q ** 2, axis=1)[:, np.newaxis]
        idx = np.argsort(b[:, 0])[::-1]

        P = np.zeros(np.shape(Q), dtype=float)
        P[idx[:nfeats], :] = Q[idx[:nfeats], :]

        return P

    def speed_up(self, C):
        """Refer to Simultaneous Clustering and Model Selection (SCAMS),
        CVPR2014.

        """
        diagmask = np.eye(np.shape(C)[0], dtype=bool)
        # Main diagonal = 0.
        C[diagmask] = 0

        # If C is (N x N), then tmp is (N*N x 1).
        tmp = np.reshape(C, (np.size(C), 1))
        # Remove the main diagonal elements of C in tmp. Then tmp has a
        # length of N * (N - 1).
        tmp = np.delete(tmp, np.where(diagmask.ravel()))
        # Scale to [0, 1] range.
        tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))

        affmaxo = C
        # affmaxo(~diagmask) is a column vector.
        affmaxo[np.logical_not(diagmask)] = tmp
        C_new = affmaxo

        return C_new

    def solve_rank_lagrange(self, A, eta):

        # Guarantee symmetry.
        A = 0.5 * (A + np.transpose(A))
        tempD, tempV = linalg.eig(A)
        # Discard the imaginary part.
        tempV = np.real(tempV)
        tmpD = np.real(tempD)

        tempD = np.real(np.diag(tempD))
        # eta * rank(P)
        tmpD[tmpD <= np.sqrt(eta)] = 0
        tempD = np.diag(tmpD)

        P = tempV.dot(tempD).dot(np.transpose(tempV))

        return P

    def solve_l0_binary(self, Q, gamma):

        P = np.copy(Q)
        # Each P_ij is in {0, 1}
        if gamma > 1:
            P[Q > 0.5 * (gamma + 1)] = 1
            P[Q <= 0.5 * (gamma + 1)] = 0
        else:
            P[Q > 1] = 1
            P[Q < np.sqrt(gamma)] = 0

        return P

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
