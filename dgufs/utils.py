# -*- coding: utf-8 -*_
#
# utils.py
# 

"""
These implementations are based on the MATLAB code:
https://github.com/eeGuoJun/AAAI2018_DGUFS/blob/master/JunGuo_AAAI_2018_DGUFS_code/files/speedUp.m

"""

# For Hungarian algorithm. See also: https://pypi.org/project/munkres/
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score
# For euclidean distance matrix. See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
from sklearn.metrics.pairwise import euclidean_distances

import numpy as np

from scipy import linalg
from scipy.spatial import distance

from sklearn.base import BaseEstimator, TransformerMixin


def similarity_matrix(X):

    S = distance.squareform(distance.pdist(np.transpose(X)))

    return -S / np.max(S)

def solve_l20(Q, nfeats):

    # b(i) is the (l2-norm)^2 of the i-th row of Q.
    b = np.sum(Q ** 2, axis=1)[:, np.newaxis]
    idx = np.argsort(b[:, 0])[::-1]

    P = np.zeros(np.shape(Q), dtype=float)
    P[idx[:nfeats], :] = Q[idx[:nfeats], :]

    return P

def speed_up(C):
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


def solve_rank_lagrange(A, eta):

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


def solve_l0_binary(Q, gamma):

    P = np.copy(Q)
    # Each P_ij is in {0, 1}
    if gamma > 1:
        P[Q > 0.5 * (gamma + 1)] = 1
        P[Q <= 0.5 * (gamma + 1)] = 0
    else:
        P[Q > 1] = 1
        P[Q < np.sqrt(gamma)] = 0

    return P


def best_map(L1, L2):
    """Permute labels of L2 match L1 as good as possible.

    """

    if np.size(L1) != np.size(L2):
        raise RuntimeError('Got sizes L1: {} and L2 {}, when should be equal'
                           ''.format(np.size(L1), np.size(L2)))

    Label1 = np.unique(L1); nClass1 = len(Label1)
    Label2 = np.unique(L2); nClass2 = len(Label2)

    nClass = max(nClass1,nClass2)
    G = zeros(nClass)
    for i in range(nClass1):
        for j in range(nClass2):
            G[i, j] = len(np.where(L1 == Label1[i] and L2 == Label2[j]))

    c, t = linear_sum_assignment(-1.0 * G);

    newL2 = np.zeros(np.size(L2))
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]

    return newL2


def normalized_mutual_information(y_true, y_pred):
    """Normalized Mutual Information between two clusterings."""

    return normalized_mutual_info_score(y_true, y_pred)
