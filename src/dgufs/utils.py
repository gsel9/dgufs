"""
The Dependence Guided Unsupervised Feature Selection algorithm by Jun Guo and
Wenwu Zhu (2018).
"""

import numpy as np

# third party
import scipy
from scipy import linalg
from scipy.spatial import distance
from sklearn.metrics import normalized_mutual_info_score


def similarity_matrix(X):
    """Compute a normalized pairwise similarity matrix.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix.

    Returns
    -------
    S : ndarray of shape (n_samples, n_samples)
        Symmetric similarity matrix with values in [-1, 0], where 0
        means identical samples.
    """
    S = distance.squareform(distance.pdist(X))
    return -S / np.max(S)


def centering_matrix(nrows):
    """Construct a scaled centering (hat) matrix of size nrows x nrows.

    Parameters
    ----------
    nrows : int
        Dimension of the square centering matrix.

    Returns
    -------
    H : ndarray of shape (nrows, nrows)
        Centering matrix scaled by 1 / (nrows - 1).
    """
    scaled = np.ones((1, nrows)) / nrows
    H = np.eye(nrows) - np.ones((nrows, 1)) * scaled
    # Experimental version where H := H / (n - 1).
    return H / (nrows - 1)


def solve_l20(Q, nfeats):
    """Solve the l_{2,0}-norm minimization by hard row selection.

    Keeps the ``nfeats`` rows of Q with the largest l2-norm and zeros
    out the remaining rows.

    Parameters
    ----------
    Q : ndarray of shape (n_features, n_samples)
        Input matrix.
    nfeats : int
        Number of rows to retain. Must satisfy ``nfeats <= Q.shape[0]``.

    Returns
    -------
    P : ndarray of shape (n_features, n_samples)
        Sparse version of Q with only ``nfeats`` non-zero rows.

    Raises
    ------
    ValueError
        If ``nfeats`` exceeds the number of rows in ``Q``.
    """
    if nfeats > Q.shape[0]:
        raise ValueError(
            "nfeats ({}) must not exceed the number of rows in Q ({}).".format(
                nfeats, Q.shape[0]
            )
        )
    # b(i) is the (l2-norm)^2 of the i-th row of Q.
    b = np.sum(Q**2, axis=1)[:, np.newaxis]
    idx = np.argsort(b[:, 0])[::-1]

    P = np.zeros(np.shape(Q), dtype=float)
    P[idx[:nfeats], :] = Q[idx[:nfeats], :]

    return P


def speed_up(C):
    """Normalize an affinity matrix off-diagonal entries to [0, 1].

    Implements the preprocessing from Simultaneous Clustering and Model
    Selection (SCAMS), CVPR 2014. The main diagonal is set to zero
    before scaling.

    Parameters
    ----------
    C : ndarray of shape (N, N)
        Square affinity matrix. The input is **not** mutated.

    Returns
    -------
    C_new : ndarray of shape (N, N)
        Copy of ``C`` with the main diagonal zeroed and all off-diagonal
        values linearly scaled to [0, 1].
    """
    C = np.array(C, copy=True)
    diagmask = np.eye(np.shape(C)[0], dtype=bool)
    # Main diagonal = 0.
    C[diagmask] = 0

    # If C is (N x N), then tmp is (N*N x 1).
    tmp = np.reshape(C, (np.size(C), 1))
    # Remove the main diagonal elements of C in tmp. Then tmp has a
    # length of N * (N - 1).
    tmp = np.delete(tmp, np.where(diagmask.ravel()))
    # Scale to [0, 1] range; guard against constant matrices.
    _min = np.min(tmp)
    _max = np.max(tmp)
    tmp = (tmp - _min) / (_max - _min + 1e-10)

    affmaxo = C
    # affmaxo(~diagmask) is a column vector.
    affmaxo[np.logical_not(diagmask)] = tmp
    C_new = affmaxo

    return C_new


def solve_rank_lagrange(A, eta):
    """Solve the rank-constrained Lagrangian sub-problem.

    Symmetrizes ``A``, computes its eigen-decomposition, and zeros out
    eigenvalues whose magnitude does not exceed ``sqrt(eta)``.

    Parameters
    ----------
    A : ndarray of shape (N, N)
        Square matrix (need not be symmetric on input).
    eta : float
        Lagrange multiplier threshold parameter.

    Returns
    -------
    P : ndarray of shape (N, N)
        Rank-reduced reconstruction of ``A``.
    """
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
    """Project each entry of Q onto {0, 1} using an l0-binary threshold.

    Parameters
    ----------
    Q : ndarray
        Input matrix whose entries are thresholded.
    gamma : float
        Regularization threshold parameter.

    Returns
    -------
    P : ndarray
        Binary matrix (same shape as ``Q``) with entries in {0, 1}.
    """
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
    """Permute labels of L2 to match L1 as closely as possible.

    Uses the Hungarian algorithm (linear sum assignment) to find the
    optimal label permutation.

    Parameters
    ----------
    L1 : array-like of shape (n_samples,)
        Reference label vector.
    L2 : array-like of shape (n_samples,)
        Label vector to be re-mapped onto the label space of ``L1``.

    Returns
    -------
    newL2 : ndarray of shape (n_samples,)
        Re-mapped version of ``L2`` with the same label space as ``L1``.

    Raises
    ------
    RuntimeError
        If ``L1`` and ``L2`` have different numbers of elements.
    """
    if np.size(L1) != np.size(L2):
        raise RuntimeError(
            "Got sizes L1: {} and L2 {}, when should be equal"
            "".format(np.size(L1), np.size(L2))
        )

    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)

    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i, j] = np.where((L1 == Label1[i]) & (L2 == Label2[j]))[0].size

    c, t = scipy.optimize.linear_sum_assignment(-1.0 * G)

    newL2 = np.zeros(np.size(L2))
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]

    return newL2


def normalized_mutual_information(y_true, y_pred):
    """Compute Normalized Mutual Information (NMI) between two clusterings.

    Thin wrapper around :func:`sklearn.metrics.normalized_mutual_info_score`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted cluster labels.

    Returns
    -------
    nmi : float
        NMI score in [0, 1]. Higher values indicate more agreement.
    """
    return normalized_mutual_info_score(y_true, y_pred)
