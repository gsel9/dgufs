# -*- coding: utf-8 -*_
#
# test_dgufs.py
#
# This module is part of dgufs
#

import pytest

import numpy as np

from dgufs.dgufs import DGUFS

from sklearn.datasets import load_iris


__author__ = 'Severin E. R. Langberg'
__email__ = 'langberg91@gmail.no'


@pytest.fixture
def data():
    """Return arbitrary test data."""
    X = np.array(
        [[ 1, -4, 22], [12,  4,  0], [12,  0, -2], [12,  15, -2], [9,  3, 0]]
    )
    return np.transpose(X)


@pytest.fixture
def iris():
    """Return the Iris data set feature matrix."""

    X, _ = load_iris(return_X_y=True)
    return X


def test_correct_num_features(data):
    """Test the exact number of features are selected."""

    for num_features in [2, 3, 4]:
        dgufs = DGUFS(num_features=num_features)
        dgufs.fit(data)
        assert len(dgufs.indicators) == num_features


def test_error_num_features(data):
    """Test an error is raised if specifying too many features."""

    with pytest.raises(ValueError):
        dgufs = DGUFS(num_features=10)
        dgufs.fit(data)


def test_X_Z_norm(data):

    #dgufs = DGUFS(num_features=num_features)
    #dgufs.fit(data)

    # test: norm(X - Z)_(2, 0) = d - m
    pass


def test_Y_norm(data):
    # Test norm(Y)_(2, 0) = m

    dgufs = DGUFS()
    dgufs.fit(data)

    l20_norm = np.linalg.norm(np.linalg.norm(dgufs.Y, ord=2, axis=1), ord=0)
    assert l20_norm == dgufs.num_features


def test_M(data):
    """Test M matrix is symmetric and binary."""

    nrows, _ = np.shape(data)

    dgufs = DGUFS()
    dgufs.fit(data)
    assert np.shape(dgufs.M) == (nrows, nrows)
    assert len(np.unique(dgufs.M)) == 2


# TODO: rank(L) = c, diag(L) = Identity
def test_L(data):
    """ Test properties of L matrix."""

    dgufs = DGUFS()
    dgufs.fit(data)

    # Test L is positive semi-definite.
    assert np.all(np.linalg.eigvals(dgufs.L) >= 0)
    # Test rank(L) = c
    #assert np.linalg.matrix_rank(dgufs.L) == dgufs.num_clusters
    # Test diag(L) = Identity
    #assert np.array_equal(np.diag(dgufs.L), np.eye(np.shape(dgufs.L)))


def test_V(data):
    """Test (n clusters x nrows) = shape(V)."""

    nrows, _ = np.shape(data)

    dgufs = DGUFS()
    dgufs.fit(data)
    assert np.shape(dgufs.cluster_labels) == (dgufs.num_clusters, nrows)


def test_iris(iris):
    """Select the two most important features from the Iris data set."""

    dgufs = DGUFS(num_features=2)
    dgufs.fit(iris)

    assert len(dgufs.indicators) == 2
