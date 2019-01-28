# -*- coding: utf-8 -*_
#
# test_dgufs.py
#
# This module is part of dgufs
#

import pytest

import numpy as np

from dgufs.dgufs import DGUFS


__author__ = 'Severin E. R. Langberg'
__email__ = 'langberg91@gmail.no'


@pytest.fixture
def data():
    """Create test data."""
    X = np.array(
        [[ 1, -4, 22], [12,  4,  0], [12,  0, -2], [12,  15, -2], [9,  3, 0]]
    )
    return np.transpose(X)


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


def test_sample_size(data):
    """Test an error is raised if too few samples."""
    with pytest.raises(RuntimeError):
        dgufs = DGUFS()
        dgufs.fit(data[0, :])


def test_X_Z_norm(data):

    #dgufs = DGUFS(num_features=num_features)
    #dgufs.fit(data)

    # test: norm(X - Z)_(2, 0) = d - m
    pass


def test_Y_norm(data):
    # Test norm(Y)_(2, 0) = m
    pass


def test_M(data):
    # Test M is symmetric and binary.
    pass


def test_L(data):
    # test L is symmetric and positive semi-definite.
    # test diag(L) = Identity
    # Test rank(L) = c
    pass


def test_V(data):

    # test [nrows, num_clusters] = size(V)
    pass
