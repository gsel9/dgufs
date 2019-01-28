# -*- coding: utf-8 -*_
#
# dgufs.py
#
# This module is part of dgufs
#

import pytest

import numpy as np

from scipy import linalg
from scipy.spatial import distance
from sklearn.base import BaseEstimator, TransformerMixin

from dgufs.dgufs import DGUFS

"""
The Dependence Guided Unsupervised Feature Selection algorithm by Jun Guo and
Wenwu Zhu (2018).

"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'langberg91@gmail.no'


@pytest.fixture
def data():
    # Creates test data.
    X = np.array(
        [[ 1, -4, 22], [12,  4,  0], [12,  0, -2], [12,  15, -2], [9,  3, 0]]
    )
    return np.transpose(X)


def test_num_features(data):
    """Test the exact number of features are selected."""

    for num_features in [2, 3, 4]:
        dgufs = DGUFS(num_features=num_features)
        dgufs.fit(data)
        assert len(dgufs.indicators) == num_features
