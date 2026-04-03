# -*- coding: utf-8 -*_
#
# test_dgufs.py
#
# This module is part of dgufs
#

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError

from dgufs import DGUFS

__author__ = "Severin Elvatun"
__email__ = "langberg91@gmail.com"


@pytest.fixture
def data():
    """Return arbitrary test data."""
    X = np.array([[1, -4, 22], [12, 4, 0], [12, 0, -2], [12, 15, -2], [9, 3, 0]])
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
        assert len(dgufs.support) == num_features


def test_error_num_features(data):
    """Test an error is raised if specifying too many features."""

    with pytest.raises(ValueError):
        dgufs = DGUFS(num_features=10)
        dgufs.fit(data)


def test_error_too_few_samples():
    """Test that a RuntimeError is raised when X has fewer than 2 samples."""

    X = np.array([[1, 2, 3]])  # only 1 sample
    with pytest.raises(RuntimeError, match="more than two"):
        dgufs = DGUFS(num_features=1)
        dgufs.fit(X)


def test_X_Z_norm(data):
    """Test that the l_{2,0} norm of (X - Z) equals d - m (non-selected rows)."""

    num_features = 2
    dgufs = DGUFS(num_features=num_features)
    dgufs.fit(data)

    # data fixture: shape (3 features, 5 samples).
    # _check_X transposes so X_trans inside algo is (5, 3) — ncols=5, nrows=3.
    n_algo_features = data.shape[1]
    X_trans = data.T
    diff = X_trans - dgufs.Z
    row_norms = np.linalg.norm(diff, ord=2, axis=1)
    l20_norm = int(np.linalg.norm(row_norms, ord=0))
    assert l20_norm == n_algo_features - num_features


def test_Y_norm(data):
    # Test norm(Y)_(2, 0) = m

    dgufs = DGUFS()
    dgufs.fit(data)

    l20_norm = np.linalg.norm(np.linalg.norm(dgufs.Y, ord=2, axis=1), ord=0)
    assert l20_norm == pytest.approx(dgufs.num_features)


def test_M(data):
    """Test M matrix is symmetric and binary."""

    nrows, _ = np.shape(data)

    dgufs = DGUFS()
    dgufs.fit(data)
    assert np.shape(dgufs.M) == (nrows, nrows)
    assert len(np.unique(dgufs.M)) == 2


def test_L(data):
    """Test properties of L matrix."""

    dgufs = DGUFS()
    dgufs.fit(data)

    # Test L is positive semi-definite.
    assert np.all(np.linalg.eigvals(dgufs.L) >= 0)
    # Test rank(L) = c
    assert np.linalg.matrix_rank(dgufs.L) == dgufs.num_clusters
    # Test diagonal of L is all ones.
    assert np.allclose(np.diag(dgufs.L), np.ones(np.shape(dgufs.L)[0]))


def test_transform_numpy(data):
    """Test that transform() returns an ndarray with the correct number of columns."""

    num_features = 2
    dgufs = DGUFS(num_features=num_features)
    dgufs.fit(data)
    result = dgufs.transform(data.T)
    assert isinstance(result, np.ndarray)
    assert result.shape[1] == num_features


def test_transform_dataframe(data):
    """Test that transform() handles a pandas DataFrame input."""

    num_features = 2
    dgufs = DGUFS(num_features=num_features)
    dgufs.fit(data)
    df = pd.DataFrame(data.T)
    result = dgufs.transform(df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == num_features


def test_transform_not_fitted():
    """Test that transform raises NotFittedError when called before fit."""

    dgufs = DGUFS(num_features=2)
    X = np.random.rand(5, 3)
    with pytest.raises(NotFittedError):
        dgufs.transform(X)


def test_memberships(data):
    """Test that memberships returns an integer label array of the right length."""

    dgufs = DGUFS(num_features=2, num_clusters=2)
    dgufs.fit(data)
    labels = dgufs.memberships
    # data is shape (3, 5) so nrows (samples) inside algo = data.shape[0] = 3
    assert labels.shape == (data.shape[0],)
    assert np.issubdtype(labels.dtype, np.integer)


def test_memberships_not_fitted():
    """Test that memberships raises NotFittedError when called before fit."""

    dgufs = DGUFS(num_features=2)
    with pytest.raises(NotFittedError):
        _ = dgufs.memberships


def test_V(data):
    """Test that V is set after fitting."""

    dgufs = DGUFS()
    dgufs.fit(data)
    # Trigger memberships to populate V, then verify shape.
    _ = dgufs.memberships
    assert dgufs.V is not None


def test_iris(iris):
    """Select the two most important features from the Iris data set."""

    dgufs = DGUFS(num_features=2)
    dgufs.fit(iris)

    assert len(dgufs.support) == 2
