# Dependence Guided Unsupervised Feature Selection (DGUFS)

![CI](https://github.com/gsel9/dgufs/actions/workflows/ci.yml/badge.svg)
![Docs](https://readthedocs.org/projects/dgufs/badge/?version=latest)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

A Python implementation of the **Dependence Guided Unsupervised Feature Selection (DGUFS)** algorithm by Jun Guo and Wenwu Zhu (AAAI 2018).

DGUFS jointly selects the most informative features and partitions data into clusters, enhancing the interdependence among the original data, cluster labels, and selected features — all without requiring labelled data.

## Installation

```bash
pip install dgufs
```

For development, clone the repo and install with the testing extras:

```bash
git clone https://github.com/gsel9/dgufs.git
cd dgufs
pip install -e ".[testing]"
```

## Usage

`DGUFS` follows the scikit-learn `fit` / `transform` API.

### Basic feature selection

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from dgufs import DGUFS

X, y = load_iris(return_X_y=True)
X_std = StandardScaler().fit_transform(X)

selector = DGUFS(num_features=2, num_clusters=3)
selector.fit(X_std)

print(selector.support)          # indices of selected features
X_reduced = selector.transform(X_std)   # shape (150, 2)
```

### Cluster memberships

DGUFS simultaneously learns a graph structure over the samples. Cluster labels can be retrieved after fitting:

```python
labels = selector.memberships    # integer array of shape (n_samples,)
```

### scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", DGUFS(num_features=2, num_clusters=3)),
])
X_reduced = pipeline.fit_transform(X)
```

### Evaluate clustering

```python
from dgufs.utils import normalized_mutual_information

nmi = normalized_mutual_information(y, selector.memberships)
print(f"NMI: {nmi:.4f}")
```

## Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `num_features` | `2` | Number of features to select |
| `num_clusters` | `2` | Expected number of clusters |
| `alpha` | `0.5` | Controls rank of the graph matrix L |
| `beta` | `0.9` | Balances similarity vs. feature-dependence terms |
| `max_iter` | `100` | Maximum ADMM iterations |
| `tol` | `1e-6` | Convergence tolerance |

## Examples

A Jupyter notebook with a full walkthrough is available in [`examples/basic_usage.ipynb`](examples/basic_usage.ipynb).

## License

`dgufs` was created by Severin Elvatun and is licensed under the [MIT License](LICENSE).

## References

* Guo, Jun, and Wenwu Zhu. "Dependence guided unsupervised feature selection." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 32. No. 1. 2018.
