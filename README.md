# Dependence Guided Unsupervised Feature Selection (DGUFS)

The Dependence Guided Unsupervised Feature Selection (DGUFS) method select features and partition data in a joint manner to enhance the interdependence among original data, cluster labels, and selected features [1]. 

## Installation

```bash
$ pip install dgufs
```

## Usage

The DGUFS implementation conform to the `scikit-learn` API:

```python
from dgufs.dgufs import DGUFS
# third party
from sklearn.datasets import load_iris

iris = load_iris(return_X_y=False)

X, y = iris.data, iris.target

# Select a subset of features 
dgufs = DGUFS(num_features=2)
X_sub = dgufs.fit_transform(X)
```

## License

`dgufs` was created by Severin Elvatun. It is licensed under the terms of the MIT license.


References
----------

* [1]: Guo, Jun, and Wenwu Zhu. "Dependence guided unsupervised feature selection." Proceedings of the AAAI conference on artificial intelligence. Vol. 32. No. 1. 2018.
