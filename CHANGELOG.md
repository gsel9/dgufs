# Changelog

<!--next-version-placeholder-->

## v0.2.0 (03/04/2026)

### Bug fixes
- Fixed circular import in `dgufs.py` (`from dgufs import utils` → `from . import utils`)
- Fixed `best_map()`: wrong 1-D cost matrix shape, broken element-wise boolean comparison, incorrect match count
- Fixed `speed_up()`: no longer mutates caller's input; added division-by-zero guard
- Fixed `memberships` property: replaced `linalg.eig` with `linalg.eigh`, corrected eigenvalue reshape, clipped near-zero eigenvalues before `sqrt`
- Fixed `support` property: wrapped `np.squeeze` with `np.atleast_1d` to prevent crash when exactly one feature is selected

### Improvements
- `transform()` and `memberships` now raise `NotFittedError` if called before `fit()`
- `fit()` emits a `RuntimeWarning` if ADMM reaches `max_iter` without converging
- `max_iter` default changed from `1e2` (float) to `100` (int)
- `solve_l20()` raises `ValueError` if `nfeats > Q.shape[0]`
- `__init__.py` now exports `DGUFS` so `from dgufs import DGUFS` works
- Completed all docstrings in `dgufs.py` and `utils.py`

### Tests
- Implemented previously empty `test_X_Z_norm`
- Fixed fragile float equality in `test_Y_norm` (uses `pytest.approx`)
- Uncommented and corrected `test_L` rank and diagonal assertions
- Added tests for `transform()`, `memberships`, `NotFittedError`, and too-few-samples error

### Infrastructure
- Enabled Ruff linting and Codecov reporting in CI
- Synced `REQUIREMENTS.txt` with `setup.cfg` (added `scipy`, `pandas`)
- Cleaned up `setup.py`: removed dead code, fixed bare `except:`

### Examples
- Added `examples/basic_usage.ipynb` notebook

## v0.1.0 (11/11/2023)

- First release of `dgufs`!