# Exact environment

The results recorded with P2P `v0.1.0` were produced in one conda environment,
pinned exactly by the two files here. Use it to reproduce those results; any
Python 3.12+ environment works for everything else.

| File | Contents |
|---|---|
| `conda-osx-arm64.lock` | The 21 conda packages (Python 3.13.14 and its base), exact URLs and MD5 checksums. Platform-specific: macOS, Apple Silicon. |
| `pip.lock.txt` | The 88 pip packages at exact versions, including NumPy 2.5.1, scikit-learn 1.9.0, SciPy 1.18.0, CVXPY 1.9.2 (OSQP 1.0.4, Clarabel 0.11.1), pandas 2.3.3, openai 1.109.1 and google-genai 1.75.0. `pip`, `setuptools`, `wheel` and `packaging` are left to conda. |

P2P itself is not in either file: it is installed from this checkout in editable mode.

## Recreate

```bash
conda create --name p2p --file environment/conda-osx-arm64.lock
conda activate p2p
pip install -r environment/pip.lock.txt
pip install -e . --no-deps
```

## Why it matters

Library versions can change fitted results. For example, scikit-learn 1.7 and 1.9
assign cross-validation folds differently. P2P now records each run's fold
assignment, which removes that particular dependence, but solver and library
versions can still move coefficients, so the recorded runs used exactly this
environment. Reproduce a fit with the same solver as well (`--solver`; see the
CLI documentation).

The documentation theme (`furo`, in the `docs` extra) is not part of the lock: it
does not affect any result.

## Recreation check

Exported 28 Sep 2026 with `conda list --explicit --md5` and
`pip list --format=freeze --exclude-editable`. A fresh environment built from
these two files (pip with `--no-deps`, so only pinned packages) matched the lock
package for package, and `pip check` was clean.
