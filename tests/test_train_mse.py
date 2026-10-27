"""Contracts for the ``train_mse`` fit diagnostic (A.9).

``train_mse`` is the in-sample MSE of the final model: raw predictions on exactly
the rows it was fitted to, unweighted over option rows whatever the fit's own loss
weighting. With cross-validation that is every training split (the final refit);
with holdout validation only the fit splits, since nothing is refitted on the
holdout split; for the simple average, the training splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p2p.metrics import mse
from p2p.stages import fit
from p2p.stages.report import _fit_diagnostics
from tests.test_stage_fit import make_regression_data

SOLVER = "CLARABEL"


def _rows(data, names):
    X = pd.concat([data.get_split(name).X for name in names])
    y = np.concatenate([data.get_split(name).y for name in names])
    return X, y


def test_cross_validation_scores_the_final_refit_on_all_training_splits() -> None:
    data = make_regression_data()
    result = fit.fit_model(data, {"method": "lasso", "alpha": (0.001, 0.01),
                                  "validation": {"strategy": "cv", "cv_folds": 2}},
                           solver=SOLVER, seed=101)
    X, y = _rows(data, ("train", "valid"))
    assert result.diagnostics["train_mse"] == pytest.approx(mse(y, result.model.predict(X)),
                                                            abs=1e-15)


def test_holdout_scores_only_the_fit_splits() -> None:
    data = make_regression_data()
    result = fit.fit_model(data, {"method": "lasso", "alpha": (0.001, 0.01),
                                  "validation": {"strategy": "holdout", "holdout_split": "valid"}},
                           solver=SOLVER, seed=101)
    X_fit, y_fit = _rows(data, ("train",))
    X_all, y_all = _rows(data, ("train", "valid"))
    expected = mse(y_fit, result.model.predict(X_fit))
    assert result.diagnostics["train_mse"] == pytest.approx(expected, abs=1e-15)
    assert expected != pytest.approx(mse(y_all, result.model.predict(X_all)))


def test_weighted_fits_report_the_unweighted_mean() -> None:
    data = make_regression_data()
    result = fit.fit_model(data, {"method": "lasso", "alpha": (0.001, 0.01), "block_weighting": True,
                                  "validation": {"strategy": "cv", "cv_folds": 2}},
                           solver=SOLVER, seed=101)
    X, y = _rows(data, ("train", "valid"))
    residuals = y - result.model.predict(X)
    assert result.diagnostics["train_mse"] == pytest.approx(float(np.mean(residuals ** 2)),
                                                            abs=1e-15)


def test_intercept_is_part_of_the_prediction_scored() -> None:
    data = make_regression_data()
    result = fit.fit_model(data, {"method": "lasso", "alpha": (0.001,), "intercept": True,
                                  "group_penalty_strength": 0.0,
                                  "validation": {"strategy": "cv", "cv_folds": 2}},
                           solver=SOLVER, seed=101)
    X, y = _rows(data, ("train", "valid"))
    predictions = X.to_numpy() @ result.model.beta_ + result.model.intercept_
    assert result.diagnostics["train_mse"] == pytest.approx(mse(y, predictions), abs=1e-15)


def test_simple_average_reports_it_on_the_training_splits() -> None:
    data = make_regression_data()
    result = fit.fit_model(data, {"method": "simple_average"}, solver=SOLVER, seed=101)
    X, y = _rows(data, ("train", "valid"))
    assert result.diagnostics["train_mse"] == pytest.approx(mse(y, X.to_numpy().mean(axis=1)),
                                                            abs=1e-15)


def test_the_run_report_records_it() -> None:
    data = make_regression_data()
    result = fit.fit_model(data, {"method": "lasso", "alpha": (0.001, 0.01),
                                  "validation": {"strategy": "cv", "cv_folds": 2}},
                           solver=SOLVER, seed=101)
    payload = _fit_diagnostics(result, result.model.coef_dict_)
    assert payload["train_mse"] == result.diagnostics["train_mse"]
