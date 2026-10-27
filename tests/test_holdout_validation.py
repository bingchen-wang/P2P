"""Contracts for holdout validation.

``model.validation.strategy: holdout`` fits every penalty on the training splits
other than ``holdout_split`` (default ``valid``), chooses the penalty with the lowest
MSE on ``holdout_split``, and keeps that fit: the holdout split only selects and is
not refitted on: validation is used only for hyperparameter selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p2p.models.constrained_lasso import ConstrainedLasso
from p2p.stages import fit
from tests.test_stage_fit import make_regression_data

ALPHAS = (0.0001, 0.001, 0.01, 0.1)
HOLDOUT = {"method": "lasso", "alpha": ALPHAS, "validation": {"strategy": "holdout"}}


def _by_hand(data):
    """Fit each penalty on train, score on valid, keep the best train-only fit."""

    train, valid = data.get_split("train"), data.get_split("valid")
    groups = dict(data.binary_survey.original_to_binary_map)
    best = None
    for alpha in ALPHAS:
        model = ConstrainedLasso(alpha=alpha, constraint_groups=groups, solver="CLARABEL",
                                 zero_threshold=1e-5, group_penalty_strength=100.0)
        model.fit(train.X, train.y)
        score = model.score(valid.X, np.asarray(valid.y))
        if best is None or score < best[0]:
            best = (score, alpha, model)
    return best


def test_selects_on_valid_and_keeps_the_train_only_fit() -> None:
    data = make_regression_data()
    result = fit.fit_model(data, HOLDOUT, solver="CLARABEL", seed=101)
    score, alpha, expected = _by_hand(data)

    assert result.best_alpha == alpha
    np.testing.assert_array_equal(result.model.beta_, expected.beta_)
    assert result.diagnostics["holdout_best_mse"] == pytest.approx(score)


def test_the_holdout_split_is_not_refitted_on() -> None:
    data = make_regression_data()
    result = fit.fit_model(data, HOLDOUT, solver="CLARABEL", seed=101)

    both = pd.concat([data.get_split("train").X, data.get_split("valid").X])
    y_both = np.concatenate([data.get_split("train").y, data.get_split("valid").y])
    refit = ConstrainedLasso(alpha=result.best_alpha, solver="CLARABEL",
                             constraint_groups=dict(data.binary_survey.original_to_binary_map))
    refit.fit(both, y_both)
    assert not np.allclose(result.model.beta_, refit.beta_)


def test_records_the_strategy_and_writes_no_fold_record() -> None:
    data = make_regression_data()
    result = fit.fit_model(data, HOLDOUT, solver="CLARABEL", seed=101)

    assert result.folds is None
    assert result.diagnostics["validation_strategy"] == "holdout"
    assert result.diagnostics["holdout_split"] == "valid"
    assert result.diagnostics["holdout_fit_splits"] == "train"
    assert result.diagnostics["holdout_fits"] == len(ALPHAS)
    assert "cv_fits" not in result.diagnostics


def test_elastic_net_scores_every_alpha_and_l1_ratio() -> None:
    config = {"method": "elasticnet", "alpha": ALPHAS, "l1_ratio": (0.5, 1.0),
              "validation": {"strategy": "holdout"}}
    result = fit.fit_model(make_regression_data(), config, solver="CLARABEL", seed=101)

    assert result.diagnostics["holdout_fits"] == len(ALPHAS) * 2
    assert result.best_l1_ratio in (0.5, 1.0)


@pytest.mark.parametrize("validation, message", [
    ({"strategy": "holdout", "cv_folds": 5}, "cv_folds does not apply to holdout"),
    ({"strategy": "holdout", "folds": {"method": "hashed"}}, "folds does not apply to holdout"),
    ({"strategy": "holdout", "holdout_split": "test"}, "must be one of the training splits"),
    ({"strategy": "bootstrap"}, "strategy must be one of"),
])
def test_rejects_settings_that_do_not_apply(validation, message) -> None:
    with pytest.raises(ValueError, match=message):
        fit.fit_model(make_regression_data(), {**HOLDOUT, "validation": validation},
                      solver="CLARABEL", seed=101)


def test_fold_assignments_apply_to_cv_only() -> None:
    with pytest.raises(ValueError, match="cv validation only"):
        fit.fit_model(make_regression_data(), HOLDOUT, solver="CLARABEL", seed=101,
                      fold_assignments={"q": 0})
