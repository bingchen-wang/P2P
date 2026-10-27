"""Contracts for the optional intercept (S.23).

``model.intercept: true`` adds an unpenalised constant ``b`` to every option row, as
scikit-learn's ``fit_intercept`` does. It is off by default and independent of the
reference respondent. A question's mass then includes ``k * b`` for its ``k`` options,
and the per-question mass penalty, when on, sees that full mass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p2p.models.constrained_elasticnet import ConstrainedElasticNet
from p2p.models.constrained_lasso import ConstrainedLasso
from p2p.models.weighted_constrained_lasso import WeightedConstrainedLasso
from p2p.stages import evaluate, fit
from tests.test_stage_fit import make_regression_data

SOLVER = "CLARABEL"


def _design(seed: int = 0):
    """Questions with 2 and 4 options; three agents with one-hot answers."""

    rng = np.random.default_rng(seed)
    rows, index, groups = [], [], {}
    for q, k in enumerate((2, 4, 2, 4, 3, 2, 4, 3)):
        members = [f"q{q}__{o}" for o in range(k)]
        groups[f"q{q}"] = members
        answers = rng.integers(0, k, size=3)
        for o, name in enumerate(members):
            index.append(name)
            rows.append([float(a == o) for a in answers])
    X = pd.DataFrame(rows, index=index, columns=["a", "b", "c"])
    return X, groups


def _question_masses(model, X, groups):
    predictions = pd.Series(model.predict(X), index=X.index)
    return np.array([predictions[members].sum() for members in groups.values()])


def test_off_by_default_and_identical_to_not_asking() -> None:
    X, groups = _design()
    y = X.to_numpy() @ np.array([0.5, 0.3, 0.2])
    default = ConstrainedLasso(alpha=0.001, constraint_groups=groups, solver=SOLVER).fit(X, y)
    explicit = ConstrainedLasso(alpha=0.001, constraint_groups=groups, solver=SOLVER,
                                intercept=False).fit(X, y)

    assert default.intercept_ == 0.0
    np.testing.assert_array_equal(default.beta_, explicit.beta_)
    np.testing.assert_array_equal(default.predict(X), X.to_numpy() @ default.beta_)


def test_recovers_a_constant_offset_and_predict_includes_it() -> None:
    X, groups = _design()
    y = X.to_numpy() @ np.array([0.5, 0.3, 0.2]) + 0.07
    model = ConstrainedLasso(alpha=1e-6, constraint_groups=groups, solver=SOLVER,
                             group_penalty_strength=0.0, intercept=True).fit(X, y)

    assert model.intercept_ == pytest.approx(0.07, abs=1e-4)
    np.testing.assert_allclose(model.predict(X), X.to_numpy() @ model.beta_ + model.intercept_)


def test_the_intercept_is_not_shrunk_by_the_penalty() -> None:
    # With an overwhelming penalty every weight is zero; an unpenalised intercept
    # still fits the mean of the targets.
    X, groups = _design()
    y = X.to_numpy() @ np.array([0.5, 0.3, 0.2]) + 0.07
    model = ConstrainedLasso(alpha=1e3, constraint_groups=groups, solver=SOLVER,
                             group_penalty_strength=0.0, intercept=True).fit(X, y)

    assert np.all(model.beta_ == 0.0)
    assert model.intercept_ == pytest.approx(float(np.mean(y)), abs=1e-6)


def test_the_mass_penalty_sees_the_intercept() -> None:
    # A strong mass penalty drives every question's full predicted mass,
    # weights and intercept together, to one, whatever its number of options.
    X, groups = _design()
    y = X.to_numpy() @ np.array([0.4, 0.3, 0.2]) + 0.05
    model = ConstrainedLasso(alpha=1e-4, constraint_groups=groups, solver=SOLVER,
                             group_penalty_strength=1e4, intercept=True).fit(X, y)

    np.testing.assert_allclose(_question_masses(model, X, groups), 1.0, atol=1e-3)
    violations = model.compute_group_constraint_violations(X)
    assert max(violations.values()) < 1e-3


@pytest.mark.parametrize("model_cls, extra", [
    (ConstrainedElasticNet, {"l1_ratio": 0.5}),
    (WeightedConstrainedLasso, {"question_weights": {"q0": 0.5}}),
])
def test_elastic_net_and_weighted_models_fit_it_too(model_cls, extra) -> None:
    X, groups = _design()
    y = X.to_numpy() @ np.array([0.5, 0.3, 0.2]) + 0.07
    model = model_cls(alpha=1e3, constraint_groups=groups, solver=SOLVER,
                      group_penalty_strength=0.0, intercept=True, **extra).fit(X, y)

    assert np.all(model.beta_ == 0.0)
    assert model.intercept_ != 0.0


_CONFIG = {
    "method": "lasso",
    "alpha": (0.001, 0.01),
    "validation": {"strategy": "cv", "cv_folds": 2},
}


def test_fit_model_reads_the_option_and_records_the_intercept() -> None:
    data = make_regression_data()
    off = fit.fit_model(data, _CONFIG, solver="OSQP", seed=101)
    on = fit.fit_model(data, {**_CONFIG, "intercept": True}, solver="OSQP", seed=101)

    assert "intercept" not in off.diagnostics
    assert off.model.intercept_ == 0.0
    assert on.diagnostics["intercept"] == on.model.intercept_
    assert on.model.intercept is True


def test_the_option_must_be_boolean_and_spelled_right() -> None:
    data = make_regression_data()
    with pytest.raises(ValueError, match="model.intercept"):
        fit.fit_model(data, {**_CONFIG, "intercept": "yes"}, solver="OSQP", seed=101)
    with pytest.raises(ValueError, match="did you mean 'intercept'"):
        fit.fit_model(data, {**_CONFIG, "intercep": True}, solver="OSQP", seed=101)


def test_the_answer_file_route_refuses_a_model_with_an_intercept() -> None:
    data = make_regression_data()
    on = fit.fit_model(data, {**_CONFIG, "intercept": True}, solver="OSQP", seed=101)
    assert on.model.intercept_ != 0.0

    with pytest.raises(ValueError, match="intercept"):
        evaluate.answer_file_predictions(data, on.model)
