"""Contracts for registered Stage-2 model fitting."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from p2p.stages import fit
from p2p.stages.split import RegressionData, RegressionSplit


class BinarySurveyStub:
    original_to_binary_map = {
        "Q1": ["Q1_0", "Q1_1"],
        "Q2": ["Q2_0", "Q2_1"],
        "Q3": ["Q3_0", "Q3_1"],
        "Q4": ["Q4_0", "Q4_1"],
    }


def make_regression_data() -> RegressionData:
    columns = ["agent_a", "agent_b", "agent_c"]

    train_X = pd.DataFrame(
        [
            [0.9, 0.1, 0.5],
            [0.1, 0.9, 0.5],
            [0.7, 0.2, 0.6],
            [0.3, 0.8, 0.4],
        ],
        index=["Q1_0", "Q1_1", "Q2_0", "Q2_1"],
        columns=columns,
    )
    train_y = np.array([0.75, 0.25, 0.60, 0.40])

    valid_X = pd.DataFrame(
        [
            [0.8, 0.3, 0.4],
            [0.2, 0.7, 0.6],
            [0.6, 0.1, 0.7],
            [0.4, 0.9, 0.3],
        ],
        index=["Q3_0", "Q3_1", "Q4_0", "Q4_1"],
        columns=columns,
    )
    valid_y = np.array([0.65, 0.35, 0.55, 0.45])

    test_X = pd.DataFrame(
        [
            [0.85, 0.15, 0.5],
            [0.15, 0.85, 0.5],
        ],
        index=["Q1_0", "Q1_1"],
        columns=columns,
    )
    test_y = np.array([0.70, 0.30])

    splits = {
        "train": RegressionSplit(
            name="train",
            qids=tuple(train_X.index),
            X=train_X,
            y=train_y,
        ),
        "valid": RegressionSplit(
            name="valid",
            qids=tuple(valid_X.index),
            X=valid_X,
            y=valid_y,
        ),
        "test": RegressionSplit(
            name="test",
            qids=tuple(test_X.index),
            X=test_X,
            y=test_y,
        ),
    }

    return RegressionData(
        survey=None,
        binary_survey=BinarySurveyStub(),
        answer_responses=None,
        binary_responses=None,
        endowments=None,
        targets=None,
        splits=MappingProxyType(splits),
        train_splits=("train", "valid"),
        test_split="test",
        feature_names=tuple(columns),
        cleaning_policy="column_drop_v2",
        source_artifacts=MappingProxyType({}),
        source_digests=MappingProxyType({}),
    )


@pytest.mark.parametrize(
    ("model_config", "expected_method"),
    [
        (
            {
                "method": "lasso",
                "alpha": (0.001, 0.01),
                "validation": {
                    "strategy": "cv",
                    "cv_folds": 2,
                },
                "zero_threshold": 1e-5,
            },
            "lasso",
        ),
        (
            {
                "method": "elasticnet",
                "alpha": (0.001, 0.01),
                "l1_ratio": (0.2, 0.8),
                "validation": {
                    "strategy": "cv",
                    "cv_folds": 2,
                },
                "zero_threshold": 1e-5,
            },
            "elasticnet",
        ),
    ],
)
def test_fit_model_uses_explicit_solver_and_returns_metadata(
    model_config: dict,
    expected_method: str,
    cvxpy_solver: str,
) -> None:
    data = make_regression_data()

    result = fit.fit_model(
        data,
        model_config,
        solver=cvxpy_solver,
        seed=101,
        verbose=False,
    )

    assert result.method == expected_method
    assert result.solver == cvxpy_solver
    assert result.status in {"optimal", "optimal_inaccurate"}
    assert result.best_alpha in model_config["alpha"]
    assert result.model.feature_names_ == list(data.feature_names)
    assert result.model.solver == cvxpy_solver
    assert result.model.num_nonzero_ >= 0
    assert np.all(
        np.isfinite(
            result.model.predict(data.get_split("test").X)
        )
    )

    if expected_method == "lasso":
        assert result.best_l1_ratio is None
    else:
        assert result.best_l1_ratio in model_config["l1_ratio"]

    assert set(result.diagnostics) >= {
        "alphas",
        "mean_mse",
        "std_mse",
    }


def test_fit_model_rejects_unknown_method(
    cvxpy_solver: str,
) -> None:
    with pytest.raises(ValueError, match="unknown model method"):
        fit.fit_model(
            make_regression_data(),
            {
                "method": "unknown",
                "alpha": (0.1,),
            },
            solver=cvxpy_solver,
            seed=101,
            verbose=False,
        )