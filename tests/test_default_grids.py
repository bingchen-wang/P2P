"""Contracts for the default penalty grids (decision S.8b).

A manifest that gives no ``model.alpha`` (or, for elastic net, no
``model.l1_ratio``) is fitted on P2P's default grid, and the run records which
grids were defaulted. A manifest that states a grid is fitted exactly as before.
"""

from __future__ import annotations

import pytest

from p2p.config import expand_grid
from p2p.stages import fit
from p2p.stages.report import _fit_diagnostics
from tests.test_stage_fit import make_regression_data

SOLVER = "CLARABEL"
CV = {"strategy": "cv", "cv_folds": 2}


def test_default_alpha_grid_is_wide_and_matches_the_documented_spec() -> None:
    values = expand_grid(fit.DEFAULT_ALPHA_GRID)
    assert len(values) == 30
    assert values[0] == pytest.approx(1e-5)
    assert values[-1] == pytest.approx(1e5)
    l1 = expand_grid(fit.DEFAULT_L1_RATIO_GRID)
    assert len(l1) == 10 and l1[0] == pytest.approx(0.1) and l1[-1] == pytest.approx(1.0)


def test_lasso_without_a_grid_uses_the_default_and_records_it() -> None:
    data = make_regression_data()
    defaulted = fit.fit_model(data, {"method": "lasso", "validation": CV},
                              solver=SOLVER, seed=101)
    explicit = fit.fit_model(
        data,
        {"method": "lasso", "alpha": expand_grid(fit.DEFAULT_ALPHA_GRID), "validation": CV},
        solver=SOLVER, seed=101,
    )

    assert defaulted.best_alpha == explicit.best_alpha
    assert defaulted.diagnostics["cv_config_count"] == 30
    assert defaulted.diagnostics["default_grids"] == "alpha"
    assert "default_grids" not in explicit.diagnostics
    assert _fit_diagnostics(defaulted, defaulted.model.coef_dict_)["default_grids"] == "alpha"


def test_elastic_net_defaults_both_grids_independently() -> None:
    data = make_regression_data()
    both = fit.fit_model(data, {"method": "elasticnet", "validation": CV},
                         solver=SOLVER, seed=101)
    ratio_only = fit.fit_model(
        data, {"method": "elasticnet", "alpha": (0.001, 0.01), "validation": CV},
        solver=SOLVER, seed=101,
    )

    assert both.diagnostics["default_grids"] == "alpha, l1_ratio"
    assert both.diagnostics["cv_config_count"] == 300
    assert ratio_only.diagnostics["default_grids"] == "l1_ratio"
    assert ratio_only.diagnostics["cv_config_count"] == 20


def test_a_stated_grid_is_used_unchanged() -> None:
    data = make_regression_data()
    result = fit.fit_model(data, {"method": "lasso", "alpha": (0.001, 0.01), "validation": CV},
                           solver=SOLVER, seed=101)
    assert result.diagnostics["cv_config_count"] == 2
    assert "default_grids" not in result.diagnostics
