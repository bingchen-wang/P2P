"""Contracts for the solver fallback and its record (S.6-F5).

A solve that fails with the configured solver is retried with a fallback solver,
and every such fallback -- in any cross-validation fold or in the final fit --
is recorded, because which solver produced a result affects it (S.6-F2).

The first two tests moved here from the removed ``test_constraint_modes.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p2p.stages import fit
from tests.test_stage_fit import make_regression_data


def _blocks(values: list[list[list[float]]]) -> tuple[pd.DataFrame, dict]:
    """Build a design from per-question option rows, two agents wide."""

    groups = {
        f"q{q}": [f"q{q}__{o}" for o in range(len(block))]
        for q, block in enumerate(values)
    }
    index: list[str] = []
    rows: list[list[float]] = []
    for (_, members), block in zip(groups.items(), values):
        for name, row in zip(members, block):
            index.append(name)
            rows.append(row)
    frame = pd.DataFrame(rows, index=index, columns=["a", "b"])
    return frame, groups


def _fail_osqp(monkeypatch) -> dict:
    import cvxpy as cp

    real_solve = cp.Problem.solve
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        if kwargs.get("solver") == "OSQP":
            calls["n"] += 1
            raise cp.error.SolverError("simulated OSQP failure")
        return real_solve(self, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", flaky)
    return calls


def test_solver_fallback_recovers_a_failed_solve(monkeypatch) -> None:
    """A solver failure on one fold must not abort the run.

    First-order solvers occasionally fail on a single cross-validation fold
    while succeeding on its neighbours. Before the fallback, one such fold
    raised and lost the whole fit.
    """

    from p2p.models.constrained_lasso import ConstrainedLasso

    X, groups = _blocks(
        [
            [[1, 0], [0, 1], [0, 0]],
            [[0, 1], [1, 0], [0, 0]],
        ]
    )
    y = np.array([0.5, 0.3, 0.2, 0.4, 0.4, 0.2])
    calls = _fail_osqp(monkeypatch)

    model = ConstrainedLasso(
        alpha=0.01, constraint_groups=groups, solver="OSQP"
    )
    model.fit(X, y)

    assert calls["n"] == 1
    assert model.solver_used_ == "CLARABEL"          # first fallback after OSQP
    assert model.solver_fallback_ is True
    assert model.status_ == "optimal"


def test_a_successful_solve_does_not_fall_back() -> None:
    from p2p.models.constrained_lasso import ConstrainedLasso

    X, groups = _blocks(
        [
            [[1, 0], [0, 1], [0, 0]],
            [[0, 1], [1, 0], [0, 0]],
        ]
    )
    model = ConstrainedLasso(
        alpha=0.01, constraint_groups=groups, solver="OSQP"
    )
    model.fit(X, np.array([0.5, 0.3, 0.2, 0.4, 0.4, 0.2]))

    assert model.solver_used_ == "OSQP"
    assert model.solver_fallback_ is False


_CONFIG = {
    "method": "lasso",
    "alpha": (0.001, 0.01),
    "validation": {"strategy": "cv", "cv_folds": 2},
}


def test_fit_model_records_every_fallback(monkeypatch) -> None:
    _fail_osqp(monkeypatch)

    result = fit.fit_model(make_regression_data(), _CONFIG, solver="OSQP", seed=101)
    diagnostics = result.diagnostics

    # 2 penalties x 2 folds, every one of them rescued by a fallback solver.
    assert diagnostics["cv_fits"] == 4
    assert diagnostics["cv_solver_fallbacks"] == 4
    assert diagnostics["solver_fallback"] is True
    assert diagnostics["solver_used"] != "OSQP"


def test_fit_model_records_no_fallback_as_zero() -> None:
    result = fit.fit_model(make_regression_data(), _CONFIG, solver="OSQP", seed=101)
    diagnostics = result.diagnostics

    assert diagnostics["cv_fits"] == 4
    assert diagnostics["cv_solver_fallbacks"] == 0
    assert diagnostics["solver_fallback"] is False
    assert diagnostics["solver_used"] == "OSQP"


def test_the_removed_total_weight_mode_is_rejected() -> None:
    """S.6-F4: `model.constraint` no longer exists; asking for it fails loudly."""

    with pytest.raises(ValueError, match="unknown model option"):
        fit.fit_model(
            make_regression_data(),
            {**_CONFIG, "constraint": "total_weight"},
            solver="OSQP",
            seed=101,
        )


def test_default_solver_for_new_runs_is_clarabel() -> None:
    """S.6-F2 decision: Clarabel by default; OSQP only to reproduce the paper."""

    import argparse

    from p2p import cli
    from p2p.models import DEFAULT_SOLVER

    assert DEFAULT_SOLVER == "CLARABEL"
    parser = argparse.ArgumentParser()
    cli._add_solver_argument(parser)
    assert parser.parse_args([]).solver == "CLARABEL"


def test_clarabel_falls_back_to_osqp_before_scs(monkeypatch) -> None:
    import cvxpy as cp

    from p2p.models.constrained_lasso import ConstrainedLasso

    real_solve = cp.Problem.solve
    tried = []

    def failing_clarabel(self, *args, **kwargs):
        tried.append(kwargs.get("solver"))
        if kwargs.get("solver") == "CLARABEL":
            raise cp.error.SolverError("simulated Clarabel failure")
        return real_solve(self, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", failing_clarabel)
    X, groups = _blocks([[[1, 0], [0, 1], [0, 0]], [[0, 1], [1, 0], [0, 0]]])
    model = ConstrainedLasso(alpha=0.01, constraint_groups=groups, solver="CLARABEL")
    model.fit(X, np.array([0.5, 0.3, 0.2, 0.4, 0.4, 0.2]))

    assert tried == ["CLARABEL", "OSQP"]
    assert model.solver_used_ == "OSQP" and model.solver_fallback_ is True


@pytest.mark.parametrize("model_path", [
    "p2p.models.weighted_constrained_lasso.WeightedConstrainedLasso",
    "p2p.models.weighted_constrained_elasticnet.WeightedConstrainedElasticNet",
])
def test_weighted_models_use_and_record_the_configured_solver(model_path, monkeypatch) -> None:
    """S.24: the block-weighted models used to call ``problem.solve`` directly,
    ignoring the configured solver, with no fallback and no record."""

    import importlib

    module, name = model_path.rsplit(".", 1)
    model_cls = getattr(importlib.import_module(module), name)
    X, groups = _blocks([[[1, 0], [0, 1], [0, 0]], [[0, 1], [1, 0], [0, 0]]])
    y = np.array([0.5, 0.3, 0.2, 0.4, 0.4, 0.2])
    weights = {"q0": 1 / 3, "q1": 1 / 3}

    model = model_cls(alpha=0.01, constraint_groups=groups, question_weights=weights,
                      solver="CLARABEL").fit(X, y)
    assert model.solver_used_ == "CLARABEL" and model.solver_fallback_ is False

    calls = _fail_osqp(monkeypatch)
    rescued = model_cls(alpha=0.01, constraint_groups=groups, question_weights=weights,
                        solver="OSQP").fit(X, y)
    assert calls["n"] == 1
    assert rescued.solver_used_ == "CLARABEL" and rescued.solver_fallback_ is True
