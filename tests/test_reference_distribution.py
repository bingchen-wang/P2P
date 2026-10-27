"""Contracts for Stage 2 fitted around a reference distribution.

``data.reference_distribution`` names a file in the target format. It enters every
prediction with weight one, so the agents explain each target's deviation from it:
the drift refit across survey waves is the case where it holds the earlier wave's
marginals. These tests pin the behaviour the design promises: the fit is the
preprocessing formulation (fit ``y - a``, mass target lowered by one), predictions
are full distributions so every metric applies, and options that would change
what the reference's weight means are refused.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from p2p.commands import regress
from p2p.config import Config
from p2p.models.constrained_elasticnet import ConstrainedElasticNet
from p2p.models.constrained_lasso import ConstrainedLasso
from p2p.models.reference_distribution import (
    check_reference_masses,
    with_reference_distribution,
)
from p2p.stages import fit
from tests.test_stage_fit import make_regression_data

N_AGENTS = 30
N_QUESTIONS = 20
N_OPTIONS = 4
REPO = Path(__file__).resolve().parents[1]


def _one_hot_basis(seed: int = 0):
    """One-hot answers, a reference distribution, and a target near it."""

    rng = np.random.default_rng(seed)
    index, groups, blocks, reference, target = [], {}, [], {}, []
    for question in range(N_QUESTIONS):
        members = [f"q{question}_{option}" for option in range(N_OPTIONS)]
        groups[f"q{question}"] = members
        index.extend(members)
        block = np.zeros((N_OPTIONS, N_AGENTS))
        block[rng.integers(0, N_OPTIONS, N_AGENTS), np.arange(N_AGENTS)] = 1.0
        blocks.append(block)
        base = rng.dirichlet(np.ones(N_OPTIONS))
        later = 0.7 * base + 0.3 * rng.dirichlet(np.ones(N_OPTIONS))
        reference.update(dict(zip(members, base)))
        target.extend(later)
    frame = pd.DataFrame(
        np.vstack(blocks), index=index, columns=[f"agent_{i}" for i in range(N_AGENTS)]
    )
    return frame, groups, reference, np.asarray(target)


def _kwargs(groups, **extra):
    return dict(
        alpha=0.001,
        constraint_groups=groups,
        zero_threshold=1e-8,
        solver="CLARABEL",
        group_penalty_strength=100.0,
        **extra,
    )


def _mass(values, frame, groups):
    series = pd.Series(values, index=frame.index)
    return np.asarray([series.loc[members].sum() for members in groups.values()])


def test_zero_weights_return_the_reference() -> None:
    frame, groups, reference, _ = _one_hot_basis()
    model = with_reference_distribution(ConstrainedLasso, reference)(**_kwargs(groups))
    model.beta_ = np.zeros(frame.shape[1])
    model.feature_names_ = list(frame.columns)

    expected = np.asarray([reference[row] for row in frame.index])
    np.testing.assert_array_equal(model.predict(frame), expected)


@pytest.mark.parametrize("soft", [True, False])
def test_fit_equals_the_preprocessing_formulation(soft: bool) -> None:
    """Wrapped fit on y == plain fit on y - a with the mass target lowered to zero."""

    frame, groups, reference, target = _one_hot_basis()
    offset = np.asarray([reference[row] for row in frame.index])

    wrapped = with_reference_distribution(ConstrainedLasso, reference)(
        **_kwargs(groups, soft_group_sum=soft)
    )
    wrapped.fit(frame, target)
    plain = ConstrainedLasso(**_kwargs(groups, soft_group_sum=soft, group_total=0.0))
    plain.fit(frame, target - offset)

    np.testing.assert_allclose(wrapped.beta_, plain.beta_, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        wrapped.predict(frame), offset + frame.to_numpy() @ plain.beta_, rtol=0, atol=1e-10
    )


@pytest.mark.parametrize(
    "model_cls,extra",
    [(ConstrainedLasso, {}), (ConstrainedElasticNet, {"l1_ratio": 0.0})],
    ids=["lasso", "ridge"],
)
def test_hard_constraint_gives_zero_sum_weights_and_unit_mass(model_cls, extra) -> None:
    frame, groups, reference, target = _one_hot_basis()
    model = with_reference_distribution(model_cls, reference)(
        **_kwargs(groups, soft_group_sum=False, **extra)
    )
    model.fit(frame, target)

    assert abs(float(np.sum(model.beta_raw_))) < 1e-7
    np.testing.assert_allclose(_mass(model.predict(frame), frame, groups), 1.0, atol=1e-7)


def test_score_is_the_error_of_the_full_prediction() -> None:
    frame, groups, reference, target = _one_hot_basis()
    model = with_reference_distribution(ConstrainedLasso, reference)(**_kwargs(groups))
    model.fit(frame, target)

    assert model.score(frame, target) == pytest.approx(
        float(np.mean((target - model.predict(frame)) ** 2)), abs=1e-15
    )


def test_reference_masses_must_be_one() -> None:
    frame, groups, reference, _ = _one_hot_basis()
    check_reference_masses(reference, groups, frame.index)

    broken = dict(reference)
    broken["q3_0"] += 0.05
    with pytest.raises(ValueError, match="'q3' sums to"):
        check_reference_masses(broken, groups, frame.index)


# fit_model: refusals and records ------------------------------------------------

def _with_reference(data, reference):
    bundle = SimpleNamespace(binary=reference, source_digest="d" * 64)
    return dataclasses.replace(data, reference=bundle)


def _fixture_reference():
    return {row: 0.5 for row in ("Q1_0", "Q1_1", "Q2_0", "Q2_1", "Q3_0", "Q3_1", "Q4_0", "Q4_1")}


@pytest.mark.parametrize(
    "option", ["intercept", "sum_to_one", "nonneg"]
)
def test_conflicting_model_options_are_refused(option: str) -> None:
    data = _with_reference(make_regression_data(), _fixture_reference())
    config = {"method": "lasso", "alpha": [0.01], option: True}
    with pytest.raises(ValueError, match=f"model.{option}"):
        fit.fit_model(data, config, solver="CLARABEL", seed=101)


def test_simple_average_is_refused() -> None:
    data = _with_reference(make_regression_data(), _fixture_reference())
    with pytest.raises(ValueError, match="simple_average"):
        fit.fit_model(data, {"method": "simple_average"}, solver="CLARABEL", seed=101)


def test_a_reference_not_summing_to_one_is_refused() -> None:
    reference = _fixture_reference()
    reference["Q2_0"] = 0.7
    data = _with_reference(make_regression_data(), reference)
    with pytest.raises(ValueError, match="'Q2' sums to"):
        fit.fit_model(data, {"method": "lasso", "alpha": [0.01]}, solver="CLARABEL", seed=101)


def test_fit_records_the_reference_digest() -> None:
    data = _with_reference(make_regression_data(), _fixture_reference())
    result = fit.fit_model(
        data, {"method": "lasso", "alpha": [0.01], "validation": {"strategy": "holdout"}},
        solver="CLARABEL", seed=101,
    )
    assert result.diagnostics["reference_distribution_digest"] == "d" * 64


def test_without_a_reference_the_model_is_unwrapped() -> None:
    result = fit.fit_model(
        make_regression_data(),
        {"method": "lasso", "alpha": [0.01], "validation": {"strategy": "holdout"}},
        solver="CLARABEL", seed=101,
    )
    assert type(result.model) is ConstrainedLasso
    assert "reference_distribution_digest" not in result.diagnostics


# p2p regress end to end -----------------------------------------------------------

def _regress_config(tmp_path: Path, *, route: str = "model") -> Config:
    manifest = yaml.safe_load((REPO / "config/central/regress_fixture.yaml").read_text())
    base = REPO / "config/central"
    for key in ("survey",):
        for field in ("source", "schema"):
            manifest["data"][key][field] = str((base / manifest["data"][key][field]).resolve())
    for key in ("basis", "target"):
        manifest["data"][key] = str((base / manifest["data"][key]).resolve())
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({q: {"Option A": 0.5, "Option B": 0.5} for q in ("Q1", "Q2", "Q3")}))
    manifest["data"]["reference_distribution"] = "reference.json"
    manifest["evaluation"]["distribution_route"] = route
    path = tmp_path / "regress.yaml"
    path.write_text(yaml.safe_dump(manifest))
    return Config.load(path)


def test_regress_scores_every_metric_and_records_the_reference(
    tmp_path: Path, cvxpy_solver: str
) -> None:
    result = regress.run(_regress_config(tmp_path), solver=cvxpy_solver, run_dir=tmp_path / "run")

    run = json.loads(result.report.run_json.read_text())
    assert Path(run["source_artifacts"]["reference_distribution"]["path"]) == (
        tmp_path / "reference.json"
    ).resolve()
    metrics = {entry["metric"]: entry for entry in json.loads(result.report.metrics_json.read_text())["metrics"]}
    for name in ("mse", "one_minus_jsd", "emd", "mcpa"):
        assert np.isfinite(metrics[name]["value"])
        assert metrics[name]["qids_used"] == metrics[name]["qids_requested"]
    diagnostics = json.loads((tmp_path / "run" / "diagnostics.json").read_text())
    assert len(diagnostics["model"]["reference_distribution_digest"]) == 64


def test_regress_refuses_the_answer_file_route(tmp_path: Path, cvxpy_solver: str) -> None:
    with pytest.raises(ValueError, match="reference distribution"):
        regress.run(
            _regress_config(tmp_path, route="legacy_answer_file"),
            solver=cvxpy_solver,
            run_dir=tmp_path / "run",
        )
