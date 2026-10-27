"""Contracts for ``distribution_mse`` (S.22).

``distribution_mse`` is raw ``mse`` with one change: each question's predictions are
first transformed by the run's distribution prediction policy (e.g. clipped at zero
and rescaled to sum to one). Same cells, same targets. So it equals ``mse`` when the
predictions already form distributions, and the gap between the two is the effect
of the transformation -- for a model without the mass penalty, mostly missing mass.

In the fixture each question has two options and each of the two agents picks a
different one, so a question's predictions are exactly the two weights.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from p2p.stages import evaluate
from p2p.stages.report import _metric_records
from tests.test_stage_evaluate import FixedModel, write_evaluation_fixture

TARGET = (0.60, 0.40)          # the fixture's test question, Q3
METRICS = ("mse", "distribution_mse")


def _evaluate(tmp_path: Path, weights, policy="clip_and_normalize", invalid="allow"):
    data = write_evaluation_fixture(tmp_path)
    fit = SimpleNamespace(method="lasso", model=FixedModel(data.feature_names, weights),
                          best_alpha=0.01, best_l1_ratio=None, status="optimal")
    return evaluate.evaluate_model(data, fit, split="test", metrics=METRICS,
                                   distribution_prediction_policy=policy,
                                   invalid_distribution=invalid)


def _squared_error(predicted) -> float:
    return sum((p - t) ** 2 for p, t in zip(predicted, TARGET)) / len(TARGET)


def test_equals_raw_mse_when_predictions_are_already_distributions(tmp_path: Path) -> None:
    result = _evaluate(tmp_path, (0.75, 0.25))

    assert result.values["mse"] == pytest.approx(_squared_error((0.75, 0.25)))
    assert result.values["distribution_mse"] == pytest.approx(result.values["mse"])


def test_rescales_a_question_whose_mass_falls_short(tmp_path: Path) -> None:
    # Weights sum to 0.8: raw predictions (0.6, 0.2), rescaled (0.75, 0.25).
    result = _evaluate(tmp_path, (0.6, 0.2))

    assert result.values["mse"] == pytest.approx(_squared_error((0.6, 0.2)))           # 0.02
    assert result.values["distribution_mse"] == pytest.approx(_squared_error((0.75, 0.25)))  # 0.0225
    assert result.coverage["distribution_mse"] == evaluate.MetricCoverage(
        qids_requested=1, qids_used=1)


def test_clips_a_negative_prediction_before_rescaling(tmp_path: Path) -> None:
    result = _evaluate(tmp_path, (1.25, -0.25))

    assert result.values["mse"] == pytest.approx(_squared_error((1.25, -0.25)))
    assert result.values["distribution_mse"] == pytest.approx(_squared_error((1.0, 0.0)))


def test_a_question_that_cannot_be_transformed_is_left_out_or_raises(tmp_path: Path) -> None:
    # Both predictions negative: nothing is left to rescale.
    allowed = _evaluate(tmp_path, (-0.5, -0.5))
    assert allowed.coverage["distribution_mse"] == evaluate.MetricCoverage(
        qids_requested=1, qids_used=0)

    strict = tmp_path / "strict"
    strict.mkdir()
    with pytest.raises(ValueError, match="Q3"):
        _evaluate(strict, (-0.5, -0.5), invalid="error")


def test_standalone_evaluation_uses_the_same_definition(tmp_path: Path) -> None:
    data = write_evaluation_fixture(tmp_path)
    result = evaluate.evaluate_predictions(
        data.survey, data.targets, {"Q3": {"Alpha": 0.6, "Beta": 0.2}},
        model_name="lasso", split="test", metrics=METRICS,
        distribution_prediction_policy="clip_and_normalize", invalid_distribution="error")

    assert result.values["mse"] == pytest.approx(_squared_error((0.6, 0.2)))
    assert result.values["distribution_mse"] == pytest.approx(_squared_error((0.75, 0.25)))


def test_metrics_json_records_its_cells_and_policy(tmp_path: Path) -> None:
    records = {r.metric: r for r in _metric_records(_evaluate(tmp_path, (0.6, 0.2)))}

    assert records["mse"].unit == "binary_option"
    assert records["mse"].prediction_policy == "raw"
    assert records["distribution_mse"].unit == "binary_option"
    assert records["distribution_mse"].prediction_policy == "clip_and_normalize"
