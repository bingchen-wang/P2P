"""Contracts for single-run Stage-2 evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from p2p.metrics import emd, mcpa, one_minus_jsd
from p2p.stages import evaluate
from p2p.stages.split import load_regression_data


class FixedModel:
    """Small fitted-model stand-in with deterministic coefficients."""

    def __init__(
        self,
        feature_names: tuple[str, ...],
        coefficients: tuple[float, ...],
    ) -> None:
        self.feature_names_ = list(feature_names)
        self.beta_ = np.asarray(coefficients, dtype=float)
        self.status_ = "optimal"

    def predict(self, frame) -> np.ndarray:
        return frame.to_numpy(dtype=float) @ self.beta_

    @property
    def coef_dict_(self) -> dict[str, float]:
        return dict(zip(self.feature_names_, self.beta_))

    @property
    def num_nonzero_(self) -> int:
        return int(np.count_nonzero(self.beta_))


def write_evaluation_fixture(tmp_path: Path):
    schema_path = tmp_path / "survey_schema.yaml"
    schema_path.write_text(
        "\n".join(
            [
                "question_id: id",
                "question_text: question",
                "code_to_answer: option_mapping",
                "split: split",
                "",
            ]
        ),
        encoding="utf-8",
    )

    survey_path = tmp_path / "survey.csv"
    with survey_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "question",
                "option_mapping",
                "split",
            ],
        )
        writer.writeheader()
        for qid, split in (
            ("Q1", "train"),
            ("Q2", "valid"),
            ("Q3", "test"),
        ):
            writer.writerow(
                {
                    "id": qid,
                    "question": f"Question {qid}",
                    "option_mapping": {
                        "A": "Alpha",
                        "B": "Beta",
                    },
                    "split": split,
                }
            )

    responses_path = tmp_path / "responses.csv"
    with responses_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["qid", "agent_a", "agent_b"])
        writer.writerow(["Q1", "Alpha", "Beta"])
        writer.writerow(["Q2", "Alpha", "Beta"])
        writer.writerow(["Q3", "Alpha", "Beta"])

    endowments_path = tmp_path / "endowments.csv"
    with endowments_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "eid",
                "endow_text",
                "role",
                "weight",
                "mode",
                "attributes",
            ],
        )
        writer.writeheader()
        for eid in ("agent_a", "agent_b"):
            writer.writerow(
                {
                    "eid": eid,
                    "endow_text": eid,
                    "role": "proxy",
                    "weight": "",
                    "mode": "core",
                    "attributes": "",
                }
            )

    target_path = tmp_path / "target.json"
    target_path.write_text(
        json.dumps(
            {
                "Q1": {
                    "Alpha": 0.50,
                    "Beta": 0.50,
                },
                "Q2": {
                    "Alpha": 0.55,
                    "Beta": 0.45,
                },
                "Q3": {
                    "Alpha": 0.60,
                    "Beta": 0.40,
                },
            }
        ),
        encoding="utf-8",
    )

    return load_regression_data(
        survey_source=survey_path,
        survey_schema=schema_path,
        responses_path=responses_path,
        endowments_path=endowments_path,
        target_path=target_path,
        train_splits=("train", "valid"),
        test_split="test",
        use_proxy_only=True,
    )


def test_evaluate_model_reproduces_raw_mse_and_distribution_metrics(
    tmp_path: Path,
) -> None:
    data = write_evaluation_fixture(tmp_path)
    model = FixedModel(
        data.feature_names,
        (0.75, 0.25),
    )
    fit_result = SimpleNamespace(
        method="lasso",
        model=model,
        best_alpha=0.01,
        best_l1_ratio=None,
        status="optimal",
    )

    result = evaluate.evaluate_model(
        data,
        fit_result,
        split="test",
        metrics=(
            "mse",
            "one_minus_jsd",
            "emd",
            "mcpa",
        ),
        distribution_prediction_policy="legacy_signed_normalize",
    )

    observed = np.array([0.60, 0.40])
    predicted = np.array([0.75, 0.25])

    assert result.model_name == "lasso"
    assert result.split == "test"
    assert result.prediction_policy == "legacy_signed_normalize"

    assert result.values["mse"] == pytest.approx(0.0225)
    assert result.values["one_minus_jsd"] == pytest.approx(
        one_minus_jsd(observed, predicted)
    )
    assert result.values["emd"] == pytest.approx(
        emd(observed, predicted)
    )
    assert result.values["mcpa"] == pytest.approx(
        mcpa(observed, predicted)
    )

    assert result.predictions.tolist() == pytest.approx(
        [0.75, 0.25]
    )

    for metric in (
        "mse",
        "one_minus_jsd",
        "emd",
        "mcpa",
    ):
        coverage = result.coverage[metric]
        assert coverage.qids_requested == 1
        assert coverage.qids_used == 1

    assert result.diagnostics == {
        "distribution_route": "model",
        "canonical_answers": True,
        "distribution_prediction_policy": (
            "legacy_signed_normalize"
        ),
        "invalid_distribution_policy": "allow",
        "invalid_qids_before_transform": 0,
        "invalid_qids_after_transform": 0,
        "qids_requested": 1,
        "qids_used": 1,
        "negative_entry_count": 0,
        "negative_mass": 0.0,
        "raw_sum_deviation": pytest.approx(0.0),
        "transformation_l1": pytest.approx(0.0),
    }


def test_evaluate_model_rejects_unknown_metric(
    tmp_path: Path,
) -> None:
    data = write_evaluation_fixture(tmp_path)
    fit_result = SimpleNamespace(
        method="lasso",
        model=FixedModel(
            data.feature_names,
            (0.5, 0.5),
        ),
        best_alpha=0.01,
        best_l1_ratio=None,
        status="optimal",
    )

    with pytest.raises(ValueError, match="unknown metric"):
        evaluate.evaluate_model(
            data,
            fit_result,
            split="test",
            metrics=("calibration",),
            distribution_prediction_policy=(
                "legacy_signed_normalize"
            ),
        )


def test_evaluate_predictions_preserves_raw_option_mse(
    tmp_path: Path,
) -> None:
    data = write_evaluation_fixture(tmp_path)

    result = evaluate.evaluate_predictions(
        data.survey,
        data.targets,
        {
            "Q3": {
                "Alpha": 0.75,
                "Beta": 0.25,
            },
        },
        model_name="lasso",
        split="test",
        metrics=(
            "mse",
            "one_minus_jsd",
            "emd",
            "mcpa",
        ),
        distribution_prediction_policy=(
            "clip_and_normalize"
        ),
        invalid_distribution="error",
    )

    assert result.values["mse"] == pytest.approx(0.0225)
    assert result.coverage["mse"] == (
        evaluate.MetricCoverage(
            qids_requested=1,
            qids_used=1,
        )
    )
    assert result.prediction_policy == (
        "clip_and_normalize"
    )


def test_evaluate_predictions_enforces_invalid_policy(
    tmp_path: Path,
) -> None:
    data = write_evaluation_fixture(tmp_path)
    invalid = {
        "Q3": {
            "Alpha": -0.25,
            "Beta": 1.25,
        },
    }

    with pytest.raises(
        ValueError,
        match="Q3",
    ):
        evaluate.evaluate_predictions(
            data.survey,
            data.targets,
            invalid,
            model_name="lasso",
            split="test",
            metrics=("one_minus_jsd",),
            distribution_prediction_policy="validate",
            invalid_distribution="error",
        )

    allowed = evaluate.evaluate_predictions(
        data.survey,
        data.targets,
        invalid,
        model_name="lasso",
        split="test",
        metrics=("one_minus_jsd",),
        distribution_prediction_policy="validate",
        invalid_distribution="allow",
    )
    assert allowed.coverage["one_minus_jsd"] == (
        evaluate.MetricCoverage(
            qids_requested=1,
            qids_used=0,
        )
    )
    assert allowed.values["one_minus_jsd"] is None


def test_signed_sum_policy_matches_its_earlier_name() -> None:
    from p2p.stages.evaluate import SIGNED_SUM_POLICIES, _transform_prediction

    values = np.asarray([30.0, 50.0, 20.0])
    assert SIGNED_SUM_POLICIES == {"signed_sum_normalize", "legacy_signed_normalize"}
    assert np.allclose(
        _transform_prediction(values, policy="signed_sum_normalize", qid="Q"),
        _transform_prediction(values, policy="legacy_signed_normalize", qid="Q"),
    )
