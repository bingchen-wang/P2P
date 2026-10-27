"""Canonical run and long-form metric record contracts."""

from __future__ import annotations

import math

import pytest

from p2p import results


def test_run_record_serializes_required_provenance() -> None:
    survey = results.SourceArtifact(
        path="/fixtures/survey.csv",
        digest="a" * 64,
    )
    target = results.SourceArtifact(
        path="/fixtures/target.json",
        digest="b" * 64,
    )

    record = results.RunRecord(
        run_id="fixture-run-0001",
        repeat=1,
        seed=101,
        conditions={
            "dataset": "fixture",
            "method": "lasso",
        },
        config_digest="c" * 64,
        cleaning_policy="column_drop_v2",
        source_artifacts={
            "survey": survey,
            "target": target,
        },
    )

    assert record.to_dict() == {
        "schema_version": 1,
        "run_id": "fixture-run-0001",
        "repeat": 1,
        "seed": 101,
        "conditions": {
            "dataset": "fixture",
            "method": "lasso",
        },
        "config_digest": "c" * 64,
        "cleaning_policy": "column_drop_v2",
        "source_artifacts": {
            "survey": {
                "path": "/fixtures/survey.csv",
                "digest": "a" * 64,
            },
            "target": {
                "path": "/fixtures/target.json",
                "digest": "b" * 64,
            },
        },
    }


def test_metric_artifact_serializes_long_form_rows() -> None:
    mse = results.MetricRecord(
        metric="mse",
        value=0.0125,
        split="test",
        model="lasso",
        unit="binary_option",
        prediction_policy="raw",
        qids_requested=3,
        qids_used=3,
    )
    jsd = results.MetricRecord(
        metric="one_minus_jsd",
        value=0.91,
        split="test",
        model="lasso",
        unit="question",
        prediction_policy="clip_and_normalize",
        qids_requested=3,
        qids_used=2,
        diagnostics={
            "invalid_qids_before_transform": 1,
            "invalid_qids_after_transform": 0,
            "negative_entry_count": 2,
            "negative_mass": 0.125,
            "raw_sum_deviation": 0.25,
            "transformation_l1": 0.375,
        },
    )

    artifact = results.MetricArtifact(
        run_id="fixture-run-0001",
        metrics=(mse, jsd),
    )

    assert artifact.to_dict() == {
        "schema_version": 1,
        "run_id": "fixture-run-0001",
        "metrics": [
            {
                "metric": "mse",
                "value": 0.0125,
                "split": "test",
                "model": "lasso",
                "unit": "binary_option",
                "prediction_policy": "raw",
                "qids_requested": 3,
                "qids_used": 3,
                "diagnostics": {},
            },
            {
                "metric": "one_minus_jsd",
                "value": 0.91,
                "split": "test",
                "model": "lasso",
                "unit": "question",
                "prediction_policy": (
                    "clip_and_normalize"
                ),
                "qids_requested": 3,
                "qids_used": 2,
                "diagnostics": {
                    "invalid_qids_before_transform": 1,
                    "invalid_qids_after_transform": 0,
                    "negative_entry_count": 2,
                    "negative_mass": 0.125,
                    "raw_sum_deviation": 0.25,
                    "transformation_l1": 0.375,
                },
            },
        ],
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "value": float("nan"),
                "qids_requested": 1,
                "qids_used": 1,
            },
            "finite",
        ),
        (
            {
                "value": 0.5,
                "qids_requested": 1,
                "qids_used": 2,
            },
            "qids_used",
        ),
    ],
)
def test_metric_record_rejects_invalid_values_and_coverage(
    kwargs,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        results.MetricRecord(
            metric="mcpa",
            split="test",
            model="lasso",
            unit="question",
            prediction_policy="clip_and_normalize",
            **kwargs,
        )


def test_metric_artifact_rejects_duplicate_rows() -> None:
    row = results.MetricRecord(
        metric="mse",
        value=0.1,
        split="test",
        model="lasso",
        unit="binary_option",
        prediction_policy="raw",
        qids_requested=1,
        qids_used=1,
    )

    with pytest.raises(ValueError, match="duplicate"):
        results.MetricArtifact(
            run_id="fixture-run-0001",
            metrics=(row, row),
        )


def test_metric_values_are_finite() -> None:
    row = results.MetricRecord(
        metric="emd",
        value=0.25,
        split="test",
        model="elasticnet",
        unit="question",
        prediction_policy="project_to_simplex",
        qids_requested=4,
        qids_used=4,
    )

    assert math.isfinite(row.value)