"""Canonical regression report artifact contracts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from p2p.config import Config
from p2p.stages import report
from p2p.stages.evaluate import MetricCoverage


def test_write_regression_report_emits_canonical_artifacts(
    tmp_path: Path,
) -> None:
    config = Config.from_mapping(
        {
            "run": {
                "name": "fixture_report",
                "seed": 101,
                "repeat": 2,
                "conditions": {
                    "dataset": "fixture",
                },
            }
        },
        declared_in=tmp_path / "report.yaml",
    )

    survey_path = tmp_path / "survey.csv"
    survey_path.write_text(
        "id,question\n",
        encoding="utf-8",
    )
    survey_digest = hashlib.sha256(
        survey_path.read_bytes()
    ).hexdigest()

    data = SimpleNamespace(
        cleaning_policy="column_drop_v2",
        source_artifacts={
            "survey": survey_path.resolve(),
        },
        source_digests={
            "survey": survey_digest,
        },
    )
    fit = SimpleNamespace(
        method="lasso",
    )
    evaluation = SimpleNamespace(
        model_name="lasso",
        split="test",
        prediction_policy="clip_and_normalize",
        values={
            "mse": 0.0125,
            "one_minus_jsd": 0.91,
        },
        coverage={
            "mse": MetricCoverage(
                qids_requested=3,
                qids_used=3,
            ),
            "one_minus_jsd": MetricCoverage(
                qids_requested=3,
                qids_used=2,
            ),
        },
        diagnostics={
            "distribution_prediction_policy": (
                "clip_and_normalize"
            ),
            "invalid_qids_before_transform": 1,
            "invalid_qids_after_transform": 0,
            "qids_requested": 3,
            "qids_used": 2,
            "negative_entry_count": 2,
            "negative_mass": 0.125,
            "raw_sum_deviation": 0.25,
            "transformation_l1": 0.375,
        },
    )
    regression = SimpleNamespace(
        data=data,
        fit=fit,
        evaluation=evaluation,
    )

    run_dir = tmp_path / "run"
    artifacts = report.write_regression_report(
        config,
        regression,
        run_dir=run_dir,
    )

    assert isinstance(
        artifacts,
        report.ReportArtifacts,
    )
    assert artifacts.run_json == (
        run_dir / "run.json"
    ).resolve()
    assert artifacts.metrics_json == (
        run_dir / "metrics.json"
    ).resolve()
    assert artifacts.metrics_csv == (
        run_dir / "metrics.csv"
    ).resolve()
    assert artifacts.config_snapshot == (
        run_dir / "config.resolved.yaml"
    ).resolve()

    assert all(
        path.is_file()
        for path in (
            artifacts.run_json,
            artifacts.metrics_json,
            artifacts.metrics_csv,
            artifacts.config_snapshot,
        )
    )

    run_payload = json.loads(
        artifacts.run_json.read_text(
            encoding="utf-8"
        )
    )
    assert run_payload == {
        "schema_version": 1,
        "run_id": "fixture_report-r0002",
        "repeat": 2,
        "seed": 101,
        "conditions": {
            "dataset": "fixture",
            "method": "lasso",
        },
        "config_digest": config.digest,
        "cleaning_policy": "column_drop_v2",
        "source_artifacts": {
            "survey": {
                "path": str(survey_path.resolve()),
                "digest": survey_digest,
            },
        },
    }

    metrics_payload = json.loads(
        artifacts.metrics_json.read_text(
            encoding="utf-8"
        )
    )
    assert metrics_payload == {
        "schema_version": 1,
        "run_id": "fixture_report-r0002",
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

    with artifacts.metrics_csv.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert [row["metric"] for row in rows] == [
        "mse",
        "one_minus_jsd",
    ]
    assert rows[0]["unit"] == "binary_option"
    assert rows[0]["prediction_policy"] == "raw"
    assert rows[0][
        "invalid_qids_before_transform"
    ] == ""

    assert rows[1]["unit"] == "question"
    assert rows[1]["prediction_policy"] == (
        "clip_and_normalize"
    )
    assert rows[1][
        "invalid_qids_before_transform"
    ] == "1"
    assert rows[1]["negative_mass"] == "0.125"
    assert rows[1]["transformation_l1"] == "0.375"


def test_write_regression_report_rejects_model_mismatch(
    tmp_path: Path,
) -> None:
    config = Config.from_mapping(
        {
            "run": {
                "name": "mismatch",
                "seed": 101,
            }
        },
        declared_in=tmp_path / "report.yaml",
    )
    regression = SimpleNamespace(
        data=SimpleNamespace(
            cleaning_policy="column_drop_v2",
            source_artifacts={
                "survey": tmp_path / "survey.csv",
            },
            source_digests={
                "survey": "a" * 64,
            },
        ),
        fit=SimpleNamespace(
            method="lasso",
        ),
        evaluation=SimpleNamespace(
            model_name="elasticnet",
            split="test",
            prediction_policy="clip_and_normalize",
            values={},
            coverage={},
            diagnostics={},
        ),
    )

    with pytest.raises(
        ValueError,
        match="model",
    ):
        report.write_regression_report(
            config,
            regression,
            run_dir=tmp_path / "run",
        )


def test_write_regression_report_rejects_config_drift(
    tmp_path: Path,
) -> None:
    config = Config.from_mapping(
        {"run": {"name": "drift"}},
        declared_in=tmp_path / "report.yaml",
    )
    regression = SimpleNamespace(
        config_digest="0" * 64,
    )

    with pytest.raises(
        ValueError,
        match="digest",
    ):
        report.write_regression_report(
            config,
            regression,
            run_dir=tmp_path / "run",
        )

    assert not (tmp_path / "run").exists()
