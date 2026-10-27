"""Standalone evaluation command and artifact contracts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from p2p.commands import evaluate
from p2p.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_predictions_runs_all_configured_splits_and_reports(
    tmp_path: Path,
) -> None:
    config = Config.load(
        PROJECT_ROOT
        / "config"
        / "central"
        / "evaluate_fixture.yaml"
    )
    run_dir = tmp_path / "evaluate-run"

    result = evaluate.run(
        config,
        run_dir=run_dir,
    )

    assert result.config_digest == config.digest
    assert result.run_dir == run_dir.resolve()
    assert result.input_kind == "predictions"
    assert result.model_name == "lasso"
    assert result.cleaning_policy == "column_drop_v2"
    assert [
        evaluation.split
        for evaluation in result.evaluations
    ] == ["train", "valid", "test"]

    assert result.report.run_json.is_file()
    assert result.report.metrics_json.is_file()
    assert result.report.metrics_csv.is_file()
    assert result.report.config_snapshot.is_file()

    payload = json.loads(
        result.report.metrics_json.read_text(
            encoding="utf-8"
        )
    )
    assert len(payload["metrics"]) == 12
    assert {
        row["split"]
        for row in payload["metrics"]
    } == {"train", "valid", "test"}
    assert {
        row["prediction_policy"]
        for row in payload["metrics"]
        if row["metric"] == "mse"
    } == {"raw"}
    assert {
        row["prediction_policy"]
        for row in payload["metrics"]
        if row["metric"] != "mse"
    } == {"clip_and_normalize"}

    run_payload = json.loads(
        result.report.run_json.read_text(
            encoding="utf-8"
        )
    )
    assert run_payload["conditions"]["method"] == "lasso"
    assert run_payload["cleaning_policy"] == (
        "column_drop_v2"
    )
    assert "predictions" in run_payload["source_artifacts"]


def test_evaluate_accepts_joblib_fitted_output_with_predictions(
    tmp_path: Path,
) -> None:
    base = Config.load(
        PROJECT_ROOT
        / "config"
        / "central"
        / "evaluate_fixture.yaml"
    ).to_dict()
    predictions_path = Path(
        base["data"].pop("predictions")
    )
    fitted_path = tmp_path / "fitted.joblib"
    joblib.dump(
        json.loads(
            predictions_path.read_text(encoding="utf-8")
        ),
        fitted_path,
    )
    base["data"]["fitted"] = str(fitted_path)

    config = Config.from_mapping(
        base,
        declared_in=tmp_path / "evaluate-fitted.yaml",
    )
    result = evaluate.run(
        config,
        run_dir=tmp_path / "fitted-run",
    )

    assert result.input_kind == "fitted"
    assert result.report.run_json.is_file()
    assert (
        result.report.run_record.source_artifacts[
            "fitted"
        ].path
        == str(fitted_path.resolve())
    )

