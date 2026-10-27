"""Production full-pipeline composition contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from p2p.commands import pipeline
from p2p.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_threads_generated_basis_into_regression(
    tmp_path: Path,
    monkeypatch,
) -> None:
    survey_source = tmp_path / "survey.csv"
    survey_schema = tmp_path / "survey.yaml"
    target_path = tmp_path / "target.json"

    for path in (
        survey_source,
        survey_schema,
        target_path,
    ):
        path.write_text("", encoding="utf-8")

    config = Config.from_mapping(
        {
            "run": {
                "name": "pipeline_contract",
                "seed": 31,
            },
            "data": {
                "survey": {
                    "source": survey_source,
                    "schema": survey_schema,
                },
                "target": target_path,
            },
            "generation": {
                "mode": "vanilla",
                "target_n": 2,
            },
            "backend": {
                "response": {
                    "type": "dummy",
                    "model_name": "dummy",
                    "kwargs": {},
                },
                "endowment": {
                    "type": "dummy",
                    "model_name": "dummy",
                    "kwargs": {},
                },
            },
            "model": {
                "method": "lasso",
                "alpha": [0.01],
                "validation": {
                    "strategy": "cv",
                    "cv_folds": 2,
                },
            },
            "split": {
                "train_val": ["train", "valid"],
                "test": "test",
                "use_proxy_only": True,
            },
            "metrics": [
                "mse",
                "one_minus_jsd",
            ],
        },
        declared_in=tmp_path / "pipeline.yaml",
    )

    run_dir = tmp_path / "run"
    endowments_path = run_dir / "endowments.csv"
    responses_path = run_dir / "responses.csv"

    generated = SimpleNamespace(
        endowments_path=endowments_path,
    )
    elicited = SimpleNamespace(
        responses_path=responses_path,
    )
    regression = object()
    canonical_report = object()
    calls: dict[str, Any] = {}

    def fake_generate(
        received_config,
        *,
        run_dir: Path,
        verbose: bool,
    ):
        calls["generate"] = {
            "config": received_config,
            "run_dir": run_dir,
            "verbose": verbose,
        }
        endowments_path.write_text(
            "eid,endow_text,role,weight\n",
            encoding="utf-8",
        )
        return generated

    def fake_elicit(
        received_config,
        *,
        generation,
        run_dir: Path,
        verbose: bool,
    ):
        calls["elicit"] = {
            "config": received_config,
            "generation": generation,
            "run_dir": run_dir,
            "verbose": verbose,
        }
        responses_path.write_text(
            "eid,qid,response\n",
            encoding="utf-8",
        )
        return elicited

    def fake_regress(
        received_config,
        *,
        solver: str,
        verbose: bool,
        basis,
    ):
        calls["regress"] = {
            "config": received_config,
            "solver": solver,
            "verbose": verbose,
            "basis": dict(basis),
        }
        return regression

    def fake_report(
        received_config,
        received_regression,
        *,
        run_dir: Path,
    ):
        calls["report"] = {
            "config": received_config,
            "regression": received_regression,
            "run_dir": run_dir,
        }
        return canonical_report

    monkeypatch.setattr(
        pipeline,
        "generate_stage",
        SimpleNamespace(run=fake_generate),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "elicit_stage",
        SimpleNamespace(run=fake_elicit),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "regress_command",
        SimpleNamespace(run=fake_regress),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "report_stage",
        SimpleNamespace(
            write_regression_report=fake_report
        ),
        raising=False,
    )

    result = pipeline.run(
        config,
        solver="OSQP",
        run_dir=run_dir,
        verbose=True,
    )

    assert isinstance(result, pipeline.PipelineResult)
    assert result.config_digest == config.digest
    assert result.run_dir == run_dir.resolve()
    assert result.generation is generated
    assert result.elicitation is elicited
    assert result.regression is regression
    assert result.report is canonical_report
    assert run_dir.is_dir()

    assert calls["generate"] == {
        "config": config,
        "run_dir": run_dir.resolve(),
        "verbose": True,
    }
    assert calls["elicit"] == {
        "config": config,
        "generation": generated,
        "run_dir": run_dir.resolve(),
        "verbose": True,
    }
    assert calls["regress"] == {
        "config": config,
        "solver": "OSQP",
        "verbose": True,
        "basis": {
            "responses": responses_path.resolve(),
            "endowments": endowments_path.resolve(),
        },
    }
    assert calls["report"] == {
        "config": config,
        "regression": regression,
        "run_dir": run_dir.resolve(),
    }


def test_pipeline_runs_vanilla_dummy_end_to_end(
    tmp_path: Path,
    cvxpy_solver: str,
) -> None:
    config = Config.load(
        PROJECT_ROOT
        / "config"
        / "central"
        / "pipeline_fixture.yaml"
    )
    run_dir = tmp_path / "pipeline-run"

    result = pipeline.run(
        config,
        solver=cvxpy_solver,
        run_dir=run_dir,
        verbose=False,
    )

    assert result.config_digest == config.digest
    assert result.run_dir == run_dir.resolve()

    assert result.generation.mode == "vanilla"
    assert len(result.generation.endowments) == 2
    assert result.generation.endowments_path.is_file()

    assert len(result.elicitation.records) == 6
    assert result.elicitation.responses_path.is_file()

    assert result.regression.fit.status == "optimal"
    assert result.regression.fit.method == "lasso"
    assert result.regression.config_digest == config.digest

    assert set(result.regression.evaluation.values) == {
        "mse",
        "one_minus_jsd",
        "emd",
        "mcpa",
    }
    assert all(
        math.isfinite(value)
        for value
        in result.regression.evaluation.values.values()
    )

    source_artifacts = (
        result.regression.data.source_artifacts
    )
    assert source_artifacts["responses"] == (
        result.elicitation.responses_path
    )
    assert source_artifacts["endowments"] == (
        result.generation.endowments_path
    )

    assert result.report.run_json.is_file()
    assert result.report.metrics_json.is_file()
    assert result.report.metrics_csv.is_file()
    assert result.report.config_snapshot.is_file()
    assert result.usage is not None
    assert result.usage.csv_path.is_file()
    assert result.usage.json_path.is_file()
    assert result.log_path is not None
    assert result.log_path.is_file()

    run_payload = json.loads(
        result.report.run_json.read_text(
            encoding="utf-8"
        )
    )
    assert run_payload["run_id"] == (
        "fixture_vanilla_pipeline-r0001"
    )
    assert run_payload["conditions"] == {
        "method": "lasso",
    }
    assert run_payload["config_digest"] == config.digest
    assert run_payload["cleaning_policy"] == (
        "column_drop_v2"
    )

    metrics_payload = json.loads(
        result.report.metrics_json.read_text(
            encoding="utf-8"
        )
    )
    assert metrics_payload["run_id"] == (
        run_payload["run_id"]
    )
    assert len(metrics_payload["metrics"]) == 4

    rows = {
        row["metric"]: row
        for row in metrics_payload["metrics"]
    }
    assert rows["mse"]["unit"] == "binary_option"
    assert rows["mse"]["prediction_policy"] == "raw"

    for metric in (
        "one_minus_jsd",
        "emd",
        "mcpa",
    ):
        assert rows[metric]["unit"] == "question"
        assert rows[metric]["prediction_policy"] == (
            "clip_and_normalize"
        )
        assert {
            "invalid_qids_before_transform",
            "invalid_qids_after_transform",
            "negative_entry_count",
            "negative_mass",
            "raw_sum_deviation",
            "transformation_l1",
        } <= set(rows[metric]["diagnostics"])


def test_active_pipeline_reuses_adaptive_generation_responses(
    tmp_path: Path,
    cvxpy_solver: str,
) -> None:
    pipeline_config = Config.load(
        PROJECT_ROOT
        / "config"
        / "central"
        / "pipeline_fixture.yaml"
    ).to_dict()
    active_config = Config.load(
        PROJECT_ROOT
        / "config"
        / "central"
        / "generate_active_fixture.yaml"
    ).to_dict()
    pipeline_config["run"]["name"] = (
        "active-pipeline-reuse"
    )
    pipeline_config["generation"] = (
        active_config["generation"]
    )
    pipeline_config["backend"] = active_config["backend"]
    config = Config.from_mapping(
        pipeline_config,
        declared_in=tmp_path / "active-pipeline.yaml",
    )

    result = pipeline.run(
        config,
        solver=cvxpy_solver,
        run_dir=tmp_path / "active-pipeline",
    )

    assert result.generation.mode == "active"
    assert (
        result.elicitation.reused_generation_responses
        is True
    )
    assert result.elicitation.responses_path == (
        result.generation.responses_path
    )
    tracker = result.elicitation.usage_tracker
    assert tracker is not None
    survey_calls = [
        record
        for record in tracker.records
        if record.module_name == "survey_conductor"
    ]
    assert len(survey_calls) == 66
    assert result.log_path is not None
    assert "Reused responses" in (
        result.log_path.read_text(encoding="utf-8")
    )
