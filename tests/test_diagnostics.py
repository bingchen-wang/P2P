"""Interactive diagnostics artifact contracts."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from p2p.commands import regress
from p2p.config import Config
from p2p.diagnostics import build_dashboard
from p2p.earlier_runs import load_earlier_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_regression_report_feeds_full_interactive_dashboard(
    tmp_path: Path,
) -> None:
    config = Config.load(
        PROJECT_ROOT
        / "config"
        / "central"
        / "regress_fixture.yaml"
    )
    run_dir = tmp_path / "regression"
    result = regress.run(
        config,
        solver="OSQP",
        run_dir=run_dir,
    )

    assert result.report.diagnostics_json is not None
    payload = json.loads(
        result.report.diagnostics_json.read_text(encoding="utf-8")
    )
    assert payload["artifact_type"] == "regression_diagnostics"
    assert len(payload["questions"]) == 3
    assert payload["model"]["coefficients"]

    dashboard = build_dashboard(
        run_dir,
        advanced=True,
    )
    html = dashboard.output_path.read_text(encoding="utf-8")

    assert dashboard.output_path == run_dir / "diagnostics.html"
    assert {
        "metrics",
        "question-entropy",
        "response-distribution",
        "prediction-scatter",
        "prediction-distribution",
        "weights",
        "singular_values",
        "mca_agents",
    } <= set(dashboard.sections)
    assert "const responseDetails" in html
    assert "const predictionDetails" in html
    assert "Plotly.react('p2p-prediction-distribution'" in html
    assert "No backend or network request is made" in html


def test_dashboard_handles_a_partially_written_run(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "running"
    run_dir.mkdir()
    (run_dir / "pipeline.log").write_text(
        "starting generation\n",
        encoding="utf-8",
    )

    dashboard = build_dashboard(
        run_dir,
        refresh_seconds=2,
    )
    html = dashboard.output_path.read_text(encoding="utf-8")

    assert dashboard.sections == ("waiting",)
    assert dashboard.warnings
    assert "in progress / legacy" in html
    assert "http-equiv='refresh' content='2'" in html
    assert "starting generation" in html


def test_dashboard_reconstructs_legacy_regression_without_pickle(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "alignment"
    data_dir = legacy_root / "data" / "fixture"
    run_dir = legacy_root / "outputs" / "legacy_run"
    data_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    shutil.copyfile(
        PROJECT_ROOT / "tests" / "fixtures" / "data" / "survey.csv",
        data_dir / "survey.csv",
    )
    shutil.copyfile(
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / "data"
        / "survey_schema.yaml",
        data_dir / "survey_schema.yaml",
    )
    shutil.copyfile(
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / "commands"
        / "pipeline"
        / "target.json",
        data_dir / "target.json",
    )
    shutil.copyfile(
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / "commands"
        / "regress"
        / "basis"
        / "responses.csv",
        run_dir / "responses.csv",
    )
    (run_dir / "updated_endowments.csv").write_text(
        "eid,endow_text,role,weight,mode,attributes\n"
        "agent_a,A,proxy,0.75,legacy,\n"
        "agent_b,B,proxy,0.25,legacy,\n",
        encoding="utf-8",
    )
    (run_dir / "regression_config.yaml").write_text(
        "paths:\n"
        "  aggregate_json: data/fixture/target.json\n"
        "  endowments_csv: outputs/legacy_run/endowments.csv\n"
        "  responses_csv: outputs/legacy_run/responses.csv\n"
        "  survey_csv: data/fixture/survey.csv\n"
        "  survey_yaml: data/fixture/survey_schema.yaml\n"
        "weight_assignment_model: elasticnet\n",
        encoding="utf-8",
    )
    (run_dir / "summary_aggregate.csv").write_text(
        "model,trainval_mse,test_mse,test_r2\n"
        "elasticnet,0.02,0.03,0.7\n",
        encoding="utf-8",
    )
    (run_dir / "token_usage_summary.csv").write_text(
        "module_name,input_tokens,output_tokens\n"
        "SurveyConductor,10,5\n",
        encoding="utf-8",
    )
    # A PNG signature is sufficient for the self-contained archival gallery.
    (run_dir / "predictions_elasticnet.png").write_bytes(
        b"\x89PNG\r\n\x1a\nlegacy"
    )
    # If the adapter ever starts unpickling implicitly, this fixture must fail.
    (run_dir / "elasticnet_model.joblib").write_bytes(
        b"not a trusted pickle"
    )
    original_names = sorted(path.name for path in run_dir.iterdir())

    bundle = load_earlier_run(run_dir)
    assert bundle.diagnostics is not None
    assert bundle.diagnostics["artifact_type"] == (
        "legacy_regression_diagnostics"
    )
    assert bundle.diagnostics["legacy_adapter"]["joblib_loaded"] is False
    assert len(bundle.diagnostics["questions"]) == 3
    assert bundle.diagnostics["questions"][0]["prediction"] == {
        "Option A": 0.75,
        "Option B": 0.25,
    }

    output = tmp_path / "scratch" / "legacy.html"
    dashboard = build_dashboard(
        run_dir,
        output_path=output,
        advanced=True,
    )
    html = output.read_text(encoding="utf-8")

    assert {
        "metrics",
        "question-entropy",
        "response-distribution",
        "prediction-scatter",
        "prediction-distribution",
        "weights",
        "usage",
        "singular_values",
        "mca_agents",
        "legacy-images",
    } <= set(dashboard.sections)
    assert "legacy parsed" in html
    assert "const predictionDetails" in html
    assert "human probability" in html
    assert "no joblib file was loaded" in html
    assert "data:image/png;base64," in html
    assert sorted(path.name for path in run_dir.iterdir()) == original_names
