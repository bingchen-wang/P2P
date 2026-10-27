"""Tests for the local-only Experiment Studio bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p2p.commands import evaluate as evaluate_command
from p2p.config import Config
from p2p.studio import StudioState, run_studio


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_bootstrap_discovers_local_workspace_artifacts(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "config" / "central" / "pipeline.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "run:\n  name: studio_fixture\n  seed: 7\n"
        "generation:\n  mode: vanilla\n  target_n: 2\n"
        "backend:\n  response:\n    type: dummy\n    model_name: dummy\n"
        "model:\n  method: lasso\n",
        encoding="utf-8",
    )
    survey = tmp_path / "config" / "surveys" / "fixture.yaml"
    survey.parent.mkdir(parents=True)
    survey.write_text(
        "question_id: qid\nquestion_text: question\ncode_to_answer: options\n",
        encoding="utf-8",
    )
    backend = tmp_path / "config" / "backends" / "dummy.yaml"
    backend.parent.mkdir(parents=True)
    backend.write_text(
        "response:\n  type: dummy\n  model_name: dummy\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "outputs" / "fixture"
    _write_json(run_dir / "run.json", {"run_id": "fixture-r0001", "seed": 7})
    _write_json(
        run_dir / "metrics.json",
        {"metrics": [{"metric": "mse", "value": 0.1, "split": "test"}]},
    )
    _write_json(
        run_dir / "diagnostics.json",
        {
            "model": {
                "method": "lasso",
                "status": "optimal",
                "coefficients": {"agent_a": 0.4, "agent_b": 0.0},
            },
            "questions": [],
        },
    )
    (run_dir / "endowments.csv").write_text(
        'eid,endow_text,role,weight,mode,attributes\n'
        'agent_a,"Current profile A",proxy,,core,"gender; income"\n'
        'agent_b,"Current profile B",proxy,,core,"education"\n',
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "outputs" / "budget" / "budget.json",
        {"confidence": 0.8, "scenarios": []},
    )
    report_dir = tmp_path / "outputs" / "paper" / "legacy_fixture"
    report_dir.mkdir(parents=True)
    (report_dir / "summary.md").write_text(
        "# Regression Experiment Report\n\n**Experiment Name**: `legacy_fixture`\n",
        encoding="utf-8",
    )
    (report_dir / "predictions_lasso.png").write_bytes(b"fake-png")
    data_dir = tmp_path / "data" / "legacy_fixture"
    data_dir.mkdir(parents=True)
    (data_dir / "survey.csv").write_text(
        "id,question,option_mapping,split\n"
        'Q1,Question 1,"{\'A\': \'Option A\', \'B\': \'Option B\'}",train\n'
        'Q2,Question 2,"{\'A\': \'Option A\', \'B\': \'Option B\'}",valid\n'
        'Q3,Question 3,"{\'A\': \'Option A\', \'B\': \'Option B\'}",test\n',
        encoding="utf-8",
    )
    (data_dir / "survey.yaml").write_text(
        "question_id: id\nquestion_text: question\n"
        "code_to_answer: option_mapping\nsplit: split\n",
        encoding="utf-8",
    )
    _write_json(
        data_dir / "target.json",
        {
            qid: {"Option A": 0.6, "Option B": 0.4}
            for qid in ("Q1", "Q2", "Q3")
        },
    )
    (report_dir / "responses.csv").write_text(
        "qid,agent_a,agent_b\n"
        "Q1,Option A,Option B\n"
        "Q2,Option A,Option B\n"
        "Q3,Option A,Option B\n",
        encoding="utf-8",
    )
    (report_dir / "updated_endowments.csv").write_text(
        'eid,endow_text,role,weight,mode,attributes\n'
        'agent_a,"Profile A",proxy,0.75,core,"gender; income"\n'
        'agent_b,"Profile B",proxy,0.25,core+economics,"risk aversion"\n',
        encoding="utf-8",
    )
    (report_dir / "summary_aggregate.csv").write_text(
        "model,test_mse,test_one_minus_jsd,test_emd,test_accuracy\n"
        "lasso,0.04,0.88,0.12,0.75\n"
        "elasticnet,0.03,0.9,0.1,0.8\n",
        encoding="utf-8",
    )
    (report_dir / "summary_by_mode.csv").write_text(
        "model,mode,num_selected,total_weight,fraction_selected,avg_weight\n"
        "elasticnet,core,1,0.75,0.5,0.75\n"
        "elasticnet,core+economics,1,0.25,0.5,0.25\n",
        encoding="utf-8",
    )
    (report_dir / "mode_summary.csv").write_text(
        "mode,n,attributes\n"
        'core,12,"gender; income"\n'
        'core+economics,8,"risk aversion; price sensitivity"\n',
        encoding="utf-8",
    )
    (report_dir / "persistent_mode_entropy.csv").write_text(
        "step,core,core+economics\n0,0.2,0.3\n1,0.25,0.4\n",
        encoding="utf-8",
    )
    (report_dir / "regression_config.yaml").write_text(
        "experiment:\n  name: legacy_fixture\n  seed: 11\n"
        "paths:\n"
        "  survey_csv: data/legacy_fixture/survey.csv\n"
        "  survey_yaml: data/legacy_fixture/survey.yaml\n"
        "  aggregate_json: data/legacy_fixture/target.json\n"
        "  responses_csv: outputs/paper/legacy_fixture/responses.csv\n"
        "  endowments_csv: outputs/paper/legacy_fixture/endowments.csv\n"
        "weight_assignment_model: elasticnet\n",
        encoding="utf-8",
    )
    (report_dir / "_freeze.yaml").write_text(
        "source_repository: /example/alignment\n"
        "source_path: outputs/legacy_fixture\n",
        encoding="utf-8",
    )

    payload = StudioState(tmp_path).bootstrap()

    assert payload["localOnly"] is True
    assert payload["manifests"][0]["name"] == "studio_fixture"
    current = next(item for item in payload["runs"] if item["origin"] == "current")
    historical = next(item for item in payload["runs"] if item["origin"] == "legacy")
    assert current["runId"] == "fixture-r0001"
    assert current["metrics"][0]["value"] == pytest.approx(0.1)
    assert current["population"] == {
        "generated": 2,
        "selected": 1,
        "antiTypes": 0,
        "sourcePath": "outputs/fixture",
        "modes": [
            {
                "mode": "core",
                "family": "Core",
                "generated": 2,
                "selected": 1,
                "antiTypes": 0,
            }
        ],
    }
    assert historical["runId"] == "legacy_fixture"
    assert historical["questionCount"] == 3
    assert historical["metrics"][0]["value"] == pytest.approx(0.03)
    assert historical["population"]["generated"] == 2
    assert historical["population"]["selected"] == 2
    assert {
        item["mode"]: item["family"]
        for item in historical["population"]["modes"]
    } == {"core": "Core", "core+economics": "Thematic"}
    assert payload["budgets"][0]["path"] == "outputs/budget"
    assert payload["surveys"][0]["questionId"] == "qid"
    assert payload["backends"][0]["local"] is True
    assert payload["reports"][0]["kind"] == "regression"
    assert payload["reports"][0]["imageCount"] == 1

    report = StudioState(tmp_path).report_detail(
        "outputs/paper/legacy_fixture"
    )
    assert report["documents"][0]["title"] == (
        "Regression Experiment Report"
    )
    assert report["images"][0]["category"] == "predictions"
    assert report["summary"]["models"] == ["lasso", "elasticnet"]
    assert report["modeEntropyRows"][1]["core+economics"] == "0.4"
    assert report["modeSummaryRows"][0]["n"] == "12"
    assert report["modeRows"][0]["total_weight"] == "0.75"
    assert report["endowmentRows"][0]["endow_text"] == "Profile A"

    current_detail = StudioState(tmp_path).detail("outputs/fixture", "run")
    assert current_detail["endowmentRows"][0]["endow_text"] == (
        "Current profile A"
    )
    assert current_detail["modeSelectionSource"] == "derived_coefficients"
    assert current_detail["modeRows"] == [
        {
            "model": "lasso",
            "mode": "core",
            "num_selected": "1",
            "num_positive": "1",
            "num_negative": "0",
            "total_weight": "0.4",
            "total_abs_weight": "0.4",
            "fraction_selected": "0.5",
            "avg_weight": "0.4",
        }
    ]

    historical_detail = StudioState(tmp_path).detail(
        "outputs/paper/legacy_fixture",
        "run",
    )
    assert historical_detail["run"]["origin"] == "legacy"
    assert historical_detail["model"]["method"] == "elasticnet"
    assert {
        item["model"] for item in historical_detail["metrics"]["metrics"]
    } == {"lasso", "elasticnet"}
    assert [item["split"] for item in historical_detail["questions"]] == [
        "train",
        "valid",
        "test",
    ]
    assert historical_detail["questions"][0]["prediction"] == {
        "Option A": 0.75,
        "Option B": 0.25,
    }
    assert historical_detail["archive"]["summary"]["name"] == "legacy_fixture"
    assert historical_detail["modeEntropyRows"][1]["core"] == "0.25"
    assert historical_detail["endowmentRows"][1]["mode"] == "core+economics"
    assert historical_detail["modeSelectionSource"] == "summary_by_mode"


def test_detail_rejects_paths_outside_outputs(tmp_path: Path) -> None:
    state = StudioState(tmp_path)

    with pytest.raises(ValueError, match="inside outputs"):
        state.detail("config", "run")

    with pytest.raises(ValueError, match="inside outputs"):
        state.report_detail("config")

    with pytest.raises(ValueError, match="inside outputs"):
        state.artifact_path("config/example.png")


def test_detail_resolves_stage_one_basis_for_stage_two_only_run(
    tmp_path: Path,
) -> None:
    basis = tmp_path / "outputs" / "basis" / "wave_fixture"
    basis.mkdir(parents=True)
    (basis / "updated_endowments.csv").write_text(
        "eid,endow_text,role,weight,mode,attributes\n"
        'agent_core,"Core profile",proxy,,core,"age"\n'
        'agent_survey,"Survey profile",proxy,,survey,"answer"\n'
        'agent_mixed,"Mixed profile",proxy,,core+economics+Q1_W9,"income"\n',
        encoding="utf-8",
    )
    (basis / "persistent_mode_entropy.csv").write_text(
        "step,core,survey,core+economics+Q1_W9\n"
        "0,0.2,0.3,0.4\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "outputs" / "stage2" / "wave_fixture"
    _write_json(run_dir / "run.json", {"run_id": "stage2-wave-fixture"})
    _write_json(run_dir / "metrics.json", {"metrics": []})
    _write_json(
        run_dir / "diagnostics.json",
        {
            "model": {
                "method": "lasso",
                "coefficients": {
                    "agent_core": 0.4,
                    "agent_survey": 0.0,
                    "agent_mixed": -0.2,
                },
            },
            "questions": [],
        },
    )
    (run_dir / "config.resolved.yaml").write_text(
        "config:\n"
        "  data:\n"
        f'    basis: "{basis}"\n',
        encoding="utf-8",
    )

    detail = StudioState(tmp_path).detail(
        "outputs/stage2/wave_fixture",
        "run",
    )

    assert [row["mode"] for row in detail["endowmentRows"]] == [
        "core",
        "survey",
        "core+economics+Q1_W9",
    ]
    assert detail["modeSelectionSource"] == "derived_coefficients"
    assert detail["modeEntropyRows"][0]["core+economics+Q1_W9"] == "0.4"
    mixed = next(
        row
        for row in detail["modeRows"]
        if row["mode"] == "core+economics+Q1_W9"
    )
    assert mixed["num_selected"] == "1"
    assert mixed["num_negative"] == "1"


def test_manifest_summary_resolves_inherited_plan(tmp_path: Path) -> None:
    base = tmp_path / "config" / "central" / "base.yaml"
    base.parent.mkdir(parents=True)
    base.write_text(
        "run:\n  name: base\n  seed: 101\n"
        "generation:\n  mode: vanilla\n  target_n: 12\n"
        "backend:\n  response:\n    type: dummy\n    model_name: dummy\n"
        "model:\n  method: lasso\n  alpha: [0.01]\n",
        encoding="utf-8",
    )
    child = tmp_path / "config" / "experiments" / "child.yaml"
    child.parent.mkdir(parents=True)
    child.write_text(
        "extends: ../central/base.yaml\nrun:\n  name: child\n  seed: 202\n",
        encoding="utf-8",
    )

    payload = StudioState(tmp_path).bootstrap()
    summary = next(item for item in payload["manifests"] if item["path"].endswith("child.yaml"))

    assert summary["name"] == "child"
    assert summary["seed"] == 202
    assert summary["mode"] == "vanilla"
    assert summary["targetN"] == 12
    assert summary["method"] == "lasso"


def test_guided_manifest_is_validated_and_saved(tmp_path: Path) -> None:
    survey = tmp_path / "data" / "survey.csv"
    schema = tmp_path / "config" / "surveys" / "survey.yaml"
    target = tmp_path / "data" / "target.json"
    backend = tmp_path / "config" / "backends" / "dummy.yaml"
    for path in (survey, schema, target, backend):
        path.parent.mkdir(parents=True, exist_ok=True)
    survey.write_text("qid,question,split,options\n", encoding="utf-8")
    schema.write_text(
        "question_id: qid\nquestion_text: question\ncode_to_answer: options\n",
        encoding="utf-8",
    )
    target.write_text("{}", encoding="utf-8")
    backend.write_text(
        "response:\n  type: dummy\n  model_name: dummy\n",
        encoding="utf-8",
    )
    config = {
        "run": {"name": "Guided Example", "seed": 7},
        "data": {
            "survey": {
                "source": "data/survey.csv",
                "schema": "config/surveys/survey.yaml",
            },
            "target": "data/target.json",
        },
        "generation": {"mode": "vanilla", "target_n": 2},
        "backend": {"profile": "config/backends/dummy.yaml"},
        "model": {
            "method": "lasso",
            "alpha": [0.01],
            "validation": {"strategy": "cv", "cv_folds": 2},
        },
        "split": {"train_val": ["train", "valid"], "test": "test"},
        "metrics": ["mse"],
    }

    created = StudioState(tmp_path).create_manifest(
        {"command": "pipeline", "config": config}
    )

    assert created["path"] == "config/experiments/studio/guided_example.yaml"
    assert (tmp_path / created["path"]).is_file()
    assert created["manifest"]["mode"] == "vanilla"
    assert "../../../data/survey.csv" in created["content"]

    with pytest.raises(ValueError, match="already exists"):
        StudioState(tmp_path).create_manifest(
            {"command": "pipeline", "config": config}
        )


def test_ui_server_rejects_non_loopback_hosts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="local-only"):
        run_studio(root=tmp_path, host="0.0.0.0", open_browser=False)


def test_backend_readiness_check_is_local_and_secret_safe(tmp_path: Path) -> None:
    backend = tmp_path / "config" / "backends" / "dummy.yaml"
    backend.parent.mkdir(parents=True)
    backend.write_text(
        "response:\n  type: dummy\n  model_name: dummy\n",
        encoding="utf-8",
    )

    result = StudioState(tmp_path).check_backend(
        {"path": "config/backends/dummy.yaml"}
    )

    assert result["ready"] is True
    assert result["networkCallMade"] is False
    assert {item["label"] for item in result["checks"]} == {
        "Profile",
        "Python dependency",
        "Provider boundary",
    }

    outside = tmp_path / "config" / "surveys" / "not-a-backend.yaml"
    outside.parent.mkdir(parents=True)
    outside.write_text("response: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inside config/backends"):
        StudioState(tmp_path).check_backend(
            {"path": "config/surveys/not-a-backend.yaml"}
        )


def test_remote_backend_readiness_reports_credential_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = tmp_path / "config" / "backends" / "openai.yaml"
    backend.parent.mkdir(parents=True)
    backend.write_text(
        "response:\n  type: openai\n  model_name: gpt-5-mini\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = StudioState(tmp_path).check_backend(
        {"path": "config/backends/openai.yaml"}
    )

    credential = next(
        item for item in result["checks"] if item["label"] == "Credential"
    )
    assert credential == {
        "label": "Credential",
        "ok": False,
        "detail": "OPENAI_API_KEY",
    }
    assert result["networkCallMade"] is False


def test_manifest_detail_supports_safe_duplicate_setup(tmp_path: Path) -> None:
    survey = tmp_path / "data" / "survey.csv"
    schema = tmp_path / "config" / "surveys" / "survey.yaml"
    target = tmp_path / "data" / "target.json"
    predictions = tmp_path / "outputs" / "prior" / "predictions.json"
    manifest = tmp_path / "config" / "experiments" / "source.yaml"
    for path in (survey, schema, target, predictions, manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    survey.write_text("qid,question,split,options\n", encoding="utf-8")
    schema.write_text(
        "question_id: qid\nquestion_text: question\ncode_to_answer: options\n",
        encoding="utf-8",
    )
    target.write_text("{}", encoding="utf-8")
    predictions.write_text("{}", encoding="utf-8")
    manifest.write_text(
        "run:\n  name: source\n  seed: 9\n"
        "data:\n"
        "  survey:\n    source: ../../data/survey.csv\n"
        "    schema: ../surveys/survey.yaml\n"
        "  target: ../../data/target.json\n"
        "  predictions: ../../outputs/prior/predictions.json\n"
        "split:\n  train_val: [train, valid]\n  test: test\n"
        "metrics: [mse]\n",
        encoding="utf-8",
    )

    result = StudioState(tmp_path).manifest_detail(
        "config/experiments/source.yaml"
    )

    assert result["command"] == "evaluate"
    assert result["config"]["data"]["survey"]["source"] == "data/survey.csv"
    assert result["config"]["data"]["predictions"] == (
        "outputs/prior/predictions.json"
    )


def test_workspace_capabilities_and_pagination_use_normalized_contract(
    tmp_path: Path,
) -> None:
    for index in range(4):
        run_dir = tmp_path / "outputs" / f"run_{index}"
        _write_json(
            run_dir / "run.json",
            {"run_id": f"run-{index}", "seed": index},
        )
        _write_json(
            run_dir / "metrics.json",
            {
                "metrics": [
                    {
                        "metric": "mse",
                        "value": index / 10,
                        "split": "test",
                        "unit": "binary_option",
                        "prediction_policy": "raw",
                        "qids_requested": 3,
                        "qids_used": 3,
                    }
                ]
            },
        )

    state = StudioState(tmp_path)
    workspace = state.workspace()
    first_page = state.list_runs(limit=2)
    second_page = state.list_runs(cursor=first_page["nextCursor"], limit=2)
    best_mse = state.list_runs(limit=4, sort="mse")
    capabilities = state.capabilities()

    assert workspace["counts"]["runs"] == 4
    assert workspace["counts"]["currentRuns"] == 4
    assert first_page["total"] == 4
    assert len(first_page["items"]) == 2
    assert len(second_page["items"]) == 2
    assert [item["runId"] for item in best_mse["items"]] == [
        "run-0",
        "run-1",
        "run-2",
        "run-3",
    ]
    assert first_page["items"][0]["sourceFormat"] == "p2p-canonical-v1"
    assert first_page["items"][0]["metrics"][0]["qidsUsed"] == 3
    assert {item["id"] for item in capabilities["models"]} >= {
        "lasso",
        "elasticnet",
    }
    assert capabilities["defaults"]["distributionPredictionPolicy"] == (
        "clip_and_normalize"
    )


def test_dataset_inspection_and_registration_are_local_and_digest_pinned(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data" / "user" / "survey.csv"
    schema = tmp_path / "config" / "surveys" / "survey.yaml"
    target = tmp_path / "data" / "user" / "target.json"
    for path in (data, schema, target):
        path.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(
        "qid,question,options,split\n"
        "Q1,One?,\"{'A': 'Yes', 'B': 'No'}\",train\n"
        "Q2,Two?,\"{'A': 'Yes', 'B': 'No'}\",valid\n"
        "Q3,Three?,\"{'A': 'Yes', 'B': 'No'}\",test\n",
        encoding="utf-8",
    )
    schema.write_text(
        "question_id: qid\nquestion_text: question\n"
        "code_to_answer: options\nsplit: split\n",
        encoding="utf-8",
    )
    _write_json(
        target,
        {
            qid: {"Yes": 0.6, "No": 0.4}
            for qid in ("Q1", "Q2", "Q3")
        },
    )
    state = StudioState(tmp_path)
    body = {
        "name": "User survey",
        "source": "data/user/survey.csv",
        "schema": "config/surveys/survey.yaml",
        "target": "data/user/target.json",
    }

    inspected = state.inspect_dataset(body)
    registered = state.register_dataset(body)

    assert inspected["valid"] is True
    assert inspected["splitCounts"] == {"train": 1, "valid": 1, "test": 1}
    assert inspected["targetCoverage"]["matched"] == 3
    assert registered["path"] == "config/datasets/user_survey.yaml"
    assert len(registered["digests"]["source"]) == 64
    assert StudioState(tmp_path).datasets()[0]["name"] == "User survey"

    with pytest.raises(ValueError, match="inside the repository"):
        state.inspect_dataset({"source": str(tmp_path.parent / "outside.csv")})


def test_mapping_wizard_can_write_schema_and_register_dataset(
    tmp_path: Path,
) -> None:
    survey = tmp_path / "data" / "mapped.csv"
    target = tmp_path / "data" / "target.json"
    survey.parent.mkdir(parents=True)
    survey.write_text(
        "key,prompt,answers,partition\n"
        "Q1,One?,\"{'A': 'Yes', 'B': 'No'}\",train\n"
        "Q2,Two?,\"{'A': 'Yes', 'B': 'No'}\",valid\n"
        "Q3,Three?,\"{'A': 'Yes', 'B': 'No'}\",test\n",
        encoding="utf-8",
    )
    _write_json(
        target,
        {qid: {"Yes": 0.5, "No": 0.5} for qid in ("Q1", "Q2", "Q3")},
    )
    body = {
        "name": "Mapped survey",
        "source": "data/mapped.csv",
        "target": "data/target.json",
        "mapping": {
            "question_id": "key",
            "question_text": "prompt",
            "code_to_answer": "answers",
            "split": "partition",
        },
    }

    registered = StudioState(tmp_path).register_dataset(body)

    assert registered["schema"] == "config/surveys/mapped_survey.yaml"
    assert (tmp_path / registered["schema"]).is_file()
    assert registered["splitCounts"] == {"train": 1, "valid": 1, "test": 1}


def test_annotations_and_downloads_stay_inside_output_records(tmp_path: Path) -> None:
    first = tmp_path / "outputs" / "first"
    second = tmp_path / "outputs" / "second"
    _write_json(first / "run.json", {"run_id": "first"})
    _write_json(first / "metrics.json", {"metrics": []})
    _write_json(second / "run.json", {"run_id": "second"})
    _write_json(second / "metrics.json", {"metrics": []})
    (first / "summary.md").write_text("# First\n", encoding="utf-8")
    (second / "summary.md").write_text("# Second\n", encoding="utf-8")
    state = StudioState(tmp_path)

    updated = state.update_annotation(
        {"path": "outputs/second", "label": "Candidate", "pinned": True}
    )
    pinned = state.list_runs(limit=10, sort="pinned")

    assert updated["annotation"]["label"] == "Candidate"
    assert pinned["items"][0]["path"] == "outputs/second"
    assert state.list_reports(limit=10, sort="pinned")["items"][0]["path"] == "outputs/second"
    assert state.list_runs(limit=10, query="candidate")["total"] == 1
    assert state.download_path("outputs/first/run.json") == first / "run.json"
    model = first / "model.joblib"
    model.write_bytes(b"local model artifact")
    assert state.download_path("outputs/first/model.joblib") == model
    with pytest.raises(ValueError, match="public output records"):
        state.download_path("outputs/.studio/annotations.json")


def test_durable_job_record_becomes_interrupted_after_restart(
    tmp_path: Path,
) -> None:
    record = tmp_path / "outputs" / ".studio" / "jobs" / "job123.json"
    _write_json(
        record,
        {
            "id": "job123",
            "command": "pipeline",
            "config": "config/example.yaml",
            "output": "outputs/incomplete",
            "status": "running",
            "createdTimestamp": 1000.0,
            "log": ["started"],
            "recovery": "live",
        },
    )

    job = StudioState(tmp_path).bootstrap()["jobs"][0]

    assert job["status"] == "interrupted"
    assert job["recovery"] == "new-output"
    assert job["durable"] is True
    assert "no longer attached" in job["log"][-1]


def test_bundled_offline_demo_writes_a_canonical_result(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = Config.load(root / "config" / "smoke" / "studio_offline_demo.yaml")

    result = evaluate_command.run(config, run_dir=tmp_path / "outputs" / "demo")
    run = json.loads(result.report.run_json.read_text(encoding="utf-8"))
    metrics = json.loads(result.report.metrics_json.read_text(encoding="utf-8"))

    assert run["run_id"] == "studio-offline-demo-r0001"
    test_rows = [row for row in metrics["metrics"] if row["split"] == "test"]
    assert {row["metric"] for row in test_rows} == {
        "mse",
        "one_minus_jsd",
        "emd",
        "mcpa",
    }
    # The demo survey's test split has four questions (examples/studio_demo).
    assert all(row["qids_requested"] == 4 for row in test_rows)
    assert all(row["qids_used"] == 4 for row in test_rows)

    detail = StudioState(tmp_path).detail("outputs/demo", "run")
    assert detail["run"]["run_id"] == "studio-offline-demo-r0001"
    assert detail["model"]["method"] == "offline-demo"
    assert detail["diagnosticsAvailable"] is False
    assert detail["metrics"]["metrics"] == metrics["metrics"]


def test_static_page_is_revalidated_and_hashed_assets_are_cached(tmp_path: Path) -> None:
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from p2p.studio import _handler_factory

    static = tmp_path / "dist"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>Studio</title>")
    (static / "assets" / "index-abc123.css").write_text("body{}")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_factory(StudioState(tmp_path), static))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for path in ("/", "/?view=runs"):
            with urllib.request.urlopen(base + path) as response:
                assert response.headers["Cache-Control"] == "no-cache"
        with urllib.request.urlopen(base + "/assets/index-abc123.css") as response:
            assert response.headers["Cache-Control"] == "public, max-age=31536000, immutable"
        with urllib.request.urlopen(base + "/api/jobs") as response:
            assert response.headers["Cache-Control"] != "no-cache"
    finally:
        server.shutdown()
