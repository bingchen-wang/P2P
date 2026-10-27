"""Studio contracts for Stage 2: capabilities, builder manifests, health, summaries.

The builder in studio/app/page.tsx writes model blocks from /api/capabilities.
These tests write the same blocks from the same capabilities and fit them with
``p2p regress`` on the license-free demo basis, so a configuration Studio offers
is one P2P runs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from p2p.commands import regress as regress_command
from p2p.config import Config, expand_grid
from p2p.stages.evaluate import SUPPORTED_METRICS
from p2p.stages.fit import DEFAULT_ALPHA_GRID, validate_model_options
from p2p.studio import METRIC_DESCRIPTIONS, StudioState

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "examples" / "studio_demo", tmp_path / "examples" / "studio_demo")
    (tmp_path / "config" / "experiments" / "studio").mkdir(parents=True)
    return tmp_path


def _builder_config(name: str, model: dict, metrics: list[str]) -> dict:
    """The manifest page.tsx builds for a regress run on the demo data."""

    return {
        "run": {"name": name, "seed": 101},
        "data": {
            "survey": {
                "source": "examples/studio_demo/survey.csv",
                "schema": "examples/studio_demo/survey.yaml",
            },
            "target": "examples/studio_demo/targets.json",
            "basis": "examples/studio_demo/basis",
        },
        "model": model,
        "split": {"train_val": ["train", "valid"], "test": "test", "use_proxy_only": True},
        "metrics": metrics,
        "evaluation": {
            "mse_prediction_policy": "raw",
            "distribution_prediction_policy": "clip_and_normalize",
            "invalid_distribution": "error",
        },
    }


def _fit(state: StudioState, root: Path, name: str, model: dict, *, solver: str | None = None):
    caps = state.capabilities()
    config = _builder_config(name, model, caps["defaults"]["metrics"])
    preview = state.preview_manifest(
        {"command": "regress", "config": config, "output": f"outputs/studio/{name}", "solver": solver}
    )
    assert preview["valid"], preview["warnings"]
    created = state.create_manifest({"command": "regress", "config": config})
    run_dir = root / "outputs" / "studio" / name
    regress_command.run(Config.load(root / created["path"]), run_dir=run_dir, solver=solver or "CLARABEL")
    return preview, run_dir


def test_capabilities_metrics_are_exactly_the_evaluators() -> None:
    caps = StudioState(ROOT).capabilities()
    assert {metric["id"] for metric in caps["metrics"]} == set(SUPPORTED_METRICS)
    assert set(METRIC_DESCRIPTIONS) >= set(SUPPORTED_METRICS)
    rescaled = next(metric for metric in caps["metrics"] if metric["id"] == "distribution_mse")
    assert rescaled["direction"] == "lower"
    assert caps["defaults"]["metrics"][:2] == ["mse", "distribution_mse"]


def test_capabilities_offer_simple_average_without_penalty_controls() -> None:
    caps = StudioState(ROOT).capabilities()
    models = {model["id"]: model for model in caps["models"]}
    assert models["simple_average"]["guided"] is True
    assert models["simple_average"]["controls"] == []
    assert "alpha" in models["lasso"]["controls"]
    assert models["weighted-lasso"]["guided"] is False


def test_capabilities_stage2_matches_p2p_defaults() -> None:
    stage2 = StudioState(ROOT).capabilities()["stage2"]
    assert stage2["alphaGrid"] == DEFAULT_ALPHA_GRID
    assert stage2["alphaGridValues"] == [
        min(expand_grid(DEFAULT_ALPHA_GRID)),
        max(expand_grid(DEFAULT_ALPHA_GRID)),
    ]
    assert stage2["defaultSolver"] in stage2["solvers"]
    assert stage2["solverCommands"] == ["pipeline", "regress"]
    ids = [item["id"] for item in stage2["configurations"]]
    assert ids == [
        "mass_penalty",
        "mass_penalty_reference",
        "mass_penalty_intercept",
        "no_mass_penalty",
        "no_mass_penalty_reference",
        "no_mass_penalty_intercept",
    ]
    for item in stage2["configurations"]:
        validate_model_options({"method": "lasso", **item["options"]})


def test_builder_no_longer_writes_options_p2p_ignores(workspace: Path) -> None:
    page = (ROOT / "studio" / "app" / "page.tsx").read_text(encoding="utf-8")
    assert "max_iter" not in page
    assert "post_selection_refit" not in page
    assert "logspace: [-5, 1, 20]" not in page


@pytest.mark.parametrize(
    "configuration",
    [
        "mass_penalty",
        "mass_penalty_reference",
        "mass_penalty_intercept",
        "no_mass_penalty",
        "no_mass_penalty_reference",
        "no_mass_penalty_intercept",
    ],
)
def test_every_offered_configuration_fits(workspace: Path, configuration: str) -> None:
    state = StudioState(workspace)
    stage2 = state.capabilities()["stage2"]
    options = next(item["options"] for item in stage2["configurations"] if item["id"] == configuration)
    model = {
        "method": "lasso",
        "alpha": stage2["alphaGrid"],
        **options,
        "validation": {"strategy": "cv", "cv_folds": 5},
    }
    _, run_dir = _fit(state, workspace, configuration, model)
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text())
    health = state.detail(f"outputs/studio/{configuration}", "run")["health"]
    facts = {fact["label"]: fact for fact in health["facts"]}

    assert facts["Stage-2 configuration"]["hint"] == configuration
    assert ("reference_respondent_weight" in diagnostics["model"]) == bool(
        options.get("reference_respondent")
    )
    assert ("intercept" in diagnostics["model"]) == bool(options.get("intercept"))
    assert "ignored_model_options" not in diagnostics["model"]


def test_simple_average_manifest_writes_only_its_method_and_fits(workspace: Path) -> None:
    state = StudioState(workspace)
    preview, run_dir = _fit(state, workspace, "simple", {"method": "simple_average"})
    written = yaml.safe_load(preview["content"])["model"]
    health = state.detail("outputs/studio/simple", "run")["health"]

    assert written == {"method": "simple_average"}
    assert json.loads((run_dir / "run.json").read_text())["run_id"]
    assert not any(check["label"].startswith("Penalty") for check in health["checks"])
    assert {fact["label"] for fact in health["facts"]} >= {"Method", "Train MSE"}


def test_holdout_and_fold_replay_manifests_fit(workspace: Path) -> None:
    state = StudioState(workspace)
    alpha = state.capabilities()["stage2"]["alphaGrid"]
    _, first = _fit(state, workspace, "cv_first", {
        "method": "lasso", "alpha": alpha, "validation": {"strategy": "cv", "cv_folds": 5},
    })
    record = f"outputs/studio/cv_first/folds.json"
    preview, replayed = _fit(state, workspace, "cv_replay", {
        "method": "lasso", "alpha": alpha,
        "validation": {"strategy": "cv", "cv_folds": 5, "folds": {"method": "replay", "file": record}},
    })
    _, holdout = _fit(state, workspace, "holdout", {
        "method": "lasso", "alpha": alpha,
        "validation": {"strategy": "holdout", "holdout_split": "valid"},
    })

    assert "../../../outputs/studio/cv_first/folds.json" in preview["content"]
    first_folds = json.loads((first / "folds.json").read_text())
    replay_folds = json.loads((replayed / "folds.json").read_text())
    assert replay_folds["partition_digest"] == first_folds["partition_digest"]
    assert replay_folds["method"] == "replay"
    assert not (holdout / "folds.json").exists()
    holdout_facts = {fact["label"]: fact["value"] for fact in
                     state.detail("outputs/studio/holdout", "run")["health"]["facts"]}
    assert holdout_facts["Penalty selection"] == "Holdout on valid"
    duplicated = state.manifest_detail("config/experiments/studio/cv_replay.yaml")
    assert duplicated["config"]["model"]["validation"]["folds"]["file"] == record


def test_solver_choice_reaches_the_cli_and_is_validated(workspace: Path) -> None:
    state = StudioState(workspace)
    alpha = state.capabilities()["stage2"]["alphaGrid"]
    preview, run_dir = _fit(state, workspace, "osqp", {
        "method": "lasso", "alpha": alpha, "validation": {"strategy": "cv", "cv_folds": 5},
    }, solver="OSQP")

    assert preview["cli"].endswith("--solver OSQP")
    assert preview["solver"] == "OSQP"
    assert json.loads((run_dir / "diagnostics.json").read_text())["model"]["solver"] == "OSQP"
    config = _builder_config("x", {"method": "simple_average"}, ["mse"])
    with pytest.raises(ValueError, match="solver must be one of"):
        state.preview_manifest({"command": "regress", "config": config, "solver": "MOSEK"})
    evaluate_config = {key: value for key, value in config.items() if key != "model"}
    evaluate_config["data"] = {**config["data"], "predictions": "examples/studio_demo/predictions.json"}
    with pytest.raises(ValueError, match="takes no solver"):
        state.preview_manifest({"command": "evaluate", "config": evaluate_config, "solver": "OSQP"})


def test_demo_manifests_fit_and_show_why_fitting_matters(tmp_path: Path) -> None:
    results = {}
    for method in ("lasso", "simple_average"):
        config = Config.load(ROOT / "config" / "smoke" / f"studio_demo_fit_{method}.yaml")
        run_dir = tmp_path / "outputs" / method
        regress_command.run(config, run_dir=run_dir, solver="CLARABEL")
        metrics = json.loads((run_dir / "metrics.json").read_text())["metrics"]
        results[method] = {row["metric"]: row for row in metrics if row["split"] == "test"}

    assert set(results["lasso"]) == set(SUPPORTED_METRICS)
    assert all(row["qids_used"] == row["qids_requested"] == 4 for row in results["lasso"].values())
    assert results["lasso"]["mse"]["value"] < results["simple_average"]["mse"]["value"]
    health = StudioState(tmp_path).detail("outputs/lasso", "run")["health"]
    assert any(check["label"] == "Solver" and check["level"] == "ok" for check in health["checks"])
    assert health["warningCount"] == 0


def test_health_reports_failed_answers_and_grid_edges(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"run_id": "r", "conditions": {"method": "lasso"}}))
    (run_dir / "responses.csv").write_text("qid,a,b\nQ1,Yes,[ERROR] OpenAI refused: no\n")
    (run_dir / "metrics.json").write_text(json.dumps({"metrics": [
        {"metric": "mse", "value": 0.1, "split": "test", "qids_requested": 3, "qids_used": 2},
    ]}))
    (run_dir / "diagnostics.json").write_text(json.dumps({"model": {
        "method": "lasso", "solver": "CLARABEL", "solver_used": "CLARABEL", "status": "optimal",
        "best_alpha": 10.0, "alpha_at_grid_edge": "ceiling", "cv_configs_within_one_se": 4,
        "cv_config_count": 20, "ignored_model_options": "max_iter",
    }}))

    health = StudioState(tmp_path).detail("outputs/run", "run")["health"]
    labels = {check["label"]: check["level"] for check in health["checks"]}

    assert health["status"] == "attention"
    assert labels["Failed answers"] == "warning"
    assert labels["Penalty at grid edge"] == "warning"
    assert labels["Incomplete coverage"] == "warning"
    assert labels["Options with no effect"] == "warning"
    assert labels["Flat validation curve"] == "info"
    assert health["warningCount"] == 4


def test_pooled_summaries_are_discovered_and_paths_stay_inside_outputs(tmp_path: Path) -> None:
    directory = tmp_path / "outputs" / "study" / "_summary"
    directory.mkdir(parents=True)
    (directory / "aggregation_validation.json").write_text(json.dumps({
        "status": "valid", "run_count": 6, "included_run_count": 6, "metric_row_count": 12,
        "checks": {"completeness": "pass"}, "missing_rows": [], "aggregation_digest": "abc",
    }))
    (directory / "summary.json").write_text(json.dumps({"summary": [
        {"arm": arm, "metric": metric, "center": 0.1, "spread": 0.01, "n": 3,
         "unit": "binary_option", "prediction_policy": "raw", "missing_count": 0, "estimand": "pooled_run"}
        for arm in ("a", "b") for metric in ("mse", "emd")
    ]}))
    (directory / "aggregation.resolved.yaml").write_text(
        "aggregate:\n  group_by: [arm]\n  statistics: {center: mean, spread: sample_std}\n")

    state = StudioState(tmp_path)
    listing = state.list_aggregates()
    detail = state.aggregate_detail("outputs/study/_summary")

    assert state.workspace()["counts"]["aggregates"] == 1
    assert listing["items"][0]["name"] == "study/_summary"
    assert listing["items"][0]["groupKeys"] == ["arm"]
    assert listing["items"][0]["groupCount"] == 2
    assert len(detail["rows"]) == 4
    assert detail["checks"] == {"completeness": "pass"}
    with pytest.raises(ValueError, match="inside outputs"):
        state.aggregate_detail("../outside")


def test_backend_capabilities_come_from_adapter_constructors() -> None:
    backends = {item["id"]: item for item in StudioState(ROOT).capabilities()["backends"]}
    openai = {option["name"]: option for option in backends["openai"]["options"]}
    ollama = {option["name"] for option in backends["ollama"]["options"]}

    assert backends["openai"]["modelRequired"] is True
    assert backends["gemini"]["modelRequired"] is True
    assert backends["ollama"]["modelRequired"] is False
    assert openai["transport"]["default"] == "chat_completions"
    assert openai["store"]["default"] is False
    assert not {"client", "api_key"} & set(openai)
    assert "transport" not in ollama  # an injected HTTP hook, not a setting
    assert "config_factory" not in {option["name"] for option in backends["gemini"]["options"]}


@pytest.mark.parametrize("profile", ["openai", "gemini", "ollama", "deepseek", "dummy", "simulated_demo"])
def test_shipped_backend_profiles_are_local_checks_only(profile: str) -> None:
    result = StudioState(ROOT).check_backend({"path": f"config/backends/{profile}.yaml"})
    assert result["networkCallMade"] is False
    assert next(check for check in result["checks"] if check["label"] == "Profile")["ok"]


def _regress(state: StudioState, root: Path, name: str, config: dict):
    """Preview, save and run a builder manifest as page.tsx does."""

    preview = state.preview_manifest(
        {"command": "regress", "config": config, "output": f"outputs/studio/{name}"}
    )
    assert preview["valid"], preview["warnings"]
    created = state.create_manifest({"command": "regress", "config": config})
    run_dir = root / "outputs" / "studio" / name
    regress_command.run(Config.load(root / created["path"]), run_dir=run_dir, solver="CLARABEL")
    return preview, created, run_dir


def _uniform_reference(workspace: Path) -> str:
    targets = json.loads((workspace / "examples/studio_demo/targets.json").read_text())
    reference = {
        qid: {label: 1 / len(shares) for label in shares} for qid, shares in targets.items()
    }
    path = workspace / "examples" / "studio_demo" / "reference.json"
    path.write_text(json.dumps(reference))
    return "examples/studio_demo/reference.json"


def test_missing_parent_package_reads_as_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    from p2p import studio

    def find_spec(name: str):
        raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")

    monkeypatch.setattr(studio.importlib.util, "find_spec", find_spec)
    assert studio._module_installed("google.genai") is False
    backends = {item["id"]: item for item in StudioState(ROOT).capabilities()["backends"]}
    assert set(backends) >= {"gemini", "openai", "ollama"}


def test_ridge_is_offered_as_elastic_net_with_l1_ratio_zero(workspace: Path) -> None:
    state = StudioState(workspace)
    ridge = next(model for model in state.capabilities()["models"] if model["id"] == "ridge")
    assert ridge["guided"] is True
    assert ridge["method"] == "elasticnet"
    assert ridge["options"] == {"l1_ratio": [0.0]}
    model = {
        "method": ridge["method"], "alpha": {"logspace": [-3, 1, 5]},
        **ridge["options"], "zero_threshold": ridge["zeroThreshold"],
    }
    validate_model_options(model)
    _, created, run_dir = _regress(
        state, workspace, "ridge", _builder_config("ridge", model, ["mse", "one_minus_jsd"])
    )
    saved = yaml.safe_load((workspace / created["path"]).read_text())
    assert saved["model"]["l1_ratio"] == [0.0]
    assert (run_dir / "metrics.json").is_file()


def test_reference_distribution_refit_runs_from_the_builder(workspace: Path) -> None:
    state = StudioState(workspace)
    reference = _uniform_reference(workspace)
    config = _builder_config(
        "reanchored",
        {"method": "lasso", "alpha": {"logspace": [-3, 1, 5]}, "soft_group_sum": False},
        ["mse", "one_minus_jsd"],
    )
    config["data"]["reference_distribution"] = reference
    _, created, run_dir = _regress(state, workspace, "reanchored", config)

    saved = yaml.safe_load((workspace / created["path"]).read_text())
    assert saved["data"]["reference_distribution"] == "../../../examples/studio_demo/reference.json"
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text())
    assert diagnostics["model"]["reference_distribution_digest"]
    detail = state.manifest_detail(created["path"])
    assert detail["config"]["data"]["reference_distribution"] == reference


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ({"method": "lasso", "reference_respondent": True}, "model.reference_respondent"),
        ({"method": "lasso", "intercept": True}, "model.intercept"),
        ({"method": "simple_average"}, "fits no weights"),
    ],
)
def test_reference_distribution_refuses_what_the_fit_refuses(
    workspace: Path, model: dict, message: str
) -> None:
    state = StudioState(workspace)
    config = _builder_config("refused", model, ["mse"])
    config["data"]["reference_distribution"] = _uniform_reference(workspace)
    with pytest.raises(ValueError, match=message):
        state.preview_manifest({"command": "regress", "config": config, "output": "outputs/studio/refused"})
    with pytest.raises(ValueError, match=message):
        state.create_manifest({"command": "regress", "config": config})


def test_grouped_target_subgroup_is_listed_checked_and_run(workspace: Path) -> None:
    state = StudioState(workspace)
    targets = json.loads((workspace / "examples/studio_demo/targets.json").read_text())
    grouped = {"schema_version": 1, "subgroups": {"north": targets, "south": targets}}
    (workspace / "examples/studio_demo/grouped.json").write_text(json.dumps(grouped))
    model = {"method": "lasso", "alpha": {"logspace": [-3, 1, 5]}}

    config = _builder_config("grouped", model, ["mse"])
    config["data"]["target"] = "examples/studio_demo/grouped.json"
    with pytest.raises(ValueError, match="north, south"):
        state.preview_manifest({"command": "regress", "config": config, "output": "outputs/studio/grouped"})

    config["data"]["target"] = {"path": "examples/studio_demo/grouped.json", "subgroup": "west"}
    with pytest.raises(ValueError, match="unknown target subgroup 'west'"):
        state.preview_manifest({"command": "regress", "config": config, "output": "outputs/studio/grouped"})

    config["data"]["target"] = {"path": "examples/studio_demo/grouped.json", "subgroup": "south"}
    preview, created, run_dir = _regress(state, workspace, "grouped", config)
    assert preview["targetSubgroups"] == {"subgroups": ["north", "south"], "default": None}
    saved = yaml.safe_load((workspace / created["path"]).read_text())
    assert saved["data"]["target"] == {
        "path": "../../../examples/studio_demo/grouped.json", "subgroup": "south",
    }
    assert (run_dir / "metrics.json").is_file()
    detail = state.manifest_detail(created["path"])
    assert detail["config"]["data"]["target"]["path"] == "examples/studio_demo/grouped.json"


def test_evaluation_policies_offered_are_ones_the_evaluator_runs(workspace: Path) -> None:
    from p2p.stages.evaluate import _SUPPORTED_DISTRIBUTION_POLICIES

    state = StudioState(workspace)
    evaluation = state.capabilities()["evaluation"]
    policies = [item["id"] for item in evaluation["distributionPolicies"]]
    assert set(policies) <= _SUPPORTED_DISTRIBUTION_POLICIES
    assert policies[0] == state.capabilities()["defaults"]["distributionPredictionPolicy"]
    assert [item["id"] for item in evaluation["invalidDistribution"]] == ["error", "allow"]

    config = _builder_config(
        "projected", {"method": "lasso", "alpha": {"logspace": [-3, 1, 5]}}, ["mse", "emd"]
    )
    config["evaluation"]["distribution_prediction_policy"] = "project_to_simplex"
    config["evaluation"]["invalid_distribution"] = "allow"
    _, _, run_dir = _regress(state, workspace, "projected", config)
    metrics = json.loads((run_dir / "metrics.json").read_text())["metrics"]
    emd = next(item for item in metrics if item["metric"] == "emd" and item["split"] == "test")
    assert emd["prediction_policy"] == "project_to_simplex"
