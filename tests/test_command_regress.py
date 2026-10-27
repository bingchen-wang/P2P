"""Production regression-command composition contracts."""

from pathlib import Path
from typing import Any

import pytest

from p2p.commands import regress
from p2p.config import Config


def test_run_regression_composes_fit_and_evaluation(
    monkeypatch,
) -> None:
    data = object()
    fitted = object()
    evaluated = object()
    fold_assignments = {
        "Q1": 0,
        "Q2": 1,
    }
    calls: dict[str, Any] = {}

    def fake_fit_model(
        received_data,
        model_config,
        **kwargs,
    ):
        calls["fit"] = {
            "data": received_data,
            "model_config": dict(model_config),
            **kwargs,
        }
        return fitted

    def fake_evaluate_model(
        received_data,
        fit_result,
        **kwargs,
    ):
        calls["evaluate"] = {
            "data": received_data,
            "fit_result": fit_result,
            **kwargs,
        }
        return evaluated

    monkeypatch.setattr(
        regress,
        "fit_model",
        fake_fit_model,
    )
    monkeypatch.setattr(
        regress,
        "evaluate_model",
        fake_evaluate_model,
    )

    result = regress.run_regression(
        data,
        {
            "method": "lasso",
            "alpha": (0.01,),
        },
        metrics=("mse", "one_minus_jsd"),
        solver="OSQP",
        seed=17,
        test_split="test",
        distribution_prediction_policy=(
            "clip_and_normalize"
        ),
        verbose=True,
        fold_assignments=fold_assignments,
    )

    assert isinstance(result, regress.RegressionResult)
    assert result.data is data
    assert result.fit is fitted
    assert result.evaluation is evaluated
    assert result.config_digest is None

    assert calls["fit"] == {
        "data": data,
        "model_config": {
            "method": "lasso",
            "alpha": (0.01,),
        },
        "solver": "OSQP",
        "seed": 17,
        "verbose": True,
        "fold_assignments": fold_assignments,
    }
    assert calls["evaluate"] == {
        "data": data,
        "fit_result": fitted,
        "split": "test",
        "metrics": ("mse", "one_minus_jsd"),
        "distribution_prediction_policy": (
            "clip_and_normalize"
        ),
        "invalid_distribution": "allow",
        "distribution_route": "model",
    }


def test_run_resolves_central_regress_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    survey_source = tmp_path / "survey.csv"
    survey_schema = tmp_path / "survey.yaml"
    target_path = tmp_path / "target.json"
    basis_path = tmp_path / "basis"
    responses_path = basis_path / "responses.csv"
    endowments_path = basis_path / "endowments.csv"

    basis_path.mkdir()
    for path in (
        survey_source,
        survey_schema,
        target_path,
        responses_path,
        endowments_path,
    ):
        path.write_text("", encoding="utf-8")

    config = Config.from_mapping(
        {
            "run": {
                "name": "regress_contract",
                "seed": 23,
            },
            "data": {
                "survey": {
                    "source": survey_source,
                    "schema": survey_schema,
                },
                "basis": basis_path,
                "target": target_path,
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
                "use_proxy_only": False,
            },
            "metrics": [
                "mse",
                "one_minus_jsd",
            ],
            "evaluation": {
                "distribution_prediction_policy": (
                    "clip_and_normalize"
                ),
            },
        },
        declared_in=tmp_path / "regress.yaml",
    )

    loaded_data = object()
    command_result = object()
    calls: dict[str, Any] = {}

    def fake_load_regression_data(**kwargs):
        calls["load"] = kwargs
        return loaded_data

    def fake_run_regression(
        received_data,
        model_config,
        **kwargs,
    ):
        calls["run_regression"] = {
            "data": received_data,
            "model_config": dict(model_config),
            **kwargs,
        }
        return command_result

    monkeypatch.setattr(
        regress,
        "load_regression_data",
        fake_load_regression_data,
    )
    monkeypatch.setattr(
        regress,
        "run_regression",
        fake_run_regression,
    )

    result = regress.run(
        config,
        solver="OSQP",
        verbose=True,
    )

    assert result is command_result
    assert calls["load"] == {
        "survey_source": survey_source.resolve(),
        "survey_schema": survey_schema.resolve(),
        "responses_path": responses_path.resolve(),
        "endowments_path": endowments_path.resolve(),
            "target_path": target_path.resolve(),
            "target_subgroup": None,
            "train_splits": ("train", "valid"),
        "test_split": "test",
        "use_proxy_only": False,
        "reference_respondent": False,
        "renormalize_targets": False,
        "canonical_answers": True,
        "reference_distribution_path": None,
    }
    assert calls["run_regression"] == {
        "data": loaded_data,
        "model_config": {
            "method": "lasso",
            "alpha": (0.01,),
            "validation": {
                "strategy": "cv",
                "cv_folds": 2,
            },
        },
        "metrics": ("mse", "one_minus_jsd"),
        "solver": "OSQP",
        "seed": 23,
        "test_split": "test",
        "distribution_prediction_policy": (
            "clip_and_normalize"
        ),
        "invalid_distribution": "error",
        "verbose": True,
        "config_digest": config.digest,
        "distribution_route": "model",
    }


def test_run_with_output_emits_canonical_report(
    tmp_path: Path,
    cvxpy_solver: str,
) -> None:
    config = Config.load(
        Path(__file__).resolve().parents[1]
        / "config"
        / "central"
        / "regress_fixture.yaml"
    )
    result = regress.run(
        config,
        solver=cvxpy_solver,
        run_dir=tmp_path / "regress-run",
    )

    assert result.run_dir == (
        tmp_path / "regress-run"
    ).resolve()
    assert result.report is not None
    assert result.report.run_json.is_file()
    assert result.report.metrics_json.is_file()
    assert result.report.metrics_csv.is_file()
    assert result.report.config_snapshot.is_file()


def test_reference_respondent_reaches_the_loader(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """model.reference_respondent must travel from config to the split stage."""

    survey_source = tmp_path / "survey.csv"
    survey_schema = tmp_path / "survey.yaml"
    target_path = tmp_path / "target.json"
    basis_path = tmp_path / "basis"
    responses_path = basis_path / "responses.csv"
    endowments_path = basis_path / "endowments.csv"

    basis_path.mkdir()
    for path in (
        survey_source,
        survey_schema,
        target_path,
        responses_path,
        endowments_path,
    ):
        path.write_text("", encoding="utf-8")

    def build(reference_respondent):
        model: dict = {
            "method": "lasso",
            "alpha": [0.01],
            "validation": {"strategy": "cv", "cv_folds": 2},
        }
        if reference_respondent is not None:
            model["reference_respondent"] = reference_respondent
        return Config.from_mapping(
            {
                "run": {"name": "reference_contract", "seed": 23},
                "data": {
                    "survey": {
                        "source": survey_source,
                        "schema": survey_schema,
                    },
                    "basis": basis_path,
                    "target": target_path,
                },
                "model": model,
                "split": {
                    "train_val": ["train", "valid"],
                    "test": "test",
                    "use_proxy_only": False,
                },
                "metrics": ["mse"],
            },
            declared_in=tmp_path / "regress.yaml",
        )

    seen: dict = {}

    def fake_load_regression_data(**kwargs):
        seen["reference_respondent"] = kwargs["reference_respondent"]
        return object()

    monkeypatch.setattr(
        regress,
        "load_regression_data",
        fake_load_regression_data,
    )
    monkeypatch.setattr(
        regress,
        "run_regression",
        lambda *args, **kwargs: object(),
    )

    for configured, expected in ((None, False), (False, False), (True, True)):
        regress.run(build(configured), solver="OSQP", verbose=False)
        assert seen["reference_respondent"] is expected

    with pytest.raises(ValueError, match="must be boolean"):
        regress.run(build("yes"), solver="OSQP", verbose=False)


def test_target_renormalize_reaches_the_loader(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """data.target.renormalize must travel from config to the split stage."""

    survey_source = tmp_path / "survey.csv"
    survey_schema = tmp_path / "survey.yaml"
    target_path = tmp_path / "target.json"
    basis_path = tmp_path / "basis"
    responses_path = basis_path / "responses.csv"
    endowments_path = basis_path / "endowments.csv"

    basis_path.mkdir()
    for path in (
        survey_source,
        survey_schema,
        target_path,
        responses_path,
        endowments_path,
    ):
        path.write_text("", encoding="utf-8")

    def build(target):
        return Config.from_mapping(
            {
                "run": {"name": "renormalize_contract", "seed": 23},
                "data": {
                    "survey": {
                        "source": survey_source,
                        "schema": survey_schema,
                    },
                    "basis": basis_path,
                    "target": target,
                },
                "model": {
                    "method": "lasso",
                    "alpha": [0.01],
                    "validation": {"strategy": "cv", "cv_folds": 2},
                },
                "split": {
                    "train_val": ["train", "valid"],
                    "test": "test",
                    "use_proxy_only": False,
                },
                "metrics": ["mse"],
            },
            declared_in=tmp_path / "regress.yaml",
        )

    seen: dict = {}

    def fake_load_regression_data(**kwargs):
        seen["renormalize_targets"] = kwargs["renormalize_targets"]
        return object()

    monkeypatch.setattr(
        regress,
        "load_regression_data",
        fake_load_regression_data,
    )
    monkeypatch.setattr(
        regress,
        "run_regression",
        lambda *args, **kwargs: object(),
    )

    cases = (
        (target_path, False),                                  # scalar entry
        ({"path": target_path}, False),                        # mapping, silent
        ({"path": target_path, "renormalize": False}, False),
        ({"path": target_path, "renormalize": True}, True),
    )
    for configured, expected in cases:
        regress.run(build(configured), solver="OSQP", verbose=False)
        assert seen["renormalize_targets"] is expected

    with pytest.raises(ValueError, match="must be boolean"):
        regress.run(
            build({"path": target_path, "renormalize": "yes"}),
            solver="OSQP",
            verbose=False,
        )


def test_diagnostics_artifact_carries_stage_two_summary(
    tmp_path: Path,
    cvxpy_solver: str,
) -> None:
    """The fit's own diagnostics must reach disk, not just live in memory."""

    import json

    import yaml

    source = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "central"
        / "regress_fixture.yaml"
    )

    def run(extra: dict, run_dir: Path) -> dict:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        data["model"].update(extra)
        result = regress.run(
            Config.from_mapping(data, declared_in=source.resolve()),
            solver=cvxpy_solver,
            run_dir=run_dir,
        )
        payload = json.loads(
            result.report.diagnostics_json.read_text(encoding="utf-8")
        )
        return payload["model"]

    default = run({}, tmp_path / "default")
    assert default["anti_type_mass"] == 0.0
    assert default["anti_type_count"] == 0
    assert default["coefficient_sum"] == pytest.approx(1.0, abs=1e-3)
    assert "reference_respondent_weight" not in default
    # S.6-F5: which solver produced the fit, and any fallbacks, are recorded.
    assert isinstance(default["solver_used"], str)
    assert default["solver_fallback"] is False
    assert default["cv_fits"] >= 1
    assert default["cv_solver_fallbacks"] == 0
    # S.6-F6: the fixture config carries the legacy no-op `max_iter`.
    assert default["ignored_model_options"] == "max_iter"

    enabled = run({"reference_respondent": True}, tmp_path / "enabled")
    assert 0.0 <= enabled["reference_respondent_weight"] <= 1.0


def test_misspelled_model_option_fails_before_loading_the_basis(
    tmp_path: Path,
    cvxpy_solver: str,
) -> None:
    import yaml

    source = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "central"
        / "regress_fixture.yaml"
    )
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["model"]["reference_respondant"] = True

    with pytest.raises(ValueError, match="did you mean 'reference_respondent'"):
        regress.run(
            Config.from_mapping(data, declared_in=source.resolve()),
            solver=cvxpy_solver,
            run_dir=tmp_path / "never",
        )
