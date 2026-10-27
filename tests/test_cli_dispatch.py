"""CLI parsing and registered-command dispatch contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from p2p.models import DEFAULT_SOLVER
from p2p import cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_initial_cli_commands_are_registered() -> None:
    assert {
        "generate",
        "elicit",
        "regress",
        "pipeline",
        "evaluate",
        "aggregate",
        "budget",
        "simulate",
        "ablate",
        "panel",
        "cluster",
        "validate",
    } <= set(cli.COMMANDS.names())


def test_cli_dispatches_regress_through_registry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "regress.yaml"
    loaded_config = object()
    calls: dict[str, Any] = {}

    def fake_load(path):
        calls["config_path"] = path
        return loaded_config

    def handler(config, **kwargs):
        calls["handler"] = {
            "config": config,
            **kwargs,
        }
        return object()

    monkeypatch.setattr(
        cli.Config,
        "load",
        staticmethod(fake_load),
    )
    monkeypatch.setattr(
        cli.COMMANDS,
        "get",
        lambda name: (
            handler
            if name == "regress"
            else None
        ),
    )

    exit_code = cli.main(
        [
            "regress",
            "--config",
            str(config_path),
            "--solver",
            "OSQP",
            "--output",
            str(tmp_path / "run"),
            "--verbose",
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == config_path
    assert calls["handler"] == {
        "config": loaded_config,
        "solver": "OSQP",
        "run_dir": tmp_path / "run",
        "verbose": True,
    }


def test_cli_dispatches_pipeline_with_run_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "pipeline.yaml"
    run_dir = tmp_path / "run"
    loaded_config = object()
    calls: dict[str, Any] = {}

    monkeypatch.setattr(
        cli.Config,
        "load",
        staticmethod(
            lambda path: (
                calls.setdefault("config_path", path),
                loaded_config,
            )[1]
        ),
    )

    def handler(config, **kwargs):
        calls["handler"] = {
            "config": config,
            **kwargs,
        }
        return object()

    monkeypatch.setattr(
        cli.COMMANDS,
        "get",
        lambda name: (
            handler
            if name == "pipeline"
            else None
        ),
    )

    exit_code = cli.main(
        [
            "pipeline",
            "--config",
            str(config_path),
            "--output",
            str(run_dir),
        ]
    )

    assert exit_code == 0
    assert calls["config_path"] == config_path
    assert calls["handler"] == {
        "config": loaded_config,
        "solver": DEFAULT_SOLVER,
        "verbose": False,
        "run_dir": run_dir,
    }


def test_cli_dispatches_budget_forecast(
    tmp_path: Path,
    monkeypatch,
) -> None:
    loaded_config = object()
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        cli.Config,
        "load",
        staticmethod(lambda path: loaded_config),
    )

    def handler(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = kwargs

    monkeypatch.setattr(cli.COMMANDS, "get", lambda name: handler)

    assert cli.main(
        [
            "budget",
            "--config",
            "fixture.yaml",
            "--history",
            "outputs/paper",
            "--pilot-run",
            "pilot",
            "--compare-model",
            "alternative",
            "--budget-usd",
            "2.5",
            "--output",
            str(tmp_path / "budget"),
        ]
    ) == 0
    assert calls["config"] is loaded_config
    assert calls["kwargs"] == {
        "output_dir": tmp_path / "budget",
        "history": [Path("outputs/paper")],
        "target_command": "pipeline",
        "comparison_models": ["alternative"],
        "pricing_path": None,
        "pilot_run": Path("pilot"),
        "confidence": 0.8,
        "budget_usd": 2.5,
        "serve": False,
        "host": "127.0.0.1",
        "port": 8060,
        "refresh_seconds": 2.0,
        "verbose": False,
    }


def test_cli_rejects_unknown_command() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["unknown"])

    assert error.value.code == 2


def test_cli_dispatches_evaluate_without_solver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "evaluate.yaml"
    run_dir = tmp_path / "run"
    loaded_config = object()
    calls: dict[str, Any] = {}

    monkeypatch.setattr(
        cli.Config,
        "load",
        staticmethod(lambda path: loaded_config),
    )

    def handler(config, **kwargs):
        calls["handler"] = {
            "config": config,
            **kwargs,
        }

    monkeypatch.setattr(
        cli.COMMANDS,
        "get",
        lambda name: handler,
    )

    assert cli.main(
        [
            "evaluate",
            "--config",
            str(config_path),
            "--output",
            str(run_dir),
        ]
    ) == 0
    assert calls["handler"] == {
        "config": loaded_config,
        "run_dir": run_dir,
        "verbose": False,
    }


@pytest.mark.parametrize(
    ("command", "extra_arguments", "expected_kwargs"),
    [
        (
            "generate",
            ["--output", "run"],
            {
                "run_dir": Path("run"),
                "verbose": False,
            },
        ),
        (
            "elicit",
            [
                "--endowments",
                "endowments.csv",
                "--output",
                "run",
            ],
            {
                "endowments_path": Path("endowments.csv"),
                "run_dir": Path("run"),
                "verbose": False,
            },
        ),
        (
            "simulate",
            ["--output", "run"],
            {
                "run_dir": Path("run"),
                "solver": DEFAULT_SOLVER,
                "verbose": False,
            },
        ),
        (
            "ablate",
            ["--output", "run"],
            {
                "run_dir": Path("run"),
                "solver": DEFAULT_SOLVER,
                "verbose": False,
            },
        ),
        (
            "panel",
            ["--output", "run"],
            {
                "run_dir": Path("run"),
                "solver": DEFAULT_SOLVER,
                "verbose": False,
            },
        ),
        (
            "cluster",
            ["--output", "run"],
            {
                "run_dir": Path("run"),
                "verbose": False,
            },
        ),
        (
            "validate",
            ["--target-verb", "regress"],
            {
                "target_verb": "regress",
                "verbose": False,
            },
        ),
    ],
)
def test_cli_dispatches_sweep_cluster_and_validate_commands(
    monkeypatch,
    command: str,
    extra_arguments: list[str],
    expected_kwargs: dict[str, Any],
) -> None:
    loaded_config = object()
    calls: dict[str, Any] = {}

    monkeypatch.setattr(
        cli.Config,
        "load",
        staticmethod(lambda path: loaded_config),
    )

    def handler(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = kwargs

    monkeypatch.setattr(
        cli.COMMANDS,
        "get",
        lambda name: handler,
    )

    assert cli.main(
        [
            command,
            "--config",
            "fixture.yaml",
            *extra_arguments,
        ]
    ) == 0
    assert calls == {
        "config": loaded_config,
        "kwargs": expected_kwargs,
    }


@pytest.mark.parametrize(
    ("command", "manifest_name"),
    [
        ("regress", "regress_fixture.yaml"),
        ("evaluate", "evaluate_fixture.yaml"),
    ],
)
def test_cli_offline_artifact_commands_emit_canonical_bundle(
    tmp_path: Path,
    cvxpy_solver: str,
    command: str,
    manifest_name: str,
) -> None:
    run_dir = tmp_path / command
    arguments = [
        command,
        "--config",
        str(
            PROJECT_ROOT
            / "config"
            / "central"
            / manifest_name
        ),
        "--output",
        str(run_dir),
    ]
    if command == "regress":
        arguments.extend(
            ["--solver", cvxpy_solver]
        )

    assert cli.main(arguments) == 0
    expected = {
        "run.json",
        "metrics.json",
        "metrics.csv",
        "config.resolved.yaml",
    }
    if command == "regress":
        expected.add("diagnostics.json")
        # Every regression records its cross-validation folds (S.2).
        expected.add("folds.json")
    assert {
            path.name
            for path in run_dir.iterdir()
        } == expected
