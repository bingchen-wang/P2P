"""Execution contracts for simulate, ablate, and panel sweep commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from p2p.commands import ablate, panel, simulate
from p2p.config import (
    AggregateRun,
    AggregateSpec,
    Config,
)
from p2p.registry import Registry
from p2p.simulations import runner


def regress_manifest(tmp_path: Path) -> dict[str, Any]:
    """Return a path-valid regression manifest for command composition."""

    survey_source = tmp_path / "survey.csv"
    survey_schema = tmp_path / "survey.yaml"
    target = tmp_path / "target.json"
    basis = tmp_path / "basis"
    basis.mkdir()
    for path in (survey_source, survey_schema, target):
        path.write_text("", encoding="utf-8")

    return {
        "run": {
            "name": "sweep-command",
            "seed": 200,
        },
        "data": {
            "survey": {
                "source": survey_source,
                "schema": survey_schema,
            },
            "basis": basis,
            "target": target,
        },
        "split": {
            "train_val": ["train", "valid"],
            "test": "test",
        },
        "model": {
            "method": "lasso",
            "alpha": [0.5],
        },
        "metrics": ["mse"],
    }


def recording_registry(
    calls: list[dict[str, Any]],
) -> Registry:
    registry = Registry("test sweep commands")

    def target(
        config: Config,
        *,
        solver: str,
        run_dir: Path,
        verbose: bool,
    ) -> dict[str, Any]:
        call = {
            "config": config,
            "solver": solver,
            "run_dir": run_dir,
            "verbose": verbose,
        }
        calls.append(call)
        (run_dir / "target.txt").write_text(
            config.digest + "\n",
            encoding="utf-8",
        )
        return call

    registry.register("regress", target)
    return registry


def test_ablate_executes_repeat_major_children_with_snapshots(
    tmp_path: Path,
) -> None:
    manifest = regress_manifest(tmp_path)
    manifest["ablate"] = {
        "target": "regress",
        "over": "model.alpha",
        "values": [0.01, 0.1],
        "repeats": 2,
    }
    config = Config.from_mapping(
        manifest,
        declared_in=tmp_path / "ablate.yaml",
    )
    calls: list[dict[str, Any]] = []

    result = ablate.run(
        config,
        run_dir=tmp_path / "output",
        solver="CLARABEL",
        verbose=True,
        registry=recording_registry(calls),
    )

    assert result.kind == "ablate"
    assert result.target_verb == "regress"
    assert result.aggregation is None
    assert result.aggregation_artifacts is None
    assert [child.run_dir.name for child in result.children] == [
        "repeat-0001-case-0001",
        "repeat-0001-case-0002",
        "repeat-0002-case-0001",
        "repeat-0002-case-0002",
    ]
    assert [
        child.config.data["run"]["repeat"]
        for child in result.children
    ] == [1, 1, 2, 2]
    assert [
        child.config.data["run"]["seed"]
        for child in result.children
    ] == [200, 200, 201, 201]
    assert [
        child.config.data["model"]["alpha"]
        for child in result.children
    ] == [
        pytest.approx((0.01,)),
        pytest.approx((0.1,)),
        pytest.approx((0.01,)),
        pytest.approx((0.1,)),
    ]
    assert all(
        child.snapshot_path
        == child.run_dir / "config.resolved.yaml"
        for child in result.children
    )
    assert all(
        child.snapshot_path.is_file()
        and (child.run_dir / "target.txt").is_file()
        for child in result.children
    )
    assert len(calls) == 4
    assert all(call["solver"] == "CLARABEL" for call in calls)
    assert all(call["verbose"] is True for call in calls)


def test_panel_executes_explicit_cases_through_injected_registry(
    tmp_path: Path,
) -> None:
    manifest = regress_manifest(tmp_path)
    manifest["panel"] = {
        "target": "regress",
        "overrides": [
            {"run": {"wave": "W1"}},
            {"run": {"wave": "W2"}},
        ],
    }
    config = Config.from_mapping(
        manifest,
        declared_in=tmp_path / "panel.yaml",
    )
    calls: list[dict[str, Any]] = []

    result = panel.run(
        config,
        run_dir=tmp_path / "panel-output",
        registry=recording_registry(calls),
    )

    assert result.kind == "panel"
    assert [
        child.config.data["run"]["wave"]
        for child in result.children
    ] == ["W1", "W2"]
    assert [child.case for child in result.children] == [1, 2]
    assert len(calls) == 2


def test_simulate_default_runner_records_settings_not_fake_results(
    tmp_path: Path,
) -> None:
    config = Config.from_mapping(
        {
            "run": {
                "name": "offline-simulation",
                "seed": 10,
            },
            "simulation": {
                "kind": "fraction",
                "invokes_generation": False,
                "sweep": {
                    "target": "simulate",
                    "overrides": [
                        {
                            "simulation": {
                                "fraction": 0.25,
                            }
                        },
                        {
                            "simulation": {
                                "fraction": 0.5,
                            }
                        },
                    ],
                },
            },
        },
        declared_in=tmp_path / "simulate.yaml",
    )

    result = simulate.run(
        config,
        run_dir=tmp_path / "simulation-output",
    )

    assert result.kind == "simulate"
    assert result.target_verb == "simulate"
    assert len(result.children) == 2
    assert [
        child.result.settings["fraction"]
        for child in result.children
    ] == [0.25, 0.5]

    for child in result.children:
        artifact = json.loads(
            child.result.artifact_path.read_text(
                encoding="utf-8"
            )
        )
        assert artifact["artifact_type"] == (
            "declared_simulation_settings"
        )
        assert "configuration record only" in artifact["boundary"]
        assert "metrics" not in artifact
        assert "results" not in artifact
        assert child.snapshot_path.is_file()


def test_offline_simulation_runner_rejects_non_scalar_settings(
    tmp_path: Path,
) -> None:
    config = Config.from_mapping(
        {
            "simulation": {
                "kind": "fraction",
                "invokes_generation": False,
                "schedule": [0.25, 0.5],
            },
        },
        declared_in=tmp_path / "nested-simulation.yaml",
    )

    with pytest.raises(ValueError, match="JSON scalars"):
        runner.run_declared_simulation(
            config,
            run_dir=tmp_path / "nested-output",
        )


def test_requested_aggregation_uses_shared_aggregation_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = regress_manifest(tmp_path)
    manifest["ablate"] = {
        "target": "regress",
        "over": "model.alpha",
        "values": [0.01, 0.1],
    }
    config = Config.from_mapping(
        manifest,
        declared_in=tmp_path / "aggregate-ablate.yaml",
    )
    calls: list[dict[str, Any]] = []
    registry = recording_registry(calls)

    placeholders = []
    for name in ("first", "second"):
        path = tmp_path / name
        path.mkdir()
        placeholders.append(
            AggregateRun(path=path, labels={})
        )
    aggregation_spec = AggregateSpec(
        runs=tuple(placeholders),
        group_by=("model",),
        metrics=("mse",),
        expected_repeats=None,
        expected_cells=(),
        require_complete=True,
        center="mean",
        spread="sample_std",
        hierarchy_reduce=(),
        hierarchy_weighting="pooled",
        missing_policy="error",
        missing_reason=None,
        export_formats=("csv", "json"),
        export_precision=4,
    )

    aggregate_result = object()
    aggregate_artifacts = object()
    shared_calls: dict[str, Any] = {}

    def fake_aggregate(spec):
        shared_calls["spec"] = spec
        return aggregate_result

    def fake_write(result, spec, **kwargs):
        shared_calls["write"] = {
            "result": result,
            "spec": spec,
            **kwargs,
        }
        return aggregate_artifacts

    monkeypatch.setattr(runner, "aggregate_runs", fake_aggregate)
    monkeypatch.setattr(
        runner,
        "write_aggregation_artifacts",
        fake_write,
    )

    result = ablate.run(
        config,
        run_dir=tmp_path / "aggregate-output",
        registry=registry,
        aggregation_spec=aggregation_spec,
    )

    assert result.aggregation is aggregate_result
    assert result.aggregation_artifacts is aggregate_artifacts
    delegated_spec = shared_calls["spec"]
    assert [run.path for run in delegated_spec.runs] == [
        child.run_dir
        for child in result.children
    ]
    assert shared_calls["write"]["result"] is aggregate_result
    assert shared_calls["write"]["spec"] is delegated_spec
    assert shared_calls["write"]["output_dir"] == (
        result.run_dir / "aggregation"
    )
