"""Contracts for aggregation configuration schema validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from p2p.config import (
    AggregateSpec,
    Config,
    ConfigValidationError,
)


def write_yaml(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )
    return path


def make_run_dirs(tmp_path: Path) -> tuple[Path, Path]:
    first = tmp_path / "runs" / "r1"
    second = tmp_path / "runs" / "r2"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    return first, second


def base_aggregate(
    tmp_path: Path,
) -> tuple[dict[str, Any], Path, Path]:
    first, second = make_run_dirs(tmp_path)
    aggregate = {
        "runs": [
            {
                "path": "runs/r1",
                "labels": {
                    "method": "p2p",
                    "wave": "W42",
                    "model": "elasticnet",
                    "split": "test",
                    "repeat": 1,
                },
            },
            {
                "path": "runs/r2",
                "labels": {
                    "method": "p2p",
                    "wave": "W42",
                    "model": "elasticnet",
                    "split": "test",
                    "repeat": 2,
                },
            },
        ],
        "group_by": [
            "method",
            "wave",
            "model",
            "split",
        ],
        "metrics": [
            "mse",
            "one_minus_jsd",
        ],
        "expected": {
            "repeats": 2,
        },
    }
    return aggregate, first, second


def test_aggregate_spec_resolves_runs_and_applies_defaults(
    tmp_path: Path,
) -> None:
    aggregate, first, second = base_aggregate(tmp_path)
    manifest = write_yaml(
        tmp_path / "aggregate.yaml",
        {
            "run": {
                "name": "aggregate-test",
            },
            "aggregate": aggregate,
        },
    )

    config = Config.load(manifest)

    assert config.validate_for("aggregate") is config
    assert isinstance(config.aggregate_spec, AggregateSpec)

    spec = config.aggregate_spec
    assert spec is not None
    assert tuple(run.path for run in spec.runs) == (
        first.resolve(),
        second.resolve(),
    )
    assert spec.group_by == (
        "method",
        "wave",
        "model",
        "split",
    )
    assert spec.metrics == (
        "mse",
        "one_minus_jsd",
    )
    assert spec.expected_repeats == 2
    assert spec.expected_cells == ()
    assert spec.require_complete is True
    assert spec.center == "mean"
    assert spec.spread == "sample_std"
    assert spec.hierarchy_reduce == ()
    assert spec.hierarchy_weighting == "equal"
    assert spec.missing_policy == "error"
    assert spec.missing_reason is None
    assert spec.export_formats == (
        "csv",
        "json",
    )
    assert spec.export_precision == 4

    with pytest.raises(TypeError):
        spec.runs[0].labels["method"] = "changed"


def test_aggregate_spec_accepts_explicit_cells_hierarchy_and_exports(
    tmp_path: Path,
) -> None:
    aggregate, _, _ = base_aggregate(tmp_path)
    aggregate["expected"] = {
        "repeats": 2,
        "require_complete": True,
        "cells": [
            {
                "method": "p2p",
                "wave": "W42",
            },
        ],
    }
    aggregate["statistics"] = {
        "center": "mean",
        "spread": "population_std",
    }
    aggregate["hierarchy"] = {
        "reduce": [
            "repeat",
            "wave",
        ],
        "weighting": "equal",
    }
    aggregate["missing"] = {
        "policy": "allow",
    }
    aggregate["export"] = {
        "formats": [
            "csv",
            "markdown",
            "latex",
        ],
        "precision": 3,
    }

    manifest = write_yaml(
        tmp_path / "aggregate.yaml",
        {
            "aggregate": aggregate,
        },
    )

    spec = Config.load(
        manifest
    ).validate_for("aggregate").aggregate_spec

    assert spec is not None
    assert spec.expected_cells == (
        {
            "method": "p2p",
            "wave": "W42",
        },
    )
    assert spec.spread == "population_std"
    assert spec.hierarchy_reduce == (
        "repeat",
        "wave",
    )
    assert spec.missing_policy == "allow"
    assert spec.export_formats == (
        "csv",
        "markdown",
        "latex",
    )
    assert spec.export_precision == 3


def test_duplicate_run_paths_are_rejected(
    tmp_path: Path,
) -> None:
    aggregate, _, _ = base_aggregate(tmp_path)
    aggregate["runs"][1]["path"] = "runs/r1"
    manifest = write_yaml(
        tmp_path / "aggregate.yaml",
        {
            "aggregate": aggregate,
        },
    )

    with pytest.raises(
        ConfigValidationError,
        match="duplicate",
    ):
        Config.load(manifest)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("empty_runs", "runs"),
        ("empty_group_by", "group_by"),
        ("empty_metrics", "metrics"),
        ("invalid_repeats", "repeats"),
        ("exclude_without_reason", "reason"),
        ("ambiguous_spread", "spread"),
        ("invalid_export", "format"),
        ("duplicate_hierarchy", "hierarchy"),
    ],
)
def test_invalid_aggregate_schema_is_rejected(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    aggregate, _, _ = base_aggregate(tmp_path)
    aggregate = deepcopy(aggregate)

    if case == "empty_runs":
        aggregate["runs"] = []
    elif case == "empty_group_by":
        aggregate["group_by"] = []
    elif case == "empty_metrics":
        aggregate["metrics"] = []
    elif case == "invalid_repeats":
        aggregate["expected"]["repeats"] = 0
    elif case == "exclude_without_reason":
        aggregate["missing"] = {
            "policy": "exclude",
        }
    elif case == "ambiguous_spread":
        aggregate["statistics"] = {
            "center": "mean",
            "spread": "std",
        }
    elif case == "invalid_export":
        aggregate["export"] = {
            "formats": ["xlsx"],
        }
    elif case == "duplicate_hierarchy":
        aggregate["hierarchy"] = {
            "reduce": [
                "repeat",
                "repeat",
            ],
            "weighting": "equal",
        }
    else:
        raise AssertionError(f"unhandled case: {case}")

    manifest = write_yaml(
        tmp_path / "aggregate.yaml",
        {
            "aggregate": aggregate,
        },
    )

    with pytest.raises(
        ConfigValidationError,
        match=message,
    ):
        Config.load(manifest)