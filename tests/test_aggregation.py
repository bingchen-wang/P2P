"""Validated aggregation, hierarchy, and export contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import pytest
import yaml

from p2p import aggregation
from p2p.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "commands"
    / "aggregation"
)


def copied_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "aggregation"
    shutil.copytree(FIXTURE, destination)
    return destination


def load_result(root: Path):
    config = Config.load(root / "aggregate.yaml")
    spec = config.validate_for("aggregate").aggregate_spec
    assert spec is not None
    return config, spec, aggregation.aggregate_runs(spec)


def test_three_repeat_fixture_matches_frozen_full_precision_summary(
    tmp_path: Path,
) -> None:
    root = copied_fixture(tmp_path)
    config, spec, result = load_result(root)

    assert result.validation["status"] == "valid"
    assert result.validation["run_count"] == 3
    assert result.validation["metric_row_count"] == 6
    assert result.summary_rows == (
        {
            "method": "p2p",
            "wave": "W42",
            "model": "lasso",
            "split": "test",
            "metric": "mse",
            "unit": "binary_option",
            "prediction_policy": "raw",
            "center": 2.0,
            "spread": 1.0,
            "n": 3,
            "missing_count": 0,
            "estimand": "pooled_run",
        },
        {
            "method": "p2p",
            "wave": "W42",
            "model": "lasso",
            "split": "test",
            "metric": "one_minus_jsd",
            "unit": "question",
            "prediction_policy": "clip_and_normalize",
            "center": 0.9,
            "spread": pytest.approx(
                0.09999999999999998
            ),
            "n": 3,
            "missing_count": 0,
            "estimand": "pooled_run",
        },
    )

    artifacts = aggregation.write_aggregation_artifacts(
        result,
        spec,
        output_dir=tmp_path / "report",
        source_config=config.source_path,
    )
    assert {
        path.name
        for path in artifacts.required_files
    } == {
        "aggregation.resolved.yaml",
        "aggregation_inputs.csv",
        "aggregation_validation.json",
        "runs.csv",
        "summary.csv",
        "summary.json",
    }
    assert artifacts.markdown is not None
    assert artifacts.latex is not None

    assert artifacts.summary_csv.read_text(
        encoding="utf-8"
    ) == (
        root / "expected" / "summary.csv"
    ).read_text(encoding="utf-8")
    assert json.loads(
        artifacts.summary_json.read_text(
            encoding="utf-8"
        )
    ) == json.loads(
        (
            root
            / "expected"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )

    assert "0.9000" in artifacts.markdown.read_text(
        encoding="utf-8"
    )
    assert "0.9000" in artifacts.latex.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("duplicate_run_id", "duplicate run_id"),
        ("label_repeat", "asserted label"),
        ("label_model", "asserted label"),
        ("mixed_split", "split"),
        ("mixed_policy", "prediction policy"),
        ("mixed_cleaning", "cleaning"),
        ("artifact_run_id", "run_id"),
        ("missing_metric", "missing"),
        ("expected_repeats", "repeats"),
    ],
)
def test_invalid_aggregation_inputs_fail_before_reduction(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    root = copied_fixture(tmp_path)
    manifest_path = root / "aggregate.yaml"
    manifest = yaml.safe_load(
        manifest_path.read_text(encoding="utf-8")
    )

    if case == "duplicate_run_id":
        run = json.loads(
            (root / "r2" / "run.json").read_text()
        )
        run["run_id"] = "fixture-r0001"
        (root / "r2" / "run.json").write_text(
            json.dumps(run),
            encoding="utf-8",
        )
        metrics = json.loads(
            (root / "r2" / "metrics.json").read_text()
        )
        metrics["run_id"] = "fixture-r0001"
        (root / "r2" / "metrics.json").write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )
    elif case == "label_repeat":
        manifest["aggregate"]["runs"][1][
            "labels"
        ]["repeat"] = 1
    elif case == "label_model":
        manifest["aggregate"]["runs"][1][
            "labels"
        ]["model"] = "elasticnet"
    elif case == "mixed_split":
        metrics = json.loads(
            (root / "r2" / "metrics.json").read_text()
        )
        metrics["metrics"][0]["split"] = "valid"
        (root / "r2" / "metrics.json").write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )
    elif case == "mixed_policy":
        metrics = json.loads(
            (root / "r2" / "metrics.json").read_text()
        )
        metrics["metrics"][1][
            "prediction_policy"
        ] = "project_to_simplex"
        (root / "r2" / "metrics.json").write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )
    elif case == "mixed_cleaning":
        run = json.loads(
            (root / "r2" / "run.json").read_text()
        )
        run["cleaning_policy"] = "row_drop_v1"
        (root / "r2" / "run.json").write_text(
            json.dumps(run),
            encoding="utf-8",
        )
    elif case == "artifact_run_id":
        metrics = json.loads(
            (root / "r2" / "metrics.json").read_text()
        )
        metrics["run_id"] = "wrong"
        (root / "r2" / "metrics.json").write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )
    elif case == "missing_metric":
        metrics = json.loads(
            (root / "r2" / "metrics.json").read_text()
        )
        metrics["metrics"] = metrics["metrics"][:1]
        (root / "r2" / "metrics.json").write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )
    elif case == "expected_repeats":
        manifest["aggregate"]["expected"]["repeats"] = 4
    else:
        raise AssertionError(case)

    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )

    config = Config.load(manifest_path)
    spec = config.aggregate_spec
    assert spec is not None
    with pytest.raises(
        aggregation.AggregationValidationError,
        match=message,
    ):
        aggregation.aggregate_runs(spec)


@pytest.mark.parametrize(
    ("policy", "expected_n", "expected_missing"),
    [
        ("allow", 2, 1),
        ("exclude", 2, 1),
    ],
)
def test_explicit_missing_policies_are_recorded(
    tmp_path: Path,
    policy: str,
    expected_n: int,
    expected_missing: int,
) -> None:
    root = copied_fixture(tmp_path)
    metrics_path = root / "r2" / "metrics.json"
    metrics = json.loads(
        metrics_path.read_text(encoding="utf-8")
    )
    metrics["metrics"] = metrics["metrics"][:1]
    metrics_path.write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )

    manifest_path = root / "aggregate.yaml"
    manifest = yaml.safe_load(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest["aggregate"]["missing"] = {
        "policy": policy,
    }
    if policy == "exclude":
        manifest["aggregate"]["missing"]["reason"] = (
            "fixture incomplete output"
        )
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )

    _, _, result = load_result(root)
    jsd = next(
        row
        for row in result.summary_rows
        if row["metric"] == "one_minus_jsd"
    )
    assert jsd["n"] == expected_n
    assert jsd["missing_count"] == expected_missing
    assert result.validation["missing_policy"] == policy
    assert any(
        not row["included"]
        for row in result.input_rows
    ) is (policy == "exclude")


def _write_hierarchy_fixture(
    tmp_path: Path,
    *,
    weighting: str,
) -> Config:
    root = copied_fixture(tmp_path)
    shutil.rmtree(root / "r3")
    manifest = yaml.safe_load(
        (root / "aggregate.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest["aggregate"]["runs"] = (
        manifest["aggregate"]["runs"][:2]
    )
    for index, (wave, value) in enumerate(
        (("W1", 0.0), ("W1", 0.0), ("W2", 9.0)),
        start=1,
    ):
        source = root / ("r1" if index == 1 else "r2")
        target = root / f"h{index}"
        shutil.copytree(source, target)
        run_path = target / "run.json"
        run = json.loads(run_path.read_text())
        run["run_id"] = f"hierarchy-r{index:04d}"
        run["repeat"] = index if wave == "W1" else 1
        run["conditions"]["wave"] = wave
        run_path.write_text(json.dumps(run))
        metrics_path = target / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["run_id"] = run["run_id"]
        metrics["metrics"] = [metrics["metrics"][0]]
        metrics["metrics"][0]["value"] = value
        metrics_path.write_text(json.dumps(metrics))

    manifest["aggregate"]["runs"] = [
        {
            "path": f"h{index}",
            "labels": {
                "method": "p2p",
                "wave": wave,
                "model": "lasso",
                "split": "test",
                "repeat": (
                    index
                    if wave == "W1"
                    else 1
                ),
            },
        }
        for index, wave in (
            (1, "W1"),
            (2, "W1"),
            (3, "W2"),
        )
    ]
    manifest["aggregate"]["metrics"] = ["mse"]
    manifest["aggregate"]["expected"] = {
        "require_complete": True,
    }
    manifest["aggregate"]["hierarchy"] = {
        "reduce": ["repeat", "wave"],
        "weighting": weighting,
    }
    path = root / "hierarchy.yaml"
    path.write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )
    return Config.load(path)


def test_equal_hierarchy_differs_from_pooled_run_estimand(
    tmp_path: Path,
) -> None:
    equal_config = _write_hierarchy_fixture(
        tmp_path / "equal",
        weighting="equal",
    )
    pooled_config = _write_hierarchy_fixture(
        tmp_path / "pooled",
        weighting="pooled",
    )
    equal = aggregation.aggregate_runs(
        equal_config.aggregate_spec
    )
    pooled = aggregation.aggregate_runs(
        pooled_config.aggregate_spec
    )

    assert equal.summary_rows[0]["center"] == 4.5
    assert equal.summary_rows[0]["estimand"] == (
        "equal_wave"
    )
    assert pooled.summary_rows[0]["center"] == 3.0
    assert pooled.summary_rows[0]["estimand"] == (
        "pooled_run"
    )
