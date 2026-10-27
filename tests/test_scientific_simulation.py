"""Integration contracts for cached-basis scientific simulations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random

from p2p.commands import simulate
from p2p.config import Config
from p2p.simulations.scientific import ScientificSimulationResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_scientific_fraction_sweep_fits_and_summarizes(
    tmp_path: Path,
) -> None:
    config = Config.load(
        PROJECT_ROOT
        / "config"
        / "central"
        / "simulate_scientific_fixture.yaml"
    )
    random.seed(991)
    expected = random.random()
    random.seed(991)

    result = simulate.run(
        config,
        run_dir=tmp_path / "simulation",
        solver="OSQP",
    )

    assert len(result.children) == 2
    assert all(
        isinstance(child.result, ScientificSimulationResult)
        for child in result.children
    )
    assert result.summary_csv is not None
    assert result.summary_csv.is_file()
    assert result.summary_json is not None
    assert result.summary_json.is_file()
    assert result.aggregate_csv is not None
    assert result.aggregate_csv.is_file()
    assert random.random() == expected

    payload = json.loads(
        result.summary_json.read_text(encoding="utf-8")
    )
    assert payload["artifact_type"] == (
        "scientific_simulation_summary"
    )
    assert len(payload["children"]) == 2
    assert {row["fraction"] for row in payload["children"]} == {
        0.5,
        1.0,
    }
    assert all(row["mse_test"] is not None for row in payload["children"])

    with result.aggregate_csv.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert {"mse", "mse_test"} <= {row["metric"] for row in rows}


def test_scientific_target_entropy_is_explicit_and_audited(
    tmp_path: Path,
) -> None:
    base = Config.load(
        PROJECT_ROOT
        / "config"
        / "central"
        / "regress_fixture.yaml"
    ).to_dict()
    base["run"]["name"] = "target-entropy"
    base["split"]["use_proxy_only"] = False
    base["simulation"] = {
        "executor": "scientific",
        "kind": "target_entropy",
        "invokes_generation": False,
        "target_source": "configured",
        "ground_truth": {"source": "random", "count": 1},
        "sample_size": 1,
        "target_mean_entropy": 0.0,
        "max_evaluations": 10,
        "sweep": {
            "target": "simulate",
            "overrides": [
                {"simulation": {"target_mean_entropy": 0.0}}
            ],
        },
    }
    config = Config.from_mapping(
        base,
        declared_in=tmp_path / "target-entropy.yaml",
    )

    result = simulate.run(
        config,
        run_dir=tmp_path / "target-entropy",
    )
    stats = result.children[0].result.stats

    assert stats["target_mean_entropy"] == 0.0
    assert stats["achieved_mean_entropy"] == 0.0
    assert stats["entropy_search_method"] == "exact"


def test_all_legacy_fraction_and_mode_paths_execute(
    tmp_path: Path,
) -> None:
    base = Config.load(
        PROJECT_ROOT
        / "config"
        / "central"
        / "regress_fixture.yaml"
    ).to_dict()
    base["run"]["name"] = "legacy-simulation-modes"
    base["split"]["use_proxy_only"] = False
    base["simulation"] = {
        "executor": "scientific",
        "kind": "endowment",
        "invokes_generation": False,
        "target_source": "configured",
        "ground_truth": {"source": "existing"},
        "fraction": 0.5,
        "included_mode_count": 1,
        "n_proxies": 1,
        "sweep": {
            "target": "simulate",
            "overrides": [
                {"simulation": {"kind": "endowment"}},
                {"simulation": {"kind": "response"}},
                {"simulation": {"kind": "trainvalid"}},
                {"simulation": {"kind": "entropy_modes"}},
            ],
        },
    }
    config = Config.from_mapping(
        base,
        declared_in=tmp_path / "legacy-modes.yaml",
    )

    result = simulate.run(
        config,
        run_dir=tmp_path / "legacy-modes",
    )

    assert {
        child.result.stats["kind"]
        for child in result.children
    } == {
        "endowment",
        "response",
        "trainvalid",
        "entropy_modes",
    }
    entropy_child = next(
        child.result
        for child in result.children
        if child.result.stats["kind"] == "entropy_modes"
    )
    assert entropy_child.stats["n_modes"] == 1
    assert entropy_child.stats["n_proxy"] == 1


def test_fixed_ground_truth_targets_are_recomputed_from_basis(
    tmp_path: Path,
) -> None:
    base = Config.load(
        PROJECT_ROOT
        / "config"
        / "central"
        / "regress_fixture.yaml"
    ).to_dict()
    base["run"]["name"] = "fixed-ground-truth"
    base["split"]["use_proxy_only"] = False
    base["data"].pop("target")
    base["simulation"] = {
        "executor": "scientific",
        "kind": "proxy",
        "invokes_generation": False,
        "target_source": "ground_truth",
        "ground_truth": {"source": "random", "count": 1},
        "fraction": 1.0,
        "sweep": {
            "target": "simulate",
            "overrides": [
                {"simulation": {"fraction": 1.0}}
            ],
        },
    }
    config = Config.from_mapping(
        base,
        declared_in=tmp_path / "fixed-ground-truth.yaml",
    )

    result = simulate.run(
        config,
        run_dir=tmp_path / "fixed-ground-truth",
    )
    child = result.children[0].result

    assert child.stats["n_gt"] == 1
    assert child.stats["n_proxy"] == 1
    for split in child.data.splits.values():
        assert set(split.y) <= {0.0, 1.0}
