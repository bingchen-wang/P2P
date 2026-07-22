"""Golden regression tests against the legacy implementation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from p2p.commands import regress


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"

DATASETS = ("atp_w42", "wvs_gb")

REQUIRED_METRICS = {
    "test_mse",
    "test_one_minus_jsd",
    "test_emd",
    "test_mcpa",
    "nnz",
    "best_alpha",
}

FIXTURE_FILES = {
    "atp_w42": (
        "outputs/fixtures/atp_w42/responses.csv",
        "outputs/fixtures/atp_w42/endowments.csv",
        "outputs/fixtures/atp_w42/legacy_regression_config.yaml",
        "data/w42/info.csv",
        "data/w42/W42_aggregate_results.json",
        "config/surveys/american_trend_panel.yaml",
    ),
    "wvs_gb": (
        "outputs/fixtures/wvs_gb/responses.csv",
        "outputs/fixtures/wvs_gb/endowments.csv",
        "outputs/fixtures/wvs_gb/legacy_regression_config.yaml",
        "data/wvs_gb/info.csv",
        "data/wvs_gb/aggregate_results.json",
        "config/surveys/wvs.yaml",
    ),
}


def load_golden(dataset: str) -> dict[str, Any]:
    path = GOLDEN_DIR / f"{dataset}.json"
    with path.open(encoding="utf-8") as file:
        return json.load(file)


@pytest.mark.parity
@pytest.mark.parametrize("dataset", DATASETS)
def test_golden_baseline_is_complete(dataset: str) -> None:
    golden = load_golden(dataset)

    assert golden["schema_version"] == 1
    assert golden["dataset"] == dataset
    assert golden["environment"]["solver"] == "OSQP"
    assert set(golden["models"]) == {"lasso", "elasticnet"}

    for model_name, result in golden["models"].items():
        assert result["status"] == "optimal"
        assert REQUIRED_METRICS <= result.keys()

        for metric in REQUIRED_METRICS - {"nnz"}:
            assert math.isfinite(result[metric])

        assert result["test_mse"] >= 0
        assert 0 <= result["test_one_minus_jsd"] <= 1
        assert result["test_emd"] >= 0
        assert 0 <= result["test_mcpa"] <= 1
        assert result["nnz"] > 0
        assert result["best_alpha"] > 0

        if model_name == "elasticnet":
            assert 0 < result["best_l1_ratio"] <= 1


@pytest.mark.parity
@pytest.mark.parametrize("dataset", DATASETS)
def test_minimal_fixture_is_complete(dataset: str) -> None:
    missing = [
        relative_path
        for relative_path in FIXTURE_FILES[dataset]
        if not (PROJECT_ROOT / relative_path).is_file()
    ]

    if missing:
        pytest.skip(
            "Licensed external fixtures are not distributed with the repository: "
            + ", ".join(missing)
        )

    assert not missing, f"Missing fixture files: {missing}"


@pytest.mark.parity
@pytest.mark.parametrize("dataset", DATASETS)
def test_regression_matches_golden(
    dataset: str,
    cvxpy_solver: str,
) -> None:
    runner = getattr(regress, "run_fixture", None)

    if not callable(runner):
        pytest.skip("The regress fixture runner lands in Session 4")

    expected = load_golden(dataset)["models"]
    actual = runner(dataset=dataset, solver=cvxpy_solver)

    for model_name in ("lasso", "elasticnet"):
        for metric in REQUIRED_METRICS - {"nnz"}:
            assert actual[model_name][metric] == pytest.approx(
                expected[model_name][metric],
                rel=1e-5,
                abs=1e-8,
            )

        assert actual[model_name]["nnz"] == expected[model_name]["nnz"]