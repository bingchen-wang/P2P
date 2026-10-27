"""Pre-run token and cost forecasting contracts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import yaml

from p2p.budget_dashboard import build_budget_dashboard
from p2p.budgeting import (
    collect_budget_history,
    forecast_budget,
    write_budget_artifacts,
)
from p2p.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SURVEY = PROJECT_ROOT / "tests" / "fixtures" / "data" / "survey.csv"
SCHEMA = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "data"
    / "survey_schema.yaml"
)


def _config_mapping(target_n: int) -> dict:
    return {
        "run": {"name": "budget_fixture", "seed": 17},
        "data": {
            "survey": {
                "source": str(SURVEY),
                "schema": str(SCHEMA),
            }
        },
        "generation": {
            "mode": "active",
            "target_n": target_n,
            "initial_n": 1,
            "num_update_steps": 2,
            "endowment_model": {"batch_size": 5},
            "question_patch": {
                "fraction": 0.5,
                "top_k": 1,
            },
        },
        "backend": {
            "response": {
                "type": "openai",
                "model_name": "paid-model",
                "kwargs": {"max_tokens": 16},
            },
            "endowment": {
                "type": "openai",
                "model_name": "paid-model",
                "kwargs": {"max_tokens": 128},
            },
            "attribute_learner": {
                "type": "openai",
                "model_name": "paid-model",
                "kwargs": {"max_tokens": 32},
            },
        },
        "model": {
            "method": "lasso",
            "alpha": [0.01],
        },
    }


def _write_usage(
    run_dir: Path,
    *,
    target_n: int,
    pilot: bool = False,
) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "config": _config_mapping(target_n),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    rows = []
    for index in range(2):
        rows.append(
            {
                "model_name": "paid-model",
                "module_name": "attribute_learner",
                "agent_id": f"attribute-{index}",
                "input_tokens": 180 + target_n,
                "output_tokens": 24,
                "cached_input_tokens": 0,
                "cost_usd": 0,
            }
        )
    endowment_calls = math.ceil(target_n / 5)
    for index in range(endowment_calls):
        rows.append(
            {
                "model_name": "paid-model",
                "module_name": "endowment_model",
                "agent_id": f"endowment-{index}",
                "input_tokens": 220,
                "output_tokens": 90,
                "cached_input_tokens": 0,
                "cost_usd": 0,
            }
        )
    response_calls = target_n * 3
    if pilot:
        response_calls = max(1, response_calls // 2)
    for index in range(response_calls):
        rows.append(
            {
                "model_name": "paid-model",
                "module_name": "survey_conductor",
                "agent_id": f"response-{index}",
                "input_tokens": 120,
                "output_tokens": 8,
                "cached_input_tokens": 0,
                "cost_usd": 0,
            }
        )
    with (run_dir / "usage.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_budget_forecast_uses_history_pricing_and_pilot(
    tmp_path: Path,
) -> None:
    history_root = tmp_path / "history"
    for index, target_n in enumerate(range(8, 20)):
        _write_usage(
            history_root / f"run-{index:02d}",
            target_n=target_n,
        )
    pilot_run = tmp_path / "pilot"
    _write_usage(pilot_run, target_n=4, pilot=True)
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        "models:\n"
        "  paid-model:\n"
        "    input_per_million_usd: 2\n"
        "    output_per_million_usd: 8\n"
        "    cached_input_per_million_usd: 1\n",
        encoding="utf-8",
    )
    config = Config.from_mapping(
        _config_mapping(24),
        declared_in=tmp_path / "target.yaml",
    )

    history, digest, roots = collect_budget_history([history_root])
    assert history["run_dir"].nunique() == 12
    assert len(digest) == 64
    assert roots == (str(history_root.resolve()),)

    forecast = forecast_budget(
        config,
        history_roots=[history_root],
        pricing_path=pricing,
        pilot_run=pilot_run,
        confidence=0.8,
        budget_usd=1.0,
    )
    configured = forecast.scenarios[0]

    assert forecast.historical_runs == 12
    assert forecast.historical_records > 0
    assert forecast.pilot_records > 0
    assert configured.total_tokens.low > 0
    assert (
        configured.total_tokens.low
        <= configured.total_tokens.point
        <= configured.total_tokens.high
    )
    assert configured.total_cost_usd is not None
    assert configured.total_cost_usd.point > 0
    assert configured.budget_status == "within budget at interval high"
    assert all(
        module.training_scope == "exact model history"
        for module in configured.modules
    )
    assert all(
        module.pilot_calibration is not None
        for module in configured.modules
    )
    assert all(
        module.pricing.source.startswith("explicit pricing file")
        for module in configured.modules
    )

    with (pilot_run / "usage.csv").open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        csv.DictWriter(
            file,
            fieldnames=[
                "model_name",
                "module_name",
                "agent_id",
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "cost_usd",
            ],
        ).writerow(
            {
                "model_name": "paid-model",
                "module_name": "survey_conductor",
                "agent_id": "live-update",
                "input_tokens": 130,
                "output_tokens": 9,
                "cached_input_tokens": 0,
                "cost_usd": 0,
            }
        )
    updated = forecast_budget(
        config,
        history_roots=[history_root],
        pricing_path=pricing,
        pilot_run=pilot_run,
        confidence=0.8,
        budget_usd=1.0,
    )
    assert updated.pilot_records == forecast.pilot_records + 1
    response_before = next(
        module
        for module in configured.modules
        if module.module_name == "survey_conductor"
    )
    response_after = next(
        module
        for module in updated.scenarios[0].modules
        if module.module_name == "survey_conductor"
    )
    assert (
        response_after.pilot_calibration["observed_calls"]
        == response_before.pilot_calibration["observed_calls"] + 1
    )

    output = tmp_path / "budget"
    artifacts = write_budget_artifacts(updated, output)
    dashboard = build_budget_dashboard(
        updated,
        output / "budget.html",
        refresh_seconds=2,
    )
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    html = dashboard.output_path.read_text(encoding="utf-8")

    assert payload["artifact_type"] == "budget_forecast"
    assert artifacts.csv_path.is_file()
    assert artifacts.history_csv_path.is_file()
    assert {
        "scenario-tokens",
        "module-tokens",
        "scenario-cost",
        "history",
        "pilot",
    } <= set(dashboard.sections)
    assert "PRE-RUN BUDGET PLANNER" in html
    assert "Live pilot usage trajectory" in html
    assert "http-equiv='refresh' content='2'" in html
    assert "No API call is made by this command" in html


def test_budget_rejects_empty_history(tmp_path: Path) -> None:
    config = Config.from_mapping(
        _config_mapping(5),
        declared_in=tmp_path / "target.yaml",
    )

    try:
        forecast_budget(
            config,
            history_roots=[tmp_path / "missing"],
        )
    except ValueError as error:
        assert "no usable tracked usage history" in str(error)
    else:
        raise AssertionError("empty history must fail explicitly")
