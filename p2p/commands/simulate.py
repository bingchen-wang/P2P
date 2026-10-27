"""Execute configured simulation sweeps with an explicit offline boundary."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from typing import Any

from p2p.models.constrained_model import DEFAULT_SOLVER
from p2p.config import AggregateSpec, Config
from p2p.registry import COMMANDS, Registry
from p2p.simulations.runner import (
    CommandHandler,
    SweepCommandResult,
    run_declared_simulation,
    run_sweep,
)
from p2p.simulations.scientific import (
    ScientificSimulationResult,
    run_scientific_simulation,
)


_SUMMARY_METRICS = (
    "mse",
    "r2",
    "mse_test",
    "r2_test",
    "precision",
    "recall",
    "selected",
    "gt_entropy",
    "proxy_entropy",
    "achieved_mean_entropy",
    "entropy_absolute_error",
)


def _write_scientific_summary(
    result: SweepCommandResult,
) -> SweepCommandResult:
    scientific = [
        child
        for child in result.children
        if isinstance(child.result, ScientificSimulationResult)
    ]
    if not scientific:
        return result

    records: list[dict[str, Any]] = []
    for child in scientific:
        row = dict(child.result.stats)
        row["repeat"] = child.repeat
        row["case"] = child.case
        row["run_dir"] = str(child.run_dir)
        records.append(row)

    summary_csv = result.run_dir / "simulation_summary.csv"
    summary_json = result.run_dir / "simulation_summary.json"
    aggregate_csv = result.run_dir / "simulation_aggregate.csv"
    fieldnames = list(
        dict.fromkeys(key for row in records for key in row)
    )
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (list, tuple, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )

    grouping_fields = (
        "kind",
        "model_type",
        "fraction",
        "target_mean_entropy",
        "n_modes",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in records:
        key = tuple(row.get(field) for field in grouping_fields)
        groups.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for key, rows in groups.items():
        labels = dict(zip(grouping_fields, key, strict=True))
        for metric in _SUMMARY_METRICS:
            values = [
                float(row[metric])
                for row in rows
                if (
                    row.get(metric) is not None
                    and math.isfinite(float(row[metric]))
                )
            ]
            if not values:
                continue
            count = len(values)
            mean = sum(values) / count
            variance = (
                sum((value - mean) ** 2 for value in values)
                / (count - 1)
                if count > 1
                else 0.0
            )
            aggregates.append(
                {
                    **labels,
                    "metric": metric,
                    "mean": mean,
                    "std": math.sqrt(variance),
                    "n": count,
                }
            )

    aggregate_fields = [
        *grouping_fields,
        "metric",
        "mean",
        "std",
        "n",
    ]
    with aggregate_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregates)

    summary_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "scientific_simulation_summary",
                "children": records,
                "aggregates": aggregates,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return replace(
        result,
        summary_csv=summary_csv,
        summary_json=summary_json,
        aggregate_csv=aggregate_csv,
    )


def run(
    config: Config,
    *,
    run_dir: str | os.PathLike[str],
    solver: str = DEFAULT_SOLVER,
    verbose: bool = False,
    registry: Registry[CommandHandler] = COMMANDS,
    simulation_runner: CommandHandler = run_declared_simulation,
    aggregation_spec: AggregateSpec | None = None,
) -> SweepCommandResult:
    """Expand a simulation sweep and execute its declared target.

    ``simulation.executor: scientific`` selects the built-in cached-basis
    executor.  Other manifests retain the configuration-only boundary unless
    a caller injects a runner.
    """

    selected_runner = simulation_runner
    simulation = config.data.get("simulation")
    if (
        simulation_runner is run_declared_simulation
        and isinstance(simulation, Mapping)
        and simulation.get("executor") == "scientific"
    ):
        selected_runner = run_scientific_simulation

    result = run_sweep(
        config,
        kind="simulate",
        run_dir=run_dir,
        solver=solver,
        verbose=verbose,
        registry=registry,
        same_verb_runner=selected_runner,
        aggregation_spec=aggregation_spec,
    )
    return _write_scientific_summary(result)


__all__ = ["run"]
