"""User-facing aggregation over canonical result directories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path

from p2p.aggregation import (
    AggregationArtifacts,
    AggregationResult,
    aggregate_runs,
    write_aggregation_artifacts,
)
from p2p.config import (
    AggregateRun,
    AggregateSpec,
    Config,
)


@dataclass(frozen=True, slots=True)
class AggregateCommandResult:
    """One completed aggregation and its written artifacts."""

    result: AggregationResult
    artifacts: AggregationArtifacts
    output_dir: Path


def _quick_spec(
    runs: Sequence[str | os.PathLike[str]],
    *,
    group_by: Sequence[str],
    metrics: Sequence[str],
) -> AggregateSpec:
    if not runs:
        raise ValueError("runs must not be empty")
    if not group_by:
        raise ValueError("group_by must not be empty")
    if not metrics:
        raise ValueError("metrics must not be empty")
    return AggregateSpec(
        runs=tuple(
            AggregateRun(
                path=Path(path).resolve(),
                labels={},
            )
            for path in runs
        ),
        group_by=tuple(group_by),
        metrics=tuple(metrics),
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


def run(
    config: Config | None,
    *,
    output_dir: str | os.PathLike[str],
    runs: Sequence[str | os.PathLike[str]] | None = None,
    group_by: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    verbose: bool = False,
) -> AggregateCommandResult:
    """Validate, aggregate, and export configured or explicit runs."""

    if config is not None:
        if not isinstance(config, Config):
            raise TypeError("config must be Config or None")
        if runs is not None:
            raise ValueError(
                "runs must not be provided with config"
            )
        config.validate_for("aggregate")
        spec = config.aggregate_spec
        source_config = config.source_path
    else:
        if runs is None:
            raise ValueError(
                "runs are required without config"
            )
        spec = _quick_spec(
            runs,
            group_by=group_by or (),
            metrics=metrics or (),
        )
        source_config = None

    if spec is None:
        raise ValueError("aggregate spec is unavailable")
    result = aggregate_runs(spec)
    resolved_output = Path(output_dir).resolve()
    artifacts = write_aggregation_artifacts(
        result,
        spec,
        output_dir=resolved_output,
        source_config=source_config,
    )
    return AggregateCommandResult(
        result=result,
        artifacts=artifacts,
        output_dir=resolved_output,
    )


__all__ = [
    "AggregateCommandResult",
    "run",
]
