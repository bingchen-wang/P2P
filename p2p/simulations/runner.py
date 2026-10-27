"""Deterministic execution plumbing for configured sweeps and simulations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import inspect
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

from p2p.models.constrained_model import DEFAULT_SOLVER
from p2p.aggregation import (
    AggregationArtifacts,
    AggregationResult,
    aggregate_runs,
    write_aggregation_artifacts,
)
from p2p.config import AggregateRun, AggregateSpec, Config
from p2p.registry import COMMANDS, Registry
from p2p.sweep import SweepSpec, expand_sweep


CommandHandler = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class SimulationArtifactResult:
    """A deterministic record of declared scalar simulation settings.

    This fallback deliberately records configuration only.  It does not claim
    to execute a scientific simulation or emit model/metric results.
    """

    config_digest: str
    run_dir: Path
    artifact_path: Path
    settings: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SweepChildResult:
    """One expanded, snapshotted, and executed sweep child."""

    index: int
    repeat: int
    case: int
    run_dir: Path
    config: Config
    snapshot_path: Path
    result: Any


@dataclass(frozen=True, slots=True)
class SweepCommandResult:
    """Completed child runs and optional shared aggregation artifacts."""

    kind: str
    target_verb: str
    run_dir: Path
    children: tuple[SweepChildResult, ...]
    aggregation: AggregationResult | None = None
    aggregation_artifacts: AggregationArtifacts | None = None
    summary_csv: Path | None = None
    summary_json: Path | None = None
    aggregate_csv: Path | None = None


def _resolved_directory(
    value: str | os.PathLike[str],
    *,
    field: str,
) -> Path:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty path")

    path = Path(value).resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(f"{field} is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _scalar_settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("simulation must be a mapping")

    settings: dict[str, Any] = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                "simulation setting names must be nonempty strings"
            )
        if child is None or isinstance(child, (str, bool, int)):
            settings[key] = child
            continue
        if isinstance(child, float) and math.isfinite(child):
            settings[key] = child
            continue
        raise ValueError(
            "offline simulation settings must be finite JSON scalars; "
            f"simulation.{key} is unsupported"
        )
    return settings


def run_declared_simulation(
    config: Config,
    *,
    run_dir: str | os.PathLike[str],
    verbose: bool = False,
) -> SimulationArtifactResult:
    """Record scalar simulation settings without fabricating scientific output."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")
    config.validate_for("simulate")

    settings = _scalar_settings(config.data.get("simulation"))
    if settings.get("invokes_generation", False):
        raise RuntimeError(
            "the offline simulation artifact runner does not execute "
            "generation; inject an executable simulation runner"
        )

    resolved_run_dir = _resolved_directory(
        run_dir,
        field="run_dir",
    )
    run = config.data.get("run", {})
    if not isinstance(run, Mapping):
        raise ValueError("run must be a mapping")

    artifact_path = resolved_run_dir / "simulation.json"
    payload = {
        "schema_version": 1,
        "artifact_type": "declared_simulation_settings",
        "boundary": (
            "configuration record only; no scientific simulation, "
            "model result, or metric result was produced"
        ),
        "config_digest": config.digest,
        "run": {
            "name": run.get("name"),
            "repeat": run.get("repeat", 1),
            "seed": run.get("seed", 101),
        },
        "simulation": settings,
    }
    artifact_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return SimulationArtifactResult(
        config_digest=config.digest,
        run_dir=resolved_run_dir,
        artifact_path=artifact_path,
        settings=MappingProxyType(dict(settings)),
    )


def _registered_handler(
    registry: Registry[CommandHandler],
    name: str,
) -> CommandHandler:
    if not hasattr(registry, "get"):
        raise TypeError("registry must provide get(name)")
    handler = registry.get(name)
    if not callable(handler):
        raise TypeError(
            f"registered command {name!r} must be callable"
        )
    return handler


def _invoke_handler(
    handler: CommandHandler,
    config: Config,
    *,
    run_dir: Path,
    solver: str,
    verbose: bool,
) -> Any:
    candidates = {
        "run_dir": run_dir,
        "solver": solver,
        "verbose": verbose,
    }
    try:
        parameters = inspect.signature(handler).parameters
    except (TypeError, ValueError):
        parameters = {}

    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs = {
        name: value
        for name, value in candidates.items()
        if accepts_kwargs or name in parameters
    }
    return handler(config, **kwargs)


def _aggregation_for_children(
    children: tuple[SweepChildResult, ...],
    template: AggregateSpec,
    *,
    output_dir: Path,
    source_config: Path,
) -> tuple[AggregationResult, AggregationArtifacts]:
    if len(template.runs) != len(children):
        raise ValueError(
            "aggregation_spec must declare one run entry per sweep child"
        )

    spec = replace(
        template,
        runs=tuple(
            AggregateRun(
                path=child.run_dir,
                labels=declared.labels,
            )
            for child, declared in zip(
                children,
                template.runs,
                strict=True,
            )
        ),
    )
    aggregated = aggregate_runs(spec)
    artifacts = write_aggregation_artifacts(
        aggregated,
        spec,
        output_dir=output_dir / "aggregation",
        source_config=source_config,
    )
    return aggregated, artifacts


def run_sweep(
    config: Config,
    *,
    kind: str,
    run_dir: str | os.PathLike[str],
    solver: str = DEFAULT_SOLVER,
    verbose: bool = False,
    registry: Registry[CommandHandler] = COMMANDS,
    same_verb_runner: CommandHandler | None = None,
    aggregation_spec: AggregateSpec | None = None,
) -> SweepCommandResult:
    """Expand, snapshot, and execute one configured sweep."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")
    config.validate_for(kind)

    spec = SweepSpec.from_config(config, kind)
    expanded = expand_sweep(config, spec)
    resolved_run_dir = _resolved_directory(
        run_dir,
        field="run_dir",
    )

    if (
        same_verb_runner is not None
        and spec.target_verb == kind
    ):
        handler = same_verb_runner
    else:
        handler = _registered_handler(
            registry,
            spec.target_verb,
        )

    case_count = len(spec.overrides)
    children: list[SweepChildResult] = []
    for index, child_config in enumerate(expanded, start=1):
        repeat = (index - 1) // case_count + 1
        case = (index - 1) % case_count + 1
        child_dir = (
            resolved_run_dir
            / f"repeat-{repeat:04d}-case-{case:04d}"
        )
        child_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = child_config.write_snapshot(
            child_dir
        ).resolve()
        result = _invoke_handler(
            handler,
            child_config,
            run_dir=child_dir,
            solver=solver,
            verbose=verbose,
        )
        children.append(
            SweepChildResult(
                index=index,
                repeat=repeat,
                case=case,
                run_dir=child_dir.resolve(),
                config=child_config,
                snapshot_path=snapshot_path,
                result=result,
            )
        )

    child_results = tuple(children)
    aggregated = None
    aggregation_artifacts = None
    if aggregation_spec is not None:
        if spec.target_verb not in {"regress", "pipeline"}:
            raise ValueError(
                "aggregation is supported only for completed "
                "regress or pipeline sweep children"
            )
        aggregated, aggregation_artifacts = (
            _aggregation_for_children(
                child_results,
                aggregation_spec,
                output_dir=resolved_run_dir,
                source_config=config.source_path,
            )
        )

    return SweepCommandResult(
        kind=kind,
        target_verb=spec.target_verb,
        run_dir=resolved_run_dir,
        children=child_results,
        aggregation=aggregated,
        aggregation_artifacts=aggregation_artifacts,
    )


__all__ = [
    "SimulationArtifactResult",
    "SweepChildResult",
    "SweepCommandResult",
    "run_declared_simulation",
    "run_sweep",
]
