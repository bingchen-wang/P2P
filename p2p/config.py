"""Central configuration loading and deterministic resolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib import resources
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml


class ConfigError(ValueError):
    """Base error for central configuration failures."""


class ConfigLoadError(ConfigError):
    """Raised when a manifest cannot be loaded or inherited."""


class ConfigValidationError(ConfigError):
    """Raised when resolved configuration data is invalid."""


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge mappings; lists and scalars replace."""
    result = deepcopy(dict(base))
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = deepcopy(value)
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze(child)
                for key, child in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _thaw(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return deepcopy(value)


def _validate_grid_bound(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(
            f"{name} must be a finite number"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ConfigValidationError(
            f"{name} must be a finite number"
        )
    return numeric


def expand_grid(spec: Any) -> tuple[Any, ...]:
    """Expand a literal list, linspace, or base-10 logspace grid."""
    if isinstance(spec, str):
        raise ConfigValidationError(
            "grid expressions are not supported; use a literal list, "
            "{linspace: [start, stop, count]}, or "
            "{logspace: [start, stop, count]}"
        )

    if isinstance(spec, Sequence) and not isinstance(
        spec,
        (str, bytes, bytearray),
    ):
        values = tuple(spec)
        if not values:
            raise ConfigValidationError(
                "grid values must not be empty"
            )
        return values

    if not isinstance(spec, Mapping):
        raise ConfigValidationError(
            "grid must be a literal list or a structured grid mapping"
        )
    if len(spec) != 1:
        raise ConfigValidationError(
            "structured grid must contain exactly one operator"
        )

    operator, arguments = next(iter(spec.items()))
    if operator not in {"linspace", "logspace"}:
        raise ConfigValidationError(
            f"unknown structured-grid operator {operator!r}"
        )
    if (
        not isinstance(arguments, Sequence)
        or isinstance(arguments, (str, bytes, bytearray))
        or len(arguments) != 3
    ):
        raise ConfigValidationError(
            f"{operator} requires [start, stop, count]"
        )

    start = _validate_grid_bound(arguments[0], "grid start")
    stop = _validate_grid_bound(arguments[1], "grid stop")
    count = arguments[2]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ConfigValidationError(
            "grid count must be a positive integer"
        )

    values = (
        np.linspace(start, stop, count)
        if operator == "linspace"
        else np.logspace(start, stop, count)
    )
    return tuple(float(value) for value in values)


def _validate_root_mapping(
    value: Any,
    source_path: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigLoadError(
            f"{source_path}: manifest root must be a mapping"
        )
    if not all(isinstance(key, str) for key in value):
        raise ConfigLoadError(
            f"{source_path}: manifest keys must be strings"
        )
    return dict(value)


def _resolve_path(
    value: Any,
    *,
    manifest_path: Path,
    field: str,
    allow_builtin: bool = False,
) -> str:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(
            f"{manifest_path}: {field} must be a nonempty path"
        )

    if value.startswith("builtin:"):
        if allow_builtin:
            return value
        raise ConfigValidationError(
            f"{manifest_path}: {field} does not accept built-in resources"
        )

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    resolved = candidate.resolve()

    if not resolved.exists():
        raise ConfigValidationError(
            f"{manifest_path}: {field} path {value!r} does not exist"
        )
    return str(resolved)


def _resolve_resource_value(
    value: Any,
    *,
    manifest_path: Path,
    field: str,
) -> Any:
    if not isinstance(value, Mapping):
        return _resolve_path(
            value,
            manifest_path=manifest_path,
            field=field,
        )

    path_keys = {
        "path",
        "source",
        "schema",
        "responses",
        "endowments",
    }
    resolved: dict[str, Any] = {}
    for key, child in value.items():
        if key in path_keys:
            resolved[key] = _resolve_path(
                child,
                manifest_path=manifest_path,
                field=f"{field}.{key}",
            )
        else:
            resolved[key] = deepcopy(child)
    return resolved


def _resolve_references(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    resolved = deepcopy(dict(manifest))

    data = resolved.get("data")
    if data is not None:
        if not isinstance(data, Mapping):
            raise ConfigValidationError(
                f"{manifest_path}: data must be a mapping"
            )
        data = dict(data)
        for key in (
            "survey",
            "basis",
            "target",
            "reference_distribution",
            "predictions",
            "fitted",
        ):
            if key in data:
                data[key] = _resolve_resource_value(
                    data[key],
                    manifest_path=manifest_path,
                    field=f"data.{key}",
                )
        resolved["data"] = data

    backend = resolved.get("backend")
    if isinstance(backend, (str, os.PathLike)):
        resolved["backend"] = _resolve_path(
            backend,
            manifest_path=manifest_path,
            field="backend",
        )
    elif isinstance(backend, Mapping) and "profile" in backend:
        backend = dict(backend)
        backend["profile"] = _resolve_path(
            backend["profile"],
            manifest_path=manifest_path,
            field="backend.profile",
        )
        resolved["backend"] = backend

    generation = resolved.get("generation")
    if isinstance(generation, Mapping) and "attribute_bank" in generation:
        generation = dict(generation)
        bank = generation["attribute_bank"]
        if isinstance(bank, Mapping):
            bank = dict(bank)
            if "source" in bank:
                bank["source"] = _resolve_path(
                    bank["source"],
                    manifest_path=manifest_path,
                    field="generation.attribute_bank.source",
                    allow_builtin=True,
                )
        elif isinstance(bank, (str, os.PathLike)):
            bank = {
                "source": _resolve_path(
                    bank,
                    manifest_path=manifest_path,
                    field="generation.attribute_bank.source",
                    allow_builtin=True,
                )
            }
        generation["attribute_bank"] = bank
        resolved["generation"] = generation

    aggregate = resolved.get("aggregate")
    if isinstance(aggregate, Mapping):
        aggregate = dict(aggregate)
        runs = aggregate.get("runs")
        if isinstance(runs, Sequence) and not isinstance(
            runs,
            (str, bytes, bytearray),
        ):
            resolved_runs = []
            for index, run in enumerate(runs):
                if not isinstance(run, Mapping):
                    resolved_runs.append(deepcopy(run))
                    continue
                run = dict(run)
                if "path" in run:
                    run["path"] = _resolve_path(
                        run["path"],
                        manifest_path=manifest_path,
                        field=f"aggregate.runs[{index}].path",
                    )
                resolved_runs.append(run)
            aggregate["runs"] = resolved_runs
        resolved["aggregate"] = aggregate

    # A replayed fold record is a path like any other: resolved against the
    # config's own location, and required to exist.
    model = resolved.get("model")
    if isinstance(model, Mapping):
        validation = model.get("validation")
        if isinstance(validation, Mapping):
            folds = validation.get("folds")
            if isinstance(folds, Mapping) and folds.get("file") is not None:
                folds = dict(folds)
                folds["file"] = _resolve_path(
                    folds["file"],
                    manifest_path=manifest_path,
                    field="model.validation.folds.file",
                )
                validation = dict(validation)
                validation["folds"] = folds
                model = dict(model)
                model["validation"] = validation
                resolved["model"] = model

    return resolved


def _load_manifest(
    manifest_path: Path,
    stack: tuple[Path, ...],
) -> dict[str, Any]:
    source_path = manifest_path.resolve()

    if source_path in stack:
        cycle = stack + (source_path,)
        chain = " -> ".join(str(path) for path in cycle)
        raise ConfigLoadError(
            f"configuration inheritance cycle: {chain}"
        )
    if not source_path.is_file():
        raise ConfigLoadError(
            f"manifest does not exist: {source_path}"
        )

    try:
        with source_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except OSError as error:
        raise ConfigLoadError(
            f"could not read manifest {source_path}: {error}"
        ) from error
    except yaml.YAMLError as error:
        raise ConfigLoadError(
            f"invalid YAML in {source_path}: {error}"
        ) from error

    local = _validate_root_mapping(loaded, source_path)
    extends = local.pop("extends", None)

    base: dict[str, Any] = {}
    if extends is not None:
        if not isinstance(extends, str) or not extends.strip():
            raise ConfigLoadError(
                f"{source_path}: extends must be a nonempty path string"
            )
        base_path = Path(extends)
        if not base_path.is_absolute():
            base_path = source_path.parent / base_path
        base = _load_manifest(
            base_path,
            stack + (source_path,),
        )

    resolved_local = _resolve_references(local, source_path)
    return _deep_merge(base, resolved_local)


def _apply_defaults(
    manifest: Mapping[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    defaults = {
        "run": {
            "name": source_path.stem,
            "seed": 101,
        },
        "evaluation": {
            "mse_prediction_policy": "raw",
            "distribution_prediction_policy": "clip_and_normalize",
            "invalid_distribution": "error",
        },
    }
    return _deep_merge(defaults, manifest)


def _expand_known_grids(
    manifest: Mapping[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    resolved = deepcopy(dict(manifest))
    model = resolved.get("model")
    if model is None:
        return resolved
    if not isinstance(model, Mapping):
        raise ConfigValidationError(
            f"{source_path}: model must be a mapping"
        )

    model = dict(model)
    for field in ("alpha", "l1_ratio"):
        if field not in model:
            continue
        try:
            model[field] = expand_grid(model[field])
        except ConfigValidationError as error:
            raise ConfigValidationError(
                f"{source_path}: model.{field}: {error}"
            ) from error
    resolved["model"] = model
    return resolved


_RESERVED_BANK_PREFIXES = (
    "survey.",
    "question.",
)


def _is_reserved_mode(value: str) -> bool:
    return value.startswith(_RESERVED_BANK_PREFIXES)


def _validate_nonempty_string(
    value: Any,
    *,
    source: str,
    field: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(
            f"{source}: {field} must be a nonempty string"
        )
    return value


def _validate_attribute_bank(
    value: Any,
    *,
    source: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(
            f"{source}: attribute bank root must be a mapping"
        )

    unknown = set(value) - {"templates", "attributes"}
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ConfigValidationError(
            f"{source}: unknown attribute-bank fields: {names}"
        )

    templates = value.get("templates")
    attributes = value.get("attributes")
    if not isinstance(templates, Mapping):
        raise ConfigValidationError(
            f"{source}: attribute bank requires a templates mapping"
        )
    if not isinstance(attributes, Mapping):
        raise ConfigValidationError(
            f"{source}: attribute bank requires an attributes mapping"
        )

    normalized_attributes: dict[str, list[str]] = {}
    for mode, mode_attributes in attributes.items():
        mode = _validate_nonempty_string(
            mode,
            source=source,
            field="attribute mode",
        )
        if _is_reserved_mode(mode):
            raise ConfigValidationError(
                f"{source}: static attribute mode {mode!r} "
                "uses a reserved namespace"
            )
        if (
            not isinstance(mode_attributes, Sequence)
            or isinstance(mode_attributes, (str, bytes, bytearray))
        ):
            raise ConfigValidationError(
                f"{source}: attributes.{mode} must be a list"
            )
        normalized_attributes[mode] = [
            _validate_nonempty_string(
                attribute,
                source=source,
                field=f"attributes.{mode} entry",
            )
            for attribute in mode_attributes
        ]

    normalized_templates: dict[str, list[str]] = {}
    seen_modes: set[str] = set()

    for group, modes in templates.items():
        group = _validate_nonempty_string(
            group,
            source=source,
            field="template group",
        )
        if _is_reserved_mode(group):
            raise ConfigValidationError(
                f"{source}: static template group {group!r} "
                "uses a reserved namespace"
            )
        if (
            not isinstance(modes, Sequence)
            or isinstance(modes, (str, bytes, bytearray))
        ):
            raise ConfigValidationError(
                f"{source}: templates.{group} must be a list"
            )

        normalized_modes = []
        for mode in modes:
            mode = _validate_nonempty_string(
                mode,
                source=source,
                field=f"templates.{group} entry",
            )
            if _is_reserved_mode(mode):
                raise ConfigValidationError(
                    f"{source}: static template mode {mode!r} "
                    "uses a reserved namespace"
                )
            if mode in seen_modes:
                raise ConfigValidationError(
                    f"{source}: duplicate template mode {mode!r}"
                )
            if mode not in normalized_attributes:
                raise ConfigValidationError(
                    f"{source}: template mode {mode!r} is missing "
                    "from attributes"
                )
            seen_modes.add(mode)
            normalized_modes.append(mode)

        normalized_templates[group] = normalized_modes

    return {
        "templates": normalized_templates,
        "attributes": normalized_attributes,
    }


@dataclass(frozen=True, slots=True)
class ResolvedAttributeBank:
    """Immutable validated attribute-bank source."""

    source: str
    source_digest: str
    data: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.data)

    def write_snapshot(
        self,
        run_dir: str | os.PathLike[str],
        *,
        attribute_bank: Mapping[str, Any] | None = None,
    ) -> Path:
        destination = Path(run_dir)
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_path = destination / "attribute_bank.resolved.yaml"

        bank = (
            self.to_dict()
            if attribute_bank is None
            else _thaw(attribute_bank)
        )
        if not isinstance(bank, Mapping):
            raise ConfigValidationError(
                "attribute-bank snapshot must be a mapping"
            )

        payload = {
            "schema_version": 1,
            "source": self.source,
            "source_digest": self.source_digest,
            "attribute_bank": dict(bank),
        }
        snapshot_path.write_text(
            yaml.safe_dump(
                payload,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        return snapshot_path


def _read_attribute_bank(
    source: str,
) -> tuple[str, bytes]:
    if source.startswith("builtin:"):
        name = source.removeprefix("builtin:")
        if name != "default":
            raise ConfigValidationError(
                f"unknown built-in attribute bank {source!r}"
            )
        resource = resources.files(
            "p2p.attribute_banks"
        ).joinpath("default.yaml")
        try:
            raw = resource.read_bytes()
        except OSError as error:
            raise ConfigValidationError(
                "could not read builtin:default attribute bank"
            ) from error
        return "builtin:default", raw

    path = Path(source).resolve()
    if not path.is_file():
        raise ConfigValidationError(
            f"attribute-bank path does not exist: {path}"
        )
    try:
        return str(path), path.read_bytes()
    except OSError as error:
        raise ConfigValidationError(
            f"could not read attribute bank {path}: {error}"
        ) from error


def _resolve_attribute_bank(
    source: str,
) -> ResolvedAttributeBank:
    canonical_source, raw = _read_attribute_bank(source)
    try:
        loaded = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ConfigValidationError(
            f"{canonical_source}: malformed attribute-bank YAML"
        ) from error

    validated = _validate_attribute_bank(
        loaded,
        source=canonical_source,
    )
    return ResolvedAttributeBank(
        source=canonical_source,
        source_digest=hashlib.sha256(raw).hexdigest(),
        data=_freeze(validated),
    )


def _attach_attribute_bank(
    manifest: Mapping[str, Any],
    source_path: Path,
) -> tuple[dict[str, Any], ResolvedAttributeBank | None]:
    resolved = deepcopy(dict(manifest))
    generation = resolved.get("generation")

    if generation is None or not isinstance(generation, Mapping):
        return resolved, None

    generation = dict(generation)
    mode = generation.get("mode", "active")
    bank_spec = generation.get("attribute_bank")

    if mode != "active" and bank_spec is None:
        return resolved, None

    if bank_spec is None:
        bank_source = "builtin:default"
    elif isinstance(bank_spec, Mapping):
        bank_source = bank_spec.get(
            "source",
            "builtin:default",
        )
    else:
        raise ConfigValidationError(
            f"{source_path}: generation.attribute_bank "
            "must be a mapping"
        )

    if not isinstance(bank_source, str):
        raise ConfigValidationError(
            f"{source_path}: generation.attribute_bank.source "
            "must be a string"
        )

    bank = _resolve_attribute_bank(bank_source)
    generation["attribute_bank"] = {
        "source": bank.source,
        "source_digest": bank.source_digest,
    }
    resolved["generation"] = generation
    return resolved, bank


def _string_tuple(
    value: Any,
    *,
    source_path: Path,
    field: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise ConfigValidationError(
            f"{source_path}: {field} must be a list"
        )

    values = tuple(value)
    if not values and not allow_empty:
        raise ConfigValidationError(
            f"{source_path}: {field} must not be empty"
        )
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ConfigValidationError(
                f"{source_path}: {field} entries must be "
                "nonempty strings"
            )
    if len(values) != len(set(values)):
        raise ConfigValidationError(
            f"{source_path}: {field} contains duplicate values"
        )
    return values


@dataclass(frozen=True, slots=True)
class AggregateRun:
    """One resolved aggregation input and its condition labels."""

    path: Path
    labels: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "labels": _thaw(self.labels),
        }


@dataclass(frozen=True, slots=True)
class AggregateSpec:
    """Validated aggregation configuration without execution logic."""

    runs: tuple[AggregateRun, ...]
    group_by: tuple[str, ...]
    metrics: tuple[str, ...]
    expected_repeats: int | None
    expected_cells: tuple[Mapping[str, Any], ...]
    require_complete: bool
    center: str
    spread: str
    hierarchy_reduce: tuple[str, ...]
    hierarchy_weighting: str
    missing_policy: str
    missing_reason: str | None
    export_formats: tuple[str, ...]
    export_precision: int

    def to_dict(self) -> dict[str, Any]:
        expected: dict[str, Any] = {
            "cells": [
                _thaw(cell)
                for cell in self.expected_cells
            ],
            "require_complete": self.require_complete,
        }
        if self.expected_repeats is not None:
            expected["repeats"] = self.expected_repeats

        missing: dict[str, Any] = {
            "policy": self.missing_policy,
        }
        if self.missing_reason is not None:
            missing["reason"] = self.missing_reason

        return {
            "runs": [
                run.to_dict()
                for run in self.runs
            ],
            "group_by": list(self.group_by),
            "metrics": list(self.metrics),
            "expected": expected,
            "statistics": {
                "center": self.center,
                "spread": self.spread,
            },
            "hierarchy": {
                "reduce": list(self.hierarchy_reduce),
                "weighting": self.hierarchy_weighting,
            },
            "missing": missing,
            "export": {
                "formats": list(self.export_formats),
                "precision": self.export_precision,
            },
        }


def _parse_labels(
    value: Any,
    *,
    source_path: Path,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(
            f"{source_path}: {field} must be a mapping"
        )

    labels: dict[str, Any] = {}
    for key, label in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ConfigValidationError(
                f"{source_path}: {field} keys must be "
                "nonempty strings"
            )
        if isinstance(label, (Mapping, list, tuple, set)):
            raise ConfigValidationError(
                f"{source_path}: {field}.{key} must be a scalar"
            )
        labels[key] = deepcopy(label)
    return _freeze(labels)


def _parse_aggregate_spec(
    aggregate: Any,
    *,
    source_path: Path,
) -> AggregateSpec:
    if not isinstance(aggregate, Mapping):
        raise ConfigValidationError(
            f"{source_path}: aggregate must be a mapping"
        )

    raw_runs = aggregate.get("runs")
    if (
        not isinstance(raw_runs, Sequence)
        or isinstance(raw_runs, (str, bytes, bytearray))
        or not raw_runs
    ):
        raise ConfigValidationError(
            f"{source_path}: aggregate.runs must be a nonempty list"
        )

    runs: list[AggregateRun] = []
    seen_paths: set[Path] = set()
    for index, raw_run in enumerate(raw_runs):
        field = f"aggregate.runs[{index}]"
        if not isinstance(raw_run, Mapping):
            raise ConfigValidationError(
                f"{source_path}: {field} must be a mapping"
            )

        raw_path = raw_run.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ConfigValidationError(
                f"{source_path}: {field}.path must be a nonempty path"
            )
        run_path = Path(raw_path).resolve()
        if not run_path.exists():
            raise ConfigValidationError(
                f"{source_path}: {field}.path does not exist"
            )
        if run_path in seen_paths:
            raise ConfigValidationError(
                f"{source_path}: duplicate aggregate run path "
                f"{run_path}"
            )

        seen_paths.add(run_path)
        runs.append(
            AggregateRun(
                path=run_path,
                labels=_parse_labels(
                    raw_run.get("labels", {}),
                    source_path=source_path,
                    field=f"{field}.labels",
                ),
            )
        )

    group_by = _string_tuple(
        aggregate.get("group_by"),
        source_path=source_path,
        field="aggregate.group_by",
    )
    metrics = _string_tuple(
        aggregate.get("metrics"),
        source_path=source_path,
        field="aggregate.metrics",
    )

    expected = aggregate.get("expected", {})
    if not isinstance(expected, Mapping):
        raise ConfigValidationError(
            f"{source_path}: aggregate.expected must be a mapping"
        )

    repeats = expected.get("repeats")
    if repeats is not None and (
        isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or repeats <= 0
    ):
        raise ConfigValidationError(
            f"{source_path}: aggregate.expected.repeats "
            "must be a positive integer"
        )

    raw_cells = expected.get("cells", [])
    if (
        not isinstance(raw_cells, Sequence)
        or isinstance(raw_cells, (str, bytes, bytearray))
    ):
        raise ConfigValidationError(
            f"{source_path}: aggregate.expected.cells must be a list"
        )
    expected_cells = tuple(
        _parse_labels(
            cell,
            source_path=source_path,
            field=f"aggregate.expected.cells[{index}]",
        )
        for index, cell in enumerate(raw_cells)
    )

    require_complete = expected.get("require_complete", True)
    if not isinstance(require_complete, bool):
        raise ConfigValidationError(
            f"{source_path}: aggregate.expected.require_complete "
            "must be boolean"
        )

    statistics = aggregate.get("statistics", {})
    if not isinstance(statistics, Mapping):
        raise ConfigValidationError(
            f"{source_path}: aggregate.statistics must be a mapping"
        )
    center = statistics.get("center", "mean")
    if center not in {"mean", "median"}:
        raise ConfigValidationError(
            f"{source_path}: aggregate.statistics.center "
            "must be mean or median"
        )
    spread = statistics.get("spread", "sample_std")
    if spread not in {
        "sample_std",
        "population_std",
        "none",
    }:
        raise ConfigValidationError(
            f"{source_path}: aggregate.statistics.spread "
            "must explicitly select sample_std, "
            "population_std, or none"
        )

    hierarchy = aggregate.get("hierarchy", {})
    if not isinstance(hierarchy, Mapping):
        raise ConfigValidationError(
            f"{source_path}: aggregate.hierarchy must be a mapping"
        )
    hierarchy_reduce = _string_tuple(
        hierarchy.get("reduce", []),
        source_path=source_path,
        field="aggregate.hierarchy.reduce",
        allow_empty=True,
    )
    hierarchy_weighting = hierarchy.get("weighting", "equal")
    if hierarchy_weighting not in {"equal", "pooled"}:
        raise ConfigValidationError(
            f"{source_path}: aggregate.hierarchy.weighting "
            "must be equal or pooled"
        )

    missing = aggregate.get("missing", {})
    if not isinstance(missing, Mapping):
        raise ConfigValidationError(
            f"{source_path}: aggregate.missing must be a mapping"
        )
    missing_policy = missing.get("policy", "error")
    if missing_policy not in {"error", "allow", "exclude"}:
        raise ConfigValidationError(
            f"{source_path}: aggregate.missing.policy "
            "must be error, allow, or exclude"
        )
    missing_reason = missing.get("reason")
    if missing_reason is not None and (
        not isinstance(missing_reason, str)
        or not missing_reason.strip()
    ):
        raise ConfigValidationError(
            f"{source_path}: aggregate.missing.reason "
            "must be a nonempty string"
        )
    if missing_policy == "exclude" and missing_reason is None:
        raise ConfigValidationError(
            f"{source_path}: aggregate.missing.reason is "
            "required when policy is exclude"
        )

    export = aggregate.get("export", {})
    if not isinstance(export, Mapping):
        raise ConfigValidationError(
            f"{source_path}: aggregate.export must be a mapping"
        )
    export_formats = _string_tuple(
        export.get("formats", ["csv", "json"]),
        source_path=source_path,
        field="aggregate.export.formats",
    )
    unsupported = set(export_formats) - {
        "csv",
        "json",
        "markdown",
        "latex",
    }
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ConfigValidationError(
            f"{source_path}: unsupported aggregate export "
            f"format: {names}"
        )

    export_precision = export.get("precision", 4)
    if (
        isinstance(export_precision, bool)
        or not isinstance(export_precision, int)
        or export_precision < 0
    ):
        raise ConfigValidationError(
            f"{source_path}: aggregate.export.precision "
            "must be a nonnegative integer"
        )

    return AggregateSpec(
        runs=tuple(runs),
        group_by=group_by,
        metrics=metrics,
        expected_repeats=repeats,
        expected_cells=expected_cells,
        require_complete=require_complete,
        center=center,
        spread=spread,
        hierarchy_reduce=hierarchy_reduce,
        hierarchy_weighting=hierarchy_weighting,
        missing_policy=missing_policy,
        missing_reason=missing_reason,
        export_formats=export_formats,
        export_precision=export_precision,
    )


def _attach_aggregate_spec(
    manifest: Mapping[str, Any],
    *,
    source_path: Path,
) -> tuple[dict[str, Any], AggregateSpec | None]:
    resolved = deepcopy(dict(manifest))
    aggregate = resolved.get("aggregate")
    if aggregate is None:
        return resolved, None

    spec = _parse_aggregate_spec(
        aggregate,
        source_path=source_path,
    )
    resolved["aggregate"] = spec.to_dict()
    return resolved, spec


def _prepare_config(
    manifest: Mapping[str, Any],
    source_path: Path,
) -> tuple[
    dict[str, Any],
    ResolvedAttributeBank | None,
    AggregateSpec | None,
]:
    defaulted = _apply_defaults(manifest, source_path)
    gridded = _expand_known_grids(defaulted, source_path)
    with_bank, bank = _attach_attribute_bank(
        gridded,
        source_path,
    )
    with_aggregate, aggregate_spec = _attach_aggregate_spec(
        with_bank,
        source_path=source_path,
    )
    return with_aggregate, bank, aggregate_spec


def _require_mapping(
    config: Mapping[str, Any],
    field: str,
    source_path: Path,
) -> Mapping[str, Any]:
    value = config.get(field)
    if not isinstance(value, Mapping):
        raise ConfigValidationError(
            f"{source_path}: {field} is required and must be a mapping"
        )
    return value


def _require_data_fields(
    config: Mapping[str, Any],
    fields: Sequence[str],
    *,
    source_path: Path,
    verb: str,
) -> Mapping[str, Any]:
    data = _require_mapping(
        config,
        "data",
        source_path,
    )
    for field in fields:
        if field not in data:
            raise ConfigValidationError(
                f"{source_path}: data.{field} is required for {verb}"
            )
    return data


def _require_metrics(
    config: Mapping[str, Any],
    source_path: Path,
) -> tuple[str, ...]:
    return _string_tuple(
        config.get("metrics"),
        source_path=source_path,
        field="metrics",
    )


def _require_backend(
    config: Mapping[str, Any],
    source_path: Path,
) -> None:
    backend = config.get("backend")
    if isinstance(backend, Mapping):
        if not backend:
            raise ConfigValidationError(
                f"{source_path}: backend must not be empty"
            )
        return
    if isinstance(backend, str) and backend.strip():
        if not Path(backend).exists():
            raise ConfigValidationError(
                f"{source_path}: backend profile does not exist"
            )
        return
    raise ConfigValidationError(
        f"{source_path}: backend is required"
    )


def _validate_generation_requirements(
    config: "Config",
    *,
    require_survey: bool,
) -> Mapping[str, Any]:
    if require_survey:
        _require_data_fields(
            config.data,
            ("survey",),
            source_path=config.source_path,
            verb="generate",
        )

    generation = _require_mapping(
        config.data,
        "generation",
        config.source_path,
    )
    _require_backend(
        config.data,
        config.source_path,
    )

    mode = generation.get("mode", "active")
    if mode not in {"active", "vanilla", "plugin"}:
        raise ConfigValidationError(
            f"{config.source_path}: generation.mode must be "
            "active, vanilla, or plugin"
        )
    if mode == "active" and config.attribute_bank is None:
        raise ConfigValidationError(
            f"{config.source_path}: active generation requires "
            "an attribute bank"
        )
    return generation


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable resolved central configuration."""

    source_path: Path
    data: Mapping[str, Any]
    attribute_bank: ResolvedAttributeBank | None = None
    aggregate_spec: AggregateSpec | None = None

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Config":
        source_path = Path(path).resolve()
        loaded = _load_manifest(source_path, ())
        resolved, bank, aggregate_spec = _prepare_config(
            loaded,
            source_path,
        )
        return cls(
            source_path=source_path,
            data=_freeze(resolved),
            attribute_bank=bank,
            aggregate_spec=aggregate_spec,
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        declared_in: str | os.PathLike[str],
    ) -> "Config":
        source_path = Path(declared_in).resolve()
        raw = _validate_root_mapping(mapping, source_path)
        if "extends" in raw:
            raise ConfigLoadError(
                f"{source_path}: from_mapping does not accept extends"
            )

        referenced = _resolve_references(raw, source_path)
        resolved, bank, aggregate_spec = _prepare_config(
            referenced,
            source_path,
        )
        return cls(
            source_path=source_path,
            data=_freeze(resolved),
            attribute_bank=bank,
            aggregate_spec=aggregate_spec,
        )

    @property
    def digest(self) -> str:
        """Return a stable digest of the resolved configuration."""
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def write_snapshot(
        self,
        run_dir: str | os.PathLike[str],
    ) -> Path:
        """Write the resolved configuration snapshot."""
        destination = Path(run_dir)
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_path = destination / "config.resolved.yaml"
        snapshot_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "source": str(self.source_path),
                    "config_digest": self.digest,
                    "config": self.to_dict(),
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        return snapshot_path

    def validate_for(self, verb: str) -> "Config":
        """Validate the authoritative verb requirement matrix."""
        if not isinstance(verb, str) or not verb.strip():
            raise ConfigValidationError(
                "verb must be a nonempty string"
            )

        if verb == "generate":
            _validate_generation_requirements(
                self,
                require_survey=True,
            )
            return self

        if verb == "pipeline":
            _validate_generation_requirements(
                self,
                require_survey=True,
            )
            _require_data_fields(
                self.data,
                ("target",),
                source_path=self.source_path,
                verb="pipeline",
            )
            _require_mapping(
                self.data,
                "split",
                self.source_path,
            )
            _require_mapping(
                self.data,
                "model",
                self.source_path,
            )
            _require_metrics(
                self.data,
                self.source_path,
            )
            return self

        if verb == "regress":
            _require_data_fields(
                self.data,
                ("survey", "basis", "target"),
                source_path=self.source_path,
                verb="regress",
            )
            _require_mapping(
                self.data,
                "split",
                self.source_path,
            )
            _require_mapping(
                self.data,
                "model",
                self.source_path,
            )
            _require_metrics(
                self.data,
                self.source_path,
            )
            return self

        if verb == "evaluate":
            data = _require_data_fields(
                self.data,
                ("survey", "target"),
                source_path=self.source_path,
                verb="evaluate",
            )
            if "predictions" not in data and "fitted" not in data:
                raise ConfigValidationError(
                    f"{self.source_path}: data.predictions or "
                    "data.fitted is required for evaluate"
                )
            _require_mapping(
                self.data,
                "split",
                self.source_path,
            )
            _require_metrics(
                self.data,
                self.source_path,
            )
            if (
                "model" in self.data
                and not isinstance(self.data["model"], Mapping)
            ):
                raise ConfigValidationError(
                    f"{self.source_path}: model must be a mapping"
                )
            return self

        if verb == "aggregate":
            if self.aggregate_spec is None:
                raise ConfigValidationError(
                    f"{self.source_path}: aggregate section is "
                    "required for aggregate"
                )
            return self

        if verb == "simulate":
            simulation = _require_mapping(
                self.data,
                "simulation",
                self.source_path,
            )
            invokes_generation = simulation.get(
                "invokes_generation",
                False,
            )
            if not isinstance(invokes_generation, bool):
                raise ConfigValidationError(
                    f"{self.source_path}: "
                    "simulation.invokes_generation must be boolean"
                )
            if invokes_generation:
                _validate_generation_requirements(
                    self,
                    require_survey=False,
                )
            return self

        if verb in {"ablate", "panel"}:
            _require_mapping(
                self.data,
                verb,
                self.source_path,
            )
            return self

        raise ConfigValidationError(
            f"{self.source_path}: unsupported verb {verb!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.data)


__all__ = [
    "AggregateRun",
    "AggregateSpec",
    "Config",
    "ConfigError",
    "ConfigLoadError",
    "ConfigValidationError",
    "ResolvedAttributeBank",
    "expand_grid",
]