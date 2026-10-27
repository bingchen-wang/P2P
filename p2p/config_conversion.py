"""One-off migration of relevant legacy P2P configurations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import os
from pathlib import Path
import re
from typing import Any

import yaml

from p2p.config import (
    ConfigError,
    ConfigValidationError,
    expand_grid,
)


class ConfigConversionError(ConfigError):
    """Raised when a legacy configuration cannot be migrated safely."""


DEFAULT_METRICS = [
    "mse",
    "one_minus_jsd",
    "emd",
    "mcpa",
]

_SUPPORTED_FAMILIES = {
    "regress",
    "pipeline",
    "ablate",
    "panel",
    "simulate",
}

_NUMBER_PATTERN = (
    r"[+-]?(?:"
    r"(?:\d+(?:\.\d*)?)"
    r"|(?:\.\d+)"
    r")(?:[eE][+-]?\d+)?"
)

_GRID_PATTERN = re.compile(
    rf"^np\.(?P<operator>logspace|linspace)"
    rf"\(\s*(?P<start>{_NUMBER_PATTERN})"
    rf"\s*,\s*(?P<stop>{_NUMBER_PATTERN})"
    rf"\s*,\s*(?P<count>[+-]?\d+)\s*\)$"
)


def _read_legacy_yaml(path: Path) -> dict[str, Any]:
    source = path.resolve()
    if not source.is_file():
        raise ConfigConversionError(
            f"legacy configuration does not exist: {source}"
        )

    try:
        loaded = yaml.safe_load(
            source.read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as error:
        raise ConfigConversionError(
            f"could not read legacy configuration {source}: {error}"
        ) from error

    if not isinstance(loaded, Mapping):
        raise ConfigConversionError(
            f"{source}: legacy configuration root must be a mapping"
        )

    return dict(loaded)


def _require_mapping(
    mapping: Mapping[str, Any],
    field: str,
) -> Mapping[str, Any]:
    value = mapping.get(field)
    if not isinstance(value, Mapping):
        raise ConfigConversionError(
            f"legacy field {field!r} must be a mapping"
        )
    return value


def _require_nonempty_string(
    value: Any,
    field: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigConversionError(
            f"legacy field {field!r} must be a nonempty string"
        )
    return value


def _positive_integer(
    value: Any,
    field: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ConfigConversionError(
            f"legacy field {field!r} must be a positive integer"
        )
    return value


def _parse_number(value: str) -> int | float:
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    return float(value)


def _migrate_grid_expression(
    value: Any,
    *,
    field: str,
) -> Any:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        if not value:
            raise ConfigConversionError(
                f"legacy grid {field!r} must not be empty"
            )
        return deepcopy(list(value))

    if not isinstance(value, str):
        raise ConfigConversionError(
            f"legacy grid expression {field!r} must be a string, "
            "list, or structured grid"
        )

    match = _GRID_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ConfigConversionError(
            f"unsupported legacy grid expression for {field!r}: "
            f"{value!r}"
        )

    count = int(match.group("count"))
    if count <= 0:
        raise ConfigConversionError(
            f"legacy grid expression {field!r} requires a "
            "positive count"
        )

    return {
        match.group("operator"): [
            _parse_number(match.group("start")),
            _parse_number(match.group("stop")),
            count,
        ]
    }


def _select_method(
    legacy: Mapping[str, Any],
    requested: str | None,
) -> str:
    available = [
        name
        for name in ("lasso", "elasticnet")
        if isinstance(legacy.get(name), Mapping)
    ]

    if requested is None:
        if len(available) != 1:
            choices = ", ".join(available) or "<none>"
            raise ConfigConversionError(
                "legacy configuration contains an ambiguous model "
                f"set ({choices}); method must be specified"
            )
        return available[0]

    if requested not in {"lasso", "elasticnet"}:
        raise ConfigConversionError(
            f"unsupported migration method {requested!r}"
        )
    if requested not in available:
        raise ConfigConversionError(
            f"requested method {requested!r} is absent from "
            "the legacy configuration"
        )

    return requested


def _migrate_model(
    legacy: Mapping[str, Any],
    method: str,
) -> dict[str, Any]:
    source = _require_mapping(legacy, method)
    model: dict[str, Any] = {
        "method": method,
    }

    for key, value in source.items():
        if key == "alpha_expr":
            model["alpha"] = _migrate_grid_expression(
                value,
                field=f"{method}.alpha_expr",
            )
        elif key == "l1_ratio_expr":
            model["l1_ratio"] = _migrate_grid_expression(
                value,
                field=f"{method}.l1_ratio_expr",
            )
        else:
            model[key] = deepcopy(value)

    if "alpha" not in model:
        raise ConfigConversionError(
            f"legacy {method} configuration is missing alpha_expr"
        )
    if method == "elasticnet" and "l1_ratio" not in model:
        raise ConfigConversionError(
            "legacy elasticnet configuration is missing "
            "l1_ratio_expr"
        )

    return model


def _relative_reference(
    value: Any,
    *,
    field: str,
    source_root: Path,
    destination_dir: Path,
) -> str:
    raw = _require_nonempty_string(value, field)
    source_path = Path(raw)
    if not source_path.is_absolute():
        source_path = source_root / source_path

    resolved = source_path.resolve()
    if not resolved.exists():
        raise ConfigConversionError(
            f"legacy path {field!r} does not exist: {resolved}"
        )

    return os.path.relpath(
        resolved,
        destination_dir,
    )


def _migrate_run(
    legacy: Mapping[str, Any],
    family: str,
) -> dict[str, Any]:
    if isinstance(legacy.get("metadata"), Mapping):
        section_name = "metadata"
    elif isinstance(legacy.get("experiment"), Mapping):
        section_name = "experiment"
    else:
        raise ConfigConversionError(
            f"legacy {family} configuration requires metadata "
            "or experiment"
        )

    source = _require_mapping(
        legacy,
        section_name,
    )
    name = _require_nonempty_string(
        source.get("name"),
        f"{section_name}.name",
    )
    seed = source.get(
        "seed",
        source.get("random_seed", 101),
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigConversionError(
            f"legacy field {section_name}.seed must be an integer"
        )

    run: dict[str, Any] = {
        "name": name,
        "seed": seed,
    }
    for key in ("description", "version"):
        if key in source:
            run[key] = deepcopy(source[key])

    return run


def _migrate_survey(
    paths: Mapping[str, Any],
    *,
    source_root: Path,
    destination_dir: Path,
) -> dict[str, str]:
    return {
        "source": _relative_reference(
            paths.get("survey_csv"),
            field="paths.survey_csv",
            source_root=source_root,
            destination_dir=destination_dir,
        ),
        "schema": _relative_reference(
            paths.get("survey_yaml"),
            field="paths.survey_yaml",
            source_root=source_root,
            destination_dir=destination_dir,
        ),
    }


def _migrate_split(
    legacy: Mapping[str, Any],
) -> dict[str, Any]:
    source = _require_mapping(
        legacy,
        "split_settings",
    )
    split = {
        "train_val": deepcopy(
            source.get(
                "train_val_split",
                ["train", "valid"],
            )
        ),
        "test": deepcopy(
            source.get(
                "test_split",
                "test",
            )
        ),
    }
    if "use_proxy_only" in source:
        split["use_proxy_only"] = deepcopy(
            source["use_proxy_only"]
        )
    return split


def _migrate_response_backend(
    legacy: Mapping[str, Any],
) -> dict[str, Any]:
    source = _require_mapping(legacy, "agent")
    backend = {
        "type": _require_nonempty_string(
            source.get("type"),
            "agent.type",
        ),
        "model_name": _require_nonempty_string(
            source.get("model_name"),
            "agent.model_name",
        ),
    }
    for key in ("formality", "kwargs", "delay"):
        if key in source:
            backend[key] = deepcopy(source[key])
    return backend


def _migrate_component_backend(
    legacy: Mapping[str, Any],
    section_name: str,
    *,
    runtime_keys: Sequence[str] = (),
) -> dict[str, Any]:
    source = _require_mapping(
        legacy,
        section_name,
    )
    backend_type = source.get(
        "backend_type",
        source.get("type"),
    )
    model_name = source.get(
        "model_name",
        source.get("model"),
    )

    migrated: dict[str, Any] = {
        "type": _require_nonempty_string(
            backend_type,
            f"{section_name}.backend_type",
        ),
        "model_name": _require_nonempty_string(
            model_name,
            f"{section_name}.model_name",
        ),
    }

    excluded = {
        "backend_type",
        "type",
        "model_name",
        "model",
        *runtime_keys,
    }
    kwargs = {
        key: deepcopy(value)
        for key, value in source.items()
        if key not in excluded
    }
    if kwargs:
        migrated["kwargs"] = kwargs

    return migrated


def _migrate_regress(
    legacy: Mapping[str, Any],
    *,
    source_root: Path,
    destination_dir: Path,
    method: str,
) -> dict[str, Any]:
    paths = _require_mapping(legacy, "paths")

    return {
        "run": _migrate_run(
            legacy,
            "regress",
        ),
        "data": {
            "survey": _migrate_survey(
                paths,
                source_root=source_root,
                destination_dir=destination_dir,
            ),
            "basis": {
                "responses": _relative_reference(
                    paths.get("responses_csv"),
                    field="paths.responses_csv",
                    source_root=source_root,
                    destination_dir=destination_dir,
                ),
                "endowments": _relative_reference(
                    paths.get("endowments_csv"),
                    field="paths.endowments_csv",
                    source_root=source_root,
                    destination_dir=destination_dir,
                ),
            },
            "target": _relative_reference(
                paths.get("aggregate_json"),
                field="paths.aggregate_json",
                source_root=source_root,
                destination_dir=destination_dir,
            ),
        },
        "model": _migrate_model(
            legacy,
            method,
        ),
        "split": _migrate_split(legacy),
        "metrics": deepcopy(DEFAULT_METRICS),
    }


def _migrate_pipeline(
    legacy: Mapping[str, Any],
    *,
    source_root: Path,
    destination_dir: Path,
    method: str,
    family: str = "pipeline",
) -> dict[str, Any]:
    paths = _require_mapping(legacy, "paths")
    generation = deepcopy(
        dict(
            _require_mapping(
                legacy,
                "generation",
            )
        )
    )
    generation["mode"] = "active"
    generation["attribute_bank"] = {
        "source": _relative_reference(
            paths.get("attribute_bank"),
            field="paths.attribute_bank",
            source_root=source_root,
            destination_dir=destination_dir,
        )
    }
    legacy_endowment_model = _require_mapping(
        legacy,
        "endowment_model",
    )
    endowment_runtime_keys = (
        "batch_size",
        "delay",
        "retry_failed",
        "randomize_attributes",
        "max_attributes",
        "max_retries",
        "retry_base_delay",
    )
    generation["endowment_model"] = {
        key: deepcopy(legacy_endowment_model[key])
        for key in endowment_runtime_keys
        if key in legacy_endowment_model
    }
    legacy_attribute_learner = _require_mapping(
        legacy,
        "attribute_learner",
    )
    if "max_attributes" in legacy_attribute_learner:
        generation["attribute_learner"] = {
            "max_attributes": deepcopy(
                legacy_attribute_learner["max_attributes"]
            )
        }

    migrated = {
        "run": _migrate_run(
            legacy,
            family,
        ),
        "data": {
            "survey": _migrate_survey(
                paths,
                source_root=source_root,
                destination_dir=destination_dir,
            ),
            "target": _relative_reference(
                paths.get("aggregate_json"),
                field="paths.aggregate_json",
                source_root=source_root,
                destination_dir=destination_dir,
            ),
        },
        "generation": generation,
        "backend": {
            "response": _migrate_response_backend(
                legacy,
            ),
            "endowment": _migrate_component_backend(
                legacy,
                "endowment_model",
                runtime_keys=endowment_runtime_keys,
            ),
            "attribute_learner": _migrate_component_backend(
                legacy,
                "attribute_learner",
                runtime_keys=(
                    "max_attributes",
                ),
            ),
        },
        "model": _migrate_model(
            legacy,
            method,
        ),
        "split": _migrate_split(legacy),
        "metrics": deepcopy(DEFAULT_METRICS),
    }

    if isinstance(legacy.get("report"), Mapping):
        migrated["report"] = deepcopy(
            dict(legacy["report"])
        )

    return migrated


def _migrate_ablate(
    legacy: Mapping[str, Any],
    *,
    source_root: Path,
    destination_dir: Path,
    method: str,
) -> dict[str, Any]:
    migrated = _migrate_pipeline(
        legacy,
        source_root=source_root,
        destination_dir=destination_dir,
        method=method,
        family="ablate",
    )
    runner = _require_mapping(legacy, "runner")
    over = _require_nonempty_string(
        runner.get("over"),
        "runner.over",
    )
    values = _migrate_grid_expression(
        runner.get("values"),
        field="runner.values",
    )
    repeats = _positive_integer(
        runner.get("repeats", 1),
        "runner.repeats",
    )

    migrated["ablate"] = {
        "target": "pipeline",
        "over": over,
        "values": values,
        "repeats": repeats,
    }
    return migrated


def _replace_wave(
    value: Any,
    *,
    wave: str,
    field: str,
) -> str:
    raw = _require_nonempty_string(value, field)
    return raw.replace("{wave}", wave)


def _migrate_panel(
    legacy: Mapping[str, Any],
    *,
    source_root: Path,
    destination_dir: Path,
    method: str,
) -> dict[str, Any]:
    sweep = _require_mapping(legacy, "sweep")
    waves = sweep.get("waves")
    if (
        not isinstance(waves, Sequence)
        or isinstance(waves, (str, bytes, bytearray))
        or not waves
    ):
        raise ConfigConversionError(
            "legacy field 'sweep.waves' must be a nonempty list"
        )

    normalized_waves = [
        _require_nonempty_string(
            wave,
            "sweep.waves entry",
        )
        for wave in waves
    ]
    repeats = _positive_integer(
        sweep.get("repeats", 1),
        "sweep.repeats",
    )

    original_paths = _require_mapping(
        legacy,
        "paths",
    )
    first_wave = normalized_waves[0]

    base_legacy = deepcopy(dict(legacy))
    base_paths = deepcopy(dict(original_paths))
    for key in ("survey_csv", "aggregate_json"):
        base_paths[key] = _replace_wave(
            base_paths.get(key),
            wave=first_wave,
            field=f"paths.{key}",
        )
    base_legacy["paths"] = base_paths

    migrated = _migrate_pipeline(
        base_legacy,
        source_root=source_root,
        destination_dir=destination_dir,
        method=method,
        family="panel",
    )

    overrides = []
    for wave in normalized_waves:
        survey_source = _relative_reference(
            _replace_wave(
                original_paths.get("survey_csv"),
                wave=wave,
                field="paths.survey_csv",
            ),
            field=f"paths.survey_csv[{wave}]",
            source_root=source_root,
            destination_dir=destination_dir,
        )
        target = _relative_reference(
            _replace_wave(
                original_paths.get("aggregate_json"),
                wave=wave,
                field="paths.aggregate_json",
            ),
            field=f"paths.aggregate_json[{wave}]",
            source_root=source_root,
            destination_dir=destination_dir,
        )

        overrides.append(
            {
                "run": {
                    "wave": wave,
                },
                "data": {
                    "survey": {
                        "source": survey_source,
                    },
                    "target": target,
                },
            }
        )

    migrated["panel"] = {
        "target": "pipeline",
        "overrides": overrides,
        "repeats": repeats,
    }
    return migrated


def _migrate_simulate(
    legacy: Mapping[str, Any],
    *,
    source_root: Path,
    destination_dir: Path,
    method: str,
) -> dict[str, Any]:
    paths = _require_mapping(legacy, "paths")
    runner = _require_mapping(legacy, "runner")
    fraction_grid = _migrate_grid_expression(
        runner.get("fractions"),
        field="runner.fractions",
    )

    try:
        fraction_values = expand_grid(
            fraction_grid
        )
    except ConfigValidationError as error:
        raise ConfigConversionError(
            f"invalid runner.fractions grid: {error}"
        ) from error

    repeats = _positive_integer(
        runner.get(
            "n_repeats",
            runner.get("repeats", 1),
        ),
        "runner.n_repeats",
    )

    legacy_mode = deepcopy(
        runner.get("mode", "endowment")
    )
    role_assignment = legacy.get("experiment", {}).get(
        "role_assignment",
        {},
    )
    if not isinstance(role_assignment, Mapping):
        raise ConfigConversionError(
            "experiment.role_assignment must be a mapping"
        )
    simulation = {
        "executor": "scientific",
        "kind": legacy_mode,
        "invokes_generation": False,
        "target_source": "ground_truth",
        "ground_truth": {
            "source": "random",
            "proxy_ratio": deepcopy(
                role_assignment.get("proxy_ratio", 0.7)
            ),
        },
        "fraction": fraction_values[0],
        "sweep": {
            "target": "simulate",
            "over": "simulation.fraction",
            "values": fraction_grid,
            "repeats": repeats,
        },
    }
    for key in ("plot", "verbose"):
        if key in runner:
            simulation[key] = deepcopy(
                runner[key]
            )

    return {
        "run": _migrate_run(
            legacy,
            "simulate",
        ),
        "data": {
            "survey": _migrate_survey(
                paths,
                source_root=source_root,
                destination_dir=destination_dir,
            ),
            "basis": {
                "responses": _relative_reference(
                    paths.get("responses_csv"),
                    field="paths.responses_csv",
                    source_root=source_root,
                    destination_dir=destination_dir,
                ),
                "endowments": _relative_reference(
                    paths.get("endowments_csv"),
                    field="paths.endowments_csv",
                    source_root=source_root,
                    destination_dir=destination_dir,
                ),
            },
        },
        "model": _migrate_model(
            legacy,
            method,
        ),
        "split": _migrate_split(legacy),
        "metrics": deepcopy(DEFAULT_METRICS),
        "simulation": simulation,
    }


def convert_earlier_config(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    family: str,
    source_root: str | os.PathLike[str] | None = None,
    method: str | None = None,
    legacy_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Convert one earlier-format YAML configuration into a central manifest.

    ``source_root`` is the root that the earlier configuration's relative paths
    are resolved against. ``legacy_root`` is its earlier name and is still
    accepted.
    """
    if source_root is not None and legacy_root is not None:
        raise ConfigConversionError(
            "pass source_root or its earlier name legacy_root, not both"
        )
    if source_root is None:
        source_root = legacy_root
    if source_root is None:
        raise ConfigConversionError("source_root is required")
    if family not in _SUPPORTED_FAMILIES:
        raise ConfigConversionError(
            f"unsupported configuration family {family!r}"
        )

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    destination_dir = destination_path.parent
    root = Path(source_root).resolve()

    if not root.is_dir():
        raise ConfigConversionError(
            f"source_root does not exist: {root}"
        )

    legacy = _read_legacy_yaml(source_path)
    selected_method = _select_method(
        legacy,
        method,
    )

    if family == "regress":
        migrated = _migrate_regress(
            legacy,
            source_root=root,
            destination_dir=destination_dir,
            method=selected_method,
        )
    elif family == "pipeline":
        migrated = _migrate_pipeline(
            legacy,
            source_root=root,
            destination_dir=destination_dir,
            method=selected_method,
        )
    elif family == "ablate":
        migrated = _migrate_ablate(
            legacy,
            source_root=root,
            destination_dir=destination_dir,
            method=selected_method,
        )
    elif family == "panel":
        migrated = _migrate_panel(
            legacy,
            source_root=root,
            destination_dir=destination_dir,
            method=selected_method,
        )
    else:
        migrated = _migrate_simulate(
            legacy,
            source_root=root,
            destination_dir=destination_dir,
            method=selected_method,
        )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination_path.write_text(
        yaml.safe_dump(
            migrated,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    return destination_path


__all__ = [
    "ConfigConversionError",
    "convert_earlier_config",
]
