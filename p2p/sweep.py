"""Generic deterministic sweep expansion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from p2p.config import (
    Config,
    ConfigValidationError,
    expand_grid,
)


_SUPPORTED_KINDS = {
    "ablate",
    "panel",
    "simulate",
}


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


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = deepcopy(value)
    return result


def _get_dotted(
    mapping: Mapping[str, Any],
    dotted_path: str,
) -> Any:
    if not isinstance(dotted_path, str) or not dotted_path.strip():
        raise ConfigValidationError(
            "sweep over must be a nonempty dotted path"
        )

    parts = dotted_path.split(".")
    if any(not part for part in parts):
        raise ConfigValidationError(
            f"invalid sweep path {dotted_path!r}"
        )

    current: Any = mapping
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigValidationError(
                f"sweep path {dotted_path!r} does not exist "
                "in the resolved base config"
            )
        current = current[part]

    return current


def _dotted_override(
    dotted_path: str,
    value: Any,
) -> dict[str, Any]:
    parts = dotted_path.split(".")
    nested: Any = deepcopy(value)

    for part in reversed(parts):
        nested = {
            part: nested,
        }

    return nested


def _normalize_shorthand_overrides(
    config: Config,
    section: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    over = section.get("over")
    if not isinstance(over, str) or not over.strip():
        raise ConfigValidationError(
            "sweep over must be a nonempty dotted path"
        )

    existing = _get_dotted(config.data, over)

    try:
        values = expand_grid(section.get("values"))
    except ConfigValidationError as error:
        raise ConfigValidationError(
            f"invalid sweep grid for {over}: {error}"
        ) from error

    overrides = []
    for value in values:
        replacement = deepcopy(value)
        if (
            isinstance(existing, Sequence)
            and not isinstance(existing, (str, bytes, bytearray))
            and not (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
            )
        ):
            replacement = [replacement]

        overrides.append(
            _freeze(
                _dotted_override(
                    over,
                    replacement,
                )
            )
        )

    return tuple(overrides)


def _normalize_explicit_overrides(
    value: Any,
) -> tuple[Mapping[str, Any], ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
    ):
        raise ConfigValidationError(
            "sweep overrides must be a nonempty list"
        )

    overrides = []
    for index, override in enumerate(value):
        if not isinstance(override, Mapping) or not override:
            raise ConfigValidationError(
                f"sweep overrides[{index}] must be a "
                "nonempty mapping"
            )
        overrides.append(_freeze(dict(override)))

    return tuple(overrides)


def _get_sweep_section(
    config: Config,
    kind: str,
) -> Mapping[str, Any]:
    if kind not in _SUPPORTED_KINDS:
        raise ConfigValidationError(
            f"unknown sweep kind {kind!r}"
        )

    if kind in {"ablate", "panel"}:
        section = config.data.get(kind)
    else:
        simulation = config.data.get("simulation")
        if not isinstance(simulation, Mapping):
            raise ConfigValidationError(
                "simulate sweep requires a simulation mapping"
            )
        section = simulation.get("sweep")

    if not isinstance(section, Mapping):
        raise ConfigValidationError(
            f"{kind} sweep configuration must be a mapping"
        )

    return section


@dataclass(frozen=True, slots=True)
class SweepSpec:
    """Normalized sweep metadata over a resolved base config."""

    kind: str
    target_verb: str
    overrides: tuple[Mapping[str, Any], ...]
    repeats: int = 1

    @classmethod
    def from_config(
        cls,
        config: Config,
        kind: str,
    ) -> "SweepSpec":
        """Normalize one ablate, panel, or simulation sweep."""
        section = _get_sweep_section(config, kind)

        target = section.get("target")
        if not isinstance(target, str) or not target.strip():
            raise ConfigValidationError(
                f"{kind} sweep target must be a nonempty verb"
            )

        repeats = section.get("repeats", 1)
        if (
            isinstance(repeats, bool)
            or not isinstance(repeats, int)
            or repeats <= 0
        ):
            raise ConfigValidationError(
                f"{kind} sweep repeats must be a positive integer"
            )

        has_shorthand = (
            "over" in section
            or "values" in section
        )
        has_explicit = "overrides" in section

        if has_shorthand == has_explicit:
            raise ConfigValidationError(
                f"{kind} sweep must define exactly one of "
                "over/values or overrides"
            )

        if has_shorthand:
            if "over" not in section or "values" not in section:
                raise ConfigValidationError(
                    f"{kind} sweep must define exactly one complete "
                    "over/values form or overrides"
                )
            overrides = _normalize_shorthand_overrides(
                config,
                section,
            )
        else:
            overrides = _normalize_explicit_overrides(
                section["overrides"]
            )

        return cls(
            kind=kind,
            target_verb=target,
            overrides=overrides,
            repeats=repeats,
        )


def _strip_meta_layer(
    base: Mapping[str, Any],
    kind: str,
) -> dict[str, Any]:
    stripped = deepcopy(dict(base))

    if kind in {"ablate", "panel"}:
        stripped.pop(kind, None)
        return stripped

    simulation = stripped.get("simulation")
    if isinstance(simulation, Mapping):
        simulation = deepcopy(dict(simulation))
        simulation.pop("sweep", None)
        stripped["simulation"] = simulation

    return stripped


def expand_sweep(
    base: Config,
    spec: SweepSpec,
) -> tuple[Config, ...]:
    """Expand overrides into resolved target-validated configs."""
    if not isinstance(base, Config):
        raise TypeError("base must be a Config")
    if not isinstance(spec, SweepSpec):
        raise TypeError("spec must be a SweepSpec")

    base_mapping = _strip_meta_layer(
        base.to_dict(),
        spec.kind,
    )
    run = base_mapping.get("run", {})
    if not isinstance(run, Mapping):
        raise ConfigValidationError(
            "resolved base run section must be a mapping"
        )

    base_seed = run.get("seed", 101)
    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
    ):
        raise ConfigValidationError(
            "resolved base run.seed must be an integer"
        )

    expanded = []
    for repeat in range(1, spec.repeats + 1):
        for frozen_override in spec.overrides:
            override = _thaw(frozen_override)
            candidate = _deep_merge(
                base_mapping,
                override,
            )

            candidate_run = candidate.get("run", {})
            if not isinstance(candidate_run, Mapping):
                raise ConfigValidationError(
                    "sweep override produced an invalid run section"
                )
            candidate_run = dict(candidate_run)
            candidate_run["repeat"] = repeat
            candidate_run["seed"] = base_seed + repeat - 1
            candidate["run"] = candidate_run

            config = Config.from_mapping(
                candidate,
                declared_in=base.source_path,
            )
            config.validate_for(spec.target_verb)
            expanded.append(config)

    return tuple(expanded)


__all__ = [
    "SweepSpec",
    "expand_sweep",
]