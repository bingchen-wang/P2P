"""Shared backend-profile resolution for execution stages."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

import yaml

from p2p.config import Config


def _load_profile(
    value: Any,
) -> dict[str, Any]:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "backend profile must be a nonempty path"
        )

    path = Path(value).resolve()
    try:
        loaded = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )
    except OSError as error:
        raise ValueError(
            f"could not read backend profile {path}: {error}"
        ) from error
    except yaml.YAMLError as error:
        raise ValueError(
            f"invalid backend profile {path}: {error}"
        ) from error

    if not isinstance(loaded, Mapping):
        raise ValueError(
            "backend profile root must be a mapping"
        )
    return dict(loaded)


def _merge_sections(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(base)

    for key, value in override.items():
        existing = merged.get(key)
        if (
            isinstance(existing, Mapping)
            and isinstance(value, Mapping)
        ):
            section = dict(existing)
            section.update(value)
            merged[key] = section
        else:
            merged[key] = value

    return merged


def load_backend_config(
    config: Config,
) -> dict[str, Any]:
    """Resolve an inline backend config or referenced profile."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")

    backend = config.data.get("backend")

    if isinstance(backend, Mapping):
        inline = dict(backend)
        profile = inline.pop("profile", None)
        if profile is None:
            return inline
        return _merge_sections(
            _load_profile(profile),
            inline,
        )

    return _load_profile(backend)


__all__ = ["load_backend_config"]