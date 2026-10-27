"""Thin command wrapper over verb-scoped configuration validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from p2p.config import Config


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Identity of one successfully validated configuration."""

    target_verb: str
    config_digest: str
    source_path: Path


def run(
    config: Config,
    *,
    target_verb: str,
    verbose: bool = False,
) -> ValidationResult:
    """Validate one resolved config for the requested target verb."""

    del verbose
    if not isinstance(config, Config):
        raise TypeError("config must be Config")

    config.validate_for(target_verb)
    return ValidationResult(
        target_verb=target_verb,
        config_digest=config.digest,
        source_path=config.source_path,
    )


__all__ = [
    "ValidationResult",
    "run",
]
