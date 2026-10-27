"""User-facing endowment-generation command composition."""

from __future__ import annotations

import os

from p2p.config import Config
from p2p.stages import generate as generate_stage
from p2p.usage import write_usage_artifacts


def run(
    config: Config,
    *,
    run_dir: str | os.PathLike[str],
    verbose: bool = False,
) -> generate_stage.GenerationResult:
    """Validate and execute one configured generation strategy."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")
    config.validate_for("generate")

    result = generate_stage.run(
        config,
        run_dir=run_dir,
        verbose=verbose,
    )
    config.write_snapshot(run_dir)
    if result.usage_tracker is not None:
        write_usage_artifacts(
            result.usage_tracker,
            run_dir,
        )
    return result


__all__ = [
    "run",
]
