"""User-facing elicitation from an existing endowment artifact."""

from __future__ import annotations

import os
from pathlib import Path

from p2p.config import Config
from p2p.endowments import ActiveEndowments
from p2p.stages import elicit as elicit_stage
from p2p.stages import generate as generate_stage
from p2p.usage import write_usage_artifacts


def _resolved_endowments_path(
    value: str | os.PathLike[str],
) -> Path:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "endowments_path must be a nonempty path"
        )

    path = Path(value).resolve()
    if not path.is_file():
        raise ValueError(
            f"endowments_path is not a file: {path}"
        )
    return path


def _load_generation_result(
    config: Config,
    *,
    endowments_path: str | os.PathLike[str],
) -> generate_stage.GenerationResult:
    """Reconstruct the stage input without generating new personas."""

    resolved_path = _resolved_endowments_path(
        endowments_path
    )
    survey = generate_stage.load_survey(config)
    endowments = ActiveEndowments.load(
        str(resolved_path)
    )
    if len(endowments) == 0:
        raise ValueError(
            "endowments_path must contain at least one endowment"
        )

    generation = config.data.get("generation")
    mode = (
        generation.get("mode", "active")
        if hasattr(generation, "get")
        else "active"
    )
    return generate_stage.GenerationResult(
        mode=mode,
        survey=survey,
        endowments=endowments,
        endowments_path=resolved_path,
    )


def run(
    config: Config,
    *,
    endowments_path: str | os.PathLike[str],
    run_dir: str | os.PathLike[str],
    verbose: bool = False,
) -> elicit_stage.ElicitationResult:
    """Elicit responses from an existing endowments CSV."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")

    generation = _load_generation_result(
        config,
        endowments_path=endowments_path,
    )
    result = elicit_stage.run(
        config,
        generation=generation,
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
