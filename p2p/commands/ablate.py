"""Execute deterministic ablation sweeps over registered commands."""

from __future__ import annotations

import os

from p2p.models.constrained_model import DEFAULT_SOLVER
from p2p.config import AggregateSpec, Config
from p2p.registry import COMMANDS, Registry
from p2p.simulations.runner import (
    CommandHandler,
    SweepCommandResult,
    run_sweep,
)


def run(
    config: Config,
    *,
    run_dir: str | os.PathLike[str],
    solver: str = DEFAULT_SOLVER,
    verbose: bool = False,
    registry: Registry[CommandHandler] = COMMANDS,
    aggregation_spec: AggregateSpec | None = None,
) -> SweepCommandResult:
    """Expand and execute the configured ablation target."""

    return run_sweep(
        config,
        kind="ablate",
        run_dir=run_dir,
        solver=solver,
        verbose=verbose,
        registry=registry,
        aggregation_spec=aggregation_spec,
    )


__all__ = ["run"]
