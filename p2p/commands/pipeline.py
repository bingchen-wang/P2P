"""Production full-pipeline command composition."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import logging
import os
from pathlib import Path
from typing import Any

from p2p.commands import regress as regress_command
from p2p.config import Config
from p2p.stages import elicit as elicit_stage
from p2p.stages import generate as generate_stage
from p2p.stages import report as report_stage
from p2p.usage import (
    UsageArtifacts,
    UsageTracker,
    write_usage_artifacts,
)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """One completed in-memory full-pipeline execution."""

    config_digest: str
    run_dir: Path
    generation: Any
    elicitation: Any
    regression: Any
    report: Any
    usage: UsageArtifacts | None = None
    log_path: Path | None = None


def _run_directory(
    value: str | os.PathLike[str],
) -> Path:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "run_dir must be a nonempty path"
        )

    run_dir = Path(value).resolve()
    if run_dir.exists() and not run_dir.is_dir():
        raise ValueError(
            f"run_dir is not a directory: {run_dir}"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _artifact_path(
    result: Any,
    *,
    field: str,
    stage: str,
) -> Path:
    value = getattr(result, field, None)
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"{stage} result must provide {field}"
        )

    path = Path(value).resolve()
    if not path.is_file():
        raise RuntimeError(
            f"{stage} result artifact does not exist: {path}"
        )
    return path


@contextmanager
def _pipeline_logger(log_path: Path):
    package_logger = logging.getLogger("p2p")
    previous_level = package_logger.level
    handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        )
    )
    package_logger.addHandler(handler)
    if previous_level == logging.NOTSET or (
        previous_level > logging.INFO
    ):
        package_logger.setLevel(logging.INFO)
    try:
        yield logging.getLogger(__name__)
    finally:
        package_logger.removeHandler(handler)
        handler.close()
        package_logger.setLevel(previous_level)


def _call_with_usage(
    function,
    config: Config,
    *,
    usage_tracker: UsageTracker,
    **kwargs,
):
    parameters = inspect.signature(
        function
    ).parameters
    if (
        "usage_tracker" in parameters
        or any(
            parameter.kind
            is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
    ):
        kwargs["usage_tracker"] = usage_tracker
    return function(
        config,
        **kwargs,
    )


def run(
    config: Config,
    *,
    solver: str,
    run_dir: str | os.PathLike[str],
    verbose: bool = False,
) -> PipelineResult:
    """Generate, elicit, regress, and report one run."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")
    config.validate_for("pipeline")

    resolved_run_dir = _run_directory(run_dir)
    log_path = (
        resolved_run_dir / "pipeline.log"
    ).resolve()
    usage_tracker = UsageTracker(
        stream_path=resolved_run_dir / "usage.csv",
        resume_stream=True,
    )

    with _pipeline_logger(log_path) as logger:
        logger.info(
            "[Pipeline] Output directory: %s",
            resolved_run_dir,
        )
        logger.info("[Pipeline] Running endowment generation")
        try:
            generation = _call_with_usage(
                generate_stage.run,
                config,
                usage_tracker=usage_tracker,
                run_dir=resolved_run_dir,
                verbose=verbose,
            )
            endowments_path = _artifact_path(
                generation,
                field="endowments_path",
                stage="generation",
            )
            logger.info(
                "[Pipeline] Generated %s endowments",
                (
                    len(generation.endowments)
                    if hasattr(
                        generation,
                        "endowments",
                    )
                    else "unknown number of"
                ),
            )

            logger.info("[Pipeline] Preparing elicited responses")
            elicitation = _call_with_usage(
                elicit_stage.run,
                config,
                usage_tracker=usage_tracker,
                generation=generation,
                run_dir=resolved_run_dir,
                verbose=verbose,
            )
            responses_path = _artifact_path(
                elicitation,
                field="responses_path",
                stage="elicitation",
            )
            if getattr(
                elicitation,
                "reused_generation_responses",
                False,
            ):
                logger.info(
                    "[Pipeline] Reused responses collected during "
                    "active generation"
                )

            logger.info("[Pipeline] Running regression and evaluation")
            regression = regress_command.run(
                config,
                solver=solver,
                verbose=verbose,
                basis={
                    "responses": responses_path,
                    "endowments": endowments_path,
                },
            )

            logger.info("[Pipeline] Writing canonical report")
            canonical_report = (
                report_stage.write_regression_report(
                    config,
                    regression,
                    run_dir=resolved_run_dir,
                )
            )
            usage = write_usage_artifacts(
                usage_tracker,
                resolved_run_dir,
            )
            logger.info(
                "[Pipeline] Usage: %s",
                usage_tracker.report(),
            )
            logger.info("[Pipeline] All stages complete")
        except Exception:
            logger.exception("[Pipeline] Run failed")
            raise

    return PipelineResult(
        config_digest=config.digest,
        run_dir=resolved_run_dir,
        generation=generation,
        elicitation=elicitation,
        regression=regression,
        report=canonical_report,
        usage=usage,
        log_path=log_path,
    )


__all__ = [
    "PipelineResult",
    "run",
]
