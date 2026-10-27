"""Endowment-generation stage dispatch and artifact writing."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass, field
import inspect
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

from p2p.agents.factory import create_backend
from p2p.agents import AgentConfig, SurveyConductor
from p2p.config import Config
from p2p.endowments import (
    ActiveEndowments,
    Endowments,
)
from p2p.generators import (
    ActiveGenerationCheckpoint,
    ActiveEndowmentGenerator,
    AttributeLearner,
    EndowmentGenerator,
    EndowmentModel,
)
from p2p.registry import STAGES, Registry
from p2p.responses import Responses
from p2p.stages._backend import load_backend_config
from p2p.survey import Survey
from p2p.usage import UsageTracker


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """In-memory generation result and its basis artifact."""

    mode: str
    survey: Survey
    endowments: Endowments
    endowments_path: Path
    attribute_bank_snapshot: Path | None = None
    responses: Responses | None = None
    responses_path: Path | None = None
    response_records: tuple[Mapping[str, Any], ...] = ()
    diagnostic_paths: Mapping[str, Path] = field(
        default_factory=lambda: MappingProxyType({})
    )
    usage_tracker: UsageTracker | None = None


def _require_mapping(
    value: Any,
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _resolved_run_dir(
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


def _positive_integer(
    value: Any,
    *,
    field: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{field} must be a positive integer"
        )
    return value


def load_survey(
    config: Config,
) -> Survey:
    """Load the survey referenced by a resolved generation config."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")
    config.validate_for("generate")

    data = _require_mapping(
        config.data.get("data"),
        field="data",
    )
    survey_config = _require_mapping(
        data.get("survey"),
        field="data.survey",
    )

    source = survey_config.get("source")
    schema = survey_config.get("schema")
    if not isinstance(source, str) or not source:
        raise ValueError(
            "data.survey.source must be a nonempty path"
        )
    if not isinstance(schema, str) or not schema:
        raise ValueError(
            "data.survey.schema must be a nonempty path"
        )

    return Survey(
        csv_path=Path(source).resolve(),
        config_path=Path(schema).resolve(),
    )


def _generate_vanilla(
    *,
    config: Config,
    survey: Survey,
    verbose: bool,
    usage_tracker: UsageTracker,
) -> ActiveEndowments:
    generation = _require_mapping(
        config.data.get("generation"),
        field="generation",
    )
    backend = load_backend_config(config)
    endowment_backend = _require_mapping(
        backend.get("endowment"),
        field="backend.endowment",
    )

    backend_type = endowment_backend.get("type")
    model_name = endowment_backend.get("model_name")
    backend_kwargs = _require_mapping(
        endowment_backend.get("kwargs", {}),
        field="backend.endowment.kwargs",
    )

    if not isinstance(backend_type, str) or not backend_type:
        raise ValueError(
            "backend.endowment.type must be a nonempty string"
        )
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(
            "backend.endowment.model_name must be a nonempty string"
        )

    target_n = _positive_integer(
        generation.get("target_n"),
        field="generation.target_n",
    )

    topic_description = generation.get(
        "topic_description"
    )
    if (
        topic_description is not None
        and not isinstance(topic_description, str)
    ):
        raise ValueError(
            "generation.topic_description must be a string or null"
        )

    generator = EndowmentGenerator(
        backend_type=backend_type,
        model_name=model_name,
        temperature=generation.get(
            "temperature",
            0.9,
        ),
        max_tokens=generation.get(
            "max_tokens",
            1024,
        ),
        batch_size=generation.get(
            "batch_size",
            10,
        ),
        delay=generation.get(
            "delay",
            0.5,
        ),
        retry_failed=generation.get(
            "retry_failed",
            True,
        ),
        clear_after_run=False,
        token_tracker=usage_tracker,
        verbose=verbose,
        max_retries=generation.get(
            "max_retries",
            10,
        ),
        retry_base_delay=generation.get(
            "retry_base_delay",
            10.0,
        ),
        backend_factory=create_backend,
        **dict(backend_kwargs),
    )
    personas = generator.generate(
        target_n,
        topic_description=topic_description,
    )

    if len(personas) != target_n:
        raise ValueError(
            "vanilla generation returned "
            f"{len(personas)} personas; expected {target_n}"
        )

    normalized = []
    for index, persona in enumerate(personas):
        if not isinstance(persona, Mapping):
            raise TypeError(
                "generated personas must be mappings"
            )

        endow_text = persona.get("endow_text")
        if (
            not isinstance(endow_text, str)
            or not endow_text.strip()
        ):
            raise ValueError(
                "generated persona endow_text must be "
                "a nonempty string"
            )

        normalized.append(
            {
                "eid": f"vanilla_{index:04d}",
                "endow_text": endow_text.strip(),
                "role": "proxy",
                "weight": None,
                "mode": ("vanilla",),
                "attributes": [],
            }
        )

    return ActiveEndowments.from_endowment_list(
        normalized
    )


def _backend_component(
    backend: Mapping[str, Any],
    name: str,
):
    component = _require_mapping(
        backend.get(name),
        field=f"backend.{name}",
    )
    backend_type = component.get("type")
    model_name = component.get("model_name")
    kwargs = _require_mapping(
        component.get("kwargs", {}),
        field=f"backend.{name}.kwargs",
    )
    if (
        not isinstance(backend_type, str)
        or not backend_type
        or not isinstance(model_name, str)
        or not model_name
    ):
        raise ValueError(
            f"backend.{name} type and model_name "
            "must be nonempty strings"
        )
    return create_backend(
        backend_type=backend_type,
        model_name=model_name,
        **dict(kwargs),
    )


def _generate_active(
    *,
    config: Config,
    survey: Survey,
    verbose: bool,
    usage_tracker: UsageTracker,
    run_dir: Path,
) -> ActiveEndowments:
    generation = _require_mapping(
        config.data.get("generation"),
        field="generation",
    )
    backend = load_backend_config(config)
    endowment_settings = _require_mapping(
        generation.get("endowment_model", {}),
        field="generation.endowment_model",
    )
    attribute_settings = _require_mapping(
        generation.get("attribute_learner", {}),
        field="generation.attribute_learner",
    )
    checkpoint = ActiveGenerationCheckpoint(
        run_dir=run_dir,
        config_digest=config.digest,
        survey=survey,
    )
    resume_state = checkpoint.load()
    if resume_state is None and usage_tracker.records:
        raise RuntimeError(
            "This output directory contains paid usage records but no "
            "compatible active-generation checkpoint. Refusing to repeat "
            "API work. Preserve this directory and rerun with a fresh "
            "--output path."
        )
    response = _require_mapping(
        backend.get("response"),
        field="backend.response",
    )
    response_kwargs = _require_mapping(
        response.get("kwargs", {}),
        field="backend.response.kwargs",
    )
    agent_config = AgentConfig(
        agent_type=response.get("type"),
        model_name=response.get("model_name"),
        formality=response.get("formality", ""),
        agent_kwargs=response_kwargs,
        delay=response.get("delay", 0.0),
    )
    if config.attribute_bank is None:
        raise ValueError(
            "active generation requires an attribute bank"
        )

    generator = ActiveEndowmentGenerator(
        survey=survey,
        endowment_model=EndowmentModel(
            backend=_backend_component(
                backend,
                "endowment",
            ),
            tracker=usage_tracker,
            batch_size=endowment_settings.get(
                "batch_size",
                10,
            ),
            delay=endowment_settings.get(
                "delay",
                0.5,
            ),
            retry_failed=endowment_settings.get(
                "retry_failed",
                True,
            ),
            randomize_attributes=endowment_settings.get(
                "randomize_attributes",
                True,
            ),
            max_attributes=endowment_settings.get(
                "max_attributes",
                12,
            ),
            seed=config.data["run"].get("seed", 101),
            max_retries=endowment_settings.get(
                "max_retries",
                1,
            ),
            retry_base_delay=endowment_settings.get(
                "retry_base_delay",
                0.0,
            ),
        ),
        attribute_learner=AttributeLearner(
            survey=survey,
            backend=_backend_component(
                backend,
                "attribute_learner",
            ),
            tracker=usage_tracker,
            max_attributes=attribute_settings.get(
                "max_attributes"
            ),
            max_retries=attribute_settings.get(
                "max_retries",
                2,
            ),
            retry_base_delay=attribute_settings.get(
                "retry_base_delay",
                0.5,
            ),
        ),
        agent_config=agent_config,
        attribute_bank=config.attribute_bank.to_dict(),
        conductor_factory=SurveyConductor,
        target_n=generation.get("target_n", 100),
        initial_n=generation.get("initial_n", 10),
        num_update_steps=generation.get(
            "num_update_steps",
            4,
        ),
        seed=config.data["run"].get("seed", 101),
        parallel=generation.get("parallel", False),
        max_workers=generation.get("max_workers", 10),
        verbose=verbose,
        question_patch=generation.get(
            "question_patch"
        ),
        token_tracker=usage_tracker,
        enable_survey_attributes=generation.get(
            "enable_survey_attributes",
            True,
        ),
        checkpoint_callback=checkpoint.save,
        resume_state=resume_state,
    )
    endowments = generator.generate()
    setattr(
        endowments,
        "_resolved_attribute_bank",
        generator.attribute_bank,
    )
    response_records = tuple(generator.response_log)
    responses = Responses.from_agent_records(
        list(response_records),
        survey,
        output_format="answer",
        clean=True,
    )
    setattr(
        endowments,
        "_generated_responses",
        responses,
    )
    setattr(
        endowments,
        "_generated_response_records",
        response_records,
    )
    setattr(
        endowments,
        "_entropy_trajectories",
        generator.tracker.get_entropy_trajectories(),
    )
    setattr(
        endowments,
        "_mode_history",
        tuple(generator.mode_history),
    )
    return endowments


def _invoke_strategy(
    strategy,
    *,
    config: Config,
    survey: Survey,
    verbose: bool,
    usage_tracker: UsageTracker,
    run_dir: Path,
) -> Endowments:
    candidates = {
        "config": config,
        "survey": survey,
        "verbose": verbose,
        "usage_tracker": usage_tracker,
        "run_dir": run_dir,
    }
    parameters = inspect.signature(strategy).parameters
    accepts_kwargs = any(
        parameter.kind
        is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    return strategy(
        **{
            name: value
            for name, value in candidates.items()
            if accepts_kwargs or name in parameters
        }
    )


def _write_series_csv(
    path: Path,
    series: Mapping[str, Any],
) -> Path:
    names = tuple(series)
    values = {
        name: tuple(series[name])
        for name in names
    }
    length = max(
        (len(items) for items in values.values()),
        default=0,
    )
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["step", *names],
        )
        writer.writeheader()
        for index in range(length):
            row: dict[str, Any] = {
                "step": index,
            }
            for name in names:
                if index < len(values[name]):
                    row[name] = values[name][index]
            writer.writerow(row)
    return path


def _write_active_diagnostics(
    endowments: Endowments,
    run_dir: Path,
) -> Mapping[str, Path]:
    trajectories = getattr(
        endowments,
        "_entropy_trajectories",
        None,
    )
    mode_history = getattr(
        endowments,
        "_mode_history",
        (),
    )
    if not isinstance(trajectories, Mapping):
        return MappingProxyType({})

    aggregate_path = _write_series_csv(
        (run_dir / "aggregate_entropy_trajectory.csv").resolve(),
        {
            "entropy": trajectories.get(
                "aggregate",
                (),
            ),
        },
    )
    by_mode = trajectories.get("by_mode", {})
    mode_entropy_path = _write_series_csv(
        (run_dir / "persistent_mode_entropy.csv").resolve(),
        {
            "+".join(mode): values
            for mode, values in by_mode.items()
        },
    )
    by_question = trajectories.get("by_question", {})
    question_entropy_path = _write_series_csv(
        (run_dir / "question_entropy_trajectories.csv").resolve(),
        dict(by_question),
    )

    modes: dict[str, dict[str, Any]] = {}
    for entry in mode_history:
        mode = "+".join(entry.get("mode", ()))
        summary = modes.setdefault(
            mode,
            {
                "n": 0,
                "attributes": set(),
            },
        )
        summary["n"] += entry.get("n", 0)
        summary["attributes"].update(
            entry.get("attributes", ())
        )

    mode_summary_path = (
        run_dir / "mode_summary.csv"
    ).resolve()
    with mode_summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "mode",
                "n",
                "attributes",
            ],
        )
        writer.writeheader()
        for mode in sorted(modes):
            writer.writerow(
                {
                    "mode": mode,
                    "n": modes[mode]["n"],
                    "attributes": "; ".join(
                        sorted(modes[mode]["attributes"])
                    ),
                }
            )

    return MappingProxyType(
        {
            "aggregate_entropy": aggregate_path,
            "mode_entropy": mode_entropy_path,
            "question_entropy": question_entropy_path,
            "mode_summary": mode_summary_path,
        }
    )


def run(
    config: Config,
    *,
    run_dir: str | os.PathLike[str],
    verbose: bool = False,
    strategy_registry: Registry = STAGES,
    usage_tracker: UsageTracker | None = None,
) -> GenerationResult:
    """Run one registered generation strategy and save its basis."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")

    survey = load_survey(config)
    generation = _require_mapping(
        config.data.get("generation"),
        field="generation",
    )

    mode = generation.get("mode", "active")
    if mode not in {"active", "vanilla", "plugin"}:
        raise ValueError(
            f"unknown generation mode {mode!r}"
        )

    strategy = strategy_registry.get(
        f"generate.{mode}"
    )
    resolved_run_dir = _resolved_run_dir(run_dir)
    tracker = usage_tracker or UsageTracker(
        stream_path=resolved_run_dir / "usage.csv",
        resume_stream=True,
    )
    endowments = _invoke_strategy(
        strategy,
        config=config,
        survey=survey,
        verbose=verbose,
        usage_tracker=tracker,
        run_dir=resolved_run_dir,
    )

    if not isinstance(endowments, Endowments):
        raise TypeError(
            "generation strategy must return Endowments"
        )
    if len(endowments) == 0:
        raise ValueError(
            "generation strategy returned no endowments"
        )

    endowments_path = (
        resolved_run_dir / "endowments.csv"
    ).resolve()
    endowments.save(str(endowments_path))

    responses = getattr(
        endowments,
        "_generated_responses",
        None,
    )
    responses_path = None
    response_records = tuple(
        getattr(
            endowments,
            "_generated_response_records",
            (),
        )
    )
    if responses is not None:
        if not isinstance(responses, Responses):
            raise TypeError(
                "generated responses must be Responses"
            )
        responses_path = (
            resolved_run_dir / "responses.csv"
        ).resolve()
        responses.save(str(responses_path))

    attribute_bank_snapshot = None
    if mode == "active" and config.attribute_bank is not None:
        attribute_bank_snapshot = (
            config.attribute_bank.write_snapshot(
                resolved_run_dir,
                attribute_bank=getattr(
                    endowments,
                    "_resolved_attribute_bank",
                    None,
                ),
            ).resolve()
        )

    diagnostic_paths = (
        _write_active_diagnostics(
            endowments,
            resolved_run_dir,
        )
        if mode == "active"
        else MappingProxyType({})
    )

    return GenerationResult(
        mode=mode,
        survey=survey,
        endowments=endowments,
        endowments_path=endowments_path,
        attribute_bank_snapshot=(
            attribute_bank_snapshot
        ),
        responses=responses,
        responses_path=responses_path,
        response_records=response_records,
        diagnostic_paths=diagnostic_paths,
        usage_tracker=tracker,
    )


STAGES.register(
    "generate.vanilla",
    _generate_vanilla,
)
STAGES.register(
    "generate.active",
    _generate_active,
)


__all__ = [
    "GenerationResult",
    "load_survey",
    "run",
]
