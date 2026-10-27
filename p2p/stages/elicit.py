"""Survey-elicitation stage and response artifact writing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from p2p.agents import SurveyConductor
from p2p.config import Config
from p2p.endowments import Endowments
from p2p.responses import Responses
from p2p.stages._backend import load_backend_config
from p2p.survey import Survey
from p2p.usage import UsageTracker


@dataclass(frozen=True, slots=True)
class ElicitationResult:
    """In-memory elicitation result and its response artifact."""

    survey: Survey
    endowments: Endowments
    responses: Responses
    records: tuple[Mapping[str, Any], ...]
    responses_path: Path
    reused_generation_responses: bool = False
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


def _flat_records(
    responses: Responses,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "qid": qid,
            "eid": eid,
            "answer": answer,
        }
        for qid, answers in responses.responses.items()
        for eid, answer in answers.items()
    )


def _reuse_generated_responses(
    generation,
    *,
    survey: Survey,
    endowments: Endowments,
    tracker: UsageTracker,
) -> ElicitationResult | None:
    responses = getattr(
        generation,
        "responses",
        None,
    )
    responses_path = getattr(
        generation,
        "responses_path",
        None,
    )
    if responses is None and responses_path is None:
        return None
    if not isinstance(responses, Responses):
        raise TypeError(
            "generation responses must be Responses"
        )
    if not isinstance(responses_path, Path):
        raise TypeError(
            "generation responses_path must be Path"
        )
    if not responses_path.is_file():
        raise ValueError(
            "generation responses artifact does not exist: "
            f"{responses_path}"
        )

    expected_eids = set(
        endowments.get_eids_by_role(None)
    )
    expected_qids = {
        question["id"]
        for question in survey.questions
    }
    observed_qids = set(responses.responses)
    if observed_qids != expected_qids:
        raise ValueError(
            "generated response QID coverage does not match "
            "the survey"
        )
    for qid in expected_qids:
        if set(responses.responses[qid]) != expected_eids:
            raise ValueError(
                "generated response EID coverage does not match "
                f"the final endowment pool for {qid!r}"
            )

    return ElicitationResult(
        survey=survey,
        endowments=endowments,
        responses=responses,
        records=_flat_records(responses),
        responses_path=responses_path.resolve(),
        reused_generation_responses=True,
        usage_tracker=tracker,
    )


def run(
    config: Config,
    *,
    generation,
    run_dir: str | os.PathLike[str],
    verbose: bool = False,
    conductor_factory=SurveyConductor,
    usage_tracker: UsageTracker | None = None,
) -> ElicitationResult:
    """Elicit responses from generated endowments and save a matrix CSV."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")

    survey = getattr(generation, "survey", None)
    endowments = getattr(generation, "endowments", None)

    if not isinstance(survey, Survey):
        raise TypeError(
            "generation result must provide a Survey"
        )
    if not isinstance(endowments, Endowments):
        raise TypeError(
            "generation result must provide Endowments"
        )

    resolved_run_dir = _resolved_run_dir(run_dir)
    tracker = (
        usage_tracker
        or getattr(
            generation,
            "usage_tracker",
            None,
        )
        or UsageTracker(
            stream_path=resolved_run_dir / "usage.csv",
        )
    )
    reused = _reuse_generated_responses(
        generation,
        survey=survey,
        endowments=endowments,
        tracker=tracker,
    )
    if reused is not None:
        return reused

    backend = load_backend_config(config)
    response_backend = _require_mapping(
        backend.get("response"),
        field="backend.response",
    )

    agent_type = response_backend.get("type")
    model_name = response_backend.get("model_name")
    formality = response_backend.get("formality", "")
    clean = response_backend.get("clean", False)
    agent_kwargs = response_backend.get("kwargs", {})

    if not isinstance(agent_type, str) or not agent_type:
        raise ValueError(
            "backend.response.type must be a nonempty string"
        )
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(
            "backend.response.model_name must be a nonempty string"
        )
    if not isinstance(formality, str):
        raise ValueError(
            "backend.response.formality must be a string"
        )
    if not isinstance(clean, bool):
        raise ValueError(
            "backend.response.clean must be boolean"
        )
    agent_kwargs = _require_mapping(
        agent_kwargs,
        field="backend.response.kwargs",
    )

    conductor = conductor_factory(
        survey=survey,
        endowments=endowments,
        agent_type=agent_type,
        model_name=model_name,
        formality=formality,
        clean=clean,
        verbose=verbose,
        token_tracker=tracker,
    )
    records = conductor.run(
        save=False,
        **dict(agent_kwargs),
    )
    if not isinstance(records, list):
        raise TypeError(
            "conductor.run(...) must return a list"
        )

    responses = Responses(
        source=records,
        survey=survey,
        output_format="answer",
        clean=False,
    )

    responses_path = (
        resolved_run_dir / "responses.csv"
    ).resolve()
    responses.save(str(responses_path))

    return ElicitationResult(
        survey=survey,
        endowments=endowments,
        responses=responses,
        records=tuple(records),
        responses_path=responses_path,
        reused_generation_responses=False,
        usage_tracker=tracker,
    )


__all__ = [
    "ElicitationResult",
    "run",
]
