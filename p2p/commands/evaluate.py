"""Standalone evaluation over self-contained prediction artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

import joblib

from p2p.config import Config
from p2p.stages import report as report_stage
from p2p.stages.evaluate import (
    EvaluationResult,
    evaluate_predictions,
)
from p2p.stages.targets import load_targets, target_spec
from p2p.survey import BinaryExtendedSurvey, Survey


@dataclass(frozen=True, slots=True)
class EvaluationCommandResult:
    """One standalone evaluation execution and its artifacts."""

    config_digest: str
    run_dir: Path
    input_kind: str
    model_name: str
    cleaning_policy: str
    evaluations: tuple[EvaluationResult, ...]
    report: report_stage.ReportArtifacts


def _require_mapping(
    value: Any,
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _resolved_file(
    value: Any,
    *,
    field: str,
) -> Path:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty path")
    path = Path(value).resolve()
    if not path.is_file():
        raise ValueError(f"{field} file does not exist: {path}")
    return path


def _survey_paths(
    value: Any,
) -> tuple[Path, Path]:
    survey = _require_mapping(
        value,
        field="data.survey",
    )
    return (
        _resolved_file(
            survey.get("source"),
            field="data.survey.source",
        ),
        _resolved_file(
            survey.get("schema"),
            field="data.survey.schema",
        ),
    )


def _load_payload(
    path: Path,
    *,
    input_kind: str,
) -> Mapping[str, Any]:
    if (
        input_kind == "fitted"
        and path.suffix.lower() != ".json"
    ):
        loaded = joblib.load(path)
    else:
        try:
            loaded = json.loads(
                path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid evaluation input JSON {path}: {error}"
            ) from error

    if not isinstance(loaded, Mapping):
        raise ValueError(
            "evaluation input artifact must be a mapping"
        )
    if loaded.get("schema_version") != 1:
        raise ValueError(
            "evaluation input schema_version must be 1"
        )
    return loaded


def _prediction_payload(
    value: Any,
) -> Mapping[str, Mapping[str, float]]:
    predictions = _require_mapping(
        value,
        field="predictions",
    )
    if not predictions:
        raise ValueError("predictions must not be empty")

    normalized: dict[str, Mapping[str, float]] = {}
    for qid, distribution in predictions.items():
        if not isinstance(qid, str) or not qid:
            raise ValueError(
                "prediction QIDs must be nonempty strings"
            )
        distribution = _require_mapping(
            distribution,
            field=f"predictions.{qid}",
        )
        if not distribution:
            raise ValueError(
                f"predictions.{qid} must not be empty"
            )

        normalized_distribution: dict[str, float] = {}
        for label, value in distribution.items():
            if not isinstance(label, str) or not label:
                raise ValueError(
                    f"prediction labels for {qid!r} "
                    "must be nonempty strings"
                )
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise ValueError(
                    f"prediction value for {qid!r}/{label!r} "
                    "must be numeric"
                )
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(
                    f"prediction value for {qid!r}/{label!r} "
                    "must be finite"
                )
            normalized_distribution[label] = numeric

        normalized[qid] = MappingProxyType(
            normalized_distribution
        )
    return MappingProxyType(normalized)


def _model_name(
    payload: Mapping[str, Any],
    config: Config,
) -> str:
    model_name = payload.get("model")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(
            "evaluation input model must be a nonempty string"
        )

    configured = config.data.get("model")
    if configured is not None:
        configured = _require_mapping(
            configured,
            field="model",
        )
        configured_method = configured.get("method")
        if (
            configured_method is not None
            and configured_method != model_name
        ):
            raise ValueError(
                "configured model method does not match "
                "the evaluation input"
            )
    return model_name


def _split_names(value: Any) -> tuple[str, ...]:
    split = _require_mapping(value, field="split")
    train_val = split.get(
        "train_val",
        ("train", "valid"),
    )
    if (
        not isinstance(train_val, Sequence)
        or isinstance(train_val, (str, bytes, bytearray))
    ):
        raise ValueError(
            "split.train_val must be a sequence"
        )
    test = split.get("test", "test")
    names = tuple(train_val) + (test,)
    if (
        not names
        or any(
            not isinstance(name, str) or not name
            for name in names
        )
        or len(names) != len(set(names))
    ):
        raise ValueError(
            "configured evaluation splits must be unique "
            "nonempty strings"
        )
    return names


def _metric_names(value: Any) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
    ):
        raise ValueError(
            "metrics must be a nonempty sequence"
        )
    metrics = tuple(value)
    if any(
        not isinstance(metric, str) or not metric
        for metric in metrics
    ):
        raise ValueError(
            "metrics must contain nonempty strings"
        )
    return metrics


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    config: Config,
    *,
    run_dir: str | os.PathLike[str],
    verbose: bool = False,
) -> EvaluationCommandResult:
    """Evaluate declared outputs and write canonical artifacts."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")
    config.validate_for("evaluate")

    resolved = config.data
    data = _require_mapping(
        resolved.get("data"),
        field="data",
    )
    present_inputs = [
        name
        for name in ("predictions", "fitted")
        if name in data
    ]
    if len(present_inputs) != 1:
        raise ValueError(
            "evaluate requires exactly one of data.predictions "
            "or data.fitted"
        )
    input_kind = present_inputs[0]
    input_path = _resolved_file(
        data[input_kind],
        field=f"data.{input_kind}",
    )
    payload = _load_payload(
        input_path,
        input_kind=input_kind,
    )
    predictions = _prediction_payload(
        payload.get("predictions")
    )
    model_name = _model_name(payload, config)
    cleaning_policy = payload.get("cleaning_policy")
    if (
        not isinstance(cleaning_policy, str)
        or not cleaning_policy.strip()
    ):
        raise ValueError(
            "evaluation input cleaning_policy must be "
            "a nonempty string"
        )

    evaluation_config = _require_mapping(
        resolved.get("evaluation"),
        field="evaluation",
    )
    if evaluation_config.get(
        "mse_prediction_policy",
        "raw",
    ) != "raw":
        raise ValueError(
            "standalone evaluate supports only raw MSE "
            "predictions"
        )
    distribution_policy = evaluation_config.get(
        "distribution_prediction_policy",
        "clip_and_normalize",
    )
    invalid_distribution = evaluation_config.get(
        "invalid_distribution",
        "error",
    )

    survey_source, survey_schema = _survey_paths(
        data.get("survey")
    )
    target_path, target_subgroup = target_spec(
        data.get("target")
    )
    target_path = _resolved_file(
        target_path,
        field="data.target.path",
    )
    survey = Survey(
        csv_path=survey_source,
        config_path=survey_schema,
    )
    binary_survey = BinaryExtendedSurvey.from_survey(
        survey
    )
    targets = load_targets(
        target_path,
        binary_survey,
        subgroup=target_subgroup,
    )
    metrics = _metric_names(resolved.get("metrics"))

    evaluations = tuple(
        evaluate_predictions(
            survey,
            targets,
            predictions,
            model_name=model_name,
            split=split,
            metrics=metrics,
            distribution_prediction_policy=(
                distribution_policy
            ),
            invalid_distribution=invalid_distribution,
        )
        for split in _split_names(resolved.get("split"))
    )

    source_artifacts = {
        "survey": survey_source,
        "survey_schema": survey_schema,
        "target": target_path,
        input_kind: input_path,
    }
    source_digests = {
        name: _digest(path)
        for name, path in source_artifacts.items()
    }
    resolved_run_dir = Path(run_dir).resolve()
    canonical_report = (
        report_stage.write_evaluation_report(
            config,
            evaluations,
            source_artifacts=source_artifacts,
            source_digests=source_digests,
            cleaning_policy=cleaning_policy,
            run_dir=resolved_run_dir,
        )
    )

    return EvaluationCommandResult(
        config_digest=config.digest,
        run_dir=resolved_run_dir,
        input_kind=input_kind,
        model_name=model_name,
        cleaning_policy=cleaning_policy,
        evaluations=evaluations,
        report=canonical_report,
    )


__all__ = [
    "EvaluationCommandResult",
    "run",
]
