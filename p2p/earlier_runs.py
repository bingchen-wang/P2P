"""Read-only adapters for runs written in the earlier result format."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from p2p.responses import Responses
from p2p.responses.converter import ResponseUtils
from p2p.survey import Survey


@dataclass(frozen=True, slots=True)
class EarlierRunBundle:
    """Canonical diagnostic inputs recovered from one legacy run directory."""

    detected: bool
    diagnostics: dict[str, Any] | None
    metrics: dict[str, Any] | None
    survey: Survey | None
    responses: Responses | None
    archived_images: tuple[Path, ...]
    notes: tuple[str, ...]


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return value if isinstance(value, dict) else None


def _resolve_path(
    run_dir: Path,
    value: Any,
    *,
    prefer_run_copy: bool = False,
) -> Path | None:
    if not isinstance(value, (str, Path)):
        return None
    configured = Path(value).expanduser()
    if configured.is_absolute():
        return configured.resolve() if configured.is_file() else None

    candidates: list[Path] = []
    if prefer_run_copy:
        candidates.append(run_dir / configured.name)
    candidates.append(run_dir / configured)
    candidates.extend(parent / configured for parent in run_dir.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _legacy_config(run_dir: Path) -> dict[str, Any] | None:
    for filename in ("regression_config.yaml", "metadata.yaml"):
        value = _load_yaml(run_dir / filename)
        if isinstance(value, dict) and isinstance(value.get("paths"), dict):
            return value
    return None


def _archived_images(run_dir: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for pattern in ("predictions_*.png", "diagnostics_*.png"):
        paths.extend(sorted(run_dir.glob(pattern)))
    return tuple(path.resolve() for path in paths if path.is_file())


def _legacy_metrics(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "summary_aggregate.csv"
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if frame.empty or "model" not in frame:
        return None

    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        model = str(record.get("model", "legacy"))
        for column, raw_value in record.items():
            if column == "model" or not isinstance(column, str):
                continue
            split = None
            metric = None
            for prefix, resolved_split in (
                ("trainval_", "train+valid"),
                ("test_", "test"),
            ):
                if column.startswith(prefix):
                    split = resolved_split
                    metric = column.removeprefix(prefix)
                    break
            if metric is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            rows.append(
                {
                    "metric": metric,
                    "value": value,
                    "split": split,
                    "model": model,
                    "unit": "legacy summary",
                    "prediction_policy": "legacy",
                }
            )
    if not rows:
        return None
    return {
        "schema_version": 1,
        "artifact_type": "legacy_metric_adapter",
        "metrics": rows,
        "source": str(path.resolve()),
    }


def _coefficient_path(
    run_dir: Path,
    configured_endowments: Any,
) -> Path | None:
    candidates = [
        run_dir / "updated_endowments.csv",
        run_dir / "endowments.csv",
    ]
    configured = _resolve_path(
        run_dir,
        configured_endowments,
        prefer_run_copy=True,
    )
    if configured is not None:
        candidates.append(configured)
    for path in candidates:
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path, usecols=["eid", "weight"])
        except Exception:
            continue
        weights = pd.to_numeric(frame["weight"], errors="coerce")
        if weights.notna().any():
            return path.resolve()
    return None


def _coefficients(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path, usecols=["eid", "weight"])
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame = frame.dropna(subset=["eid", "weight"])
    coefficients = {
        str(row.eid): float(row.weight)
        for row in frame.itertuples(index=False)
        if math.isfinite(float(row.weight))
    }
    if not coefficients or not any(
        abs(value) > 1e-15 for value in coefficients.values()
    ):
        raise ValueError("no finite non-zero legacy weights were found")
    return coefficients


def _probabilities(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    probabilities: dict[str, float] = {}
    for label, raw_probability in value.items():
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError):
            continue
        if math.isfinite(probability):
            probabilities[str(label)] = probability
    return probabilities


def _question_payloads(
    survey: Survey,
    responses: Responses,
    targets: Mapping[str, Any],
    coefficients: Mapping[str, float],
) -> list[dict[str, Any]]:
    predicted = ResponseUtils.aggregate_weighted_responses(
        responses,
        dict(coefficients),
    )
    questions: list[dict[str, Any]] = []
    for question in survey.questions:
        qid = str(question["id"])
        target = _probabilities(targets.get(qid))
        prediction = _probabilities(predicted.get(qid))
        if not target or not prediction:
            continue
        option_order = [
            str(label)
            for label in question.get("code_to_answer", {}).values()
        ]
        option_order.extend(
            label
            for label in (*target, *prediction)
            if label not in option_order
        )
        questions.append(
            {
                "qid": qid,
                "question": question.get("question", ""),
                "split": question.get("split", ""),
                "option_order": option_order,
                "target": target,
                "prediction": prediction,
            }
        )
    return questions


def load_earlier_run(run_dir: str | Path) -> EarlierRunBundle:
    """Recover canonical diagnostics without importing or unpickling old code."""

    run_path = Path(run_dir).resolve()
    config = _legacy_config(run_path)
    images = _archived_images(run_path)
    metrics = _legacy_metrics(run_path)
    detected = config is not None or bool(images) or metrics is not None
    if not detected:
        return EarlierRunBundle(
            detected=False,
            diagnostics=None,
            metrics=None,
            survey=None,
            responses=None,
            archived_images=(),
            notes=(),
        )

    notes: list[str] = []
    if config is None:
        notes.append(
            "legacy regression configuration is unavailable; archived "
            "plots and summary metrics are shown when present"
        )
        return EarlierRunBundle(
            detected=True,
            diagnostics=None,
            metrics=metrics,
            survey=None,
            responses=None,
            archived_images=images,
            notes=tuple(notes),
        )

    paths = config.get("paths", {})
    survey_path = _resolve_path(run_path, paths.get("survey_csv"))
    schema_path = _resolve_path(run_path, paths.get("survey_yaml"))
    target_path = _resolve_path(run_path, paths.get("aggregate_json"))
    responses_path = _resolve_path(
        run_path,
        paths.get("responses_csv"),
        prefer_run_copy=True,
    )
    coefficient_path = _coefficient_path(
        run_path,
        paths.get("endowments_csv"),
    )

    basis_required = {
        "survey": survey_path,
        "survey schema": schema_path,
        "responses": responses_path,
    }
    missing_basis = [
        name
        for name, path in basis_required.items()
        if path is None
    ]
    if missing_basis:
        notes.append(
            "legacy response views are incomplete because "
            + ", ".join(missing_basis)
            + " could not be resolved"
        )
        return EarlierRunBundle(
            detected=True,
            diagnostics=None,
            metrics=metrics,
            survey=None,
            responses=None,
            archived_images=images,
            notes=tuple(notes),
        )

    try:
        survey = Survey(survey_path, schema_path)
        responses = Responses(
            str(responses_path),
            survey,
            output_format="answer",
        )
    except Exception as error:
        notes.append(f"legacy response loading failed: {error}")
        return EarlierRunBundle(
            detected=True,
            diagnostics=None,
            metrics=metrics,
            survey=None,
            responses=None,
            archived_images=images,
            notes=tuple(notes),
        )

    prediction_required = {
        "human aggregate": target_path,
        "weighted endowments": coefficient_path,
    }
    missing_prediction = [
        name
        for name, path in prediction_required.items()
        if path is None
    ]
    if missing_prediction:
        notes.append(
            "legacy prediction reconstruction is incomplete because "
            + ", ".join(missing_prediction)
            + " could not be resolved"
        )
        return EarlierRunBundle(
            detected=True,
            diagnostics=None,
            metrics=metrics,
            survey=survey,
            responses=responses,
            archived_images=images,
            notes=tuple(notes),
        )

    try:
        targets = json.loads(target_path.read_text(encoding="utf-8"))
        if not isinstance(targets, dict):
            raise ValueError("human aggregate JSON must be a mapping")
        coefficients = _coefficients(coefficient_path)
        questions = _question_payloads(
            survey,
            responses,
            targets,
            coefficients,
        )
        if not questions:
            raise ValueError(
                "no questions had both human and predicted distributions"
            )
    except Exception as error:
        notes.append(f"legacy interactive reconstruction failed: {error}")
        return EarlierRunBundle(
            detected=True,
            diagnostics=None,
            metrics=metrics,
            survey=None,
            responses=None,
            archived_images=images,
            notes=tuple(notes),
        )

    method = str(config.get("weight_assignment_model", "legacy weighted model"))
    selected = {
        eid: value
        for eid, value in coefficients.items()
        if abs(value) > 1e-15
    }
    diagnostics = {
        "schema_version": 1,
        "artifact_type": "legacy_regression_diagnostics",
        "run_id": run_path.name,
        "model": {
            "method": method,
            "coefficients": selected,
            "coefficient_source": str(coefficient_path),
        },
        "prediction_policy": "legacy weighted response aggregate",
        "target": {
            "source": str(target_path),
        },
        "legacy_adapter": {
            "joblib_loaded": False,
            "responses": str(responses_path),
            "survey": str(survey_path),
            "survey_schema": str(schema_path),
            "weighted_endowments": str(coefficient_path),
        },
        "questions": questions,
    }
    notes.append(
        f"parsed {len(questions)} legacy questions using {method} weights "
        f"from {coefficient_path.name}; no joblib file was loaded"
    )
    return EarlierRunBundle(
        detected=True,
        diagnostics=diagnostics,
        metrics=metrics,
        survey=survey,
        responses=responses,
        archived_images=images,
        notes=tuple(notes),
    )


__all__ = [
    "EarlierRunBundle",
    "load_earlier_run",
]
