"""Canonical single-run report artifact writing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from p2p.config import Config
from p2p.models.folds import (
    FOLD_RECORD_FILENAME,
    FoldAssignment,
    write_fold_record,
)
from p2p.results import (
    MetricArtifact,
    MetricRecord,
    RunRecord,
    SourceArtifact,
)
from p2p.stages.evaluate import (
    ANSWER_FILE_ROUTES,
    answer_file_predictions,
    model_distribution_predictions,
)
from p2p.stages.split import REFERENCE_FEATURE


_DIAGNOSTIC_FIELDS = (
    "distribution_route",
    "canonical_answers",
    "invalid_distribution_policy",
    "invalid_qids_before_transform",
    "invalid_qids_after_transform",
    "negative_entry_count",
    "negative_mass",
    "raw_sum_deviation",
    "transformation_l1",
)

_METRIC_FIELDS = (
    "metric",
    "value",
    "split",
    "model",
    "unit",
    "prediction_policy",
    "qids_requested",
    "qids_used",
)


@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    """Canonical files and in-memory records for one run."""

    run_record: RunRecord
    metric_artifact: MetricArtifact
    run_json: Path
    metrics_json: Path
    metrics_csv: Path
    config_snapshot: Path
    attribute_bank_snapshot: Path | None = None
    diagnostics_json: Path | None = None
    folds_json: Path | None = None


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


def _run_identity(
    config: Config,
    *,
    method: str,
) -> tuple[str, int, int, dict[str, Any]]:
    run = _require_mapping(
        config.data.get("run"),
        field="run",
    )

    name = run.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "run.name must be a nonempty string"
        )

    repeat = run.get("repeat", 1)
    seed = run.get("seed", 101)

    conditions = dict(
        _require_mapping(
            run.get("conditions", {}),
            field="run.conditions",
        )
    )
    configured_method = conditions.get("method")
    if (
        configured_method is not None
        and configured_method != method
    ):
        raise ValueError(
            "run condition model method does not match "
            "the fitted model"
        )
    conditions["method"] = method

    run_id = run.get(
        "run_id",
        f"{name}-r{repeat:04d}",
    )
    return run_id, repeat, seed, conditions


def _source_records(
    data,
) -> dict[str, SourceArtifact]:
    paths = _require_mapping(
        getattr(data, "source_artifacts", None),
        field="data.source_artifacts",
    )
    digests = _require_mapping(
        getattr(data, "source_digests", None),
        field="data.source_digests",
    )

    if set(paths) != set(digests):
        raise ValueError(
            "source artifact paths and digests must have "
            "matching names"
        )

    return {
        name: SourceArtifact(
            path=path,
            digest=digests[name],
        )
        for name, path in paths.items()
    }


def _source_records_from(
    paths: Mapping[str, Any],
    digests: Mapping[str, Any],
) -> dict[str, SourceArtifact]:
    if set(paths) != set(digests):
        raise ValueError(
            "source artifact paths and digests must have "
            "matching names"
        )
    return {
        name: SourceArtifact(
            path=path,
            digest=digests[name],
        )
        for name, path in paths.items()
    }


def _metric_records(
    evaluation,
) -> tuple[MetricRecord, ...]:
    values = _require_mapping(
        getattr(evaluation, "values", None),
        field="evaluation.values",
    )
    coverage = _require_mapping(
        getattr(evaluation, "coverage", None),
        field="evaluation.coverage",
    )
    diagnostics = _require_mapping(
        getattr(evaluation, "diagnostics", None),
        field="evaluation.diagnostics",
    )

    if set(values) != set(coverage):
        raise ValueError(
            "metric values and coverage must have "
            "matching names"
        )

    model_name = getattr(
        evaluation,
        "model_name",
        None,
    )
    split = getattr(evaluation, "split", None)
    distribution_policy = getattr(
        evaluation,
        "prediction_policy",
        None,
    )

    rows = []
    for metric, value in values.items():
        if value is None:
            continue
        metric_coverage = coverage[metric]

        if metric == "mse":
            unit = "binary_option"
            prediction_policy = "raw"
            metric_diagnostics = {}
        elif metric == "distribution_mse":
            # The same option cells as mse, predictions after the policy.
            unit = "binary_option"
            prediction_policy = distribution_policy
            metric_diagnostics = {}
        else:
            unit = "question"
            prediction_policy = distribution_policy
            metric_diagnostics = {
                field: diagnostics[field]
                for field in _DIAGNOSTIC_FIELDS
                if field in diagnostics
            }

        rows.append(
            MetricRecord(
                metric=metric,
                value=value,
                split=split,
                model=model_name,
                unit=unit,
                prediction_policy=prediction_policy,
                qids_requested=(
                    metric_coverage.qids_requested
                ),
                qids_used=metric_coverage.qids_used,
                diagnostics=metric_diagnostics,
            )
        )

    return tuple(rows)


def _write_report_artifacts(
    config: Config,
    *,
    run_record: RunRecord,
    metric_artifact: MetricArtifact,
    run_dir: str | os.PathLike[str],
) -> ReportArtifacts:
    resolved_run_dir = _resolved_run_dir(run_dir)
    run_json = (
        resolved_run_dir / "run.json"
    ).resolve()
    metrics_json = (
        resolved_run_dir / "metrics.json"
    ).resolve()
    metrics_csv = (
        resolved_run_dir / "metrics.csv"
    ).resolve()

    _write_json(
        run_json,
        run_record.to_dict(),
    )
    _write_json(
        metrics_json,
        metric_artifact.to_dict(),
    )
    _write_metrics_csv(
        metrics_csv,
        metric_artifact,
    )

    config_snapshot = config.write_snapshot(
        resolved_run_dir
    ).resolve()

    attribute_bank_snapshot = None
    if config.attribute_bank is not None:
        attribute_bank_snapshot = (
            config.attribute_bank.write_snapshot(
                resolved_run_dir
            ).resolve()
        )

    return ReportArtifacts(
        run_record=run_record,
        metric_artifact=metric_artifact,
        run_json=run_json,
        metrics_json=metrics_json,
        metrics_csv=metrics_csv,
        config_snapshot=config_snapshot,
        attribute_bank_snapshot=(
            attribute_bank_snapshot
        ),
    )


def write_evaluation_report(
    config: Config,
    evaluations: Sequence[Any],
    *,
    source_artifacts: Mapping[str, Any],
    source_digests: Mapping[str, Any],
    cleaning_policy: str,
    run_dir: str | os.PathLike[str],
) -> ReportArtifacts:
    """Write canonical artifacts for standalone evaluations."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")
    if (
        not isinstance(evaluations, Sequence)
        or isinstance(evaluations, (str, bytes, bytearray))
        or not evaluations
    ):
        raise ValueError(
            "evaluations must be a nonempty sequence"
        )

    model_names = {
        getattr(evaluation, "model_name", None)
        for evaluation in evaluations
    }
    if len(model_names) != 1:
        raise ValueError(
            "standalone evaluations must use one model"
        )
    method = next(iter(model_names))
    run_id, repeat, seed, conditions = _run_identity(
        config,
        method=method,
    )
    rows = tuple(
        row
        for evaluation in evaluations
        for row in _metric_records(evaluation)
    )
    run_record = RunRecord(
        run_id=run_id,
        repeat=repeat,
        seed=seed,
        conditions=conditions,
        config_digest=config.digest,
        cleaning_policy=cleaning_policy,
        source_artifacts=_source_records_from(
            source_artifacts,
            source_digests,
        ),
    )
    metric_artifact = MetricArtifact(
        run_id=run_id,
        metrics=rows,
    )
    return _write_report_artifacts(
        config,
        run_record=run_record,
        metric_artifact=metric_artifact,
        run_dir=run_dir,
    )


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_metrics_csv(
    path: Path,
    artifact: MetricArtifact,
) -> None:
    fieldnames = [
        *_METRIC_FIELDS,
        *_DIAGNOSTIC_FIELDS,
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for metric in artifact.metrics:
            payload = metric.to_dict()
            diagnostics = payload.pop("diagnostics")
            payload.update(diagnostics)
            writer.writerow(payload)


def write_regression_report(
    config: Config,
    regression,
    *,
    run_dir: str | os.PathLike[str],
) -> ReportArtifacts:
    """Write canonical artifacts for one regression result."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")
    regression_digest = getattr(
        regression,
        "config_digest",
        None,
    )
    if (
        regression_digest is not None
        and regression_digest != config.digest
    ):
        raise ValueError(
            "regression config digest does not match report config"
        )

    data = getattr(regression, "data", None)
    fit = getattr(regression, "fit", None)
    evaluation = getattr(
        regression,
        "evaluation",
        None,
    )

    method = getattr(fit, "method", None)
    evaluated_model = getattr(
        evaluation,
        "model_name",
        None,
    )
    if method != evaluated_model:
        raise ValueError(
            "fitted model and evaluated model do not match"
        )

    run_id, repeat, seed, conditions = _run_identity(
        config,
        method=method,
    )
    run_record = RunRecord(
        run_id=run_id,
        repeat=repeat,
        seed=seed,
        conditions=conditions,
        config_digest=config.digest,
        cleaning_policy=getattr(
            data,
            "cleaning_policy",
            None,
        ),
        source_artifacts=_source_records(data),
    )
    metric_artifact = MetricArtifact(
        run_id=run_id,
        metrics=_metric_records(evaluation),
    )

    artifacts = _write_report_artifacts(
        config,
        run_record=run_record,
        metric_artifact=metric_artifact,
        run_dir=run_dir,
    )
    # Written before the diagnostics so that a run with a fold assignment
    # always records it, whether or not full diagnostics can be written.
    fold_assignment = getattr(fit, "folds", None)
    folds_json = (
        write_fold_record(
            artifacts.run_json.parent / FOLD_RECORD_FILENAME,
            fold_assignment,
        )
        if isinstance(fold_assignment, FoldAssignment)
        else None
    )
    diagnostics_json = _write_regression_diagnostics(
        artifacts.run_json.parent / "diagnostics.json",
        run_id=run_id,
        data=data,
        fit=fit,
        evaluation=evaluation,
    )
    if diagnostics_json is None and folds_json is None:
        return artifacts
    return ReportArtifacts(
        run_record=artifacts.run_record,
        metric_artifact=artifacts.metric_artifact,
        run_json=artifacts.run_json,
        metrics_json=artifacts.metrics_json,
        metrics_csv=artifacts.metrics_csv,
        config_snapshot=artifacts.config_snapshot,
        attribute_bank_snapshot=artifacts.attribute_bank_snapshot,
        diagnostics_json=diagnostics_json,
        folds_json=folds_json,
    )


_SCALAR_FIT_DIAGNOSTICS = (
    "train_mse",
    "anti_type_mass",
    "anti_type_count",
    "coefficient_sum",
    "intercept",
    "reference_distribution_digest",
    "ignored_model_options",
    "default_grids",
    "solver_used",
    "solver_fallback",
    "validation_strategy",
    "holdout_split",
    "holdout_fit_splits",
    "holdout_fits",
    "holdout_solver_fallbacks",
    "holdout_best_mse",
    "holdout_score_spread",
    "cv_fits",
    "cv_solver_fallbacks",
    "cv_config_count",
    "cv_configs_within_one_se",
    "cv_score_spread",
)

# Recorded even when null, because absent would be ambiguous: it could mean the
# selection was interior, or that the grid was too small to have an edge.
_NULLABLE_FIT_DIAGNOSTICS = (
    "alpha_at_grid_edge",
    "l1_ratio_at_grid_edge",
)


def _fit_diagnostics(
    fit: Any,
    coefficients: Mapping[str, float],
) -> dict[str, Any]:
    """Scalar Stage-2 diagnostics worth persisting beside the coefficients.

    ``nnz`` alone is a poor summary: it counts coefficients above a numeric
    threshold, so the same model can report very different ensemble sizes.
    Anti-type mass and the total weight describe the fit itself. The reference
    respondent's weight, when present, reports how much of the target the proxy
    basis could not reach.
    """

    diagnostics = getattr(fit, "diagnostics", None)
    payload: dict[str, Any] = {}

    if isinstance(diagnostics, Mapping):
        for field in _SCALAR_FIT_DIAGNOSTICS:
            if field not in diagnostics:
                continue
            value = diagnostics[field]
            if isinstance(value, str):
                payload[field] = value
            elif isinstance(value, bool):
                payload[field] = value
            elif isinstance(value, (int, np.integer)):
                payload[field] = int(value)
            elif isinstance(value, (float, np.floating)):
                payload[field] = float(value)

    if isinstance(diagnostics, Mapping):
        for field in _NULLABLE_FIT_DIAGNOSTICS:
            if field in diagnostics:
                value = diagnostics[field]
                payload[field] = (
                    value if value is None else str(value)
                )

    if REFERENCE_FEATURE in coefficients:
        payload["reference_respondent_weight"] = float(
            coefficients[REFERENCE_FEATURE]
        )

    fold_assignment = getattr(fit, "folds", None)
    if isinstance(fold_assignment, FoldAssignment):
        # Enough to compare two runs' folds without opening folds.json.
        payload["folds"] = {
            "method": fold_assignment.method,
            "seed": fold_assignment.seed,
            "n_splits": fold_assignment.n_splits,
            "partition_digest": fold_assignment.partition_digest,
            "record": FOLD_RECORD_FILENAME,
        }

    return payload


def _write_regression_diagnostics(
    path: Path,
    *,
    run_id: str,
    data: Any,
    fit: Any,
    evaluation: Any,
) -> Path | None:
    """Write drill-down data when a full regression object is available."""

    survey = getattr(data, "survey", None)
    responses = getattr(data, "answer_responses", None)
    targets = getattr(data, "targets", None)
    model = getattr(fit, "model", None)
    coefficients = getattr(model, "coef_dict_", None)
    if (
        survey is None
        or responses is None
        or targets is None
        or not isinstance(coefficients, Mapping)
    ):
        return None

    # Show the prediction that was scored: the same route as the evaluation.
    route = _require_mapping(
        getattr(evaluation, "diagnostics", None) or {},
        field="evaluation.diagnostics",
    ).get("distribution_route", "model")
    predicted = (
        answer_file_predictions(data, model)
        if route in ANSWER_FILE_ROUTES
        else model_distribution_predictions(data, model)
    )
    questions = []
    for question in survey.questions:
        qid = question["id"]
        target = targets.raw.get(qid)
        if target is None:
            continue
        option_order = list(question.get("code_to_answer", {}).values())
        questions.append(
            {
                "qid": qid,
                "question": question.get("question", ""),
                "split": question.get("split", ""),
                "option_order": option_order,
                "target": dict(target),
                "prediction": dict(predicted.get(qid, {})),
            }
        )

    payload = {
        "schema_version": 1,
        "artifact_type": "regression_diagnostics",
        "run_id": run_id,
        "model": {
            "method": getattr(fit, "method", None),
            "solver": getattr(fit, "solver", None),
            "status": getattr(fit, "status", None),
            "best_alpha": getattr(fit, "best_alpha", None),
            "best_l1_ratio": getattr(fit, "best_l1_ratio", None),
            "coefficients": dict(coefficients),
            **_fit_diagnostics(fit, coefficients),
        },
        "prediction_policy": getattr(
            evaluation,
            "prediction_policy",
            None,
        ),
        "distribution_route": route,
        "canonical_answers": bool(getattr(data, "canonical_answers", True)),
        "target": {
            "subgroup": getattr(targets, "subgroup", None),
            "source_digest": getattr(targets, "source_digest", None),
            "selection_digest": getattr(
                targets,
                "selection_digest",
                None,
            ),
        },
        "questions": questions,
    }
    _write_json(path, payload)
    return path.resolve()


__all__ = [
    "ReportArtifacts",
    "write_evaluation_report",
    "write_regression_report",
]
