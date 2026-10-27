"""Pure validation and reduction of canonical single-run artifacts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

import yaml

from p2p.config import AggregateSpec
from p2p.results import (
    MetricArtifact,
    MetricRecord,
    RunRecord,
    SourceArtifact,
)


class AggregationValidationError(ValueError):
    """Raised when canonical aggregation inputs are incompatible."""


@dataclass(frozen=True, slots=True)
class _LoadedRun:
    index: int
    path: Path
    labels: Mapping[str, Any]
    run: RunRecord
    metrics: MetricArtifact
    run_digest: str
    metrics_digest: str
    included: bool = True
    exclusion_reason: str = ""


@dataclass(frozen=True, slots=True)
class AggregationResult:
    """Validated full-precision rows produced by the pure core."""

    input_rows: tuple[dict[str, Any], ...]
    run_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]
    validation: Mapping[str, Any]
    aggregation_digest: str


@dataclass(frozen=True, slots=True)
class AggregationArtifacts:
    """Files emitted for one validated aggregation."""

    resolved_spec: Path
    inputs_csv: Path
    validation_json: Path
    runs_csv: Path
    summary_csv: Path
    summary_json: Path
    markdown: Path | None = None
    latex: Path | None = None

    @property
    def required_files(self) -> tuple[Path, ...]:
        return (
            self.resolved_spec,
            self.inputs_csv,
            self.validation_json,
            self.runs_csv,
            self.summary_csv,
            self.summary_json,
        )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec_digest(spec: AggregateSpec) -> str:
    canonical = json.dumps(
        spec.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise AggregationValidationError(
            f"{label} does not exist: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AggregationValidationError(
            f"invalid {label}: {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise AggregationValidationError(
            f"{label} root must be a mapping: {path}"
        )
    return value


def _source_artifacts(
    value: Any,
) -> dict[str, SourceArtifact]:
    if not isinstance(value, Mapping):
        raise AggregationValidationError(
            "run source_artifacts must be a mapping"
        )
    artifacts: dict[str, SourceArtifact] = {}
    for name, payload in value.items():
        if not isinstance(payload, Mapping):
            raise AggregationValidationError(
                f"source artifact {name!r} must be a mapping"
            )
        artifacts[name] = SourceArtifact(
            path=payload.get("path"),
            digest=payload.get("digest"),
        )
    return artifacts


def _run_record(payload: Mapping[str, Any]) -> RunRecord:
    if payload.get("schema_version") != 1:
        raise AggregationValidationError(
            "unsupported run.json schema_version"
        )
    try:
        return RunRecord(
            run_id=payload.get("run_id"),
            repeat=payload.get("repeat"),
            seed=payload.get("seed"),
            conditions=payload.get("conditions"),
            config_digest=payload.get("config_digest"),
            cleaning_policy=payload.get(
                "cleaning_policy"
            ),
            source_artifacts=_source_artifacts(
                payload.get("source_artifacts")
            ),
        )
    except (TypeError, ValueError) as error:
        raise AggregationValidationError(
            f"invalid run.json record: {error}"
        ) from error


def _metric_artifact(
    payload: Mapping[str, Any],
) -> MetricArtifact:
    if payload.get("schema_version") != 1:
        raise AggregationValidationError(
            "unsupported metrics.json schema_version"
        )
    raw_metrics = payload.get("metrics")
    if (
        not isinstance(raw_metrics, Sequence)
        or isinstance(
            raw_metrics,
            (str, bytes, bytearray),
        )
    ):
        raise AggregationValidationError(
            "metrics.json metrics must be a list"
        )

    rows = []
    try:
        for raw in raw_metrics:
            if not isinstance(raw, Mapping):
                raise TypeError(
                    "metric rows must be mappings"
                )
            rows.append(
                MetricRecord(
                    metric=raw.get("metric"),
                    value=raw.get("value"),
                    split=raw.get("split"),
                    model=raw.get("model"),
                    unit=raw.get("unit"),
                    prediction_policy=raw.get(
                        "prediction_policy"
                    ),
                    qids_requested=raw.get(
                        "qids_requested"
                    ),
                    qids_used=raw.get("qids_used"),
                    diagnostics=raw.get(
                        "diagnostics",
                        {},
                    ),
                )
            )
        return MetricArtifact(
            run_id=payload.get("run_id"),
            metrics=tuple(rows),
        )
    except (TypeError, ValueError) as error:
        raise AggregationValidationError(
            f"invalid metrics.json artifact: {error}"
        ) from error


def _load_run(index: int, declared) -> _LoadedRun:
    path = Path(declared.path).resolve()
    if not path.is_dir():
        raise AggregationValidationError(
            f"aggregate run path is not a directory: {path}"
        )
    run_path = path / "run.json"
    metrics_path = path / "metrics.json"
    run = _run_record(
        _read_json(run_path, label="run.json")
    )
    metrics = _metric_artifact(
        _read_json(metrics_path, label="metrics.json")
    )
    if metrics.run_id != run.run_id:
        raise AggregationValidationError(
            "run.json and metrics.json run_id do not match "
            f"for {path}"
        )
    return _LoadedRun(
        index=index,
        path=path,
        labels=dict(declared.labels),
        run=run,
        metrics=metrics,
        run_digest=_digest(run_path),
        metrics_digest=_digest(metrics_path),
    )


def _uniform_metric_field(
    loaded: _LoadedRun,
    field: str,
) -> Any:
    values = {
        getattr(metric, field)
        for metric in loaded.metrics.metrics
    }
    if len(values) != 1:
        raise AggregationValidationError(
            f"asserted label {field!r} is ambiguous in "
            f"{loaded.run.run_id}"
        )
    return next(iter(values))


def _assert_labels(loaded: _LoadedRun) -> None:
    for name, expected in loaded.labels.items():
        if name == "repeat":
            actual = loaded.run.repeat
        elif name == "cleaning_policy":
            actual = loaded.run.cleaning_policy
        elif name in {
            "model",
            "split",
            "unit",
            "prediction_policy",
        }:
            actual = _uniform_metric_field(
                loaded,
                name,
            )
        else:
            actual = loaded.run.conditions.get(name)

        if actual != expected:
            raise AggregationValidationError(
                f"asserted label {name!r}={expected!r} "
                f"does not match artifact value {actual!r} "
                f"for run_id {loaded.run.run_id!r}"
            )


def _row_value(
    loaded: _LoadedRun,
    metric: MetricRecord,
    field: str,
) -> Any:
    if field == "run_id":
        return loaded.run.run_id
    if field == "repeat":
        return loaded.run.repeat
    if field == "seed":
        return loaded.run.seed
    if field == "cleaning_policy":
        return loaded.run.cleaning_policy
    if field in {
        "metric",
        "split",
        "model",
        "unit",
        "prediction_policy",
    }:
        return getattr(metric, field)
    if field in loaded.labels:
        return loaded.labels[field]
    if field in loaded.run.conditions:
        return loaded.run.conditions[field]
    raise AggregationValidationError(
        f"grouping field {field!r} is absent from "
        f"run_id {loaded.run.run_id!r}"
    )


def _requested_metrics(
    loaded: _LoadedRun,
    spec: AggregateSpec,
) -> dict[str, MetricRecord]:
    selected: dict[str, MetricRecord] = {}
    for metric in loaded.metrics.metrics:
        if metric.metric not in spec.metrics:
            continue
        if metric.metric in selected:
            raise AggregationValidationError(
                "multiple rows for requested metric "
                f"{metric.metric!r} in run_id "
                f"{loaded.run.run_id!r}"
            )
        selected[metric.metric] = metric
    return selected


def _compatibility(
    loaded: Sequence[_LoadedRun],
    spec: AggregateSpec,
) -> None:
    cleaning = {
        item.run.cleaning_policy
        for item in loaded
    }
    if len(cleaning) > 1:
        raise AggregationValidationError(
            "mixed cleaning policies are incompatible"
        )

    for metric_name in spec.metrics:
        records = [
            metric
            for item in loaded
            for name, metric
            in _requested_metrics(item, spec).items()
            if name == metric_name
        ]
        for field, label in (
            ("split", "split contracts"),
            ("unit", "metric units"),
            (
                "prediction_policy",
                "prediction policy values",
            ),
        ):
            values = {
                getattr(record, field)
                for record in records
            }
            if len(values) > 1:
                raise AggregationValidationError(
                    f"mixed {label} for metric "
                    f"{metric_name!r}: {sorted(values)}"
                )


def _cell_key(
    loaded: _LoadedRun,
    metric: MetricRecord,
    fields: Sequence[str],
) -> tuple[Any, ...]:
    return tuple(
        _row_value(loaded, metric, field)
        for field in fields
    )


def _expected_completeness(
    loaded: Sequence[_LoadedRun],
    spec: AggregateSpec,
) -> list[dict[str, Any]]:
    missing_rows: list[dict[str, Any]] = []
    expected_repeats = spec.expected_repeats

    for metric_name in spec.metrics:
        by_cell: dict[
            tuple[Any, ...],
            list[tuple[_LoadedRun, MetricRecord]],
        ] = defaultdict(list)
        logical_cells: dict[
            tuple[Any, ...],
            list[_LoadedRun],
        ] = defaultdict(list)

        for item in loaded:
            selected = _requested_metrics(item, spec)
            representative = (
                selected.get(metric_name)
                or next(iter(selected.values()), None)
            )
            if representative is None:
                continue
            key = _cell_key(
                item,
                representative,
                spec.group_by,
            )
            logical_cells[key].append(item)
            if metric_name in selected:
                by_cell[key].append(
                    (item, selected[metric_name])
                )

        for key, items in logical_cells.items():
            observed = by_cell.get(key, [])
            repeats = [
                item.run.repeat
                for item, _ in observed
            ]
            if len(repeats) != len(set(repeats)):
                raise AggregationValidationError(
                    "duplicate repeats in aggregation cell "
                    f"{dict(zip(spec.group_by, key))}"
                )
            if (
                expected_repeats is not None
                and len(observed) > expected_repeats
            ):
                raise AggregationValidationError(
                    "observed repeats exceed expected repeats "
                    f"for {dict(zip(spec.group_by, key))}"
                )
            missing = len(items) - len(observed)
            if expected_repeats is not None:
                missing = max(
                    missing,
                    expected_repeats - len(observed),
                )
            if missing:
                missing_rows.append(
                    {
                        "cell": dict(
                            zip(spec.group_by, key)
                        ),
                        "metric": metric_name,
                        "expected": (
                            expected_repeats
                            if expected_repeats is not None
                            else len(items)
                        ),
                        "observed": len(observed),
                        "missing": missing,
                    }
                )

    for expected_cell in spec.expected_cells:
        found = False
        for item in loaded:
            selected = _requested_metrics(item, spec)
            representative = next(
                iter(selected.values()),
                None,
            )
            if representative is None:
                continue
            if all(
                _row_value(item, representative, key)
                == value
                for key, value in expected_cell.items()
            ):
                found = True
                break
        if not found and spec.require_complete:
            raise AggregationValidationError(
                "missing expected aggregation cell "
                f"{dict(expected_cell)}"
            )
    return missing_rows


def _input_rows(
    loaded: Sequence[_LoadedRun],
) -> tuple[dict[str, Any], ...]:
    rows = []
    for item in loaded:
        row: dict[str, Any] = {
            "input_index": item.index,
            "run_path": str(item.path),
            "run_json_path": str(
                item.path / "run.json"
            ),
            "run_json_digest": item.run_digest,
            "metrics_json_path": str(
                item.path / "metrics.json"
            ),
            "metrics_json_digest": item.metrics_digest,
            "run_id": item.run.run_id,
            "repeat": item.run.repeat,
            "seed": item.run.seed,
            "config_digest": item.run.config_digest,
            "cleaning_policy": (
                item.run.cleaning_policy
            ),
            "metric_row_count": len(
                item.metrics.metrics
            ),
            "included": item.included,
            "exclusion_reason": (
                item.exclusion_reason
            ),
        }
        for name in sorted(item.labels):
            row[f"label.{name}"] = item.labels[name]
        rows.append(row)
    return tuple(rows)


def _run_rows(
    loaded: Sequence[_LoadedRun],
    spec: AggregateSpec,
) -> tuple[dict[str, Any], ...]:
    rows = []
    metric_order = {
        name: index
        for index, name in enumerate(spec.metrics)
    }
    for item in loaded:
        if not item.included:
            continue
        selected = _requested_metrics(item, spec)
        for metric in sorted(
            selected.values(),
            key=lambda row: metric_order[row.metric],
        ):
            row: dict[str, Any] = {
                "input_index": item.index,
                "run_path": str(item.path),
                "run_id": item.run.run_id,
                "repeat": item.run.repeat,
                "seed": item.run.seed,
                "config_digest": item.run.config_digest,
                "cleaning_policy": (
                    item.run.cleaning_policy
                ),
            }
            for field in spec.group_by:
                row[field] = _row_value(
                    item,
                    metric,
                    field,
                )
            row.update(
                {
                    "metric": metric.metric,
                    "value": metric.value,
                    "split": metric.split,
                    "model": metric.model,
                    "unit": metric.unit,
                    "prediction_policy": (
                        metric.prediction_policy
                    ),
                    "qids_requested": (
                        metric.qids_requested
                    ),
                    "qids_used": metric.qids_used,
                }
            )
            for name in sorted(metric.diagnostics):
                row[f"diagnostic.{name}"] = (
                    metric.diagnostics[name]
                )
            rows.append(row)
    return tuple(rows)


def _center(values: Sequence[float], method: str) -> float:
    if method == "mean":
        return statistics.fmean(values)
    return float(statistics.median(values))


def _spread(
    values: Sequence[float],
    method: str,
) -> float | None:
    if method == "none":
        return None
    if method == "sample_std":
        return (
            statistics.stdev(values)
            if len(values) >= 2
            else None
        )
    return statistics.pstdev(values)


def _effective_observations(
    rows: Sequence[dict[str, Any]],
    *,
    spec: AggregateSpec,
    output_fields: Sequence[str],
) -> dict[tuple[Any, ...], list[float]]:
    contract_fields = (
        "metric",
        "unit",
        "prediction_policy",
    )
    if (
        spec.hierarchy_weighting == "pooled"
        or not spec.hierarchy_reduce
    ):
        grouped: dict[
            tuple[Any, ...],
            list[float],
        ] = defaultdict(list)
        for row in rows:
            key = tuple(
                row[field]
                for field in (
                    *output_fields,
                    *contract_fields,
                )
            )
            grouped[key].append(row["value"])
        return grouped

    current = [dict(row) for row in rows]
    for dimension in spec.hierarchy_reduce[:-1]:
        remaining = [
            name
            for name in spec.hierarchy_reduce
            if name != dimension
        ]
        grouped_stage: dict[
            tuple[Any, ...],
            list[float],
        ] = defaultdict(list)
        key_fields = (
            *output_fields,
            *remaining,
            *contract_fields,
        )
        for row in current:
            key = tuple(
                row[field]
                for field in key_fields
            )
            grouped_stage[key].append(row["value"])
        current = [
            {
                **dict(zip(key_fields, key)),
                "value": statistics.fmean(values),
            }
            for key, values in grouped_stage.items()
        ]

    last = spec.hierarchy_reduce[-1]
    grouped: dict[
        tuple[Any, ...],
        list[float],
    ] = defaultdict(list)
    for row in current:
        key = tuple(
            row[field]
            for field in (
                *output_fields,
                *contract_fields,
            )
        )
        grouped[key].append(row["value"])
        if last not in row:
            raise AggregationValidationError(
                f"hierarchy dimension {last!r} "
                "is unavailable"
            )
    return grouped


def _summary_rows(
    rows: Sequence[dict[str, Any]],
    loaded: Sequence[_LoadedRun],
    spec: AggregateSpec,
    missing_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    output_fields = tuple(
        field
        for field in spec.group_by
        if field not in spec.hierarchy_reduce
    )
    observations = _effective_observations(
        rows,
        spec=spec,
        output_fields=output_fields,
    )
    contract_fields = (
        "metric",
        "unit",
        "prediction_policy",
    )
    metric_order = {
        name: index
        for index, name in enumerate(spec.metrics)
    }
    summary = []
    for key, values in observations.items():
        payload = dict(
            zip(
                (
                    *output_fields,
                    *contract_fields,
                ),
                key,
            )
        )
        missing_count = sum(
            int(item["missing"])
            for item in missing_rows
            if item["metric"] == payload["metric"]
            and all(
                item["cell"].get(field)
                == payload.get(field)
                for field in output_fields
                if field in item["cell"]
            )
        )
        estimand = "pooled_run"
        if (
            spec.hierarchy_reduce
            and spec.hierarchy_weighting == "equal"
        ):
            estimand = (
                f"equal_{spec.hierarchy_reduce[-1]}"
            )
        row = {
            **{
                field: payload[field]
                for field in output_fields
            },
            "metric": payload["metric"],
            "unit": payload["unit"],
            "prediction_policy": payload[
                "prediction_policy"
            ],
            "center": _center(values, spec.center),
            "spread": _spread(values, spec.spread),
            "n": len(values),
            "missing_count": missing_count,
            "estimand": estimand,
        }
        summary.append(row)

    summary.sort(
        key=lambda row: (
            tuple(str(row[field]) for field in output_fields),
            metric_order[row["metric"]],
        )
    )
    return tuple(summary)


def aggregate_runs(
    spec: AggregateSpec | None,
) -> AggregationResult:
    """Validate and reduce canonical runs without writing files."""

    if not isinstance(spec, AggregateSpec):
        raise TypeError("spec must be AggregateSpec")

    loaded = [
        _load_run(index, declared)
        for index, declared in enumerate(
            spec.runs,
            start=1,
        )
    ]
    run_ids = [item.run.run_id for item in loaded]
    duplicates = sorted(
        {
            run_id
            for run_id in run_ids
            if run_ids.count(run_id) > 1
        }
    )
    if duplicates:
        raise AggregationValidationError(
            f"duplicate run_id values: {duplicates}"
        )
    for item in loaded:
        _assert_labels(item)
    _compatibility(loaded, spec)

    missing_rows = _expected_completeness(
        loaded,
        spec,
    )
    if (
        missing_rows
        and spec.missing_policy == "error"
    ):
        raise AggregationValidationError(
            "missing requested aggregation rows or repeats: "
            f"{missing_rows}"
        )

    if spec.missing_policy == "exclude":
        missing_run_ids = {
            item.run.run_id
            for item in loaded
            if set(spec.metrics)
            - set(_requested_metrics(item, spec))
        }
        loaded = [
            _LoadedRun(
                index=item.index,
                path=item.path,
                labels=item.labels,
                run=item.run,
                metrics=item.metrics,
                run_digest=item.run_digest,
                metrics_digest=item.metrics_digest,
                included=(
                    item.run.run_id
                    not in missing_run_ids
                ),
                exclusion_reason=(
                    ""
                    if item.run.run_id
                    not in missing_run_ids
                    else spec.missing_reason or ""
                ),
            )
            for item in loaded
        ]

    run_rows = _run_rows(loaded, spec)
    summary_rows = _summary_rows(
        run_rows,
        loaded,
        spec,
        missing_rows,
    )
    validation = {
        "schema_version": 1,
        "status": "valid",
        "run_count": len(loaded),
        "included_run_count": sum(
            item.included
            for item in loaded
        ),
        "metric_row_count": len(run_rows),
        "missing_policy": spec.missing_policy,
        "missing_reason": spec.missing_reason,
        "missing_rows": list(missing_rows),
        "checks": {
            "artifact_integrity": "pass",
            "duplicate_run_ids": "pass",
            "label_assertions": "pass",
            "compatibility": "pass",
            "completeness": (
                "warn"
                if missing_rows
                else "pass"
            ),
        },
    }
    return AggregationResult(
        input_rows=_input_rows(loaded),
        run_rows=run_rows,
        summary_rows=summary_rows,
        validation=validation,
        aggregation_digest=_spec_digest(spec),
    )


def _resolved_output_dir(
    value: str | os.PathLike[str],
) -> Path:
    path = Path(value).resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(
            f"output_dir is not a directory: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fieldnames(
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    return names


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    names = _fieldnames(rows)
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=names,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
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


def _presentation_value(
    value: Any,
    *,
    precision: int,
    missing: str,
) -> str:
    if value is None:
        return missing
    if isinstance(value, float):
        if not math.isfinite(value):
            return missing
        rendered = f"{value:.{precision}f}"
        if float(rendered) == 0.0:
            rendered = f"{0.0:.{precision}f}"
        return rendered
    return str(value)


def _markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    precision: int,
) -> str:
    fields = _fieldnames(rows)
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = _presentation_value(
                row.get(field),
                precision=precision,
                missing="—",
            )
            values.append(
                value.replace("\\", "\\\\")
                .replace("|", "\\|")
                .replace("\n", " ")
            )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(
        replacements.get(character, character)
        for character in value
    )


def _latex(
    rows: Sequence[Mapping[str, Any]],
    *,
    precision: int,
) -> str:
    fields = _fieldnames(rows)
    lines = [
        r"\begin{tabular}{" + "l" * len(fields) + "}",
        " & ".join(_latex_escape(name) for name in fields)
        + r" \\",
        r"\hline",
    ]
    for row in rows:
        values = [
            _latex_escape(
                _presentation_value(
                    row.get(field),
                    precision=precision,
                    missing="--",
                )
            )
            for field in fields
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def write_aggregation_artifacts(
    result: AggregationResult,
    spec: AggregateSpec,
    *,
    output_dir: str | os.PathLike[str],
    source_config: str | os.PathLike[str] | None = None,
) -> AggregationArtifacts:
    """Write ledgers and full-precision summary artifacts."""

    if not isinstance(result, AggregationResult):
        raise TypeError("result must be AggregationResult")
    if not isinstance(spec, AggregateSpec):
        raise TypeError("spec must be AggregateSpec")

    output = _resolved_output_dir(output_dir)
    resolved_spec = output / "aggregation.resolved.yaml"
    inputs_csv = output / "aggregation_inputs.csv"
    validation_json = (
        output / "aggregation_validation.json"
    )
    runs_csv = output / "runs.csv"
    summary_csv = output / "summary.csv"
    summary_json = output / "summary.json"

    snapshot = {
        "schema_version": 1,
        "aggregation_digest": (
            result.aggregation_digest
        ),
        "aggregate": spec.to_dict(),
    }
    if source_config is not None:
        snapshot["source"] = str(
            Path(source_config).resolve()
        )
    resolved_spec.write_text(
        yaml.safe_dump(
            snapshot,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_csv(inputs_csv, result.input_rows)
    _write_json(
        validation_json,
        {
            **dict(result.validation),
            "aggregation_digest": (
                result.aggregation_digest
            ),
        },
    )
    _write_csv(runs_csv, result.run_rows)
    _write_csv(summary_csv, result.summary_rows)
    _write_json(
        summary_json,
        {
            "schema_version": 1,
            "summary": list(result.summary_rows),
        },
    )

    markdown = None
    if "markdown" in spec.export_formats:
        markdown = output / "summary.md"
        markdown.write_text(
            _markdown(
                result.summary_rows,
                precision=spec.export_precision,
            ),
            encoding="utf-8",
        )

    latex = None
    if "latex" in spec.export_formats:
        latex = output / "summary.tex"
        latex.write_text(
            _latex(
                result.summary_rows,
                precision=spec.export_precision,
            ),
            encoding="utf-8",
        )

    return AggregationArtifacts(
        resolved_spec=resolved_spec,
        inputs_csv=inputs_csv,
        validation_json=validation_json,
        runs_csv=runs_csv,
        summary_csv=summary_csv,
        summary_json=summary_json,
        markdown=markdown,
        latex=latex,
    )


__all__ = [
    "AggregationArtifacts",
    "AggregationResult",
    "AggregationValidationError",
    "aggregate_runs",
    "write_aggregation_artifacts",
]
