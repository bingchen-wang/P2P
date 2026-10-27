"""Canonical run provenance and long-form metric records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import os
import re
from types import MappingProxyType
from typing import Any


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a nonempty string"
        )
    return value


def _sha256_digest(
    value: Any,
    *,
    field_name: str,
) -> str:
    value = _nonempty_string(
        value,
        field_name=field_name,
    )
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return value


def _json_scalar(
    value: Any,
    *,
    field_name: str,
) -> Any:
    if value is None or isinstance(
        value,
        (str, bool, int),
    ):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"{field_name} must be finite"
            )
        return value

    raise ValueError(
        f"{field_name} must be a JSON scalar"
    )


def _scalar_mapping(
    value: Any,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{field_name} must be a mapping"
        )

    normalized: dict[str, Any] = {}
    for key, child in value.items():
        key = _nonempty_string(
            key,
            field_name=f"{field_name} key",
        )
        normalized[key] = _json_scalar(
            child,
            field_name=f"{field_name}.{key}",
        )

    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    """One source path and its captured SHA-256 digest."""

    path: str
    digest: str

    def __post_init__(self) -> None:
        path = self.path
        if isinstance(path, os.PathLike):
            path = os.fspath(path)
        path = _nonempty_string(
            path,
            field_name="source artifact path",
        )
        digest = _sha256_digest(
            self.digest,
            field_name="source artifact digest",
        )

        object.__setattr__(self, "path", path)
        object.__setattr__(self, "digest", digest)

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Canonical provenance stored in one run.json artifact."""

    run_id: str
    repeat: int
    seed: int
    conditions: Mapping[str, Any]
    config_digest: str
    cleaning_policy: str
    source_artifacts: Mapping[str, SourceArtifact]
    schema_version: int = field(
        init=False,
        default=1,
    )

    def __post_init__(self) -> None:
        run_id = _nonempty_string(
            self.run_id,
            field_name="run_id",
        )
        if (
            isinstance(self.repeat, bool)
            or not isinstance(self.repeat, int)
            or self.repeat < 1
        ):
            raise ValueError(
                "repeat must be a positive integer"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
        ):
            raise ValueError("seed must be an integer")

        conditions = _scalar_mapping(
            self.conditions,
            field_name="conditions",
        )
        config_digest = _sha256_digest(
            self.config_digest,
            field_name="config_digest",
        )
        cleaning_policy = _nonempty_string(
            self.cleaning_policy,
            field_name="cleaning_policy",
        )

        if not isinstance(
            self.source_artifacts,
            Mapping,
        ):
            raise TypeError(
                "source_artifacts must be a mapping"
            )

        source_artifacts: dict[
            str,
            SourceArtifact,
        ] = {}
        for name, artifact in (
            self.source_artifacts.items()
        ):
            name = _nonempty_string(
                name,
                field_name="source artifact name",
            )
            if not isinstance(
                artifact,
                SourceArtifact,
            ):
                raise TypeError(
                    "source artifact values must be "
                    "SourceArtifact"
                )
            source_artifacts[name] = artifact

        if not source_artifacts:
            raise ValueError(
                "source_artifacts must not be empty"
            )

        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(
            self,
            "conditions",
            conditions,
        )
        object.__setattr__(
            self,
            "config_digest",
            config_digest,
        )
        object.__setattr__(
            self,
            "cleaning_policy",
            cleaning_policy,
        )
        object.__setattr__(
            self,
            "source_artifacts",
            MappingProxyType(source_artifacts),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "repeat": self.repeat,
            "seed": self.seed,
            "conditions": dict(self.conditions),
            "config_digest": self.config_digest,
            "cleaning_policy": self.cleaning_policy,
            "source_artifacts": {
                name: artifact.to_dict()
                for name, artifact
                in self.source_artifacts.items()
            },
        }


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """One canonical long-form metric observation."""

    metric: str
    value: float
    split: str
    model: str
    unit: str
    prediction_policy: str
    qids_requested: int
    qids_used: int
    diagnostics: Mapping[str, Any] = field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        for field_name in (
            "metric",
            "split",
            "model",
            "unit",
            "prediction_policy",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonempty_string(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )

        if (
            isinstance(self.value, bool)
            or not isinstance(
                self.value,
                (int, float),
            )
        ):
            raise ValueError(
                "value must be a finite number"
            )

        numeric_value = float(self.value)
        if not math.isfinite(numeric_value):
            raise ValueError(
                "value must be finite"
            )

        for field_name in (
            "qids_requested",
            "qids_used",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a "
                    "nonnegative integer"
                )

        if self.qids_used > self.qids_requested:
            raise ValueError(
                "qids_used must not exceed "
                "qids_requested"
            )

        diagnostics = _scalar_mapping(
            self.diagnostics,
            field_name="diagnostics",
        )

        object.__setattr__(
            self,
            "value",
            numeric_value,
        )
        object.__setattr__(
            self,
            "diagnostics",
            diagnostics,
        )

    @property
    def identity(self) -> tuple[str, ...]:
        """Fields identifying one metric row."""

        return (
            self.metric,
            self.split,
            self.model,
            self.unit,
            self.prediction_policy,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "split": self.split,
            "model": self.model,
            "unit": self.unit,
            "prediction_policy": (
                self.prediction_policy
            ),
            "qids_requested": self.qids_requested,
            "qids_used": self.qids_used,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class MetricArtifact:
    """Canonical metrics.json payload for one run."""

    run_id: str
    metrics: Sequence[MetricRecord]
    schema_version: int = field(
        init=False,
        default=1,
    )

    def __post_init__(self) -> None:
        run_id = _nonempty_string(
            self.run_id,
            field_name="run_id",
        )
        if (
            not isinstance(self.metrics, Sequence)
            or isinstance(
                self.metrics,
                (str, bytes, bytearray),
            )
        ):
            raise TypeError(
                "metrics must be a sequence"
            )

        metrics = tuple(self.metrics)
        if not metrics:
            raise ValueError(
                "metrics must not be empty"
            )
        if any(
            not isinstance(metric, MetricRecord)
            for metric in metrics
        ):
            raise TypeError(
                "metrics must contain MetricRecord values"
            )

        identities = [
            metric.identity
            for metric in metrics
        ]
        if len(identities) != len(set(identities)):
            raise ValueError(
                "duplicate canonical metric row"
            )

        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "metrics", metrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "metrics": [
                metric.to_dict()
                for metric in self.metrics
            ],
        }


__all__ = [
    "MetricArtifact",
    "MetricRecord",
    "RunRecord",
    "SourceArtifact",
]