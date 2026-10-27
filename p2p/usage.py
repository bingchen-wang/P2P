"""Backend-neutral token usage and cost accounting."""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
import json
import math
import os
from pathlib import Path
from threading import Lock
from typing import Mapping

from p2p.backends.base import TokenUsage


@dataclass(frozen=True, slots=True)
class TokenPricing:
    """USD pricing per one million tokens."""

    input_per_million_usd: float = 0.0
    output_per_million_usd: float = 0.0
    cached_input_per_million_usd: float = 0.0

    def __post_init__(self) -> None:
        for pricing_field in fields(self):
            value = getattr(self, pricing_field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"{pricing_field.name} must be a number"
                )
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{pricing_field.name} must be finite and nonnegative"
                )


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """One accounted backend operation."""

    usage: TokenUsage
    model_name: str
    module_name: str
    agent_id: str
    cost_usd: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a JSON-serializable usage record."""

        return {
            "model_name": self.model_name,
            "module_name": self.module_name,
            "agent_id": self.agent_id,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cached_input_tokens": (
                self.usage.cached_input_tokens
            ),
            "reasoning_tokens": self.usage.reasoning_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass(frozen=True, slots=True)
class UsageArtifacts:
    """CSV and JSON usage artifacts for one command run."""

    csv_path: Path
    json_path: Path


class UsageTracker:
    """Accumulate normalized usage with required model/module metadata."""

    def __init__(
        self,
        pricing: Mapping[str, TokenPricing] | None = None,
        *,
        stream_path: str | os.PathLike[str] | None = None,
        resume_stream: bool = False,
    ) -> None:
        if not isinstance(resume_stream, bool):
            raise TypeError("resume_stream must be a bool")
        self._pricing = dict(pricing or {})
        self._records: list[UsageRecord] = []
        self._lock = Lock()
        self._stream_lock = Lock()
        self._stream_path = (
            Path(stream_path).resolve()
            if stream_path is not None
            else None
        )

        for model_name, model_pricing in self._pricing.items():
            self._validate_metadata("model_name", model_name)
            if not isinstance(model_pricing, TokenPricing):
                raise TypeError(
                    "pricing values must be TokenPricing instances"
                )
        if self._stream_path is not None:
            self._stream_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            if resume_stream and self._stream_path.exists():
                self._load_stream()
            else:
                with self._stream_path.open(
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=[
                            "model_name",
                            "module_name",
                            "agent_id",
                            "input_tokens",
                            "output_tokens",
                            "cached_input_tokens",
                            "reasoning_tokens",
                            "cost_usd",
                        ],
                    )
                    writer.writeheader()

    def _load_stream(self) -> None:
        """Restore prior streamed records before appending resumed work."""

        if self._stream_path is None:
            return
        with self._stream_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)
            required = {
                "model_name",
                "module_name",
                "agent_id",
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "cost_usd",
            }
            if reader.fieldnames is None:
                raise ValueError(
                    f"usage stream has no header: {self._stream_path}"
                )
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    "usage stream is missing required columns: "
                    + ", ".join(sorted(missing))
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    record = UsageRecord(
                        usage=TokenUsage(
                            input_tokens=int(row["input_tokens"]),
                            output_tokens=int(row["output_tokens"]),
                            cached_input_tokens=int(
                                row["cached_input_tokens"]
                            ),
                            reasoning_tokens=int(
                                row.get("reasoning_tokens") or 0
                            ),
                        ),
                        model_name=str(row["model_name"]),
                        module_name=str(row["module_name"]),
                        agent_id=str(row["agent_id"]),
                        cost_usd=float(row["cost_usd"]),
                    )
                    self._validate_metadata(
                        "model_name",
                        record.model_name,
                    )
                    self._validate_metadata(
                        "module_name",
                        record.module_name,
                    )
                    self._validate_metadata(
                        "agent_id",
                        record.agent_id,
                    )
                    if (
                        not math.isfinite(record.cost_usd)
                        or record.cost_usd < 0
                    ):
                        raise ValueError(
                            "cost_usd must be finite and nonnegative"
                        )
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(
                        "invalid usage stream row "
                        f"{row_number}: {self._stream_path}"
                    ) from exc
                self._records.append(record)

    @staticmethod
    def _validate_metadata(field_name: str, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{field_name} must be a nonempty string"
            )

    def log(
        self,
        *,
        usage: TokenUsage,
        model_name: str,
        module_name: str,
        agent_id: str,
    ) -> UsageRecord:
        """Account for one normalized backend response exactly once."""

        if not isinstance(usage, TokenUsage):
            raise TypeError("usage must be TokenUsage")

        self._validate_metadata("model_name", model_name)
        self._validate_metadata("module_name", module_name)
        self._validate_metadata("agent_id", agent_id)

        pricing = self._pricing.get(model_name, TokenPricing())
        non_cached_input = (
            usage.input_tokens - usage.cached_input_tokens
        )
        cost_usd = (
            non_cached_input * pricing.input_per_million_usd
            + usage.output_tokens * pricing.output_per_million_usd
            + usage.cached_input_tokens
            * pricing.cached_input_per_million_usd
        ) / 1_000_000

        record = UsageRecord(
            usage=usage,
            model_name=model_name,
            module_name=module_name,
            agent_id=agent_id,
            cost_usd=cost_usd,
        )

        with self._lock:
            self._records.append(record)
        if self._stream_path is not None:
            with self._stream_lock:
                with self._stream_path.open(
                    "a",
                    newline="",
                    encoding="utf-8",
                ) as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=list(record.to_dict()),
                    )
                    writer.writerow(record.to_dict())

        return record

    @property
    def records(self) -> tuple[UsageRecord, ...]:
        """Return an immutable snapshot of accounted operations."""

        with self._lock:
            return tuple(self._records)

    def summary(self) -> dict[str, int | float]:
        """Return full-precision aggregate token counts and cost."""

        records = self.records
        return {
            "input_tokens": sum(
                record.usage.input_tokens for record in records
            ),
            "output_tokens": sum(
                record.usage.output_tokens for record in records
            ),
            "cached_input_tokens": sum(
                record.usage.cached_input_tokens for record in records
            ),
            "reasoning_tokens": sum(
                record.usage.reasoning_tokens for record in records
            ),
            "cost_usd": sum(record.cost_usd for record in records),
        }

    def report(self) -> str:
        """Return a compact human-readable usage summary."""

        summary = self.summary()
        return (
            f"{summary['input_tokens']} input tokens, "
            f"{summary['output_tokens']} output tokens, "
            f"{summary['reasoning_tokens']} reasoning tokens, "
            f"{summary['cached_input_tokens']} cached input tokens, "
            f"${summary['cost_usd']:.6f}"
        )


def write_usage_artifacts(
    tracker: UsageTracker,
    run_dir: str | os.PathLike[str],
) -> UsageArtifacts:
    """Write full-precision normalized usage records and totals."""

    if not isinstance(tracker, UsageTracker):
        raise TypeError("tracker must be UsageTracker")
    path = Path(run_dir).resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(f"run_dir is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)

    records = tracker.records
    fieldnames = [
        "model_name",
        "module_name",
        "agent_id",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
        "cost_usd",
    ]
    csv_path = (path / "usage.csv").resolve()
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            record.to_dict()
            for record in records
        )

    json_path = (path / "usage.json").resolve()
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "summary": tracker.summary(),
                "records": [
                    record.to_dict()
                    for record in records
                ],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return UsageArtifacts(
        csv_path=csv_path,
        json_path=json_path,
    )


__all__ = [
    "TokenPricing",
    "UsageArtifacts",
    "UsageRecord",
    "UsageTracker",
    "write_usage_artifacts",
]
