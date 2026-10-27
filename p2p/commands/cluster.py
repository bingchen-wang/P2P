"""Deterministic offline clustering of existing endowment text."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import warnings

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from p2p.config import Config, ConfigValidationError


_ALGORITHM = "tfidf_kmeans"


@dataclass(frozen=True, slots=True)
class ClusterAssignment:
    """One endowment's canonical cluster assignment."""

    eid: str
    cluster: int


@dataclass(frozen=True, slots=True)
class ClusterResult:
    """Artifacts and assignments from one clustering run."""

    config_digest: str
    run_dir: Path
    assignments: tuple[ClusterAssignment, ...]
    assignments_path: Path
    metadata_path: Path
    config_path: Path


def _require_mapping(
    value: Any,
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(
            f"{field} is required and must be a mapping"
        )
    return value


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
        raise ConfigValidationError(
            f"{field} must be a positive integer"
        )
    return value


def _endowments_path(data: Mapping[str, Any]) -> Path:
    basis = data.get("basis")
    if isinstance(basis, Mapping):
        value = basis.get("endowments")
        field = "data.basis.endowments"
    else:
        value = basis
        field = "data.basis"

    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(
            f"{field} must be a nonempty path"
        )

    path = Path(value).resolve()
    if path.is_dir():
        path = (path / "endowments.csv").resolve()
    if not path.is_file():
        raise ConfigValidationError(
            f"{field} file does not exist: {path}"
        )
    return path


def _run_directory(
    value: str | os.PathLike[str],
) -> Path:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("run_dir must be a nonempty path")

    run_dir = Path(value).resolve()
    if run_dir.exists() and not run_dir.is_dir():
        raise ValueError(
            f"run_dir is not a directory: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_endowment_text(
    path: Path,
) -> tuple[tuple[str, str], ...]:
    try:
        with path.open(
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(
                    "endowments CSV must contain a header"
                )
            missing = {
                "eid",
                "endow_text",
            } - set(reader.fieldnames)
            if missing:
                fields = ", ".join(sorted(missing))
                raise ValueError(
                    "endowments CSV is missing required "
                    f"column(s): {fields}"
                )

            rows: list[tuple[str, str]] = []
            seen: set[str] = set()
            for row_number, row in enumerate(
                reader,
                start=2,
            ):
                eid = (row.get("eid") or "").strip()
                text = (
                    row.get("endow_text") or ""
                ).strip()
                if not eid:
                    raise ValueError(
                        "endowments CSV row "
                        f"{row_number} has an empty eid"
                    )
                if eid in seen:
                    raise ValueError(
                        f"duplicate endowment eid {eid!r}"
                    )
                if not text:
                    raise ValueError(
                        "endowments CSV row "
                        f"{row_number} has empty endow_text"
                    )
                seen.add(eid)
                rows.append((eid, text))
    except OSError as error:
        raise ValueError(
            f"could not read endowments CSV {path}: {error}"
        ) from error

    if not rows:
        raise ValueError(
            "endowments CSV must contain at least one row"
        )
    return tuple(sorted(rows))


def _canonical_labels(
    eids: tuple[str, ...],
    labels: tuple[int, ...],
) -> tuple[int, ...]:
    members: dict[int, list[str]] = {}
    for eid, label in zip(eids, labels, strict=True):
        members.setdefault(label, []).append(eid)

    ordered = sorted(
        members,
        key=lambda label: min(members[label]),
    )
    canonical = {
        label: index
        for index, label in enumerate(ordered)
    }
    return tuple(canonical[label] for label in labels)


def run(
    config: Config,
    *,
    run_dir: str | os.PathLike[str],
    verbose: bool = False,
) -> ClusterResult:
    """Cluster endowment text and write deterministic assignments."""

    del verbose
    if not isinstance(config, Config):
        raise TypeError("config must be Config")

    data = _require_mapping(
        config.data.get("data"),
        field="data",
    )
    cluster = _require_mapping(
        config.data.get("cluster"),
        field="cluster",
    )
    n_clusters = _positive_integer(
        cluster.get("n_clusters"),
        field="cluster.n_clusters",
    )

    algorithm = cluster.get(
        "algorithm",
        _ALGORITHM,
    )
    if algorithm != _ALGORITHM:
        raise ConfigValidationError(
            "cluster.algorithm must be "
            f"{_ALGORITHM!r}"
        )

    run_config = config.data.get("run", {})
    if not isinstance(run_config, Mapping):
        raise ConfigValidationError(
            "run must be a mapping"
        )
    seed = cluster.get(
        "seed",
        run_config.get("seed", 101),
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigValidationError(
            "cluster.seed must be an integer"
        )

    source_path = _endowments_path(data)
    rows = _load_endowment_text(source_path)
    if n_clusters > len(rows):
        raise ConfigValidationError(
            "cluster.n_clusters cannot exceed the number "
            "of endowments"
        )

    eids = tuple(eid for eid, _ in rows)
    texts = tuple(text for _, text in rows)
    features = TfidfVectorizer().fit_transform(texts)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "Could not find the number of physical cores"
            ),
            category=UserWarning,
            module=r"joblib\..*",
        )
        raw_labels = tuple(
            int(value)
            for value in KMeans(
                n_clusters=n_clusters,
                random_state=seed,
                n_init=10,
            ).fit_predict(features)
        )
    labels = _canonical_labels(eids, raw_labels)
    if len(set(labels)) != n_clusters:
        raise ValueError(
            "clustering produced fewer distinct clusters "
            "than requested"
        )

    assignments = tuple(
        ClusterAssignment(
            eid=eid,
            cluster=label,
        )
        for eid, label in zip(
            eids,
            labels,
            strict=True,
        )
    )

    resolved_run_dir = _run_directory(run_dir)
    assignments_path = (
        resolved_run_dir / "clusters.csv"
    ).resolve()
    with assignments_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "eid",
                "cluster",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {
                "eid": assignment.eid,
                "cluster": assignment.cluster,
            }
            for assignment in assignments
        )

    source_digest = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    cluster_sizes = Counter(labels)
    metadata_path = (
        resolved_run_dir / "clustering.json"
    ).resolve()
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": _ALGORITHM,
                "config_digest": config.digest,
                "seed": seed,
                "n_clusters": n_clusters,
                "n_endowments": len(assignments),
                "cluster_sizes": {
                    str(label): cluster_sizes[label]
                    for label in sorted(cluster_sizes)
                },
                "source": {
                    "path": str(source_path),
                    "sha256": source_digest,
                },
                "assignments": assignments_path.name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = config.write_snapshot(
        resolved_run_dir
    ).resolve()

    return ClusterResult(
        config_digest=config.digest,
        run_dir=resolved_run_dir,
        assignments=assignments,
        assignments_path=assignments_path,
        metadata_path=metadata_path,
        config_path=config_path,
    )


__all__ = [
    "ClusterAssignment",
    "ClusterResult",
    "run",
]
