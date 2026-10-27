"""Cross-validation fold assignments: generate, record and replay.

A *question* is the indivisible cross-validation unit: all of its option rows go
to the same fold. A *fold assignment* maps every question to a fold index
``0 .. n_splits - 1`` and fully determines the folds.

Three methods produce an assignment:

``hashed`` (default)
    Questions are placed largest first; ties between equal-size questions are
    broken by ``sha256("{seed}:{question_id}")``. Each question goes to the
    fold with the fewest option rows so far, the lowest fold index winning a
    tie. Only the Python standard library decides the order, so the result is
    identical on every machine and library version.
``sklearn``
    Whatever the installed ``sklearn.model_selection.GroupKFold`` does, without
    shuffling. scikit-learn up to 1.7 broke ties with an unstable sort whose
    order depended on the installation (scikit-learn issue #29495), so this is
    for quick checks, not for reproducible results.
``replay``
    The assignment is read from a fold record written by an earlier run.

Every assignment carries a ``partition_digest`` that identifies which questions
share a fold, ignoring fold numbering: renumbering folds cannot change any
result, because each penalty is scored by the average error over the folds.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
from types import MappingProxyType
from typing import Any

import numpy as np


FOLD_METHODS = ("hashed", "sklearn", "replay")
DEFAULT_FOLD_METHOD = "hashed"
# ModelCV's default ``random_state``, so a caller that sets no seed anywhere
# still gets one well-defined assignment.
DEFAULT_FOLD_SEED = 101
FOLD_RECORD_FILENAME = "folds.json"
FOLD_RECORD_SCHEMA_VERSION = 2
_READABLE_SCHEMA_VERSIONS = (1, 2)


def _producer() -> Mapping[str, str]:
    """Versions that could matter for reproducing a fold assignment."""

    import sklearn

    try:
        p2p_version = metadata.version("prompts-to-proxies")
    except metadata.PackageNotFoundError:
        p2p_version = "unknown"
    return MappingProxyType(
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "p2p": p2p_version,
        }
    )


def _check_n_splits(n_splits: Any) -> int:
    if (
        isinstance(n_splits, bool)
        or not isinstance(n_splits, (int, np.integer))
        or n_splits < 2
    ):
        raise ValueError("n_splits must be an integer of at least 2")
    return int(n_splits)


def _check_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("the fold seed must be an integer")
    return int(seed)


def count_question_sizes(group_labels: Sequence[str]) -> dict[str, int]:
    """Count the option rows of each question in a per-row label sequence."""

    if isinstance(group_labels, (str, bytes, bytearray)):
        raise ValueError("groups must be a one-dimensional sequence")

    labels = np.asarray(group_labels, dtype=object)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(
            "groups must be a nonempty one-dimensional sequence"
        )
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("group labels must be nonempty strings")

    return dict(sorted(Counter(labels.tolist()).items()))


def _check_sizes(sizes: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(sizes, Mapping) or not sizes:
        raise ValueError("question sizes must be a nonempty mapping")
    checked: dict[str, int] = {}
    for question, size in sizes.items():
        if not isinstance(question, str) or not question:
            raise ValueError("question ids must be nonempty strings")
        if (
            isinstance(size, bool)
            or not isinstance(size, (int, np.integer))
            or size < 1
        ):
            raise ValueError(
                f"question {question!r} must have a positive integer size"
            )
        checked[question] = int(size)
    return checked


def _hash_key(seed: int, question: str) -> str:
    return hashlib.sha256(f"{seed}:{question}".encode("utf-8")).hexdigest()


def _greedy(
    order: Sequence[str],
    sizes: Mapping[str, int],
    n_splits: int,
) -> dict[str, int]:
    """Place each question, in order, into the fold with the fewest rows."""

    loads = [0] * n_splits
    assignment: dict[str, int] = {}
    for question in order:
        fold = min(range(n_splits), key=lambda index: (loads[index], index))
        assignment[question] = fold
        loads[fold] += sizes[question]
    return assignment


def _hashed_assignment(
    sizes: Mapping[str, int],
    n_splits: int,
    seed: int,
) -> dict[str, int]:
    order = sorted(
        sizes,
        # The question id is a final key only for total determinism; two
        # SHA-256 values colliding is not a practical concern.
        key=lambda question: (
            -sizes[question],
            _hash_key(seed, question),
            question,
        ),
    )
    return _greedy(order, sizes, n_splits)


def _sklearn_assignment(
    sizes: Mapping[str, int],
    n_splits: int,
) -> dict[str, int]:
    from sklearn.model_selection import GroupKFold

    questions = sorted(sizes)
    labels = np.repeat(
        np.asarray(questions),
        [sizes[question] for question in questions],
    )
    assignment: dict[str, int] = {}
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(np.zeros(labels.size), groups=labels)
    ):
        for question in set(labels[validation_indices].tolist()):
            assignment[str(question)] = fold
    return assignment


def _check_assignment(
    assignment: Mapping[str, Any],
    n_splits: int,
) -> dict[str, int]:
    """Validate fold indices and that every fold is used."""

    if not isinstance(assignment, Mapping) or not assignment:
        raise ValueError("fold assignments must be a nonempty mapping")

    checked: dict[str, int] = {}
    for question, fold in assignment.items():
        if not isinstance(question, str) or not question:
            raise ValueError("fold assignment keys must be nonempty strings")
        if (
            isinstance(fold, bool)
            or not isinstance(fold, (int, np.integer))
            or not 0 <= fold < n_splits
        ):
            raise ValueError(
                "explicit fold assignment fold index must be an integer "
                f"between 0 and {n_splits - 1}"
            )
        checked[question] = int(fold)

    if set(checked.values()) != set(range(n_splits)):
        raise ValueError(
            "explicit fold assignments must assign at least one group to "
            "every fold"
        )
    return checked


def _fold_questions(
    assignment: Mapping[str, int],
    n_splits: int,
) -> list[list[str]]:
    return [
        sorted(q for q, fold in assignment.items() if fold == index)
        for index in range(n_splits)
    ]


def _list_of_names(names: Sequence[str], limit: int = 10) -> str:
    shown = ", ".join(repr(name) for name in names[:limit])
    more = len(names) - limit
    return shown + (f" and {more} more" if more > 0 else "")


@dataclass(frozen=True)
class FoldAssignment:
    """A complete question -> fold assignment and its provenance."""

    n_splits: int
    method: str
    seed: int | None
    assignment: Mapping[str, int]
    question_sizes: Mapping[str, int] | None
    producer: Mapping[str, str]
    source: str | None = None
    # Schema-1 records hold per-fold option-row totals, not per-question sizes.
    fold_row_counts: tuple[int, ...] | None = None
    # For a replayed record: how the record itself was produced.
    origin: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(
        cls,
        assignment: Mapping[str, Any],
        *,
        n_splits: int,
        source: str | None = None,
    ) -> FoldAssignment:
        """Wrap an explicit mapping, e.g. one passed in by a test."""

        n_splits = _check_n_splits(n_splits)
        return cls(
            n_splits=n_splits,
            method="replay",
            seed=None,
            assignment=MappingProxyType(_check_assignment(assignment, n_splits)),
            question_sizes=None,
            producer=_producer(),
            source=source,
        )

    @property
    def partition_digest(self) -> str:
        """SHA-256 of the sorted folds, each a sorted list of question ids.

        Fold numbering is dropped, so two assignments share a digest exactly
        when they put the same questions together.
        """

        folds = sorted(_fold_questions(self.assignment, self.n_splits))
        encoded = json.dumps(
            folds,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def indices(
        self,
        group_labels: Sequence[str],
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Training and validation row indices for each fold, in fold order.

        The labels must name exactly the assigned questions. When the
        assignment knows its question sizes (or, for schema-1 records, its
        per-fold row totals), the labels must reproduce those too: a mismatch
        means the assignment was made for a differently cleaned design.
        """

        observed = count_question_sizes(group_labels)
        assigned = set(self.assignment)
        if set(observed) != assigned:
            missing = sorted(set(observed) - assigned)
            unexpected = sorted(assigned - set(observed))
            raise ValueError(
                "explicit fold assignments must cover exactly the observed "
                f"groups; missing={missing}, unexpected={unexpected}"
            )

        if self.question_sizes is not None:
            differing = sorted(
                question
                for question, size in observed.items()
                if self.question_sizes.get(question) != size
            )
            if differing:
                raise ValueError(
                    "the fold assignment was made for a differently cleaned "
                    "design: option-row counts differ for "
                    + _list_of_names(differing)
                )
        elif self.fold_row_counts is not None:
            totals = tuple(
                sum(
                    observed[question]
                    for question, fold in self.assignment.items()
                    if fold == index
                )
                for index in range(self.n_splits)
            )
            if totals != self.fold_row_counts:
                raise ValueError(
                    "the fold assignment was made for a differently cleaned "
                    f"design: per-fold option rows are {list(totals)}, the "
                    f"record says {list(self.fold_row_counts)}"
                )

        labels = np.asarray(group_labels, dtype=object)
        row_folds = np.asarray(
            [self.assignment[label] for label in labels.tolist()],
            dtype=int,
        )
        rows = np.arange(labels.size, dtype=int)
        folds = []
        for fold in range(self.n_splits):
            validation = rows[row_folds == fold]
            training = rows[row_folds != fold]
            if validation.size == 0 or training.size == 0:
                raise ValueError(f"fold {fold} is empty")
            folds.append((training, validation))
        return tuple(folds)

    def to_record(self) -> dict[str, Any]:
        """The schema-2 fold record written beside every run."""

        sizes = self.question_sizes
        record: dict[str, Any] = {
            "schema_version": FOLD_RECORD_SCHEMA_VERSION,
            "method": self.method,
            "seed": self.seed,
            "n_splits": self.n_splits,
            "question_count": len(self.assignment),
            "option_row_count": None if sizes is None else sum(sizes.values()),
            "producer": dict(self.producer),
            "partition_digest": self.partition_digest,
            "source": self.source,
        }
        if self.origin is not None:
            record["origin"] = dict(self.origin)
        record["question_sizes"] = None if sizes is None else dict(sorted(sizes.items()))
        record["folds"] = [
            {
                "fold": index,
                "questions": questions,
                "option_rows": (
                    None if sizes is None
                    else sum(sizes[question] for question in questions)
                ),
            }
            for index, questions in enumerate(
                _fold_questions(self.assignment, self.n_splits)
            )
        ]
        return record


def assign_folds(
    sizes: Mapping[str, int],
    *,
    n_splits: int,
    method: str = DEFAULT_FOLD_METHOD,
    seed: int | None = DEFAULT_FOLD_SEED,
) -> FoldAssignment:
    """Generate a fold assignment for questions of the given sizes."""

    n_splits = _check_n_splits(n_splits)
    sizes = _check_sizes(sizes)
    if n_splits > len(sizes):
        raise ValueError("n_splits must not exceed the number of groups")

    if method == "hashed":
        seed = _check_seed(seed)
        assignment = _hashed_assignment(sizes, n_splits, seed)
    elif method == "sklearn":
        seed = None
        assignment = _sklearn_assignment(sizes, n_splits)
    elif method == "replay":
        raise ValueError("replay reads a fold record; use read_fold_record")
    else:
        raise ValueError(
            f"unknown fold method {method!r}; expected one of {FOLD_METHODS}"
        )

    return FoldAssignment(
        n_splits=n_splits,
        method=method,
        seed=seed,
        assignment=MappingProxyType(_check_assignment(assignment, n_splits)),
        question_sizes=MappingProxyType(dict(sorted(sizes.items()))),
        producer=_producer(),
    )


def resolve_fold_assignment(
    group_labels: Sequence[str],
    *,
    n_splits: int,
    assignments: FoldAssignment | Mapping[str, Any] | None = None,
    method: str = DEFAULT_FOLD_METHOD,
    seed: int | None = DEFAULT_FOLD_SEED,
) -> FoldAssignment:
    """Return the assignment a cross-validation run will use.

    An explicit ``FoldAssignment`` or plain mapping wins; otherwise one is
    generated from the labels with ``method`` and ``seed``.
    """

    n_splits = _check_n_splits(n_splits)
    sizes = count_question_sizes(group_labels)
    if n_splits > len(sizes):
        raise ValueError("n_splits must not exceed the number of groups")

    if assignments is None:
        return assign_folds(sizes, n_splits=n_splits, method=method, seed=seed)
    if isinstance(assignments, FoldAssignment):
        if assignments.n_splits != n_splits:
            raise ValueError(
                f"the fold assignment has {assignments.n_splits} folds but "
                f"{n_splits} were requested"
            )
        return assignments
    if isinstance(assignments, Mapping):
        return FoldAssignment.from_mapping(assignments, n_splits=n_splits)
    raise TypeError("assignments must be a FoldAssignment, a mapping or None")


def write_fold_record(
    path: str | os.PathLike[str],
    assignment: FoldAssignment,
) -> Path:
    """Write ``assignment`` as a schema-2 JSON fold record."""

    if not isinstance(assignment, FoldAssignment):
        raise TypeError("assignment must be a FoldAssignment")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(assignment.to_record(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target.resolve()


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(f"fold record {field} must be an integer >= {minimum}")
    return value


def read_fold_record(path: str | os.PathLike[str]) -> FoldAssignment:
    """Read a schema-1 or schema-2 fold record for replay.

    Schema 1 is the session-4 parity-fixture format (``validation_groups``,
    optional per-fold ``validation_rows``); its ``validation_qids_digest`` has
    no documented definition and is ignored. Schema 2 is what P2P writes; its
    ``partition_digest`` is recomputed and must match.
    """

    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"fold record does not exist: {source}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid fold record {source}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"fold record {source} must be a JSON object")

    version = payload.get("schema_version")
    if version not in _READABLE_SCHEMA_VERSIONS:
        raise ValueError(
            f"fold record schema_version must be one of "
            f"{_READABLE_SCHEMA_VERSIONS}; got {version!r}"
        )
    n_splits = _check_n_splits(payload.get("n_splits"))

    folds = payload.get("folds")
    if (
        not isinstance(folds, Sequence)
        or isinstance(folds, (str, bytes, bytearray))
        or len(folds) != n_splits
    ):
        raise ValueError(f"fold record folds must contain exactly {n_splits} entries")

    key = "validation_groups" if version == 1 else "questions"
    assignment: dict[str, int] = {}
    row_counts: dict[int, int] = {}
    seen_folds: set[int] = set()
    for entry in folds:
        if not isinstance(entry, Mapping):
            raise ValueError("each fold record entry must be a mapping")
        fold = _require_int(entry.get("fold"), field="fold index")
        if fold >= n_splits:
            raise ValueError("fold record fold index is out of range")
        if fold in seen_folds:
            raise ValueError(f"fold record lists fold {fold} more than once")
        seen_folds.add(fold)

        questions = entry.get(key)
        if (
            not isinstance(questions, Sequence)
            or isinstance(questions, (str, bytes, bytearray))
            or not questions
        ):
            raise ValueError(f"fold record {key} must be a nonempty sequence")
        for question in questions:
            if not isinstance(question, str) or not question:
                raise ValueError("fold record question ids must be nonempty strings")
            if question in assignment:
                raise ValueError(
                    f"fold record assigns question {question!r} more than once"
                )
            assignment[question] = fold

        rows_key = "validation_rows" if version == 1 else "option_rows"
        if entry.get(rows_key) is not None:
            row_counts[fold] = _require_int(entry[rows_key], field=rows_key, minimum=1)

    if seen_folds != set(range(n_splits)):
        raise ValueError("fold record must contain every fold index")

    sizes = None
    fold_row_counts = None
    if version == 2:
        raw_sizes = payload.get("question_sizes")
        if raw_sizes is not None:
            sizes = _check_sizes(raw_sizes)
            if set(sizes) != set(assignment):
                raise ValueError(
                    "fold record question_sizes must name exactly its assigned questions"
                )
    if sizes is None and len(row_counts) == n_splits:
        fold_row_counts = tuple(row_counts[index] for index in range(n_splits))

    origin: dict[str, Any] = {"schema_version": version}
    for field in ("method", "seed", "producer", "source_run", "dataset"):
        if field in payload:
            origin[field] = payload[field]

    record = FoldAssignment(
        n_splits=n_splits,
        method="replay",
        seed=None,
        assignment=MappingProxyType(_check_assignment(assignment, n_splits)),
        question_sizes=None if sizes is None else MappingProxyType(dict(sorted(sizes.items()))),
        producer=_producer(),
        source=str(source),
        fold_row_counts=fold_row_counts,
        origin=MappingProxyType(origin),
    )

    if version == 2:
        stated = payload.get("partition_digest")
        if stated is not None and stated != record.partition_digest:
            raise ValueError(
                "fold record partition_digest does not match its folds: "
                f"stated {stated}, recomputed {record.partition_digest}"
            )
    return record


__all__ = [
    "DEFAULT_FOLD_METHOD",
    "DEFAULT_FOLD_SEED",
    "FOLD_METHODS",
    "FOLD_RECORD_FILENAME",
    "FoldAssignment",
    "assign_folds",
    "count_question_sizes",
    "read_fold_record",
    "resolve_fold_assignment",
    "write_fold_record",
]
