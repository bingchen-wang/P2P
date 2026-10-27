"""Contracts for deterministic grouped cross-validation folds."""

from __future__ import annotations

import numpy as np
import pytest

from p2p.models.constrained_cv import build_group_folds


def _validation_group_sets(
    groups: tuple[str, ...],
    folds,
) -> list[set[str]]:
    group_array = np.asarray(groups)
    return [
        set(group_array[validation_indices])
        for _, validation_indices in folds
    ]


def test_build_group_folds_default_is_seeded_hash_order() -> None:
    """Ties between equal-size questions are broken by the seeded hash.

    Pinned to the ``hashed`` rule with the default seed 101. (The earlier
    default broke ties by descending name, which gave {b, e}, {a, d}, {c};
    that is scikit-learn 1.9's rule and is now reached with
    ``method="sklearn"``.)
    """

    groups = (
        "a",
        "a",
        "b",
        "b",
        "c",
        "c",
        "d",
        "d",
        "e",
        "e",
    )

    folds = build_group_folds(
        groups,
        n_splits=3,
    )

    assert _validation_group_sets(
        groups,
        folds,
    ) == [
        {"c", "d"},
        {"a", "b"},
        {"e"},
    ]

    all_indices = set(range(len(groups)))
    seen_validation_indices: set[int] = set()

    for train_indices, validation_indices in folds:
        train = set(train_indices.tolist())
        validation = set(validation_indices.tolist())

        assert train.isdisjoint(validation)
        assert train | validation == all_indices
        assert seen_validation_indices.isdisjoint(
            validation
        )
        seen_validation_indices.update(validation)

    assert seen_validation_indices == all_indices


def test_build_group_folds_is_invariant_to_sample_order() -> None:
    first_groups = (
        "a",
        "a",
        "b",
        "b",
        "c",
        "c",
        "d",
        "d",
        "e",
        "e",
    )
    second_groups = (
        "e",
        "c",
        "a",
        "d",
        "b",
        "e",
        "c",
        "a",
        "d",
        "b",
    )

    first = build_group_folds(
        first_groups,
        n_splits=3,
    )
    second = build_group_folds(
        second_groups,
        n_splits=3,
    )

    assert _validation_group_sets(
        first_groups,
        first,
    ) == _validation_group_sets(
        second_groups,
        second,
    )


def test_build_group_folds_honors_explicit_assignments() -> None:
    groups = (
        "q1",
        "q1",
        "q2",
        "q2",
        "q3",
        "q3",
        "q4",
        "q4",
    )

    folds = build_group_folds(
        groups,
        n_splits=2,
        assignments={
            "q1": 1,
            "q2": 0,
            "q3": 1,
            "q4": 0,
        },
    )

    assert _validation_group_sets(
        groups,
        folds,
    ) == [
        {"q2", "q4"},
        {"q1", "q3"},
    ]


@pytest.mark.parametrize(
    ("assignments", "match"),
    [
        (
            {
                "q1": 0,
                "q2": 1,
            },
            "must cover exactly",
        ),
        (
            {
                "q1": 0,
                "q2": 1,
                "q3": 2,
            },
            "fold index",
        ),
        (
            {
                "q1": 0,
                "q2": 0,
                "q3": 0,
            },
            "must assign at least one group",
        ),
    ],
)
def test_build_group_folds_rejects_invalid_assignments(
    assignments: dict[str, int],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_group_folds(
            ("q1", "q2", "q3"),
            n_splits=2,
            assignments=assignments,
        )


@pytest.mark.parametrize(
    "n_splits",
    [
        1,
        4,
    ],
)
def test_build_group_folds_rejects_invalid_fold_count(
    n_splits: int,
) -> None:
    with pytest.raises(ValueError):
        build_group_folds(
            ("q1", "q2", "q3"),
            n_splits=n_splits,
        )

# --- fold grouping is independent of the mass constraints -------------------


class _RecordingModel:
    """Minimal Model stand-in that records how it was constructed."""

    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).calls.append(kwargs)

    def fit(self, A, y):
        self.A_ = A
        self.beta_ = np.zeros(A.shape[1])
        self.status_ = "optimal"
        return self

    def score(self, A, y):
        return float(len(y))


def _toy_design() -> tuple["pd.DataFrame", np.ndarray, dict[str, list[str]]]:
    import pandas as pd

    groups = {f"q{q}": [f"q{q}__{o}" for o in range(3)] for q in range(6)}
    index = [label for members in groups.values() for label in members]
    frame = pd.DataFrame(
        np.eye(len(index), 4)[:, :4],
        index=index,
        columns=[f"agent_{i}" for i in range(4)],
    )
    return frame, np.arange(len(index), dtype=float), groups


def _run(**cv_kwargs):
    from p2p.models.constrained_cv import ConstrainedModelCV

    A, y, groups = _toy_design()
    _RecordingModel.calls = []
    cv = ConstrainedModelCV(
        model_cls=_RecordingModel,
        alphas=[0.1],
        k=3,
        verbose=False,
        **cv_kwargs,
    )
    cv.fit(A, y)
    return cv, _RecordingModel.calls, groups


def test_fold_groups_defaults_to_constraint_groups() -> None:
    _, _, groups = _toy_design()
    explicit, explicit_calls, _ = _run(
        model_kwargs={"constraint_groups": groups},
        fold_groups=groups,
    )
    implicit, implicit_calls, _ = _run(
        model_kwargs={"constraint_groups": groups},
    )

    assert (
        explicit.diagnostics_["fold_assignments"]
        == implicit.diagnostics_["fold_assignments"]
    )
    assert len(explicit_calls) == len(implicit_calls)


def test_fold_groups_survive_without_constraint_groups() -> None:
    _, _, groups = _toy_design()
    grouped, calls, _ = _run(
        model_kwargs={},
        fold_groups=groups,
    )

    # Folds are still cut along question boundaries ...
    assignments = grouped.diagnostics_["fold_assignments"]
    assert set(assignments) == set(groups)
    # ... while the model itself receives no mass constraints.
    assert all("constraint_groups" not in call for call in calls)


def test_missing_grouping_is_rejected_rather_than_cut_at_random() -> None:
    """Splitting one question's option rows across folds would leak."""

    with pytest.raises(ValueError, match="must be grouped by question"):
        _run(model_kwargs={})


def test_fold_and_constraint_groups_must_agree() -> None:
    _, _, groups = _toy_design()
    coarser = {"half_a": sum(list(groups.values())[:3], [])}
    coarser["half_b"] = sum(list(groups.values())[3:], [])

    with pytest.raises(ValueError, match="same partition"):
        _run(model_kwargs={"constraint_groups": groups}, fold_groups=coarser)


def test_fold_groups_rejects_non_mapping() -> None:
    from p2p.models.constrained_cv import ConstrainedModelCV

    with pytest.raises(TypeError, match="fold_groups must be a mapping"):
        ConstrainedModelCV(
            model_cls=_RecordingModel,
            alphas=[0.1],
            fold_groups=["q0", "q1"],
        )
