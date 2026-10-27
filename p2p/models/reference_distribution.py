"""Stage 2 fitted around a fixed reference distribution.

A reference distribution ``a`` (one value per option row, summing to one within
each question) enters every prediction with weight one:

    prediction = a + X @ beta

The agents therefore explain each target's *deviation* from the reference
rather than its level. Internally the estimator is fitted to ``y - a`` with the
question-mass target lowered by one, so that each question's prediction still
sums to the configured total; when every agent answers every question this
makes the agents' weights sum to zero. The penalty, the cross-validation and
the scoring are those of the wrapped estimator: ``score`` and ``predict`` work
on the full prediction, so the error of ``y - a`` against ``X @ beta`` and the
error of ``y`` against ``a + X @ beta`` are the same number.

The drift refit of a survey across waves is the case where ``a`` is the earlier
wave's marginals.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

TOLERANCE = 1e-9


def reference_vector(
    reference: Mapping[str, float],
    rows: pd.Index,
) -> np.ndarray:
    """The reference value of every option row, in row order."""
    missing = [row for row in rows if row not in reference]
    if missing:
        raise ValueError(
            "reference distribution has no value for option rows "
            f"{missing[:5]}{' ...' if len(missing) > 5 else ''}"
        )
    return np.asarray([float(reference[row]) for row in rows], dtype=float)


def check_reference_masses(
    reference: Mapping[str, float],
    constraint_groups: Mapping[str, list[str]],
    rows: pd.Index,
) -> None:
    """Every question's reference mass over its modelled rows must be one."""
    present = set(rows)
    for question, members in constraint_groups.items():
        modelled = [row for row in members if row in present]
        if not modelled:
            continue
        mass = sum(float(reference[row]) for row in modelled)
        if abs(mass - 1.0) > TOLERANCE:
            raise ValueError(
                f"reference distribution for {question!r} sums to {mass!r} over "
                "its modelled option rows; it must sum to one"
            )


def with_reference_distribution(model_cls: type, reference: Mapping[str, float]) -> type:
    """Return ``model_cls`` fitted around ``reference`` (see the module docstring).

    ``reference`` maps option-row ids to values. The returned class takes the
    same keyword arguments; ``group_total`` keeps its meaning as the total of
    each question's full prediction.
    """
    frozen = dict(reference)

    class ReferenceDistributionModel(model_cls):
        reference_distribution_ = frozen

        def __init__(self, *args, group_total: float = 1.0, **kwargs):
            # The estimator only sees the deviation, whose mass target is the
            # full prediction's total minus the reference's mass of one.
            super().__init__(*args, group_total=group_total - 1.0, **kwargs)
            self.prediction_group_total_ = group_total

        def fit(self, A: pd.DataFrame, y: np.ndarray):
            offset = reference_vector(frozen, A.index)
            return super().fit(A, np.asarray(y, dtype=float) - offset)

        def predict(self, A: pd.DataFrame) -> np.ndarray:
            return super().predict(A) + reference_vector(frozen, A.index)

    ReferenceDistributionModel.__name__ = f"{model_cls.__name__}WithReference"
    ReferenceDistributionModel.__qualname__ = ReferenceDistributionModel.__name__
    return ReferenceDistributionModel


__all__ = [
    "check_reference_masses",
    "reference_vector",
    "with_reference_distribution",
]
