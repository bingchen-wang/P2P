"""Pure numerical metrics and explicit probability transformations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "clip_and_normalize",
    "emd",
    "mcpa",
    "mse",
    "one_minus_jsd",
    "project_to_simplex",
    "validate_distribution",
]


def _prefix(qid: str | None) -> str:
    return f"qid {qid!r}: " if qid is not None else ""


def _as_vector(
    values: ArrayLike,
    *,
    name: str,
    qid: str | None = None,
) -> NDArray[np.float64]:
    """Convert input to a finite, nonempty, one-dimensional float array."""
    prefix = _prefix(qid)

    try:
        vector = np.array(values, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{prefix}{name} must be numeric") from exc

    if vector.ndim != 1:
        raise ValueError(f"{prefix}{name} must be one-dimensional")
    if vector.size == 0:
        raise ValueError(f"{prefix}{name} must not be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{prefix}{name} must contain only finite values")

    return vector


def validate_distribution(
    values: ArrayLike,
    *,
    name: str = "distribution",
    qid: str | None = None,
    atol: float = 1e-8,
) -> NDArray[np.float64]:
    """Validate and return a probability vector without transforming it."""
    vector = _as_vector(values, name=name, qid=qid)
    prefix = _prefix(qid)

    if np.any(vector < 0.0):
        raise ValueError(f"{prefix}{name} contains negative probability mass")
    if np.any(vector > 1.0):
        raise ValueError(f"{prefix}{name} contains a probability greater than 1")

    total = float(vector.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=atol):
        raise ValueError(
            f"{prefix}{name} must sum to 1; observed sum was {total:.17g}"
        )

    return vector


def _distribution_pair(
    observed: ArrayLike,
    predicted: ArrayLike,
    *,
    qid: str | None,
    atol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    observed_vector = validate_distribution(
        observed,
        name="observed",
        qid=qid,
        atol=atol,
    )
    predicted_vector = validate_distribution(
        predicted,
        name="predicted",
        qid=qid,
        atol=atol,
    )

    if observed_vector.shape != predicted_vector.shape:
        prefix = _prefix(qid)
        raise ValueError(
            f"{prefix}observed and predicted must have the same shape; "
            f"got {observed_vector.shape} and {predicted_vector.shape}"
        )

    return observed_vector, predicted_vector


def clip_and_normalize(
    values: ArrayLike,
    *,
    qid: str | None = None,
) -> NDArray[np.float64]:
    """Clip negative entries and normalize the remaining positive mass."""
    vector = _as_vector(values, name="values", qid=qid)
    clipped = np.maximum(vector, 0.0)
    total = float(clipped.sum())

    if total <= 0.0:
        raise ValueError(
            f"{_prefix(qid)}values have no positive mass after clipping"
        )

    return clipped / total


def project_to_simplex(
    values: ArrayLike,
    *,
    qid: str | None = None,
) -> NDArray[np.float64]:
    """Return the Euclidean projection onto the unit probability simplex."""
    vector = _as_vector(values, name="values", qid=qid)

    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, vector.size + 1, dtype=float)
    candidates = np.flatnonzero(ordered - cumulative / indices > 0.0)

    # A finite, nonempty vector always has at least one candidate.
    rho = int(candidates[-1])
    threshold = cumulative[rho] / float(rho + 1)

    return np.maximum(vector - threshold, 0.0)


def mse(observed: ArrayLike, predicted: ArrayLike) -> float:
    """Return raw option-level mean squared error without transformation."""
    try:
        observed_array = np.asarray(observed, dtype=float)
        predicted_array = np.asarray(predicted, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("observed and predicted must be numeric") from exc

    if observed_array.shape != predicted_array.shape:
        raise ValueError(
            "observed and predicted must have the same shape; "
            f"got {observed_array.shape} and {predicted_array.shape}"
        )
    if observed_array.size == 0:
        raise ValueError("observed and predicted must not be empty")
    if not np.all(np.isfinite(observed_array)):
        raise ValueError("observed must contain only finite values")
    if not np.all(np.isfinite(predicted_array)):
        raise ValueError("predicted must contain only finite values")

    return float(np.mean(np.square(predicted_array - observed_array)))


def one_minus_jsd(
    observed: ArrayLike,
    predicted: ArrayLike,
    *,
    qid: str | None = None,
    atol: float = 1e-8,
) -> float:
    """Return one minus Jensen-Shannon distance using natural logarithms."""
    p, q = _distribution_pair(observed, predicted, qid=qid, atol=atol)
    midpoint = 0.5 * (p + q)

    p_mask = p > 0.0
    q_mask = q > 0.0

    p_kl = float(np.sum(p[p_mask] * np.log(p[p_mask] / midpoint[p_mask])))
    q_kl = float(np.sum(q[q_mask] * np.log(q[q_mask] / midpoint[q_mask])))
    divergence = max(0.0, 0.5 * (p_kl + q_kl))

    # This matches scipy.spatial.distance.jensenshannon with base=None.
    return 1.0 - float(np.sqrt(divergence))


def emd(
    observed: ArrayLike,
    predicted: ArrayLike,
    *,
    qid: str | None = None,
    atol: float = 1e-8,
) -> float:
    """Return normalized EMD for vectors in canonical option order."""
    p, q = _distribution_pair(observed, predicted, qid=qid, atol=atol)

    if p.size == 1:
        return 0.0

    raw_emd = float(np.abs(np.cumsum(p - q)[:-1]).sum())
    return raw_emd / float(p.size - 1)


def mcpa(
    observed: ArrayLike,
    predicted: ArrayLike,
    *,
    qid: str | None = None,
    atol: float = 1e-8,
) -> float:
    """Return majority-category prediction accuracy for one question."""
    p, q = _distribution_pair(observed, predicted, qid=qid, atol=atol)

    # np.argmax preserves the legacy first-index behavior for ties.
    return float(np.argmax(p) == np.argmax(q))