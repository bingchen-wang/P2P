"""Deterministic, offline sampling over an existing response basis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import random
from types import MappingProxyType
from typing import Any

from p2p.endowments import ActiveEndowments


@dataclass(frozen=True, slots=True)
class EntropySample:
    """Best subset found for an explicitly requested mean entropy."""

    selected_eids: tuple[str, ...]
    target_entropy: float
    achieved_entropy: float
    absolute_error: float
    sample_size: int
    method: str
    candidates_evaluated: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_eids": list(self.selected_eids),
            "target_entropy": self.target_entropy,
            "achieved_entropy": self.achieved_entropy,
            "absolute_error": self.absolute_error,
            "sample_size": self.sample_size,
            "method": self.method,
            "candidates_evaluated": self.candidates_evaluated,
            "seed": self.seed,
        }


def _finite_unit_interval(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return number


def sample_to_target_mean_entropy(
    endowments,
    responses,
    survey,
    *,
    sample_size: int,
    target_entropy: float,
    seed: int = 101,
    max_evaluations: int = 5000,
    tolerance: float | None = None,
    require_tolerance: bool = False,
) -> EntropySample:
    """Select a fixed-size subset closest to a requested mean entropy.

    The objective calls the :meth:`ActiveEndowments.compute_entropy` kernel,
    so the estimand is the same mean of normalized per-question categorical
    entropies that active generation tracks.  Small search spaces are enumerated exactly.  Larger ones
    use a seeded random search followed by deterministic one-swap refinement.
    The global random state is never touched.
    """

    target = _finite_unit_interval(
        target_entropy,
        field="target_entropy",
    )
    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, int)
        or sample_size <= 0
    ):
        raise ValueError("sample_size must be a positive integer")
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, int)
        or max_evaluations <= 0
    ):
        raise ValueError("max_evaluations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if tolerance is not None:
        tolerance = _finite_unit_interval(
            tolerance,
            field="tolerance",
        )
    if not isinstance(require_tolerance, bool):
        raise ValueError("require_tolerance must be boolean")
    if require_tolerance and tolerance is None:
        raise ValueError(
            "require_tolerance needs an explicit tolerance"
        )

    records = list(endowments.get_endowments())
    by_eid: dict[str, MappingProxyType] = {}
    for record in records:
        eid = record.get("eid")
        if not isinstance(eid, str) or not eid:
            raise ValueError("endowments must have nonempty EIDs")
        if eid in by_eid:
            raise ValueError(f"duplicate endowment EID {eid!r}")
        by_eid[eid] = MappingProxyType(dict(record))
    eids = tuple(sorted(by_eid))
    if sample_size > len(eids):
        raise ValueError(
            f"sample_size {sample_size} exceeds {len(eids)} candidates"
        )

    evaluated: dict[tuple[str, ...], float] = {}

    def score(subset: tuple[str, ...]) -> float:
        normalized = tuple(sorted(subset))
        if normalized not in evaluated:
            selected = [by_eid[eid] for eid in normalized]
            evaluated[normalized] = float(
                ActiveEndowments.compute_entropy(
                    selected,
                    responses,
                    survey,
                    normalize=True,
                )
            )
        return evaluated[normalized]

    def ranking(subset: tuple[str, ...]) -> tuple[float, tuple[str, ...]]:
        normalized = tuple(sorted(subset))
        return abs(score(normalized) - target), normalized

    combination_count = math.comb(len(eids), sample_size)
    if combination_count <= max_evaluations:
        method = "exact"
        candidates = combinations(eids, sample_size)
        best = min(candidates, key=ranking)
    else:
        method = "seeded_search"
        rng = random.Random(seed)
        initial = tuple(sorted(rng.sample(eids, sample_size)))
        best = initial
        score(best)
        while len(evaluated) < max_evaluations:
            candidate = tuple(sorted(rng.sample(eids, sample_size)))
            candidate_rank = ranking(candidate)
            if candidate_rank < ranking(best):
                best = candidate
            if len(evaluated) >= max_evaluations:
                break

            selected = set(best)
            outside = [eid for eid in eids if eid not in selected]
            improved = False
            for removed in sorted(selected):
                for added in outside:
                    swapped = tuple(
                        sorted((selected - {removed}) | {added})
                    )
                    swapped_rank = ranking(swapped)
                    if swapped_rank < ranking(best):
                        best = swapped
                        improved = True
                    if len(evaluated) >= max_evaluations:
                        break
                if len(evaluated) >= max_evaluations:
                    break
            if (
                tolerance is not None
                and ranking(best)[0] <= tolerance
            ):
                break
            if not improved and len(evaluated) >= combination_count:
                break

    best = tuple(sorted(best))
    achieved = score(best)
    error = abs(achieved - target)
    if (
        require_tolerance
        and tolerance is not None
        and error > tolerance
    ):
        raise ValueError(
            "no entropy subset met the requested tolerance: "
            f"target={target:.6f}, achieved={achieved:.6f}, "
            f"error={error:.6f}, tolerance={tolerance:.6f}"
        )
    return EntropySample(
        selected_eids=best,
        target_entropy=target,
        achieved_entropy=achieved,
        absolute_error=error,
        sample_size=sample_size,
        method=method,
        candidates_evaluated=len(evaluated),
        seed=seed,
    )


__all__ = [
    "EntropySample",
    "sample_to_target_mean_entropy",
]
