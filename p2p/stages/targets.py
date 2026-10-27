"""Human-target loading for Stage-2 execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from p2p.metrics import validate_distribution


@dataclass(frozen=True, slots=True)
class TargetBundle:
    """Validated categorical targets and their binary-option expansion."""

    source_path: Path | None
    source_digest: str
    selection_digest: str
    subgroup: str | None
    raw: Mapping[str, Mapping[str, float]]
    binary: Mapping[str, float]
    renormalized: tuple[str, ...] = ()
    degenerate: tuple[str, ...] = ()


def _freeze_nested(
    value: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Mapping[str, float]]:
    return MappingProxyType(
        {
            qid: MappingProxyType(dict(distribution))
            for qid, distribution in value.items()
        }
    )


def load_targets(
    path: str | Path,
    binary_survey,
    *,
    subgroup: str | None = None,
    renormalize: bool = False,
) -> TargetBundle:
    """Load and validate human targets, optionally repairing their total.

    ``renormalize`` divides each target vector by its own sum (first dividing
    by 100 when the sum exceeds 1.5, i.e. when it is given in percent), so a
    target whose shares do not quite sum to one can still be scored. It is
    opt-in: by default a target that is not a distribution is refused, so a
    corrupt file is caught rather than rescaled. Every repaired qid is recorded
    on ``TargetBundle.renormalized``.

    Under ``renormalize`` an all-zero target is left unchanged and recorded on
    ``TargetBundle.degenerate`` instead of raising. Such a question enters the
    design as a block of zero target mass; callers should skip those qids when
    scoring distributions, where the metric would be non-finite.

    **The repair applies to ``raw`` only, not to ``binary``.** ``binary`` reaches
    the fit through ``split.py`` unrepaired, while ``raw`` reaches the metrics
    through ``evaluate.py`` repaired: the fit sees the target block as given and
    the scoring sees it rescaled. This is the deliberate contract of the option,
    kept so that runs made with it reproduce exactly.

    Both ``renormalized`` and ``degenerate`` list every qid in the file, not
    only the modelled ones, so a caller that cares about its own question set
    intersects them with it.

    Ungrouped files remain a flat ``{qid: {label: probability}}`` mapping.
    Grouped files use an explicit envelope::

        {
          "schema_version": 1,
          "default_subgroup": "overall",
          "subgroups": {
            "overall": {"Q1": {"yes": 0.5, "no": 0.5}},
            "party.democrat": {"Q1": {"yes": 0.7, "no": 0.3}}
          }
        }

    ``subgroup`` is deliberately opt-in.  A grouped file with multiple
    subgroups must either declare ``default_subgroup`` or be selected by the
    caller.
    """

    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise ValueError(f"target file does not exist: {source_path}")

    raw_bytes = source_path.read_bytes()
    try:
        loaded: Any = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid target JSON {source_path}: {error}"
        ) from error

    if not isinstance(loaded, Mapping):
        raise ValueError("target JSON root must be a mapping")

    selected: Any = loaded
    selected_subgroup: str | None = None
    if "subgroups" in loaded:
        if loaded.get("schema_version") != 1:
            raise ValueError(
                "grouped target schema_version must be 1"
            )
        subgroups = loaded.get("subgroups")
        if not isinstance(subgroups, Mapping) or not subgroups:
            raise ValueError(
                "grouped target subgroups must be a nonempty mapping"
            )
        selected_subgroup = subgroup
        if selected_subgroup is None:
            default = loaded.get("default_subgroup")
            if default is not None:
                if not isinstance(default, str) or not default:
                    raise ValueError(
                        "default_subgroup must be a nonempty string"
                    )
                selected_subgroup = default
            elif len(subgroups) == 1:
                selected_subgroup = next(iter(subgroups))
            else:
                raise ValueError(
                    "grouped target has multiple subgroups; set "
                    "data.target.subgroup or default_subgroup"
                )
        if (
            not isinstance(selected_subgroup, str)
            or not selected_subgroup
        ):
            raise ValueError("subgroup must be a nonempty string")
        if selected_subgroup not in subgroups:
            available = ", ".join(sorted(str(key) for key in subgroups))
            raise ValueError(
                f"unknown target subgroup {selected_subgroup!r}; "
                f"available: {available}"
            )
        selected = subgroups[selected_subgroup]
    elif subgroup is not None:
        raise ValueError(
            "data.target.subgroup requires a grouped target file"
        )

    if not isinstance(selected, Mapping):
        raise ValueError("selected target must be a mapping")

    normalized: dict[str, dict[str, float]] = {}
    for qid, distribution in selected.items():
        if not isinstance(qid, str) or not qid:
            raise ValueError("target question IDs must be nonempty strings")
        if not isinstance(distribution, Mapping) or not distribution:
            raise ValueError(
                f"target distribution for {qid!r} must be a nonempty mapping"
            )

        normalized_distribution: dict[str, float] = {}
        for label, value in distribution.items():
            if not isinstance(label, str) or not label:
                raise ValueError(
                    f"target labels for {qid!r} must be nonempty strings"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"target value for {qid!r}/{label!r} must be numeric"
                )
            normalized_distribution[label] = float(value)

        normalized[qid] = normalized_distribution

    original_questions = {
        question["id"]: question
        for question in binary_survey.original_questions
    }

    # Built before the repair below, so the fit target stays exactly what the
    # file says -- see the note in this function's docstring.
    binary: dict[str, float] = {}
    for question in binary_survey.questions:
        base_id = question.get("base_id")
        base_answer = question.get("base_answer")
        if base_id is None or base_answer is None:
            continue

        distribution = normalized.get(base_id)
        if distribution is None:
            raise ValueError(
                f"human target is missing base qid {base_id!r}"
            )

        binary[question["id"]] = float(
            distribution.get(base_answer, 0.0)
        )

    if not binary:
        raise ValueError("target conversion produced no binary targets")

    renormalized: list[str] = []
    degenerate: list[str] = []
    if renormalize:
        for qid in sorted(normalized):
            distribution = normalized[qid]
            total = sum(distribution.values())
            if total > 1.5:          # percentages, as _to_prob_vec assumes
                distribution = {
                    label: value / 100.0
                    for label, value in distribution.items()
                }
                total = sum(distribution.values())
            if total == 0.0:
                degenerate.append(qid)
            elif total < 0.0:
                raise ValueError(
                    f"target for {qid!r} sums to {total!r}; "
                    "a negative total cannot be renormalized"
                )
            elif abs(total - 1.0) > 1e-9:
                distribution = {
                    label: value / total
                    for label, value in distribution.items()
                }
                renormalized.append(qid)
            normalized[qid] = distribution

    tolerated = set(degenerate)
    for qid in original_questions:
        if qid not in normalized:
            raise ValueError(f"human target is missing survey qid {qid!r}")

        if qid in tolerated:
            continue

        validate_distribution(
            list(normalized[qid].values()),
            name="human target",
            qid=qid,
        )

    selection_bytes = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return TargetBundle(
        source_path=source_path,
        source_digest=hashlib.sha256(raw_bytes).hexdigest(),
        selection_digest=hashlib.sha256(selection_bytes).hexdigest(),
        subgroup=selected_subgroup,
        raw=_freeze_nested(normalized),
        binary=MappingProxyType(binary),
        renormalized=tuple(renormalized),
        degenerate=tuple(degenerate),
    )


def target_spec(value: Any) -> tuple[Path, str | None]:
    """Resolve a scalar or grouped target config entry."""

    subgroup = None
    if isinstance(value, Mapping):
        path_value = value.get("path", value.get("source"))
        subgroup = value.get("subgroup")
        if subgroup is not None and (
            not isinstance(subgroup, str) or not subgroup.strip()
        ):
            raise ValueError(
                "data.target.subgroup must be a nonempty string"
            )
    else:
        path_value = value

    if isinstance(path_value, Path):
        path = path_value.resolve()
    elif isinstance(path_value, str) and path_value.strip():
        path = Path(path_value).resolve()
    else:
        raise ValueError(
            "data.target must be a path or a mapping with path/source"
        )
    return path, subgroup


def target_renormalize(value: Any) -> bool:
    """Read ``data.target.renormalize`` from a scalar or grouped target entry.

    Kept separate from :func:`target_spec` so the existing two-tuple contract of
    that function is unchanged.
    """

    if not isinstance(value, Mapping):
        return False
    flag = value.get("renormalize", False)
    if not isinstance(flag, bool):
        raise ValueError(
            "data.target.renormalize must be boolean"
        )
    return flag


__all__ = [
    "TargetBundle",
    "load_targets",
    "target_renormalize",
    "target_spec",
]
