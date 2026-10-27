"""Stage-2 data loading and split preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from p2p.endowments import ActiveEndowments
from p2p.responses import BinaryExtendedResponses, Responses
from p2p.responses.cleaner import ResponseCleaner
from p2p.stages.targets import TargetBundle, load_targets
from p2p.survey import BinaryExtendedSurvey, Survey


COLUMN_DROP_V2 = "column_drop_v2"


REFERENCE_FEATURE = "__reference_respondent__"
"""Reserved feature name for the uniform reference respondent."""


@dataclass(frozen=True, slots=True)
class RegressionSplit:
    """One prepared regression split."""

    name: str
    qids: tuple[str, ...]
    X: pd.DataFrame
    y: np.ndarray


@dataclass(frozen=True, slots=True)
class RegressionData:
    """Prepared Stage-2 inputs shared by fitting and evaluation."""

    survey: Survey
    binary_survey: BinaryExtendedSurvey
    answer_responses: Responses
    binary_responses: BinaryExtendedResponses
    endowments: ActiveEndowments
    targets: TargetBundle
    splits: Mapping[str, RegressionSplit]
    train_splits: tuple[str, ...]
    test_split: str
    feature_names: tuple[str, ...]
    cleaning_policy: str
    source_artifacts: Mapping[str, Path]
    source_digests: Mapping[str, str]
    # Whether answer-format responses carry each option's own text (see
    # ResponseCleaner); False only to reproduce published scores exactly.
    canonical_answers: bool = True
    # An optional reference distribution entering every prediction with weight
    # one (p2p.models.reference_distribution). Loaded and validated exactly like
    # the targets; None when the run has none.
    reference: TargetBundle | None = None

    def get_split(self, name: str) -> RegressionSplit:
        try:
            return self.splits[name]
        except KeyError:
            available = ", ".join(sorted(self.splits))
            raise KeyError(
                f"unknown regression split {name!r}; available: {available}"
            ) from None


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} file does not exist: {resolved}")
    return resolved


def _normalize_split_names(
    train_splits: Sequence[str],
    test_split: str,
) -> tuple[tuple[str, ...], str]:
    if isinstance(train_splits, (str, bytes)) or not train_splits:
        raise ValueError("train_splits must be a nonempty sequence")

    normalized_train = tuple(train_splits)
    if any(
        not isinstance(name, str) or not name
        for name in normalized_train
    ):
        raise ValueError("train split names must be nonempty strings")
    if len(normalized_train) != len(set(normalized_train)):
        raise ValueError("train_splits contains duplicate names")
    if not isinstance(test_split, str) or not test_split:
        raise ValueError("test_split must be a nonempty string")
    if test_split in normalized_train:
        raise ValueError("test_split must not also be a train split")

    return normalized_train, test_split


def _build_augmented_matrix(
    *,
    binary_survey: BinaryExtendedSurvey,
    binary_responses: BinaryExtendedResponses,
    endowments: ActiveEndowments,
    targets: TargetBundle,
    use_proxy_only: bool,
    reference_respondent: bool = False,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    eligible_qids = {
        question["id"]
        for question in binary_survey.questions
        if len(question.get("answer_to_code", {})) <= 3
    }

    matrix = {
        qid: answers
        for qid, answers in binary_responses.responses.items()
        if qid in eligible_qids
    }
    if not matrix:
        raise ValueError("binary response loading produced no usable questions")

    all_qids = sorted(matrix)
    all_eids = sorted(
        {
            eid
            for answers in matrix.values()
            for eid in answers
        }
    )
    if not all_eids:
        raise ValueError("binary response loading produced no agents")

    rows = []
    for qid in all_qids:
        row = {
            eid: matrix[qid].get(eid)
            for eid in all_eids
        }
        row["qid"] = qid
        row["aggregate"] = targets.binary.get(qid)
        rows.append(row)

    augmented = pd.DataFrame(rows).set_index("qid")

    agent_columns = [
        column
        for column in augmented.columns
        if column != "aggregate"
    ]
    incomplete = [
        column
        for column in agent_columns
        if augmented[column].isna().any()
    ]
    augmented = augmented.drop(columns=incomplete)

    retained_agents = [
        column
        for column in augmented.columns
        if column != "aggregate"
    ]

    if use_proxy_only:
        proxy_eids = set(endowments.get_eids_by_role("proxy"))
        retained_agents = [
            eid
            for eid in retained_agents
            if eid in proxy_eids
        ]

    if not retained_agents:
        raise ValueError(
            "column-drop cleaning retained no usable agents"
        )

    if reference_respondent:
        if REFERENCE_FEATURE in retained_agents:
            raise ValueError(
                f"agent id {REFERENCE_FEATURE!r} collides with the reserved "
                "reference-respondent feature name"
            )

        # A maximally uninformative respondent: equal probability on every
        # option of every question. Weighted like any other agent, and free to
        # be dropped when the basis already covers the target. Its 1/k values
        # are taken from the option rows actually present, so its column sums
        # to one on every block by construction.
        rows_by_group: dict[str, list[str]] = {}
        for original_qid, binary_qids in (
            binary_survey.original_to_binary_map.items()
        ):
            present = [
                qid
                for qid in binary_qids
                if qid in augmented.index
            ]
            if present:
                rows_by_group[original_qid] = present

        reference = pd.Series(
            0.0,
            index=augmented.index,
            dtype=float,
        )
        for present in rows_by_group.values():
            reference.loc[present] = 1.0 / len(present)

        # A row in no group (a malformed question is kept unexpanded) keeps 0.
        augmented[REFERENCE_FEATURE] = reference
        retained_agents = retained_agents + [REFERENCE_FEATURE]

    return (
        augmented[retained_agents + ["aggregate"]],
        tuple(retained_agents),
    )


def load_regression_data(
    *,
    survey_source: str | Path,
    survey_schema: str | Path,
    responses_path: str | Path,
    endowments_path: str | Path,
    target_path: str | Path | None,
    target_subgroup: str | None = None,
    train_splits: Sequence[str] = ("train", "valid"),
    test_split: str = "test",
    use_proxy_only: bool = True,
    cleaning_policy: str = COLUMN_DROP_V2,
    allow_missing_target: bool = False,
    reference_respondent: bool = False,
    renormalize_targets: bool = False,
    canonical_answers: bool = True,
    reference_distribution_path: str | Path | None = None,
) -> RegressionData:
    """Load one Stage-2 basis using explicit column-drop cleaning."""

    if cleaning_policy != COLUMN_DROP_V2:
        raise ValueError(
            "the initial regress path supports only column_drop_v2"
        )

    normalized_train, normalized_test = _normalize_split_names(
        train_splits,
        test_split,
    )

    paths: dict[str, Path] = {
        "survey": _resolve_file(survey_source, "survey"),
        "survey_schema": _resolve_file(survey_schema, "survey schema"),
        "responses": _resolve_file(responses_path, "responses"),
        "endowments": _resolve_file(endowments_path, "endowments"),
    }
    if target_path is None:
        if not allow_missing_target:
            raise ValueError("target_path is required")
        if target_subgroup is not None:
            raise ValueError(
                "target_subgroup cannot be used without target_path"
            )
    else:
        paths["target"] = _resolve_file(target_path, "target")
    if reference_distribution_path is not None:
        paths["reference_distribution"] = _resolve_file(
            reference_distribution_path, "reference distribution"
        )

    survey = Survey(
        csv_path=paths["survey"],
        config_path=paths["survey_schema"],
    )
    binary_survey = BinaryExtendedSurvey.from_survey(survey)

    if not isinstance(canonical_answers, bool):
        raise ValueError("canonical_answers must be boolean")
    answer_responses = Responses(
        source=str(paths["responses"]),
        survey=survey,
        output_format="answer",
        cleaner=ResponseCleaner(survey=survey, canonical=canonical_answers),
    )
    binary_responses = BinaryExtendedResponses(
        source=str(paths["responses"]),
        survey=binary_survey,
        output_format="code",
    )
    endowments = ActiveEndowments.load(paths["endowments"])
    if "target" in paths:
        targets = load_targets(
            paths["target"],
            binary_survey,
            subgroup=target_subgroup,
            renormalize=renormalize_targets,
        )
    else:
        empty_digest = hashlib.sha256(b"").hexdigest()
        selection_digest = hashlib.sha256(b"{}").hexdigest()
        targets = TargetBundle(
            source_path=None,
            source_digest=empty_digest,
            selection_digest=selection_digest,
            subgroup=None,
            raw=MappingProxyType({}),
            binary=MappingProxyType(
                {
                    question["id"]: 0.0
                    for question in binary_survey.questions
                }
            ),
        )

    reference = None
    if "reference_distribution" in paths:
        # The same checks as the targets: every survey question present, labels
        # numeric, every distribution non-negative and summing to one.
        reference = load_targets(paths["reference_distribution"], binary_survey)

    augmented, feature_names = _build_augmented_matrix(
        binary_survey=binary_survey,
        binary_responses=binary_responses,
        endowments=endowments,
        targets=targets,
        use_proxy_only=use_proxy_only,
        reference_respondent=reference_respondent,
    )

    split_names = normalized_train + (normalized_test,)
    prepared: dict[str, RegressionSplit] = {}

    for split_name in split_names:
        requested_qids = [
            question["id"]
            for question in binary_survey.get_questions_by_split(split_name)
        ]
        selected_index = augmented.index.intersection(requested_qids)
        frame = augmented.loc[selected_index]

        if frame.empty:
            raise ValueError(
                f"regression split {split_name!r} contains no usable rows"
            )
        if frame["aggregate"].isna().any():
            missing = frame.index[
                frame["aggregate"].isna()
            ].tolist()
            raise ValueError(
                f"human target is missing binary qids: {missing}"
            )

        X = frame[list(feature_names)].astype(float)
        y = frame["aggregate"].astype(float).to_numpy()

        if X.isna().any().any() or np.isnan(y).any():
            raise ValueError(
                f"regression split {split_name!r} contains missing values "
                "after column_drop_v2"
            )

        prepared[split_name] = RegressionSplit(
            name=split_name,
            qids=tuple(str(qid) for qid in frame.index),
            X=X,
            y=y,
        )

    return RegressionData(
        survey=survey,
        binary_survey=binary_survey,
        answer_responses=answer_responses,
        binary_responses=binary_responses,
        endowments=endowments,
        targets=targets,
        splits=MappingProxyType(prepared),
        train_splits=normalized_train,
        test_split=normalized_test,
        feature_names=feature_names,
        cleaning_policy=cleaning_policy,
        source_artifacts=MappingProxyType(paths),
        source_digests=MappingProxyType(
            {
                name: _digest(path)
                for name, path in paths.items()
            }
        ),
        canonical_answers=canonical_answers,
        reference=reference,
    )


__all__ = [
    "COLUMN_DROP_V2",
    "RegressionData",
    "RegressionSplit",
    "load_regression_data",
]
