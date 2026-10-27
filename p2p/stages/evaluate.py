"""Single-run evaluation with explicit prediction policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.spatial import distance
from scipy.stats import wasserstein_distance

from p2p.metrics import (
    clip_and_normalize,
    emd,
    mcpa,
    mse,
    one_minus_jsd,
    project_to_simplex,
    validate_distribution,
)
from p2p.responses.converter import ResponseUtils
from p2p.stages.split import RegressionData


_SUPPORTED_METRICS = {
    "mse",
    "distribution_mse",
    "one_minus_jsd",
    "emd",
    "mcpa",
}
# Public, read-only view for interfaces that list the metrics P2P can score.
SUPPORTED_METRICS = frozenset(_SUPPORTED_METRICS)

# Scored per question on the transformed distribution and averaged over
# questions. ``mse`` and ``distribution_mse`` are pooled over cells instead.
_DISTRIBUTION_METRICS = (
    "one_minus_jsd",
    "emd",
    "mcpa",
)

# Earlier spellings of an option value, still accepted as written so that
# existing manifests run unchanged (and keep their config digests).
SIGNED_SUM_POLICIES = frozenset({"signed_sum_normalize", "legacy_signed_normalize"})
ANSWER_FILE_ROUTES = frozenset({"answer_file", "legacy_answer_file"})

_SUPPORTED_DISTRIBUTION_POLICIES = {
    "validate",
    "clip_and_normalize",
    "project_to_simplex",
    *SIGNED_SUM_POLICIES,
}


@dataclass(frozen=True, slots=True)
class MetricCoverage:
    """Question coverage for one evaluated metric."""

    qids_requested: int
    qids_used: int


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """In-memory result for one model and one split."""

    model_name: str
    split: str
    prediction_policy: str
    values: Mapping[str, float | None]
    coverage: Mapping[str, MetricCoverage]
    predictions: np.ndarray
    diagnostics: Mapping[str, Any]


def _signed_sum_normalize(
    values: np.ndarray,
) -> np.ndarray:
    """Divide by the signed sum, after dividing by 100 if that sum exceeds 1.5.

    Negative entries are kept; a zero-sum vector is returned unchanged.
    """

    vector = np.asarray(values, dtype=float).copy()
    total = float(vector.sum())

    if total == 0.0:
        return vector

    if total > 1.5:
        vector = vector / 100.0
        total = float(vector.sum())

    return vector / total if total > 0.0 else vector


def _is_valid_distribution(
    values: np.ndarray,
    *,
    qid: str,
) -> bool:
    try:
        validate_distribution(
            values,
            name="predicted",
            qid=qid,
        )
    except ValueError:
        return False
    return True


def _transform_prediction(
    values: np.ndarray,
    *,
    policy: str,
    qid: str,
) -> np.ndarray:
    if policy in SIGNED_SUM_POLICIES:
        return _signed_sum_normalize(values)
    if policy == "validate":
        return validate_distribution(
            values,
            name="predicted",
            qid=qid,
        )
    if policy == "clip_and_normalize":
        return clip_and_normalize(values, qid=qid)
    if policy == "project_to_simplex":
        return project_to_simplex(values, qid=qid)

    raise ValueError(
        f"unknown distribution prediction policy {policy!r}"
    )


def _aligned_vectors(
    human: Mapping[str, float],
    predicted: Mapping[str, float],
    *,
    order: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if order is None:
        keys = list(human)
    else:
        support = set(human) | set(predicted)
        keys = [
            key
            for key in order
            if key in support
        ]

    for key in human:
        if key not in keys:
            keys.append(key)
    for key in predicted:
        if key not in keys:
            keys.append(key)

    observed = np.asarray(
        [human.get(key, 0.0) for key in keys],
        dtype=float,
    )
    raw_prediction = np.asarray(
        [predicted.get(key, 0.0) for key in keys],
        dtype=float,
    )
    return observed, raw_prediction


def _evaluation_contract(
    metrics: Sequence[str],
    *,
    distribution_prediction_policy: str,
    invalid_distribution: str,
) -> tuple[str, ...]:
    requested_metrics = tuple(metrics)
    if not requested_metrics:
        raise ValueError("metrics must not be empty")

    unknown_metrics = set(requested_metrics) - _SUPPORTED_METRICS
    if unknown_metrics:
        names = ", ".join(sorted(unknown_metrics))
        raise ValueError(f"unknown metric: {names}")

    if distribution_prediction_policy not in (
        _SUPPORTED_DISTRIBUTION_POLICIES
    ):
        raise ValueError(
            "unknown distribution prediction policy "
            f"{distribution_prediction_policy!r}"
        )
    if invalid_distribution not in {"error", "allow"}:
        raise ValueError(
            "invalid_distribution must be 'error' or 'allow'"
        )
    return requested_metrics


def evaluate_predictions(
    survey,
    targets,
    predictions: Mapping[str, Mapping[str, float]],
    *,
    model_name: str,
    split: str,
    metrics: Sequence[str],
    distribution_prediction_policy: str,
    invalid_distribution: str,
) -> EvaluationResult:
    """Evaluate self-contained option-level predictions for one split."""

    requested_metrics = _evaluation_contract(
        metrics,
        distribution_prediction_policy=(
            distribution_prediction_policy
        ),
        invalid_distribution=invalid_distribution,
    )
    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must be a mapping")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a nonempty string")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a nonempty string")

    qids = [
        question["id"]
        for question in survey.get_questions_by_split(split)
    ]
    if not qids:
        raise ValueError(
            f"evaluation split {split!r} contains no questions"
        )

    distribution_metrics = [
        metric
        for metric in requested_metrics
        if metric in _DISTRIBUTION_METRICS
    ]
    metric_values: dict[str, list[float]] = {
        metric: []
        for metric in distribution_metrics
    }
    mse_observed: list[np.ndarray] = []
    mse_predicted: list[np.ndarray] = []
    raw_vectors: list[np.ndarray] = []
    mse_qids_used = 0
    # distribution_mse: the same cells as mse, predictions after the policy.
    transformed_observed: list[np.ndarray] = []
    transformed_predicted: list[np.ndarray] = []

    invalid_before = 0
    invalid_after = 0
    valid_after = 0
    negative_entry_count = 0
    negative_mass = 0.0
    raw_sum_deviation = 0.0
    transformation_l1 = 0.0

    for qid in qids:
        human = targets.raw.get(qid)
        if human is None:
            raise ValueError(
                f"human target is missing qid {qid!r}"
            )

        predicted = predictions.get(qid)
        if predicted is None:
            if invalid_distribution == "error":
                raise ValueError(
                    f"predictions are missing qid {qid!r}"
                )
            invalid_before += 1
            invalid_after += 1
            continue
        if not isinstance(predicted, Mapping):
            raise ValueError(
                f"predictions for {qid!r} must be a mapping"
            )

        order = survey.get_option_order_text(qid)
        observed_vector, raw_prediction = _aligned_vectors(
            human,
            predicted,
            order=order,
        )
        validate_distribution(
            observed_vector,
            name="human target",
            qid=qid,
        )
        if not np.all(np.isfinite(raw_prediction)):
            raise ValueError(
                f"predictions for {qid!r} must be finite"
            )

        raw_vectors.append(raw_prediction)
        if "mse" in requested_metrics:
            mse_observed.append(observed_vector)
            mse_predicted.append(raw_prediction)
            mse_qids_used += 1

        before_valid = _is_valid_distribution(
            raw_prediction,
            qid=qid,
        )
        if not before_valid:
            invalid_before += 1

        negative = raw_prediction[raw_prediction < 0.0]
        negative_entry_count += int(negative.size)
        negative_mass += float(-negative.sum())
        raw_sum_deviation += abs(
            float(raw_prediction.sum()) - 1.0
        )

        try:
            transformed = _transform_prediction(
                raw_prediction,
                policy=distribution_prediction_policy,
                qid=qid,
            )
        except ValueError as error:
            if invalid_distribution == "error":
                raise ValueError(
                    f"invalid prediction distribution for {qid!r}"
                ) from error
            transformed = raw_prediction.copy()

        transformation_l1 += float(
            np.abs(transformed - raw_prediction).sum()
        )
        after_valid = _is_valid_distribution(
            transformed,
            qid=qid,
        )
        if after_valid:
            valid_after += 1
        else:
            invalid_after += 1
            if invalid_distribution == "error":
                raise ValueError(
                    f"invalid prediction distribution for {qid!r}"
                )
            continue

        if "distribution_mse" in requested_metrics:
            transformed_observed.append(observed_vector)
            transformed_predicted.append(transformed)

        for metric in distribution_metrics:
            if metric == "one_minus_jsd":
                value = one_minus_jsd(
                    observed_vector,
                    transformed,
                    qid=qid,
                )
            elif metric == "emd":
                value = emd(
                    observed_vector,
                    transformed,
                    qid=qid,
                )
            elif metric == "mcpa":
                value = mcpa(
                    observed_vector,
                    transformed,
                    qid=qid,
                )
            else:
                raise AssertionError(
                    f"unhandled metric {metric!r}"
                )
            metric_values[metric].append(float(value))

    values: dict[str, float | None] = {}
    coverage: dict[str, MetricCoverage] = {}
    if "mse" in requested_metrics:
        values["mse"] = (
            mse(
                np.concatenate(mse_observed),
                np.concatenate(mse_predicted),
            )
            if mse_observed
            else None
        )
        coverage["mse"] = MetricCoverage(
            qids_requested=len(qids),
            qids_used=mse_qids_used,
        )
    if "distribution_mse" in requested_metrics:
        values["distribution_mse"] = (
            mse(
                np.concatenate(transformed_observed),
                np.concatenate(transformed_predicted),
            )
            if transformed_observed
            else None
        )
        coverage["distribution_mse"] = MetricCoverage(
            qids_requested=len(qids),
            qids_used=len(transformed_observed),
        )

    for metric in distribution_metrics:
        observed_values = metric_values[metric]
        values[metric] = (
            float(np.mean(observed_values))
            if observed_values
            else None
        )
        coverage[metric] = MetricCoverage(
            qids_requested=len(qids),
            qids_used=len(observed_values),
        )

    diagnostics = {
        "distribution_prediction_policy": (
            distribution_prediction_policy
        ),
        "invalid_distribution_policy": (
            invalid_distribution
        ),
        "invalid_qids_before_transform": invalid_before,
        "invalid_qids_after_transform": invalid_after,
        "qids_requested": len(qids),
        "qids_used": valid_after,
        "negative_entry_count": negative_entry_count,
        "negative_mass": negative_mass,
        "raw_sum_deviation": raw_sum_deviation,
        "transformation_l1": transformation_l1,
    }
    flat_predictions = (
        np.concatenate(raw_vectors)
        if raw_vectors
        else np.asarray([], dtype=float)
    )
    return EvaluationResult(
        model_name=model_name,
        split=split,
        prediction_policy=distribution_prediction_policy,
        values=MappingProxyType(values),
        coverage=MappingProxyType(coverage),
        predictions=flat_predictions.copy(),
        diagnostics=MappingProxyType(diagnostics),
    )


def _legacy_one_minus_jsd(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    with np.errstate(all="ignore"):
        return 1.0 - float(
            distance.jensenshannon(
                observed,
                predicted,
            )
        )


def _legacy_emd(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    indices = np.arange(len(observed), dtype=float)
    raw = float(
        wasserstein_distance(
            indices,
            indices,
            u_weights=observed,
            v_weights=predicted,
        )
    )
    maximum = len(indices) - 1 if len(indices) > 1 else 1
    return raw / maximum


def _legacy_mcpa(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    return float(
        np.argmax(observed) == np.argmax(predicted)
    )


DISTRIBUTION_ROUTES = ("model", "answer_file", "legacy_answer_file")


def _row_labels(data: RegressionData) -> dict[str, tuple[str, str]]:
    """Option row id -> (question id, the option's own text)."""

    labels: dict[str, tuple[str, str]] = {}
    for question in data.binary_survey.questions:
        base_id = question.get("base_id")
        base_answer = question.get("base_answer")
        if base_id is not None and base_answer is not None:
            labels[question["id"]] = (base_id, base_answer)
    return labels


def model_distribution_predictions(
    data: RegressionData,
    model: Any,
    *,
    split: str | None = None,
) -> dict[str, dict[str, float]]:
    """Each question's predicted option masses, from the model's own predictions.

    This is the ``model`` distribution route: ``model.predict`` on the design
    rows -- the same predictions MSE scores -- grouped by question and labelled
    with each option's own text. Every model feature contributes through its own
    column, so a basis element that is not an answering agent (the reference
    respondent) is included. With ``split=None`` every split is covered.
    """

    labels = _row_labels(data)
    names = [split] if split is not None else list(data.splits)
    predictions: dict[str, dict[str, float]] = {}
    for name in names:
        prepared = data.get_split(name)
        raw = np.asarray(model.predict(prepared.X), dtype=float)
        for row, value in zip(prepared.X.index, raw):
            question, label = labels.get(str(row), (None, None))
            if question is None:
                continue
            predictions.setdefault(question, {})[label] = float(value)
    return predictions


def answer_file_predictions(
    data: RegressionData,
    model: Any,
) -> dict[str, dict[str, float]]:
    """The answer-file route: re-aggregate the answer file by agent name.

    An alternative to scoring the model's own predictions. It cannot see a feature
    that is not an answering agent, so such a model is refused rather than
    silently scored without it.
    """

    if getattr(model, "intercept_", 0.0):
        raise ValueError(
            "the answer_file route cannot score a model with an intercept: "
            "re-aggregating answers by agent weight has no place for a constant. "
            "Use distribution_route='model'."
        )
    if getattr(model, "reference_distribution_", None) is not None:
        raise ValueError(
            "the answer_file route cannot score a model fitted around a "
            "reference distribution: re-aggregating answers by agent weight omits "
            "the reference. Use distribution_route='model'."
        )
    weights = model.coef_dict_
    agents = {
        eid
        for answers in data.answer_responses.responses.values()
        for eid in answers
    }
    missing = sorted(name for name in weights if name not in agents)
    if missing:
        raise ValueError(
            "the answer_file route cannot score a model whose features "
            "are not all answering agents; missing from the answer file: "
            + ", ".join(repr(name) for name in missing[:5])
            + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
            + ". Use distribution_route='model'."
        )
    return ResponseUtils.aggregate_weighted_responses(
        data.answer_responses,
        weights,
    )


def _distribution_mse_on_option_rows(
    data: RegressionData,
    prepared_split,
    predictions: np.ndarray,
    *,
    qids: Sequence[str],
    policy: str,
    invalid_distribution: str,
) -> tuple[float, int]:
    """MSE on the design's option rows after the distribution prediction policy.

    Same rows and targets as raw ``mse``; only the predictions differ: each
    question's block of option rows is transformed by ``policy`` (for
    ``clip_and_normalize``: clipped at zero and rescaled to sum to one). The gap
    between ``mse`` and ``distribution_mse`` is therefore exactly the effect of
    the transformation, e.g. of predicted mass that does not sum to one.
    A question whose block cannot be transformed into a valid distribution is
    left out (or raises, under ``invalid_distribution="error"``).
    """

    position = {
        str(label): index
        for index, label in enumerate(prepared_split.X.index)
    }
    observed = np.asarray(prepared_split.y, dtype=float)
    groups = data.binary_survey.original_to_binary_map

    observed_blocks: list[np.ndarray] = []
    predicted_blocks: list[np.ndarray] = []
    for qid in qids:
        rows = [
            position[str(member)]
            for member in groups.get(qid, ())
            if str(member) in position
        ]
        if not rows:
            continue
        try:
            transformed = _transform_prediction(
                predictions[rows],
                policy=policy,
                qid=qid,
            )
        except ValueError as error:
            if invalid_distribution == "error":
                raise ValueError(
                    f"invalid prediction distribution for {qid!r}"
                ) from error
            continue
        if not _is_valid_distribution(transformed, qid=qid):
            if invalid_distribution == "error":
                raise ValueError(
                    f"invalid prediction distribution for {qid!r}"
                )
            continue
        observed_blocks.append(observed[rows])
        predicted_blocks.append(transformed)

    if not observed_blocks:
        return float("nan"), 0
    return (
        mse(
            np.concatenate(observed_blocks),
            np.concatenate(predicted_blocks),
        ),
        len(observed_blocks),
    )


def evaluate_model(
    data: RegressionData,
    fit_result,
    *,
    split: str,
    metrics: Sequence[str],
    distribution_prediction_policy: str,
    invalid_distribution: str = "allow",
    distribution_route: str = "model",
) -> EvaluationResult:
    """Evaluate one fitted model on one configured split.

    ``distribution_route`` chooses where the distributional metrics' predicted
    distributions come from: ``"model"`` (the model's own predictions, as MSE
    uses) or ``"answer_file"`` (the agents' answer file re-aggregated by agent
    name and weight; ``"legacy_answer_file"`` is accepted as the same route).
    """

    if distribution_route not in DISTRIBUTION_ROUTES:
        raise ValueError(
            f"distribution_route must be one of {DISTRIBUTION_ROUTES}; "
            f"got {distribution_route!r}"
        )

    requested_metrics = _evaluation_contract(
        metrics,
        distribution_prediction_policy=(
            distribution_prediction_policy
        ),
        invalid_distribution=invalid_distribution,
    )

    prepared_split = data.get_split(split)
    model = fit_result.model

    predictions = np.asarray(
        model.predict(prepared_split.X),
        dtype=float,
    )
    if predictions.shape != prepared_split.y.shape:
        raise ValueError(
            "model predictions and targets must have the same shape"
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError(
            "model predictions must contain only finite values"
        )

    original_qids = [
        question["id"]
        for question in data.survey.get_questions_by_split(split)
    ]
    qids_requested = len(original_qids)

    values: dict[str, float] = {}
    coverage: dict[str, MetricCoverage] = {}

    if "mse" in requested_metrics:
        values["mse"] = mse(
            prepared_split.y,
            predictions,
        )
        coverage["mse"] = MetricCoverage(
            qids_requested=qids_requested,
            qids_used=qids_requested,
        )
    if "distribution_mse" in requested_metrics:
        value, qids_used = _distribution_mse_on_option_rows(
            data,
            prepared_split,
            predictions,
            qids=original_qids,
            policy=distribution_prediction_policy,
            invalid_distribution=invalid_distribution,
        )
        values["distribution_mse"] = value
        coverage["distribution_mse"] = MetricCoverage(
            qids_requested=qids_requested,
            qids_used=qids_used,
        )

    distribution_metrics = [
        metric
        for metric in requested_metrics
        if metric in _DISTRIBUTION_METRICS
    ]

    predicted_distributions = (
        model_distribution_predictions(data, model, split=split)
        if distribution_route == "model"
        else answer_file_predictions(data, model)
    )

    metric_values: dict[str, list[float]] = {
        metric: []
        for metric in distribution_metrics
    }

    invalid_before = 0
    invalid_after = 0
    valid_after = 0
    negative_entry_count = 0
    negative_mass = 0.0
    raw_sum_deviation = 0.0
    transformation_l1 = 0.0

    for qid in original_qids:
        human = data.targets.raw.get(qid)
        if human is None:
            raise ValueError(
                f"human target is missing qid {qid!r}"
            )

        predicted = predicted_distributions.get(qid, {})
        order = data.survey.get_option_order_text(qid)
        observed_vector, raw_prediction = _aligned_vectors(
            human,
            predicted,
            order=order,
        )

        validate_distribution(
            observed_vector,
            name="human target",
            qid=qid,
        )

        before_valid = _is_valid_distribution(
            raw_prediction,
            qid=qid,
        )
        if not before_valid:
            invalid_before += 1

        negative = raw_prediction[raw_prediction < 0.0]
        negative_entry_count += int(negative.size)
        negative_mass += float(-negative.sum())
        raw_sum_deviation += abs(
            float(raw_prediction.sum()) - 1.0
        )

        try:
            transformed = _transform_prediction(
                raw_prediction,
                policy=distribution_prediction_policy,
                qid=qid,
            )
        except ValueError as error:
            if invalid_distribution == "error":
                raise ValueError(
                    f"invalid prediction distribution for {qid!r}"
                ) from error
            transformed = raw_prediction.copy()

        transformation_l1 += float(
            np.abs(transformed - raw_prediction).sum()
        )

        after_valid = _is_valid_distribution(
            transformed,
            qid=qid,
        )
        if after_valid:
            valid_after += 1
        else:
            invalid_after += 1
            if invalid_distribution == "error":
                raise ValueError(
                    f"invalid prediction distribution for {qid!r}"
                )

        for metric in distribution_metrics:
            try:
                if (
                    distribution_prediction_policy
                    in SIGNED_SUM_POLICIES
                ):
                    if metric == "one_minus_jsd":
                        value = _legacy_one_minus_jsd(
                            observed_vector,
                            transformed,
                        )
                    elif metric == "emd":
                        value = _legacy_emd(
                            observed_vector,
                            transformed,
                        )
                    elif metric == "mcpa":
                        value = _legacy_mcpa(
                            observed_vector,
                            transformed,
                        )
                    else:
                        raise AssertionError(
                            f"unhandled metric {metric!r}"
                        )
                else:
                    if not after_valid:
                        continue
                    if metric == "one_minus_jsd":
                        value = one_minus_jsd(
                            observed_vector,
                            transformed,
                            qid=qid,
                        )
                    elif metric == "emd":
                        value = emd(
                            observed_vector,
                            transformed,
                            qid=qid,
                        )
                    elif metric == "mcpa":
                        value = mcpa(
                            observed_vector,
                            transformed,
                            qid=qid,
                        )
                    else:
                        raise AssertionError(
                            f"unhandled metric {metric!r}"
                        )

                if np.isfinite(value):
                    metric_values[metric].append(
                        float(value)
                    )
            except (TypeError, ValueError, ZeroDivisionError):
                if invalid_distribution == "error":
                    raise
                # Historical parity excludes unusable distribution QIDs.
                continue

    for metric in distribution_metrics:
        observed_values = metric_values[metric]
        values[metric] = (
            float(np.mean(observed_values))
            if observed_values
            else float("nan")
        )
        coverage[metric] = MetricCoverage(
            qids_requested=qids_requested,
            qids_used=len(observed_values),
        )

    diagnostics = {
        "distribution_route": distribution_route,
        "canonical_answers": bool(
            getattr(data, "canonical_answers", True)
        ),
        "distribution_prediction_policy": (
            distribution_prediction_policy
        ),
        "invalid_distribution_policy": (
            invalid_distribution
        ),
        "invalid_qids_before_transform": invalid_before,
        "invalid_qids_after_transform": invalid_after,
        "qids_requested": qids_requested,
        "qids_used": valid_after,
        "negative_entry_count": negative_entry_count,
        "negative_mass": negative_mass,
        "raw_sum_deviation": raw_sum_deviation,
        "transformation_l1": transformation_l1,
    }

    return EvaluationResult(
        model_name=fit_result.method,
        split=split,
        prediction_policy=distribution_prediction_policy,
        values=MappingProxyType(values),
        coverage=MappingProxyType(coverage),
        predictions=predictions.copy(),
        diagnostics=MappingProxyType(diagnostics),
    )


__all__ = [
    "DISTRIBUTION_ROUTES",
    "EvaluationResult",
    "MetricCoverage",
    "SUPPORTED_METRICS",
    "evaluate_model",
    "evaluate_predictions",
    "ANSWER_FILE_ROUTES",
    "SIGNED_SUM_POLICIES",
    "answer_file_predictions",
    "model_distribution_predictions",
]
