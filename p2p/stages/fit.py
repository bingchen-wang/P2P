"""Registered Stage-2 model fitting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import get_close_matches
import math
import warnings
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from p2p.config import expand_grid
import p2p.models  # Registers built-in model factories.
from p2p.models import ConstrainedModelCV
from p2p.models.folds import (
    DEFAULT_FOLD_METHOD,
    FOLD_METHODS,
    FoldAssignment,
    read_fold_record,
)
from p2p.registry import MODELS
from p2p.models.reference_distribution import (
    check_reference_masses,
    with_reference_distribution,
)
from p2p.stages.split import REFERENCE_FEATURE, RegressionData


@dataclass(frozen=True, slots=True)
class FitResult:
    """One fitted Stage-2 model and its selected hyperparameters."""

    method: str
    model: Any
    solver: str
    status: str
    # None for a model with no penalty to select (the simple average).
    best_alpha: float | None
    best_l1_ratio: float | None
    diagnostics: Mapping[str, Any]
    # The cross-validation fold assignment actually used, for recording and
    # replay (``p2p.models.folds``).
    folds: FoldAssignment | None = None


def _numeric_grid(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
    ):
        raise ValueError(
            f"{field} must be a nonempty numeric sequence"
        )

    normalized = []
    for item in value:
        if isinstance(item, bool) or not isinstance(
            item,
            (int, float, np.integer, np.floating),
        ):
            raise ValueError(
                f"{field} must contain only numbers"
            )

        number = float(item)
        if not math.isfinite(number):
            raise ValueError(
                f"{field} must contain only finite numbers"
            )
        if minimum is not None and number < minimum:
            raise ValueError(
                f"{field} values must be at least {minimum}"
            )
        if maximum is not None and number > maximum:
            raise ValueError(
                f"{field} values must be at most {maximum}"
            )
        normalized.append(number)

    return tuple(normalized)


def _finite_number(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{field} must be numeric")

    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(
            f"{field} must be at least {minimum}"
        )
    return number


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


class PenaltySelectionWarning(UserWarning):
    """The cross-validated penalty was not clearly determined by the data."""


# Options with no effect in P2P. Accepted so manifests that carry them still
# load, but never silently: any present is recorded in the fit diagnostics.
IGNORED_MODEL_OPTIONS = ("max_iter", "post_selection_refit")

# Fitted weights smaller than this in absolute value are set to zero.
DEFAULT_ZERO_THRESHOLD = 1e-5

# Model options refused together with data.reference_distribution, in the order
# the refusal names them.
REFERENCE_DISTRIBUTION_CONFLICTS = (
    "intercept",
    "sum_to_one",
    "nonneg",
    "reference_respondent",
)


MODEL_OPTIONS = frozenset(
    {
        # Read by this stage.
        "method",
        "alpha",
        "l1_ratio",
        "validation",
        "zero_threshold",
        "group_penalty_strength",
        "group_total",
        "soft_group_sum",
        "nonneg",
        "sum_to_one",
        "block_weighting",
        "intercept",
        # Read by the regress command when preparing the design.
        "reference_respondent",
        # Legacy options with no effect (see IGNORED_MODEL_OPTIONS).
        *IGNORED_MODEL_OPTIONS,
    }
)


class IgnoredModelOptionWarning(UserWarning):
    """A configured model option asks for behaviour P2P does not implement."""


def validate_model_options(model_config: Mapping[str, Any]) -> None:
    """Reject model options this build does not recognise.

    Unknown keys were previously ignored in silence, so a misspelled flag left
    the feature switched off while the run looked correctly configured. A
    mistyped ``reference_respondent`` would have produced an on/off comparison
    in which neither arm was on.
    """

    if not isinstance(model_config, Mapping):
        raise TypeError("model_config must be a mapping")

    unknown = sorted(set(model_config) - MODEL_OPTIONS)
    if not unknown:
        if model_config.get("method") == SIMPLE_AVERAGE:
            _check_simple_average_options(model_config)
        return

    known = ", ".join(sorted(MODEL_OPTIONS))
    suggestions = []
    for name in unknown:
        close = get_close_matches(name, MODEL_OPTIONS, n=1, cutoff=0.7)
        if close:
            suggestions.append(f"{name!r} (did you mean {close[0]!r}?)")
        else:
            suggestions.append(repr(name))

    raise ValueError(
        "unknown model option(s): "
        + ", ".join(suggestions)
        + f". Recognised options are: {known}"
    )


SIMPLE_AVERAGE = "simple_average"


# Penalty grids used when a manifest gives none. The alpha grid is wide enough
# that its ceiling is rarely selected (a grid ending at 10 is often selected at
# its ceiling). Manifests that state a grid are unaffected.
DEFAULT_ALPHA_GRID = {"logspace": [-5, 5, 30]}
DEFAULT_L1_RATIO_GRID = {"linspace": [0.1, 1.0, 10]}


def _grid_or_default(
    model_config: Mapping[str, Any],
    field: str,
    default: Mapping[str, Any],
    defaulted: list[str],
) -> Any:
    value = model_config.get(field)
    if value is not None:
        return value
    defaulted.append(field)
    return expand_grid(default)


def _train_mse(model: Any, X: pd.DataFrame, y: np.ndarray) -> float:
    """In-sample MSE: raw predictions on the rows the final model was fitted to.

    Unweighted over option rows, whatever the fit's own loss weighting, so it is
    comparable across weighting schemes.
    """

    return float(model.score(X, np.asarray(y, dtype=float)))


def _check_simple_average_options(model_config: Mapping[str, Any]) -> None:
    """The simple average has no penalty, validation or design options.

    Any such option is refused rather than ignored, so a configuration cannot
    look like it changes the baseline when it does not. The legacy no-effect
    options are accepted and recorded, as for every method.
    """

    allowed = {"method", *IGNORED_MODEL_OPTIONS}
    inapplicable = sorted(set(model_config) - allowed)
    if inapplicable:
        raise ValueError(
            f"model option(s) {', '.join(repr(name) for name in inapplicable)} "
            f"do not apply to method {SIMPLE_AVERAGE!r}: it weights every agent "
            "equally and selects nothing"
        )


def _fit_simple_average(
    data: RegressionData,
    ignored_options: Sequence[str],
) -> FitResult:
    """Uniform weights over the cleaned proxy panel, after checking the design.

    The baseline's denominator must be the same for every question: every agent
    in the design must have exactly one answer on every question block of every
    split. Column-drop cleaning guarantees this; a design that breaks it (for
    example one with zero entries for non-response) is refused before
    prediction rather than averaged over a varying set of agents.
    """

    groups = data.binary_survey.original_to_binary_map
    for name, split in data.splits.items():
        rows = set(split.X.index)
        for qid, members in groups.items():
            present = [member for member in members if member in rows]
            if not present:
                continue
            answers = split.X.loc[present].sum(axis=0).to_numpy(dtype=float)
            incomplete = int(np.sum(answers != 1.0))
            if incomplete:
                raise ValueError(
                    f"simple average: {incomplete} agent(s) do not have exactly "
                    f"one answer on question {qid!r} (split {name!r}); the "
                    "average would not have a fixed denominator"
                )

    train = pd.concat([data.get_split(name).X for name in data.train_splits])
    train_y = np.concatenate([data.get_split(name).y for name in data.train_splits])
    model = MODELS.create(SIMPLE_AVERAGE).fit(train)
    beta = np.asarray(model.beta_, dtype=float)
    diagnostics: dict[str, Any] = {
        "validation_strategy": "none",
        "train_mse": _train_mse(model, train, train_y),
        "anti_type_mass": 0.0,
        "anti_type_count": 0,
        "coefficient_sum": float(beta.sum()),
        "solver_used": None,
        "solver_fallback": False,
    }
    if ignored_options:
        diagnostics["ignored_model_options"] = ", ".join(ignored_options)
    return FitResult(
        method=SIMPLE_AVERAGE,
        model=model,
        solver="none",
        status=model.status_,
        best_alpha=None,
        best_l1_ratio=None,
        diagnostics=diagnostics,
        folds=None,
    )


_FOLD_OPTIONS = frozenset({"method", "file"})


def _fold_settings(
    value: Any,
    *,
    cv_folds: int,
) -> tuple[str, FoldAssignment | None]:
    """Read ``model.validation.folds``: the fold method and, for replay, the record.

    Returns the method used to generate folds (ignored for replay) and the
    replayed ``FoldAssignment``, or ``None`` when folds are generated.
    """

    if value is None:
        return DEFAULT_FOLD_METHOD, None
    if not isinstance(value, Mapping):
        raise ValueError("model.validation.folds must be a mapping")

    unknown = sorted(set(value) - _FOLD_OPTIONS)
    if unknown:
        described = []
        for name in unknown:
            close = get_close_matches(name, _FOLD_OPTIONS, n=1, cutoff=0.6)
            described.append(
                f"{name!r} (did you mean {close[0]!r}?)" if close else repr(name)
            )
        raise ValueError(
            "unknown model.validation.folds option(s): "
            + ", ".join(described)
            + ". Recognised options are: file, method"
        )

    method = value.get("method", DEFAULT_FOLD_METHOD)
    if method not in FOLD_METHODS:
        raise ValueError(
            f"model.validation.folds.method must be one of {FOLD_METHODS}; "
            f"got {method!r}"
        )

    file = value.get("file")
    if method != "replay":
        if file is not None:
            raise ValueError(
                "model.validation.folds.file is only used with method: replay"
            )
        return method, None

    if not isinstance(file, str) or not file.strip():
        raise ValueError(
            "model.validation.folds.method replay needs "
            "model.validation.folds.file"
        )
    record = read_fold_record(file)
    if record.n_splits != cv_folds:
        raise ValueError(
            f"the fold record has {record.n_splits} folds but "
            f"model.validation.cv_folds is {cv_folds}"
        )
    return DEFAULT_FOLD_METHOD, record


def _ignored_model_options(model_config: Mapping[str, Any]) -> tuple[str, ...]:
    """Options present in the config that have no effect.

    ``post_selection_refit: true`` asks for a refit that P2P does not perform,
    so it also warns; ``max_iter`` is only recorded.
    """

    present = tuple(name for name in IGNORED_MODEL_OPTIONS if name in model_config)
    refit = model_config.get("post_selection_refit", False)
    if not isinstance(refit, bool):
        raise ValueError("model.post_selection_refit must be boolean")
    if refit:
        warnings.warn(
            "model.post_selection_refit: true has no effect -- P2P does not "
            "implement a post-selection refit (nor did the legacy pipeline, "
            "which also ignored it); the model is fitted as usual",
            IgnoredModelOptionWarning,
            stacklevel=3,
        )
    return present


def penalty_selection_diagnostics(
    model_cv: Any,
    *,
    alphas: Sequence[float],
    l1_ratios: Sequence[float] | None,
) -> dict[str, Any]:
    """Report whether the selected penalty was actually determined by the data.

    Two ways it may not have been. The optimum can sit at an end of the searched
    range, meaning the range is probably wrong. Or many penalties can score
    within one standard error of the best, meaning the choice among them is
    noise -- which is what happens once the penalty stops discriminating, and is
    why a run can land at the grid's ceiling with nothing to distinguish it from
    a value orders of magnitude away.

    Reported rather than corrected: both are signals about the grid and the
    data, not errors.
    """

    scores = getattr(model_cv, "cv_mean_mse_", None) or {}
    deviations = getattr(model_cv, "cv_std_mse_", None) or {}
    best = getattr(model_cv, "best_params_", None)
    if not scores or best is None:
        return {}

    best_alpha, best_l1_ratio = best
    best_score = float(scores[best])
    # The standard error of the best configuration's mean CV error. The CV
    # stores the population SD of its K fold errors; the standard error of their
    # mean is the sample SD over sqrt(K), i.e. that SD over sqrt(K - 1).
    folds = getattr(model_cv, "k", None)
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError(
            "penalty selection diagnostics need the number of CV folds (k >= 2)"
        )
    best_standard_error = float(deviations.get(best, 0.0)) / math.sqrt(folds - 1)

    finite = [
        float(value)
        for value in scores.values()
        if math.isfinite(float(value))
    ]
    within_one_se = sum(
        1
        for value in finite
        if value <= best_score + best_standard_error
    )

    diagnostics: dict[str, Any] = {
        "cv_config_count": len(finite),
        "cv_configs_within_one_se": within_one_se,
        "cv_score_spread": (
            max(finite) - min(finite) if finite else 0.0
        ),
    }

    if len(alphas) > 1:
        ordered = sorted(float(value) for value in alphas)
        at_floor = float(best_alpha) <= ordered[0]
        at_ceiling = float(best_alpha) >= ordered[-1]
        diagnostics["alpha_at_grid_edge"] = (
            "floor" if at_floor else "ceiling" if at_ceiling else None
        )

    if l1_ratios is not None and len(l1_ratios) > 1 and (
        best_l1_ratio is not None
    ):
        ordered = sorted(float(value) for value in l1_ratios)
        at_floor = float(best_l1_ratio) <= ordered[0]
        at_ceiling = float(best_l1_ratio) >= ordered[-1]
        diagnostics["l1_ratio_at_grid_edge"] = (
            "floor" if at_floor else "ceiling" if at_ceiling else None
        )

    messages = []
    edge = diagnostics.get("alpha_at_grid_edge")
    if edge:
        messages.append(
            f"selected alpha {best_alpha:g} sits at the {edge} of the "
            f"searched range [{min(alphas):g}, {max(alphas):g}]; the "
            "optimum may lie outside it"
        )
    ratio_edge = diagnostics.get("l1_ratio_at_grid_edge")
    if ratio_edge:
        messages.append(
            f"selected l1_ratio {best_l1_ratio:g} sits at the {ratio_edge} "
            "of its searched range"
        )
    if len(finite) > 1 and within_one_se > 1:
        messages.append(
            f"{within_one_se} of {len(finite)} penalties score within one "
            "standard error of the best, so the selection among them is not "
            "distinguishable from noise"
        )

    if messages:
        warnings.warn(
            "penalty selection: " + "; ".join(messages),
            PenaltySelectionWarning,
            stacklevel=3,
        )

    return diagnostics


VALIDATION_STRATEGIES = ("cv", "holdout")

_HOLDOUT_KEYS = frozenset({"strategy", "holdout_split"})


def _holdout_settings(
    validation: Mapping[str, Any],
    *,
    train_splits: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    """Read ``model.validation`` for holdout: the split that selects the penalty.

    Holdout fits on the remaining training splits and scores each penalty on
    ``holdout_split`` (default ``valid``); the penalty with the lowest error is
    kept together with its fit, with no refit on the holdout split. Settings
    that belong to cross-validation are refused rather than ignored.
    """

    unknown = sorted(set(validation) - _HOLDOUT_KEYS)
    if unknown:
        raise ValueError(
            "model.validation."
            + ", model.validation.".join(unknown)
            + " does not apply to holdout validation"
        )
    holdout_split = validation.get("holdout_split", "valid")
    if not isinstance(holdout_split, str) or not holdout_split:
        raise ValueError("model.validation.holdout_split must be a nonempty string")
    if holdout_split not in train_splits:
        raise ValueError(
            f"model.validation.holdout_split {holdout_split!r} must be one of "
            f"the training splits {tuple(train_splits)}"
        )
    fit_splits = tuple(name for name in train_splits if name != holdout_split)
    if not fit_splits:
        raise ValueError(
            "holdout validation needs at least one training split besides "
            f"{holdout_split!r}"
        )
    return holdout_split, fit_splits


def _grid_edge(value: float, grid: Sequence[float]) -> str | None:
    ordered = sorted(float(item) for item in grid)
    if float(value) <= ordered[0]:
        return "floor"
    if float(value) >= ordered[-1]:
        return "ceiling"
    return None


def _fit_holdout(
    data: RegressionData,
    *,
    model_cls: Any,
    model_kwargs: Mapping[str, Any],
    alphas: Sequence[float],
    l1_ratios: Sequence[float] | None,
    fit_splits: Sequence[str],
    holdout_split: str,
) -> tuple[Any, float, float | None, dict[str, Any]]:
    """Select the penalty on a holdout split; keep the fit on the other splits.

    Every penalty (and l1 ratio) is fitted on ``fit_splits`` and scored by MSE
    on ``holdout_split``, the same score cross-validation uses. The best fit is
    returned as fitted: the holdout split only selects, it is not refitted on.
    Ties keep the first configuration in grid order, as cross-validation does.
    """

    X_fit = pd.concat([data.get_split(name).X for name in fit_splits])
    y_fit = np.concatenate([data.get_split(name).y for name in fit_splits])
    held = data.get_split(holdout_split)

    scores: dict[tuple[float, float | None], float] = {}
    fallbacks = 0
    best_model = None
    best_params: tuple[float, float | None] | None = None
    best_score = math.inf
    for alpha in alphas:
        for l1_ratio in (l1_ratios if l1_ratios is not None else (None,)):
            kwargs = {**model_kwargs, "alpha": alpha}
            if l1_ratio is not None:
                kwargs["l1_ratio"] = l1_ratio
            model = model_cls(**kwargs)
            model.fit(X_fit, y_fit)
            fallbacks += int(bool(getattr(model, "solver_fallback_", False)))
            score = float(model.score(held.X, np.asarray(held.y, dtype=float)))
            scores[(alpha, l1_ratio)] = score
            if score < best_score:
                best_score, best_params, best_model = score, (alpha, l1_ratio), model

    if best_model is None:
        raise RuntimeError("holdout validation produced no finite score")
    best_alpha, best_l1_ratio = best_params

    finite = [value for value in scores.values() if math.isfinite(value)]
    diagnostics: dict[str, Any] = {
        "validation_strategy": "holdout",
        "train_mse": _train_mse(best_model, X_fit, y_fit),
        "holdout_split": holdout_split,
        "holdout_fit_splits": ", ".join(fit_splits),
        "holdout_fits": len(scores),
        "holdout_solver_fallbacks": fallbacks,
        "holdout_best_mse": best_score,
        "holdout_score_spread": (max(finite) - min(finite)) if finite else 0.0,
    }
    messages = []
    if len(alphas) > 1:
        edge = _grid_edge(best_alpha, alphas)
        diagnostics["alpha_at_grid_edge"] = edge
        if edge:
            messages.append(
                f"selected alpha {best_alpha:g} sits at the {edge} of the "
                f"searched range [{min(alphas):g}, {max(alphas):g}]; the "
                "optimum may lie outside it"
            )
    if l1_ratios is not None and len(l1_ratios) > 1:
        ratio_edge = _grid_edge(best_l1_ratio, l1_ratios)
        diagnostics["l1_ratio_at_grid_edge"] = ratio_edge
        if ratio_edge:
            messages.append(
                f"selected l1_ratio {best_l1_ratio:g} sits at the {ratio_edge} "
                "of its searched range"
            )
    if messages:
        warnings.warn(
            "penalty selection: " + "; ".join(messages),
            PenaltySelectionWarning,
            stacklevel=3,
        )
    return best_model, best_alpha, best_l1_ratio, diagnostics


def fit_model(
    data: RegressionData,
    model_config: Mapping[str, Any],
    *,
    solver: str,
    seed: int,
    verbose: bool = False,
    fold_assignments: FoldAssignment | Mapping[str, int] | None = None,
) -> FitResult:
    """Cross-validate and refit one registered constrained model.

    Folds follow ``model.validation.folds`` (default: the ``hashed`` method
    seeded by ``seed``). ``fold_assignments`` replays a given assignment
    instead; it must not be combined with ``folds.method: replay``.
    """

    if not isinstance(data, RegressionData):
        raise TypeError("data must be RegressionData")
    if not isinstance(model_config, Mapping):
        raise TypeError("model_config must be a mapping")
    if not isinstance(solver, str) or not solver.strip():
        raise ValueError(
            "solver must be a nonempty string"
        )
    if solver not in cp.installed_solvers():
        raise ValueError(
            f"solver {solver!r} is not installed; "
            f"available: {cp.installed_solvers()}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (
        fold_assignments is not None
        and not isinstance(fold_assignments, (Mapping, FoldAssignment))
    ):
        raise TypeError(
            "fold_assignments must be a FoldAssignment, a mapping or None"
        )

    validate_model_options(model_config)
    ignored_options = _ignored_model_options(model_config)

    method = model_config.get("method")
    if data.reference is not None and method == SIMPLE_AVERAGE:
        raise ValueError(
            "data.reference_distribution cannot be used with method "
            f"{SIMPLE_AVERAGE!r}, which fits no weights"
        )
    if method == SIMPLE_AVERAGE:
        if fold_assignments is not None:
            raise ValueError(
                "fold assignments do not apply to the simple average"
            )
        return _fit_simple_average(data, ignored_options)
    if method not in {"lasso", "elasticnet"}:
        raise ValueError(
            f"unknown model method {method!r}"
        )

    defaulted_grids: list[str] = []
    alphas = _numeric_grid(
        _grid_or_default(model_config, "alpha", DEFAULT_ALPHA_GRID, defaulted_grids),
        field="model.alpha",
        minimum=0.0,
    )

    l1_ratios: tuple[float, ...] | None = None
    if method == "elasticnet":
        l1_ratios = _numeric_grid(
            _grid_or_default(
                model_config, "l1_ratio", DEFAULT_L1_RATIO_GRID, defaulted_grids
            ),
            field="model.l1_ratio",
            minimum=0.0,
            maximum=1.0,
        )

    validation = model_config.get("validation", {})
    if not isinstance(validation, Mapping):
        raise ValueError(
            "model.validation must be a mapping"
        )

    strategy = validation.get("strategy", "cv")
    if strategy not in VALIDATION_STRATEGIES:
        raise ValueError(
            f"model.validation.strategy must be one of "
            f"{VALIDATION_STRATEGIES}; got {strategy!r}"
        )

    holdout_split = None
    if strategy == "holdout":
        holdout_split, fit_splits = _holdout_settings(
            validation,
            train_splits=data.train_splits,
        )
        if fold_assignments is not None:
            raise ValueError(
                "fold assignments apply to cv validation only"
            )
    else:
        cv_folds = validation.get("cv_folds", 5)
        if (
            isinstance(cv_folds, bool)
            or not isinstance(cv_folds, int)
            or cv_folds < 2
        ):
            raise ValueError(
                "model.validation.cv_folds must be at least 2"
            )

        fold_method, replay = _fold_settings(
            validation.get("folds"),
            cv_folds=cv_folds,
        )
        if replay is not None and fold_assignments is not None:
            raise ValueError(
                "fold assignments were given twice: by model.validation.folds "
                "(replay) and by the fold_assignments argument"
            )
        if replay is not None:
            fold_assignments = replay

    zero_threshold = _finite_number(
        model_config.get("zero_threshold", DEFAULT_ZERO_THRESHOLD),
        field="model.zero_threshold",
        minimum=0.0,
    )
    group_penalty_strength = _finite_number(
        model_config.get(
            "group_penalty_strength",
            100.0,
        ),
        field="model.group_penalty_strength",
        minimum=0.0,
    )
    group_total = _finite_number(
        model_config.get("group_total", 1.0),
        field="model.group_total",
    )
    soft_group_sum = _boolean(
        model_config.get("soft_group_sum", True),
        field="model.soft_group_sum",
    )
    nonneg = _boolean(
        model_config.get("nonneg", False),
        field="model.nonneg",
    )
    sum_to_one = _boolean(
        model_config.get("sum_to_one", False),
        field="model.sum_to_one",
    )
    block_weighting = _boolean(
        model_config.get("block_weighting", False),
        field="model.block_weighting",
    )
    # An unpenalised constant on every option row; independent of the
    # reference respondent (see ConstrainedModel).
    intercept = _boolean(
        model_config.get("intercept", False),
        field="model.intercept",
    )

    train_frames = [
        data.get_split(name).X
        for name in data.train_splits
    ]
    train_targets = [
        data.get_split(name).y
        for name in data.train_splits
    ]

    X_all = pd.concat(train_frames)
    y_all = np.concatenate(train_targets)

    constraint_groups = dict(
        data.binary_survey.original_to_binary_map
    )

    registry_name = (
        f"weighted-{method}"
        if block_weighting
        else method
    )
    model_cls = MODELS.get(registry_name)

    if data.reference is not None:
        # Each option combined here would redefine what the reference's weight
        # of one means; refused rather than given an unexamined meaning.
        chosen = {
            "intercept": intercept,
            "sum_to_one": sum_to_one,
            "nonneg": nonneg,
            "reference_respondent": REFERENCE_FEATURE in data.feature_names,
        }
        conflicts = [
            f"model.{name}"
            for name in REFERENCE_DISTRIBUTION_CONFLICTS
            if chosen[name]
        ]
        if conflicts:
            raise ValueError(
                "data.reference_distribution cannot be combined with "
                + ", ".join(conflicts)
            )
        check_reference_masses(
            data.reference.binary,
            data.binary_survey.original_to_binary_map,
            pd.Index(
                [
                    row
                    for name in (*data.train_splits, data.test_split)
                    for row in data.get_split(name).X.index
                ]
            ),
        )
        model_cls = with_reference_distribution(model_cls, data.reference.binary)

    model_kwargs: dict[str, Any] = {
        "constraint_groups": constraint_groups,
        "group_total": group_total,
        "nonneg": nonneg,
        "sum_to_one": sum_to_one,
        "zero_threshold": zero_threshold,
        "soft_group_sum": soft_group_sum,
        "group_penalty_strength": (
            group_penalty_strength
        ),
        "solver": solver,
        "verbose": verbose,
    }
    if intercept:
        model_kwargs["intercept"] = True

    if block_weighting:
        model_kwargs["question_weights"] = {
            original_qid: 1.0 / len(binary_qids)
            for original_qid, binary_qids
            in constraint_groups.items()
            if binary_qids
        }

    if strategy == "holdout":
        model, best_alpha, best_l1_ratio, diagnostics = _fit_holdout(
            data,
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            alphas=alphas,
            l1_ratios=l1_ratios,
            fit_splits=fit_splits,
            holdout_split=holdout_split,
        )
        folds = None
    else:
        # Fold grouping is stated separately from the mass constraints so that
        # folds stay grouped by question regardless of how the constraints are
        # configured.
        model_cv = ConstrainedModelCV(
            model_cls=model_cls,
            alphas=alphas,
            l1_ratios=l1_ratios,
            k=cv_folds,
            model_kwargs=model_kwargs,
            random_state=seed,
            verbose=verbose,
            fold_groups=constraint_groups,
            fold_method=fold_method,
        )
        model_cv.fit(
            X_all,
            y_all,
            fold_assignments=fold_assignments,
        )
        model = model_cv.get_best_model(
            X_all,
            y_all,
        )

        best_alpha, best_l1_ratio = (
            model_cv.best_params_
        )

        diagnostics = dict(model_cv.diagnostics_)
        diagnostics["train_mse"] = _train_mse(model, X_all, y_all)
        diagnostics.update(
            penalty_selection_diagnostics(
                model_cv,
                alphas=alphas,
                l1_ratios=l1_ratios,
            )
        )
        folds = model_cv.fold_assignment_

    beta = np.asarray(model.beta_, dtype=float)
    # abs() rather than negation so an all-nonnegative fit reports 0.0 and not
    # the -0.0 that summing an empty selection produces.
    diagnostics["anti_type_mass"] = float(
        np.abs(beta[beta < 0].sum())
    )
    diagnostics["anti_type_count"] = int((beta < 0).sum())
    diagnostics["coefficient_sum"] = float(beta.sum())
    if data.reference is not None:
        diagnostics["reference_distribution_digest"] = data.reference.source_digest
    if intercept:
        diagnostics["intercept"] = float(model.intercept_)
    if ignored_options:
        diagnostics["ignored_model_options"] = ", ".join(ignored_options)
    if defaulted_grids:
        diagnostics["default_grids"] = ", ".join(defaulted_grids)
    diagnostics["solver_used"] = getattr(model, "solver_used_", None)
    diagnostics["solver_fallback"] = bool(
        getattr(model, "solver_fallback_", False)
    )

    return FitResult(
        method=method,
        model=model,
        solver=solver,
        status=model.status_,
        best_alpha=float(best_alpha),
        best_l1_ratio=(
            None
            if best_l1_ratio is None
            else float(best_l1_ratio)
        ),
        diagnostics=diagnostics,
        folds=folds,
    )


__all__ = [
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_L1_RATIO_GRID",
    "FitResult",
    "SIMPLE_AVERAGE",
    "fit_model",
]