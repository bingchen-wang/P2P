"""Production Stage-2 regression command composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any

from p2p.config import Config
from p2p.stages.evaluate import (
    DISTRIBUTION_ROUTES,
    EvaluationResult,
    evaluate_model,
)
from p2p.stages.fit import (
    FitResult,
    fit_model,
    validate_model_options,
)
from p2p.stages import report as report_stage
from p2p.stages.split import (
    RegressionData,
    load_regression_data,
)
from p2p.stages.targets import (
    target_renormalize,
    target_spec,
)


@dataclass(frozen=True, slots=True)
class RegressionResult:
    """One completed in-memory regression execution."""

    data: RegressionData
    fit: FitResult
    evaluation: EvaluationResult
    config_digest: str | None = None
    run_dir: Path | None = None
    report: Any | None = None


def _require_mapping(
    value: Any,
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _resolved_path(
    value: Any,
    *,
    field: str,
) -> Path:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field} must be a nonempty path"
        )
    return Path(value).resolve()


def _survey_paths(
    value: Any,
) -> tuple[Path, Path]:
    survey = _require_mapping(
        value,
        field="data.survey",
    )
    return (
        _resolved_path(
            survey.get("source"),
            field="data.survey.source",
        ),
        _resolved_path(
            survey.get("schema"),
            field="data.survey.schema",
        ),
    )


def _basis_paths(
    value: Any,
) -> tuple[Path, Path]:
    if isinstance(value, Mapping):
        return (
            _resolved_path(
                value.get("responses"),
                field="data.basis.responses",
            ),
            _resolved_path(
                value.get("endowments"),
                field="data.basis.endowments",
            ),
        )

    basis_path = _resolved_path(
        value,
        field="data.basis",
    )
    if not basis_path.is_dir():
        raise ValueError(
            "data.basis must be a directory or a mapping "
            "containing responses and endowments"
        )

    return (
        (basis_path / "responses.csv").resolve(),
        (basis_path / "endowments.csv").resolve(),
    )


def _split_contract(
    value: Any,
) -> tuple[tuple[str, ...], str, bool]:
    split = _require_mapping(
        value,
        field="split",
    )

    train_splits = split.get(
        "train_val",
        ("train", "valid"),
    )
    if (
        not isinstance(train_splits, Sequence)
        or isinstance(
            train_splits,
            (str, bytes, bytearray),
        )
        or not train_splits
    ):
        raise ValueError(
            "split.train_val must be a nonempty sequence"
        )

    normalized_train = tuple(train_splits)
    if any(
        not isinstance(name, str) or not name
        for name in normalized_train
    ):
        raise ValueError(
            "split.train_val must contain nonempty strings"
        )

    test_split = split.get("test", "test")
    if not isinstance(test_split, str) or not test_split:
        raise ValueError(
            "split.test must be a nonempty string"
        )

    use_proxy_only = split.get(
        "use_proxy_only",
        True,
    )
    if not isinstance(use_proxy_only, bool):
        raise ValueError(
            "split.use_proxy_only must be boolean"
        )

    return (
        normalized_train,
        test_split,
        use_proxy_only,
    )


def _metric_names(
    value: Any,
) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
    ):
        raise ValueError(
            "metrics must be a nonempty sequence"
        )

    metrics = tuple(value)
    if any(
        not isinstance(metric, str) or not metric
        for metric in metrics
    ):
        raise ValueError(
            "metrics must contain nonempty strings"
        )
    return metrics


def run_regression(
    data: RegressionData,
    model_config: Mapping[str, Any],
    *,
    metrics: Sequence[str],
    solver: str,
    seed: int,
    test_split: str,
    distribution_prediction_policy: str,
    invalid_distribution: str = "allow",
    verbose: bool = False,
    fold_assignments: Mapping[str, int] | None = None,
    config_digest: str | None = None,
    distribution_route: str = "model",
) -> RegressionResult:
    """Fit and evaluate one model on prepared regression data."""

    fitted = fit_model(
        data,
        model_config,
        solver=solver,
        seed=seed,
        verbose=verbose,
        fold_assignments=fold_assignments,
    )
    evaluated = evaluate_model(
        data,
        fitted,
        split=test_split,
        metrics=metrics,
        distribution_prediction_policy=(
            distribution_prediction_policy
        ),
        invalid_distribution=invalid_distribution,
        distribution_route=distribution_route,
    )

    return RegressionResult(
        data=data,
        fit=fitted,
        evaluation=evaluated,
        config_digest=config_digest,
    )


def run(
    config: Config,
    *,
    solver: str,
    verbose: bool = False,
    basis: Any | None = None,
    run_dir: str | os.PathLike[str] | None = None,
) -> RegressionResult:
    """Execute regression from config or generated basis artifacts."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")

    if basis is None:
        config.validate_for("regress")

    resolved = config.data
    data_config = _require_mapping(
        resolved.get("data"),
        field="data",
    )
    model_config = _require_mapping(
        resolved.get("model"),
        field="model",
    )
    run_config = _require_mapping(
        resolved.get("run"),
        field="run",
    )
    evaluation_config = _require_mapping(
        resolved.get("evaluation"),
        field="evaluation",
    )

    survey_source, survey_schema = _survey_paths(
        data_config.get("survey")
    )
    responses_path, endowments_path = _basis_paths(
        data_config.get("basis")
        if basis is None
        else basis
    )
    target_path, target_subgroup = target_spec(
        data_config.get("target")
    )
    renormalize_targets = target_renormalize(
        data_config.get("target")
    )
    train_splits, test_split, use_proxy_only = (
        _split_contract(resolved.get("split"))
    )
    metrics = _metric_names(
        resolved.get("metrics")
    )

    mse_prediction_policy = evaluation_config.get(
        "mse_prediction_policy",
        "raw",
    )
    if mse_prediction_policy != "raw":
        raise ValueError(
            "the initial regress path supports only raw MSE "
            "predictions"
        )

    distribution_prediction_policy = (
        evaluation_config.get(
            "distribution_prediction_policy",
            "clip_and_normalize",
        )
    )
    invalid_distribution = evaluation_config.get(
        "invalid_distribution",
        "error",
    )
    distribution_route = evaluation_config.get(
        "distribution_route",
        "model",
    )
    if distribution_route not in DISTRIBUTION_ROUTES:
        raise ValueError(
            f"evaluation.distribution_route must be one of "
            f"{DISTRIBUTION_ROUTES}; got {distribution_route!r}"
        )
    canonical_answers = data_config.get("canonical_answers", True)
    if not isinstance(canonical_answers, bool):
        raise ValueError("data.canonical_answers must be boolean")

    # Validated here as well as in the fit stage so a mistyped option fails
    # before the basis is loaded rather than after.
    validate_model_options(model_config)

    reference_respondent = model_config.get(
        "reference_respondent",
        False,
    )
    if not isinstance(reference_respondent, bool):
        raise ValueError(
            "model.reference_respondent must be boolean"
        )

    reference_distribution = data_config.get("reference_distribution")
    if reference_distribution is not None and not isinstance(
        reference_distribution, (str, Path)
    ):
        raise ValueError(
            "data.reference_distribution must be a path to a JSON file in the "
            "target format"
        )

    data = load_regression_data(
        survey_source=survey_source,
        survey_schema=survey_schema,
        responses_path=responses_path,
        endowments_path=endowments_path,
        target_path=target_path,
        target_subgroup=target_subgroup,
        train_splits=train_splits,
        test_split=test_split,
        use_proxy_only=use_proxy_only,
        reference_respondent=reference_respondent,
        renormalize_targets=renormalize_targets,
        canonical_answers=canonical_answers,
        reference_distribution_path=reference_distribution,
    )

    result = run_regression(
        data,
        model_config,
        metrics=metrics,
        solver=solver,
        seed=run_config.get("seed", 101),
        test_split=test_split,
        distribution_prediction_policy=(
            distribution_prediction_policy
        ),
        invalid_distribution=invalid_distribution,
        verbose=verbose,
        config_digest=config.digest,
        distribution_route=distribution_route,
    )
    if run_dir is None:
        return result

    resolved_run_dir = Path(run_dir).resolve()
    canonical_report = (
        report_stage.write_regression_report(
            config,
            result,
            run_dir=resolved_run_dir,
        )
    )
    return replace(
        result,
        run_dir=resolved_run_dir,
        report=canonical_report,
    )


__all__ = [
    "RegressionResult",
    "run",
    "run_regression",
]
