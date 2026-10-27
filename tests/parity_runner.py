"""Test-only historical regression parity harness."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

from p2p.stages.evaluate import evaluate_model
from p2p.models.folds import FoldAssignment, read_fold_record
from p2p.stages.fit import fit_model
from p2p.stages.split import RegressionData, load_regression_data


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_FIXTURES = {
    "atp_w42": {
        "source_run": "ATP_w42_20250919_022806",
        "legacy_config": (
            "outputs/fixtures/atp_w42/"
            "legacy_regression_config.yaml"
        ),
        "feature_order": (
            "tests/fixtures/commands/parity/"
            "atp_w42_feature_order.json"
        ),
        "cv_folds": (
            "tests/fixtures/commands/parity/"
            "atp_w42_cv_folds.json"
        ),
        "responses": (
            "outputs/fixtures/atp_w42/responses.csv"
        ),
        "endowments": (
            "outputs/fixtures/atp_w42/endowments.csv"
        ),
        "survey_source": "data/w42/info.csv",
        "survey_schema": (
            "config/surveys/american_trend_panel.yaml"
        ),
        "target": "data/w42/W42_aggregate_results.json",
    },
    "wvs_gb": {
        "source_run": "WVS_GB_20251116_081710",
        "legacy_config": (
            "outputs/fixtures/wvs_gb/"
            "legacy_regression_config.yaml"
        ),
        "feature_order": (
            "tests/fixtures/commands/parity/"
            "wvs_gb_feature_order.json"
        ),
        "cv_folds": (
            "tests/fixtures/commands/parity/"
            "wvs_gb_cv_folds.json"
        ),
        "responses": (
            "outputs/fixtures/wvs_gb/responses.csv"
        ),
        "endowments": (
            "outputs/fixtures/wvs_gb/endowments.csv"
        ),
        "survey_source": "data/wvs_gb/info.csv",
        "survey_schema": "config/surveys/wvs.yaml",
        "target": "data/wvs_gb/aggregate_results.json",
    },
}

_NUMBER_PATTERN = (
    r"[+-]?(?:"
    r"(?:\d+(?:\.\d*)?)"
    r"|(?:\.\d+)"
    r")(?:[eE][+-]?\d+)?"
)

_GRID_PATTERN = re.compile(
    rf"^np\.(?P<operator>logspace|linspace)"
    rf"\(\s*(?P<start>{_NUMBER_PATTERN})"
    rf"\s*,\s*(?P<stop>{_NUMBER_PATTERN})"
    rf"\s*,\s*(?P<count>[+-]?\d+)\s*\)$"
)

_EVALUATION_METRICS = (
    "mse",
    "one_minus_jsd",
    "emd",
    "mcpa",
)


def _fixture_path(relative_path: str) -> Path:
    return _PROJECT_ROOT / relative_path


def _require_mapping(
    value: Any,
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _load_legacy_config(
    path: Path,
) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(
            f"legacy regression config does not exist: {path}"
        )

    try:
        loaded = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"could not read legacy regression config {path}: "
            f"{error}"
        ) from error

    return _require_mapping(
        loaded,
        field="legacy regression config",
    )


def _load_json_mapping(
    path: Path,
    *,
    label: str,
) -> tuple[Mapping[str, Any], str]:
    if not path.is_file():
        raise ValueError(
            f"{label} does not exist: {path}"
        )

    raw_bytes = path.read_bytes()
    try:
        loaded = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid {label} {path}: {error}"
        ) from error

    payload = _require_mapping(
        loaded,
        field=label,
    )
    return (
        payload,
        hashlib.sha256(raw_bytes).hexdigest(),
    )


def _validate_fixture_identity(
    payload: Mapping[str, Any],
    *,
    label: str,
    dataset: str,
    source_run: str,
) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"{label} schema_version must be 1"
        )
    if payload.get("dataset") != dataset:
        raise ValueError(
            f"{label} dataset does not match "
            f"{dataset!r}"
        )
    if payload.get("source_run") != source_run:
        raise ValueError(
            f"{label} source_run does not match "
            f"{source_run!r}"
        )


def _load_feature_order(
    path: Path,
    *,
    dataset: str,
    source_run: str,
) -> tuple[tuple[str, ...], str]:
    payload, digest = _load_json_mapping(
        path,
        label="feature-order fixture",
    )
    _validate_fixture_identity(
        payload,
        label="feature-order fixture",
        dataset=dataset,
        source_run=source_run,
    )

    names = payload.get("feature_names")
    if (
        not isinstance(names, Sequence)
        or isinstance(names, (str, bytes, bytearray))
        or not names
    ):
        raise ValueError(
            "feature-order fixture feature_names must "
            "be a nonempty sequence"
        )

    normalized = tuple(names)
    if any(
        not isinstance(name, str) or not name
        for name in normalized
    ):
        raise ValueError(
            "feature-order entries must be nonempty strings"
        )
    if len(normalized) != len(set(normalized)):
        raise ValueError(
            "feature-order fixture contains duplicate names"
        )

    return normalized, digest


def _load_cv_fold_assignments(
    path: Path,
    *,
    dataset: str,
    source_run: str,
) -> tuple[FoldAssignment, int, str]:
    """Load a session-4 CV-fold fixture through P2P's own fold-record reader.

    The fixture's identity (dataset, source run) is checked here; the folds
    themselves are parsed and validated by ``read_fold_record``, which also
    carries the fixture's per-fold option-row totals so that replay checks
    them against the rebuilt design.
    """

    payload, digest = _load_json_mapping(
        path,
        label="CV-fold fixture",
    )
    _validate_fixture_identity(
        payload,
        label="CV-fold fixture",
        dataset=dataset,
        source_run=source_run,
    )
    record = read_fold_record(path)
    return record, record.n_splits, digest


def _record_source_artifact(
    data: RegressionData,
    *,
    name: str,
    path: Path,
    digest: str,
) -> RegressionData:
    source_artifacts = dict(data.source_artifacts)
    source_artifacts[name] = path.resolve()

    source_digests = dict(data.source_digests)
    source_digests[name] = digest

    return replace(
        data,
        source_artifacts=MappingProxyType(
            source_artifacts
        ),
        source_digests=MappingProxyType(
            source_digests
        ),
    )


def _apply_feature_order(
    data: RegressionData,
    feature_order: tuple[str, ...],
    *,
    source_path: Path,
    source_digest: str,
) -> RegressionData:
    current = set(data.feature_names)
    pinned = set(feature_order)

    if current != pinned:
        missing = sorted(current - pinned)
        unexpected = sorted(pinned - current)
        raise ValueError(
            "feature-order fixture does not match retained "
            f"regression features; missing={missing}, "
            f"unexpected={unexpected}"
        )

    ordered_splits = {
        name: replace(
            split,
            X=split.X.loc[
                :,
                list(feature_order),
            ].copy(),
        )
        for name, split in data.splits.items()
    }

    reordered = replace(
        data,
        splits=MappingProxyType(ordered_splits),
        feature_names=feature_order,
    )
    return _record_source_artifact(
        reordered,
        name="feature_order",
        path=source_path,
        digest=source_digest,
    )


def _expand_legacy_grid(
    value: Any,
    *,
    field: str,
) -> tuple[float, ...]:
    if (
        isinstance(value, Sequence)
        and not isinstance(
            value,
            (str, bytes, bytearray),
        )
    ):
        if not value:
            raise ValueError(
                f"{field} must not be empty"
            )
        return tuple(float(item) for item in value)

    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be a sequence or supported "
            "NumPy grid"
        )

    match = _GRID_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(
            f"unsupported legacy grid expression for "
            f"{field}: {value!r}"
        )

    start = float(match.group("start"))
    stop = float(match.group("stop"))
    count = int(match.group("count"))
    if count <= 0:
        raise ValueError(
            f"{field} grid count must be positive"
        )

    operator = match.group("operator")
    values = (
        np.logspace(start, stop, count)
        if operator == "logspace"
        else np.linspace(start, stop, count)
    )
    return tuple(float(item) for item in values)


def _model_config(
    legacy: Mapping[str, Any],
    method: str,
) -> dict[str, Any]:
    source = _require_mapping(
        legacy.get(method),
        field=f"legacy {method}",
    )

    config: dict[str, Any] = {
        "method": method,
        "alpha": _expand_legacy_grid(
            source.get("alpha_expr"),
            field=f"{method}.alpha_expr",
        ),
        "validation": dict(
            _require_mapping(
                source.get("validation", {}),
                field=f"{method}.validation",
            )
        ),
    }

    if method == "elasticnet":
        config["l1_ratio"] = _expand_legacy_grid(
            source.get("l1_ratio_expr"),
            field="elasticnet.l1_ratio_expr",
        )

    for field in (
        "zero_threshold",
        "soft_group_sum",
        "group_penalty_strength",
        "group_total",
        "nonneg",
        "sum_to_one",
        "block_weighting",
    ):
        if field in source:
            config[field] = source[field]

    return config


def run_fixture(
    *,
    dataset: str,
    solver: str,
) -> dict[str, dict[str, Any]]:
    """Run one frozen regression fixture through the Stage-2 path."""

    try:
        fixture = _FIXTURES[dataset]
    except KeyError:
        available = ", ".join(sorted(_FIXTURES))
        raise ValueError(
            f"unknown regression fixture {dataset!r}; "
            f"available: {available}"
        ) from None

    legacy = _load_legacy_config(
        _fixture_path(fixture["legacy_config"])
    )

    experiment = _require_mapping(
        legacy.get("experiment"),
        field="legacy experiment",
    )
    seed = experiment.get("seed", 101)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            "legacy experiment.seed must be an integer"
        )

    split_settings = _require_mapping(
        legacy.get("split_settings"),
        field="legacy split_settings",
    )
    train_splits = split_settings.get(
        "train_val_split",
        ("train", "valid"),
    )
    test_split = split_settings.get(
        "test_split",
        "test",
    )
    use_proxy_only = split_settings.get(
        "use_proxy_only",
        True,
    )
    if not isinstance(use_proxy_only, bool):
        raise ValueError(
            "legacy split_settings.use_proxy_only "
            "must be boolean"
        )

    data = load_regression_data(
        survey_source=_fixture_path(
            fixture["survey_source"]
        ),
        survey_schema=_fixture_path(
            fixture["survey_schema"]
        ),
        responses_path=_fixture_path(
            fixture["responses"]
        ),
        endowments_path=_fixture_path(
            fixture["endowments"]
        ),
        target_path=_fixture_path(
            fixture["target"]
        ),
        train_splits=train_splits,
        test_split=test_split,
        use_proxy_only=use_proxy_only,
        # Parity reproduces the earlier scoring protocol, defects included:
        # raw answer text and the answer-file route.
        canonical_answers=False,
    )

    feature_order_path = _fixture_path(
        fixture["feature_order"]
    )
    feature_order, feature_order_digest = (
        _load_feature_order(
            feature_order_path,
            dataset=dataset,
            source_run=fixture["source_run"],
        )
    )
    data = _apply_feature_order(
        data,
        feature_order,
        source_path=feature_order_path,
        source_digest=feature_order_digest,
    )

    cv_folds_path = _fixture_path(
        fixture["cv_folds"]
    )
    (
        fold_assignments,
        fold_count,
        fold_digest,
    ) = _load_cv_fold_assignments(
        cv_folds_path,
        dataset=dataset,
        source_run=fixture["source_run"],
    )
    data = _record_source_artifact(
        data,
        name="cv_folds",
        path=cv_folds_path,
        digest=fold_digest,
    )

    model_configs = {
        method: _model_config(legacy, method)
        for method in ("lasso", "elasticnet")
    }

    for method, model_config in model_configs.items():
        validation = _require_mapping(
            model_config.get("validation"),
            field=f"{method}.validation",
        )
        configured_fold_count = validation.get(
            "cv_folds",
            5,
        )
        if configured_fold_count != fold_count:
            raise ValueError(
                f"{method} requests "
                f"{configured_fold_count} folds but the "
                f"fixture contains {fold_count}"
            )

    results: dict[str, dict[str, Any]] = {}

    for method in ("lasso", "elasticnet"):
        fitted = fit_model(
            data,
            model_configs[method],
            solver=solver,
            seed=seed,
            verbose=False,
            fold_assignments=fold_assignments,
        )
        evaluated = evaluate_model(
            data,
            fitted,
            split=data.test_split,
            metrics=_EVALUATION_METRICS,
            distribution_prediction_policy=(
                "legacy_signed_normalize"
            ),
            distribution_route="legacy_answer_file",
        )

        result: dict[str, Any] = {
            "status": fitted.status,
            "test_mse": evaluated.values["mse"],
            "test_one_minus_jsd": (
                evaluated.values["one_minus_jsd"]
            ),
            "test_emd": evaluated.values["emd"],
            "test_mcpa": evaluated.values["mcpa"],
            "nnz": int(fitted.model.num_nonzero_),
            "best_alpha": fitted.best_alpha,
        }

        if fitted.best_l1_ratio is not None:
            result["best_l1_ratio"] = (
                fitted.best_l1_ratio
            )

        results[method] = result

    return results


__all__ = [
    "run_fixture",
]