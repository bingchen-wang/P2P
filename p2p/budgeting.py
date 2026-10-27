"""Historical token and cost forecasting for API-backed P2P runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yaml

from p2p.config import Config
from p2p.stages._backend import load_backend_config
from p2p.survey import Survey
from p2p.usage import TokenPricing


_MODULE_ALIASES = {
    "attributelearner": "attribute_learner",
    "attribute_learner": "attribute_learner",
    "endowmentgenerator": "endowment_generator",
    "endowment_generator": "endowment_generator",
    "endowmentmodel": "endowment_model",
    "endowment_model": "endowment_model",
    "surveyconductor": "survey_conductor",
    "survey_conductor": "survey_conductor",
}
_FEATURE_NAMES = (
    "target_n",
    "question_count",
    "survey_chars",
    "option_count",
    "initial_n",
    "update_steps",
    "batch_size",
    "patch_fraction",
    "patch_top_k",
    "max_tokens",
    "is_active",
    "is_vanilla",
)
_TARGET_NAMES = (
    "calls",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
)


@dataclass(frozen=True, slots=True)
class BudgetFeatures:
    """Numerical run features used by the forecasting model."""

    target_n: int
    question_count: int
    survey_chars: int
    option_count: int
    initial_n: int
    update_steps: int
    batch_size: int
    patch_fraction: float
    patch_top_k: int
    max_tokens: int
    is_active: int
    is_vanilla: int

    def vector(self) -> list[float]:
        """Return features in the stable model order."""

        return [float(getattr(self, name)) for name in _FEATURE_NAMES]


@dataclass(frozen=True, slots=True)
class PricingEstimate:
    """One resolved per-million-token price with provenance."""

    model_name: str
    input_per_million_usd: float | None
    output_per_million_usd: float | None
    cached_input_per_million_usd: float | None
    source: str

    @property
    def available(self) -> bool:
        return (
            self.input_per_million_usd is not None
            and self.output_per_million_usd is not None
            and self.cached_input_per_million_usd is not None
        )


@dataclass(frozen=True, slots=True)
class IntervalEstimate:
    """Point and central uncertainty interval."""

    low: float
    point: float
    high: float


@dataclass(frozen=True, slots=True)
class ModuleBudget:
    """Forecast for one API-using pipeline module."""

    module_name: str
    model_name: str
    backend_type: str
    training_scope: str
    training_runs: int
    coverage: str
    calls: IntervalEstimate
    input_tokens: IntervalEstimate
    output_tokens: IntervalEstimate
    cached_input_tokens: IntervalEstimate
    cost_usd: IntervalEstimate | None
    pricing: PricingEstimate
    pilot_calibration: dict[str, float] | None
    oob_median_absolute_percentage_error: dict[str, float] | None


@dataclass(frozen=True, slots=True)
class ScenarioBudget:
    """Forecast for one configured or comparison model scenario."""

    name: str
    modules: tuple[ModuleBudget, ...]
    total_tokens: IntervalEstimate
    total_cost_usd: IntervalEstimate | None
    budget_usd: float | None
    budget_status: str


@dataclass(frozen=True, slots=True)
class BudgetForecast:
    """Complete reproducible forecast and model-card payload."""

    schema_version: int
    artifact_type: str
    config_source: str
    config_digest: str
    target_command: str
    confidence: float
    history_roots: tuple[str, ...]
    history_digest: str
    historical_runs: int
    historical_records: int
    features: BudgetFeatures
    scenarios: tuple[ScenarioBudget, ...]
    history_rows: tuple[dict[str, Any], ...]
    pilot_run: str | None
    pilot_records: int
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible payload."""

        payload = asdict(self)
        payload.pop("history_rows")
        return payload


@dataclass(frozen=True, slots=True)
class BudgetArtifacts:
    """Files written for one forecast."""

    json_path: Path
    csv_path: Path
    history_csv_path: Path


def _normalize_module(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    compact = value.strip().replace("-", "_")
    return _MODULE_ALIASES.get(compact.lower())


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return default
    return numeric if numeric > 0 else default


def _finite_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _resolve_artifact_path(
    run_dir: Path,
    value: Any,
    *,
    prefer_run_copy: bool = False,
) -> Path | None:
    if not isinstance(value, (str, os.PathLike)):
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve() if path.is_file() else None
    candidates: list[Path] = []
    if prefer_run_copy:
        candidates.append(run_dir / path.name)
    candidates.append(run_dir / path)
    candidates.extend(parent / path for parent in run_dir.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _read_yaml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(loaded, Mapping):
        return None
    config = loaded.get("config", loaded)
    return dict(config) if isinstance(config, Mapping) else None


def _run_config(run_dir: Path) -> dict[str, Any]:
    for filename in (
        "config.resolved.yaml",
        "original_config.yaml",
        "metadata.yaml",
        "regression_config.yaml",
    ):
        loaded = _read_yaml(run_dir / filename)
        if loaded is not None:
            return loaded
    return {}


def _survey_paths(
    config: Mapping[str, Any],
    run_dir: Path,
) -> tuple[Path | None, Path | None]:
    data_survey = _mapping(_mapping(config.get("data")).get("survey"))
    legacy_paths = _mapping(config.get("paths"))
    survey_value = data_survey.get(
        "source",
        legacy_paths.get("survey_csv"),
    )
    schema_value = data_survey.get(
        "schema",
        legacy_paths.get("survey_yaml"),
    )
    return (
        _resolve_artifact_path(run_dir, survey_value),
        _resolve_artifact_path(run_dir, schema_value),
    )


def _survey_dimensions(
    config: Mapping[str, Any],
    run_dir: Path,
) -> tuple[int, int, int]:
    survey_path, schema_path = _survey_paths(config, run_dir)
    responses_path = run_dir / "responses.csv"
    return _survey_dimensions_from_paths(
        survey_path,
        schema_path,
        responses_path.resolve() if responses_path.is_file() else None,
    )


@lru_cache(maxsize=256)
def _survey_dimensions_from_paths(
    survey_path: Path | None,
    schema_path: Path | None,
    responses_path: Path | None,
) -> tuple[int, int, int]:
    if survey_path is not None and schema_path is not None:
        try:
            survey = Survey(survey_path, schema_path)
            questions = list(survey.questions)
            return (
                len(questions),
                sum(
                    len(survey.get_prompt_text(question))
                    for question in questions
                ),
                sum(
                    len(question.get("code_to_answer", {}))
                    for question in questions
                ),
            )
        except Exception:
            pass

    if responses_path is not None and responses_path.is_file():
        try:
            frame = pd.read_csv(responses_path, usecols=["qid"])
            return len(frame), 0, 0
        except Exception:
            pass
    return 0, 0, 0


def _generation_mode(config: Mapping[str, Any]) -> str:
    generation = _mapping(config.get("generation"))
    configured = generation.get("mode")
    if isinstance(configured, str) and configured:
        return configured.lower()
    if isinstance(config.get("endowment_generator"), Mapping):
        return "vanilla"
    if (
        isinstance(config.get("endowment_model"), Mapping)
        or "initial_n" in generation
    ):
        return "active"
    name = str(_mapping(config.get("metadata")).get("name", "")).lower()
    if name.startswith("vanilla"):
        return "vanilla"
    return "existing"


def _target_n(
    config: Mapping[str, Any],
    run_dir: Path,
    usage: pd.DataFrame | None = None,
) -> int:
    generation = _mapping(config.get("generation"))
    generator = _mapping(config.get("endowment_generator"))
    runner = _mapping(config.get("runner"))
    for value in (
        generation.get("target_n"),
        generator.get("n_endowments"),
        runner.get("target_n"),
    ):
        resolved = _positive_int(value, 0)
        if resolved:
            return resolved
    for filename in ("endowments.csv", "updated_endowments.csv"):
        path = run_dir / filename
        if path.is_file():
            try:
                return len(pd.read_csv(path, usecols=["eid"]))
            except Exception:
                continue
    if usage is not None and not usage.empty:
        response = usage[
            usage["normalized_module"] == "survey_conductor"
        ]
        if not response.empty:
            return len(response)
    return 1


def _module_settings(
    config: Mapping[str, Any],
    module_name: str,
) -> tuple[int, int]:
    generation = _mapping(config.get("generation"))
    backend = _mapping(config.get("backend"))
    component_name = {
        "survey_conductor": "response",
        "endowment_model": "endowment",
        "endowment_generator": "endowment",
        "attribute_learner": "attribute_learner",
    }[module_name]
    component = _mapping(backend.get(component_name))
    kwargs = _mapping(component.get("kwargs"))
    legacy = {
        "survey_conductor": _mapping(config.get("agent")),
        "endowment_model": _mapping(config.get("endowment_model")),
        "endowment_generator": _mapping(
            config.get("endowment_generator")
        ),
        "attribute_learner": _mapping(
            config.get("attribute_learner")
        ),
    }[module_name]
    legacy_kwargs = _mapping(legacy.get("kwargs"))

    if module_name == "endowment_model":
        runtime = _mapping(generation.get("endowment_model"))
    elif module_name == "attribute_learner":
        runtime = _mapping(generation.get("attribute_learner"))
    else:
        runtime = generation
    batch_default = 10 if "endowment" in module_name else 1
    batch_size = _positive_int(
        runtime.get(
            "batch_size",
            legacy.get("batch_size", batch_default),
        ),
        batch_default,
    )
    max_tokens = _positive_int(
        kwargs.get(
            "max_tokens",
            runtime.get(
                "max_tokens",
                legacy_kwargs.get(
                    "max_tokens",
                    legacy.get("max_tokens", 1024),
                ),
            ),
        ),
        1024,
    )
    return batch_size, max_tokens


def _features(
    config: Mapping[str, Any],
    run_dir: Path,
    module_name: str,
    *,
    usage: pd.DataFrame | None = None,
) -> BudgetFeatures:
    generation = _mapping(config.get("generation"))
    patch = _mapping(generation.get("question_patch"))
    mode = _generation_mode(config)
    question_count, survey_chars, option_count = _survey_dimensions(
        config,
        run_dir,
    )
    batch_size, max_tokens = _module_settings(config, module_name)
    return BudgetFeatures(
        target_n=_target_n(config, run_dir, usage),
        question_count=question_count,
        survey_chars=survey_chars,
        option_count=option_count,
        initial_n=_positive_int(generation.get("initial_n"), 1),
        update_steps=_positive_int(
            generation.get("num_update_steps"),
            1,
        ),
        batch_size=batch_size,
        patch_fraction=_finite_float(patch.get("fraction"), 0.0),
        patch_top_k=_positive_int(patch.get("top_k"), 0),
        max_tokens=max_tokens,
        is_active=int(mode == "active"),
        is_vanilla=int(mode == "vanilla"),
    )


def _usage_file(run_dir: Path) -> Path | None:
    for filename in ("usage.csv", "token_usage_summary.csv"):
        path = run_dir / filename
        if path.is_file():
            return path
    return None


def _read_usage(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "model_name",
        "module_name",
        "input_tokens",
        "output_tokens",
    }
    if not required <= set(frame.columns):
        raise ValueError(f"{path} does not have normalized usage columns")
    frame = frame.copy()
    frame["normalized_module"] = frame["module_name"].map(
        _normalize_module
    )
    frame = frame.dropna(subset=["normalized_module"])
    for column in (
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "input_cost",
        "output_cost",
        "cached_input_cost",
        "total_cost",
        "cost_usd",
    ):
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).fillna(0.0)
    return frame


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_budget_history(
    roots: Sequence[str | os.PathLike[str]],
) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    """Collect one model/module row per historical tracked run."""

    resolved_roots = tuple(
        str(Path(root).resolve())
        for root in roots
    )
    usage_paths: list[Path] = []
    seen_runs: set[Path] = set()
    for root_value in resolved_roots:
        root = Path(root_value)
        if not root.exists():
            continue
        candidates = (
            [root]
            if root.is_file()
            else sorted(root.rglob("usage.csv"))
            + sorted(root.rglob("token_usage_summary.csv"))
        )
        for candidate in candidates:
            run_dir = candidate.parent.resolve()
            if run_dir in seen_runs:
                continue
            preferred = _usage_file(run_dir)
            if preferred is not None:
                usage_paths.append(preferred.resolve())
                seen_runs.add(run_dir)

    rows: list[dict[str, Any]] = []
    digests: list[str] = []
    for path in sorted(usage_paths):
        try:
            usage = _read_usage(path)
        except Exception:
            continue
        if usage.empty:
            continue
        run_dir = path.parent
        config = _run_config(run_dir)
        digests.append(f"{path}:{_file_digest(path)}")
        for (module_name, model_name), group in usage.groupby(
            ["normalized_module", "model_name"],
            sort=True,
        ):
            feature = _features(
                config,
                run_dir,
                str(module_name),
                usage=usage,
            )
            record_count = int(len(group))
            calls = record_count
            if (
                path.name == "token_usage_summary.csv"
                and module_name == "survey_conductor"
            ):
                calls *= max(feature.question_count, 1)
            row: dict[str, Any] = {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "usage_path": str(path),
                "module_name": str(module_name),
                "model_name": str(model_name),
                "record_count": record_count,
                "calls": calls,
                "input_tokens": float(group["input_tokens"].sum()),
                "output_tokens": float(group["output_tokens"].sum()),
                "cached_input_tokens": float(
                    group["cached_input_tokens"].sum()
                ),
                "input_cost": float(group["input_cost"].sum()),
                "output_cost": float(group["output_cost"].sum()),
                "cached_input_cost": float(
                    group["cached_input_cost"].sum()
                ),
                "total_cost": float(
                    (
                        group["total_cost"]
                        if group["total_cost"].abs().sum() > 0
                        else group["cost_usd"]
                    ).sum()
                ),
            }
            row.update(asdict(feature))
            rows.append(row)

    history_digest = hashlib.sha256(
        "\n".join(sorted(digests)).encode("utf-8")
    ).hexdigest()
    return pd.DataFrame(rows), history_digest, resolved_roots


def _historical_pricing(
    history: pd.DataFrame,
) -> dict[str, PricingEstimate]:
    pricing: dict[str, PricingEstimate] = {}
    if history.empty:
        return pricing
    for model_name, group in history.groupby("model_name", sort=True):
        input_tokens = float(group["input_tokens"].sum())
        output_tokens = float(group["output_tokens"].sum())
        cached_tokens = float(group["cached_input_tokens"].sum())
        input_cost = float(group["input_cost"].sum())
        output_cost = float(group["output_cost"].sum())
        cached_cost = float(group["cached_input_cost"].sum())
        if input_cost <= 0 and output_cost <= 0 and cached_cost <= 0:
            continue
        pricing[str(model_name)] = PricingEstimate(
            model_name=str(model_name),
            input_per_million_usd=(
                input_cost / max(input_tokens - cached_tokens, 1.0)
                * 1_000_000
            ),
            output_per_million_usd=(
                output_cost / max(output_tokens, 1.0) * 1_000_000
            ),
            cached_input_per_million_usd=(
                cached_cost / cached_tokens * 1_000_000
                if cached_tokens > 0
                else 0.0
            ),
            source="historical tracked effective rate",
        )
    return pricing


def load_pricing(
    path: str | os.PathLike[str] | None,
) -> dict[str, PricingEstimate]:
    """Load explicit current prices from a small provider-neutral YAML."""

    if path is None:
        return {}
    source = Path(path).resolve()
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"could not load pricing file {source}: {error}") from error
    root = _mapping(loaded)
    models = _mapping(root.get("models", root))
    resolved: dict[str, PricingEstimate] = {}
    for model_name, raw_pricing in models.items():
        values = _mapping(raw_pricing)
        pricing = TokenPricing(
            input_per_million_usd=_finite_float(
                values.get("input_per_million_usd"),
                0.0,
            ),
            output_per_million_usd=_finite_float(
                values.get("output_per_million_usd"),
                0.0,
            ),
            cached_input_per_million_usd=_finite_float(
                values.get("cached_input_per_million_usd"),
                0.0,
            ),
        )
        resolved[str(model_name)] = PricingEstimate(
            model_name=str(model_name),
            input_per_million_usd=pricing.input_per_million_usd,
            output_per_million_usd=pricing.output_per_million_usd,
            cached_input_per_million_usd=(
                pricing.cached_input_per_million_usd
            ),
            source=f"explicit pricing file: {source}",
        )
    return resolved


def _planned_modules(
    config: Config,
    target_command: str,
    *,
    model_override: str | None = None,
) -> tuple[tuple[str, str, str], ...]:
    generation = _mapping(config.data.get("generation"))
    mode = str(generation.get("mode", "active")).lower()
    backend = load_backend_config(config)

    def component(name: str) -> tuple[str, str]:
        values = _mapping(backend.get(name))
        backend_type = str(values.get("type", "unknown"))
        model_name = str(values.get("model_name", "unknown"))
        return (
            "comparison" if model_override is not None else backend_type,
            model_override or model_name,
        )

    planned: list[tuple[str, str, str]] = []
    if mode == "vanilla":
        backend_type, model_name = component("endowment")
        planned.append(("endowment_generator", model_name, backend_type))
        if target_command == "pipeline":
            backend_type, model_name = component("response")
            planned.append(("survey_conductor", model_name, backend_type))
    elif mode == "active":
        backend_type, model_name = component("attribute_learner")
        planned.append(("attribute_learner", model_name, backend_type))
        backend_type, model_name = component("endowment")
        planned.append(("endowment_model", model_name, backend_type))
        backend_type, model_name = component("response")
        planned.append(("survey_conductor", model_name, backend_type))
    else:
        raise ValueError(
            "budget forecasting currently requires generation.mode "
            "'active' or 'vanilla'"
        )
    return tuple(planned)


def _coverage_label(
    training_runs: int,
    exact: bool,
    features: BudgetFeatures,
    training: pd.DataFrame,
) -> str:
    if training_runs < 8:
        quality = "low"
    elif training_runs < 25:
        quality = "moderate"
    else:
        quality = "strong"
    extrapolated = False
    for name in ("target_n", "question_count", "survey_chars"):
        values = training[name]
        target = getattr(features, name)
        if len(values) and (
            target < float(values.min())
            or target > float(values.max())
        ):
            extrapolated = True
    scope = "exact model" if exact else "pooled model fallback"
    suffix = "; extrapolation" if extrapolated else ""
    return f"{quality}; {scope}{suffix}"


def _fit_module(
    history: pd.DataFrame,
    *,
    module_name: str,
    model_name: str,
    features: BudgetFeatures,
    confidence: float,
    seed: int,
) -> tuple[dict[str, IntervalEstimate], str, int, str, dict[str, float] | None]:
    module_history = history[history["module_name"] == module_name]
    exact_history = module_history[
        module_history["model_name"] == model_name
    ]
    exact = len(exact_history) >= 3
    training = exact_history if exact else module_history
    if len(training) < 3:
        raise ValueError(
            f"not enough historical runs for module {module_name!r}: the "
            f"forecast needs at least 3 earlier runs that used it, found "
            f"{len(training)}. Run more small pilots first."
        )

    x = training[list(_FEATURE_NAMES)].to_numpy(dtype=float)
    if module_name == "attribute_learner":
        exposure = np.ones(len(training), dtype=float)
    elif module_name == "survey_conductor":
        exposure = np.maximum(
            training["target_n"].to_numpy(dtype=float)
            * np.maximum(
                training["question_count"].to_numpy(dtype=float),
                1.0,
            ),
            1.0,
        )
    else:
        exposure = np.maximum(
            training["target_n"].to_numpy(dtype=float),
            1.0,
        )
    raw_targets = training[list(_TARGET_NAMES)].to_numpy(dtype=float)
    normalized_targets = raw_targets / exposure[:, None]
    y = np.log1p(normalized_targets)
    use_oob = len(training) >= 10
    forest = RandomForestRegressor(
        n_estimators=256,
        min_samples_leaf=(
            1
            if len(training) < 10
            else max(2, min(5, len(training) // 20))
        ),
        max_features=0.85,
        bootstrap=True,
        oob_score=use_oob,
        random_state=seed,
        n_jobs=1,
    )
    forest.fit(x, y)
    target = np.asarray([features.vector()], dtype=float)
    tree_log_predictions = np.asarray(
        [tree.predict(target)[0] for tree in forest.estimators_]
    )
    alpha = (1.0 - confidence) / 2.0
    residual_low = np.zeros(len(_TARGET_NAMES))
    residual_high = np.zeros(len(_TARGET_NAMES))
    oob_mape = None
    if use_oob:
        valid = np.isfinite(forest.oob_prediction_).all(axis=1)
        residuals = y[valid] - forest.oob_prediction_[valid]
        if len(residuals):
            residual_low = np.quantile(residuals, alpha, axis=0)
            residual_high = np.quantile(
                residuals,
                1.0 - alpha,
                axis=0,
            )
            actual = np.expm1(y[valid])
            predicted = np.expm1(forest.oob_prediction_[valid])
            percentage = np.abs(predicted - actual) / np.maximum(
                actual,
                1.0,
            )
            oob_mape = {
                name: float(np.median(percentage[:, index]))
                for index, name in enumerate(_TARGET_NAMES)
            }

    intervals: dict[str, IntervalEstimate] = {}
    if module_name == "attribute_learner":
        target_exposure = 1.0
    elif module_name == "survey_conductor":
        target_exposure = float(
            features.target_n * max(features.question_count, 1)
        )
    else:
        target_exposure = float(features.target_n)
    for index, name in enumerate(_TARGET_NAMES):
        lower_log = (
            np.quantile(tree_log_predictions[:, index], alpha)
            + residual_low[index]
        )
        point_log = np.median(tree_log_predictions[:, index])
        upper_log = (
            np.quantile(
                tree_log_predictions[:, index],
                1.0 - alpha,
            )
            + residual_high[index]
        )
        low, point, high = np.maximum(
            np.expm1([lower_log, point_log, upper_log]),
            0.0,
        ) * target_exposure
        low, high = min(low, point, high), max(low, point, high)
        if name == "calls":
            if module_name == "survey_conductor":
                minimum_calls = float(
                    features.target_n
                    * max(features.question_count, 1)
                )
            elif module_name in {
                "endowment_model",
                "endowment_generator",
            }:
                minimum_calls = float(
                    math.ceil(features.target_n / features.batch_size)
                )
            else:
                minimum_calls = 1.0
            low = max(low, minimum_calls)
            point = max(point, minimum_calls)
            high = max(high, point)
        intervals[name] = IntervalEstimate(
            low=float(low),
            point=float(point),
            high=float(high),
        )

    return (
        intervals,
        "exact model history" if exact else "pooled cross-model history",
        int(len(training)),
        _coverage_label(
            len(training),
            exact,
            features,
            training,
        ),
        oob_mape,
    )


def _pilot_calibration(
    history: pd.DataFrame,
    pilot_run: Path | None,
    module_name: str,
    model_name: str,
    target_features: BudgetFeatures,
) -> tuple[dict[str, float] | None, int]:
    if pilot_run is None:
        return None, 0
    usage_path = _usage_file(pilot_run)
    if usage_path is None:
        return None, 0
    try:
        usage = _read_usage(usage_path)
    except Exception:
        return None, 0
    group = usage[
        (usage["normalized_module"] == module_name)
        & (usage["model_name"] == model_name)
    ]
    if group.empty:
        return None, len(usage)

    comparable = history[
        (history["module_name"] == module_name)
        & (history["model_name"] == model_name)
    ]
    if len(comparable) < 3:
        comparable = history[history["module_name"] == module_name]
    if comparable.empty:
        return None, len(usage)
    historical_input_per_call = np.median(
        comparable["input_tokens"]
        / np.maximum(comparable["calls"], 1)
    )
    historical_output_per_call = np.median(
        comparable["output_tokens"]
        / np.maximum(comparable["calls"], 1)
    )
    actual_input_per_call = float(group["input_tokens"].sum()) / len(group)
    actual_output_per_call = float(group["output_tokens"].sum()) / len(group)

    def ratio(actual: float, historical: float) -> float:
        if historical <= 0:
            return 1.0
        return float(np.clip(actual / historical, 0.05, 10.0))

    pilot_config = _run_config(pilot_run)
    pilot_features = _features(
        pilot_config,
        pilot_run,
        module_name,
        usage=usage,
    )
    if module_name == "survey_conductor":
        pilot_exposure = (
            pilot_features.target_n
            * max(pilot_features.question_count, 1)
        )
        target_exposure = (
            target_features.target_n
            * max(target_features.question_count, 1)
        )
    elif module_name == "attribute_learner":
        pilot_exposure = max(pilot_features.question_count, 1)
        target_exposure = max(target_features.question_count, 1)
    else:
        pilot_exposure = pilot_features.target_n
        target_exposure = target_features.target_n
    projected_calls = (
        len(group)
        / max(float(pilot_exposure), 1.0)
        * float(target_exposure)
    )
    return (
        {
            "observed_calls": float(len(group)),
            "projected_calls": projected_calls,
            "input_per_call_ratio": ratio(
                actual_input_per_call,
                historical_input_per_call,
            ),
            "output_per_call_ratio": ratio(
                actual_output_per_call,
                historical_output_per_call,
            ),
        },
        len(usage),
    )


def _scale_interval(
    value: IntervalEstimate,
    factor: float,
) -> IntervalEstimate:
    return IntervalEstimate(
        low=value.low * factor,
        point=value.point * factor,
        high=value.high * factor,
    )


def _cost_interval(
    input_tokens: IntervalEstimate,
    output_tokens: IntervalEstimate,
    cached_tokens: IntervalEstimate,
    pricing: PricingEstimate,
) -> IntervalEstimate | None:
    if not pricing.available:
        return None

    def cost(position: str) -> float:
        input_value = getattr(input_tokens, position)
        output_value = getattr(output_tokens, position)
        cached_value = min(
            getattr(cached_tokens, position),
            input_value,
        )
        return (
            (input_value - cached_value)
            * float(pricing.input_per_million_usd)
            + output_value * float(pricing.output_per_million_usd)
            + cached_value
            * float(pricing.cached_input_per_million_usd)
        ) / 1_000_000

    return IntervalEstimate(
        low=cost("low"),
        point=cost("point"),
        high=cost("high"),
    )


def _sum_intervals(
    values: Sequence[IntervalEstimate],
) -> IntervalEstimate:
    return IntervalEstimate(
        low=sum(value.low for value in values),
        point=sum(value.point for value in values),
        high=sum(value.high for value in values),
    )


def _resolve_pricing(
    model_name: str,
    backend_type: str,
    explicit: Mapping[str, PricingEstimate],
    historical: Mapping[str, PricingEstimate],
) -> PricingEstimate:
    if model_name in explicit:
        return explicit[model_name]
    if backend_type.lower() in {"ollama", "dummy", "simulated", "local"}:
        return PricingEstimate(
            model_name=model_name,
            input_per_million_usd=0.0,
            output_per_million_usd=0.0,
            cached_input_per_million_usd=0.0,
            source=f"local backend ({backend_type})",
        )
    if model_name in historical:
        return historical[model_name]
    return PricingEstimate(
        model_name=model_name,
        input_per_million_usd=None,
        output_per_million_usd=None,
        cached_input_per_million_usd=None,
        source="unpriced; provide --pricing",
    )


def forecast_budget(
    config: Config,
    *,
    history_roots: Sequence[str | os.PathLike[str]],
    target_command: str = "pipeline",
    comparison_models: Sequence[str] = (),
    pricing_path: str | os.PathLike[str] | None = None,
    pilot_run: str | os.PathLike[str] | None = None,
    confidence: float = 0.8,
    budget_usd: float | None = None,
) -> BudgetForecast:
    """Train historical models and forecast one configured API run."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")
    if target_command not in {"generate", "pipeline"}:
        raise ValueError("target_command must be 'generate' or 'pipeline'")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not math.isfinite(float(confidence))
        or not 0.5 <= float(confidence) < 1.0
    ):
        raise ValueError("confidence must be in [0.5, 1)")
    if budget_usd is not None and (
        isinstance(budget_usd, bool)
        or not isinstance(budget_usd, (int, float))
        or not math.isfinite(float(budget_usd))
        or budget_usd < 0
    ):
        raise ValueError("budget_usd must be finite and nonnegative")

    history, history_digest, roots = collect_budget_history(
        history_roots
    )
    if history.empty:
        searched = ", ".join(str(root) for root in history_roots)
        raise ValueError(
            "no usable tracked usage history was found under "
            f"{searched}. The planner learns from earlier runs' usage.csv; run a "
            "small pilot first (or pass --history with folders of earlier runs), "
            "then add --pilot-run to calibrate the forecast."
        )
    explicit_pricing = load_pricing(pricing_path)
    historical_pricing = _historical_pricing(history)
    pilot_path = (
        Path(pilot_run).resolve()
        if pilot_run is not None
        else None
    )
    if pilot_path is not None and not pilot_path.is_dir():
        raise ValueError(f"pilot run directory does not exist: {pilot_path}")

    config_run_dir = Path(config.source_path).parent.resolve()
    scenarios: list[ScenarioBudget] = []
    notes: list[str] = [
        "Forecast intervals combine random-forest tree dispersion with "
        "out-of-bag residuals when at least ten training runs are available.",
        "Historical prices are observed effective rates, not a promise of "
        "current provider pricing; use --pricing for an explicit current rate.",
    ]
    pilot_records = 0
    scenario_specs = [("configured", None)]
    scenario_specs.extend(
        (f"compare:{model}", str(model))
        for model in dict.fromkeys(comparison_models)
    )

    base_modules = _planned_modules(config, target_command)
    base_feature = _features(
        config.data,
        config_run_dir,
        base_modules[0][0],
    )
    for scenario_name, override in scenario_specs:
        module_budgets: list[ModuleBudget] = []
        planned = _planned_modules(
            config,
            target_command,
            model_override=override,
        )
        for index, (module_name, model_name, backend_type) in enumerate(
            planned
        ):
            feature = _features(
                config.data,
                config_run_dir,
                module_name,
            )
            seed_material = (
                f"{config.digest}:{scenario_name}:{module_name}"
            )
            seed = int(
                hashlib.sha256(seed_material.encode()).hexdigest()[:8],
                16,
            )
            (
                intervals,
                training_scope,
                training_runs,
                coverage,
                oob_mape,
            ) = _fit_module(
                history,
                module_name=module_name,
                model_name=model_name,
                features=feature,
                confidence=float(confidence),
                seed=seed,
            )
            calibration, observed_records = _pilot_calibration(
                history,
                pilot_path,
                module_name,
                model_name,
                feature,
            )
            pilot_records = max(pilot_records, observed_records)
            if calibration is not None:
                projected_calls = calibration["projected_calls"]
                intervals["calls"] = IntervalEstimate(
                    low=min(intervals["calls"].low, projected_calls),
                    point=projected_calls,
                    high=max(intervals["calls"].high, projected_calls),
                )
                intervals["input_tokens"] = _scale_interval(
                    intervals["input_tokens"],
                    calibration["input_per_call_ratio"],
                )
                intervals["output_tokens"] = _scale_interval(
                    intervals["output_tokens"],
                    calibration["output_per_call_ratio"],
                )
            pricing = _resolve_pricing(
                model_name,
                backend_type,
                explicit_pricing,
                historical_pricing,
            )
            cost = _cost_interval(
                intervals["input_tokens"],
                intervals["output_tokens"],
                intervals["cached_input_tokens"],
                pricing,
            )
            module_budgets.append(
                ModuleBudget(
                    module_name=module_name,
                    model_name=model_name,
                    backend_type=backend_type,
                    training_scope=training_scope,
                    training_runs=training_runs,
                    coverage=coverage,
                    calls=intervals["calls"],
                    input_tokens=intervals["input_tokens"],
                    output_tokens=intervals["output_tokens"],
                    cached_input_tokens=intervals[
                        "cached_input_tokens"
                    ],
                    cost_usd=cost,
                    pricing=pricing,
                    pilot_calibration=calibration,
                    oob_median_absolute_percentage_error=oob_mape,
                )
            )

        total_tokens = _sum_intervals(
            [
                _sum_intervals(
                    [module.input_tokens, module.output_tokens]
                )
                for module in module_budgets
            ]
        )
        costs = [
            module.cost_usd
            for module in module_budgets
            if module.cost_usd is not None
        ]
        total_cost = (
            _sum_intervals(costs)
            if len(costs) == len(module_budgets)
            else None
        )
        if budget_usd is None:
            status = "no budget ceiling supplied"
        elif total_cost is None:
            status = "cannot assess: one or more models are unpriced"
        elif total_cost.high <= budget_usd:
            status = "within budget at interval high"
        elif total_cost.point <= budget_usd:
            status = "point estimate within budget; interval crosses ceiling"
        else:
            status = "point estimate exceeds budget"
        scenarios.append(
            ScenarioBudget(
                name=scenario_name,
                modules=tuple(module_budgets),
                total_tokens=total_tokens,
                total_cost_usd=total_cost,
                budget_usd=(
                    float(budget_usd)
                    if budget_usd is not None
                    else None
                ),
                budget_status=status,
            )
        )

    if any(
        "pooled model fallback" in module.coverage
        for scenario in scenarios
        for module in scenario.modules
    ):
        notes.append(
            "At least one configured model lacks three matching historical "
            "runs; its token forecast uses pooled cross-model history."
        )
    if pilot_path is not None and pilot_records == 0:
        notes.append(
            "The pilot directory has no usable usage records yet; live mode "
            "will recalibrate as usage.csv appears."
        )

    history_rows = tuple(
        {
            key: (
                value.item()
                if isinstance(value, np.generic)
                else value
            )
            for key, value in row.items()
        }
        for row in history.to_dict(orient="records")
    )
    return BudgetForecast(
        schema_version=1,
        artifact_type="budget_forecast",
        config_source=str(config.source_path),
        config_digest=config.digest,
        target_command=target_command,
        confidence=float(confidence),
        history_roots=roots,
        history_digest=history_digest,
        historical_runs=int(history["run_dir"].nunique()),
        historical_records=int(history["record_count"].sum()),
        features=base_feature,
        scenarios=tuple(scenarios),
        history_rows=history_rows,
        pilot_run=str(pilot_path) if pilot_path is not None else None,
        pilot_records=pilot_records,
        notes=tuple(notes),
    )


def write_budget_artifacts(
    forecast: BudgetForecast,
    output_dir: str | os.PathLike[str],
) -> BudgetArtifacts:
    """Write canonical JSON/CSV forecast and training provenance."""

    if not isinstance(forecast, BudgetForecast):
        raise TypeError("forecast must be BudgetForecast")
    path = Path(output_dir).resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(f"output_dir is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)

    json_path = path / "budget.json"
    json_path.write_text(
        json.dumps(
            forecast.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = path / "budget.csv"
    fieldnames = [
        "scenario",
        "module_name",
        "model_name",
        "backend_type",
        "training_scope",
        "training_runs",
        "coverage",
        "calls_low",
        "calls_point",
        "calls_high",
        "input_tokens_low",
        "input_tokens_point",
        "input_tokens_high",
        "output_tokens_low",
        "output_tokens_point",
        "output_tokens_high",
        "cost_usd_low",
        "cost_usd_point",
        "cost_usd_high",
        "pricing_source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for scenario in forecast.scenarios:
            for module in scenario.modules:
                cost = module.cost_usd
                writer.writerow(
                    {
                        "scenario": scenario.name,
                        "module_name": module.module_name,
                        "model_name": module.model_name,
                        "backend_type": module.backend_type,
                        "training_scope": module.training_scope,
                        "training_runs": module.training_runs,
                        "coverage": module.coverage,
                        "calls_low": module.calls.low,
                        "calls_point": module.calls.point,
                        "calls_high": module.calls.high,
                        "input_tokens_low": module.input_tokens.low,
                        "input_tokens_point": module.input_tokens.point,
                        "input_tokens_high": module.input_tokens.high,
                        "output_tokens_low": module.output_tokens.low,
                        "output_tokens_point": module.output_tokens.point,
                        "output_tokens_high": module.output_tokens.high,
                        "cost_usd_low": cost.low if cost else "",
                        "cost_usd_point": cost.point if cost else "",
                        "cost_usd_high": cost.high if cost else "",
                        "pricing_source": module.pricing.source,
                    }
                )

    history_csv_path = path / "budget_history.csv"
    history = pd.DataFrame(forecast.history_rows)
    history.to_csv(history_csv_path, index=False)
    return BudgetArtifacts(
        json_path=json_path.resolve(),
        csv_path=csv_path.resolve(),
        history_csv_path=history_csv_path.resolve(),
    )


__all__ = [
    "BudgetArtifacts",
    "BudgetFeatures",
    "BudgetForecast",
    "IntervalEstimate",
    "ModuleBudget",
    "PricingEstimate",
    "ScenarioBudget",
    "collect_budget_history",
    "forecast_budget",
    "load_pricing",
    "write_budget_artifacts",
]
