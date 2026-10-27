"""Faithful offline scientific simulations over cached P2P bases."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import csv
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import random
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from p2p.models.constrained_model import DEFAULT_SOLVER
from p2p.config import Config
from p2p.endowments import ActiveEndowments
from p2p.sampling import EntropySample, sample_to_target_mean_entropy
from p2p.stages.fit import FitResult, fit_model
from p2p.stages.split import (
    REFERENCE_FEATURE,
    RegressionData,
    RegressionSplit,
    load_regression_data,
)
from p2p.stages.targets import target_spec


_KINDS = {
    "endowment",
    "response",
    "proxy",
    "trainvalid",
    "entropy_modes",
    "target_entropy",
}


@dataclass(frozen=True, slots=True)
class ScientificSimulationResult:
    """One fitted scientific simulation child and its artifact."""

    config_digest: str
    run_dir: Path
    artifact_path: Path
    csv_path: Path
    data: RegressionData
    fit: FitResult
    stats: Mapping[str, Any]
    entropy_sample: EntropySample | None = None


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _path(value: Any, *, field: str) -> Path:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty path")
    path = Path(value).resolve()
    if not path.exists():
        raise ValueError(f"{field} does not exist: {path}")
    return path


def _fraction(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("simulation.fraction must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError("simulation.fraction must be in (0, 1]")
    return result


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _load_base(config: Config) -> RegressionData:
    data = _mapping(config.data.get("data"), field="data")
    survey = _mapping(data.get("survey"), field="data.survey")
    basis = data.get("basis")
    if isinstance(basis, Mapping):
        responses_path = _path(
            basis.get("responses"),
            field="data.basis.responses",
        )
        endowments_path = _path(
            basis.get("endowments"),
            field="data.basis.endowments",
        )
    else:
        basis_path = _path(basis, field="data.basis")
        responses_path = basis_path / "responses.csv"
        endowments_path = basis_path / "endowments.csv"

    split = _mapping(config.data.get("split"), field="split")
    train_splits = split.get("train_val", ("train", "valid"))
    if (
        not isinstance(train_splits, Sequence)
        or isinstance(train_splits, (str, bytes, bytearray))
    ):
        raise ValueError("split.train_val must be a sequence")
    target_value = data.get("target")
    target_source = config.data.get("simulation", {}).get(
        "target_source",
        "configured",
    )
    if target_value is None:
        if target_source != "ground_truth":
            raise ValueError(
                "data.target is required when simulation.target_source "
                "is configured"
            )
        target_path = None
        target_subgroup = None
    else:
        target_path, target_subgroup = target_spec(target_value)
    return load_regression_data(
        survey_source=_path(
            survey.get("source"),
            field="data.survey.source",
        ),
        survey_schema=_path(
            survey.get("schema"),
            field="data.survey.schema",
        ),
        responses_path=responses_path,
        endowments_path=endowments_path,
        target_path=target_path,
        target_subgroup=target_subgroup,
        train_splits=tuple(train_splits),
        test_split=split.get("test", "test"),
        use_proxy_only=split.get("use_proxy_only", True),
        allow_missing_target=(target_path is None),
    )


def _mode_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return "+".join(str(part) for part in value)
    return str(value)


def _mode_entropy(data: RegressionData) -> list[tuple[str, float, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for endowment in data.endowments.get_endowments():
        grouped[_mode_name(endowment.get("mode", ()))].append(endowment)

    rows = [
        (
            mode,
            float(
                ActiveEndowments.compute_entropy(
                    endowments,
                    data.answer_responses,
                    data.survey,
                )
            ),
            len(endowments),
        )
        for mode, endowments in grouped.items()
    ]
    return sorted(rows, key=lambda row: (row[1], row[0]))


def _assign_ground_truth(
    data: RegressionData,
    simulation: Mapping[str, Any],
    *,
    seed: int,
) -> ActiveEndowments:
    settings = simulation.get("ground_truth", {"source": "existing"})
    settings = _mapping(
        settings,
        field="simulation.ground_truth",
    )
    source = settings.get("source", "existing")
    if source not in {"existing", "random", "modes"}:
        raise ValueError(
            "simulation.ground_truth.source must be existing, random, "
            "or modes"
        )

    records = deepcopy(data.endowments.get_endowments())
    rng = random.Random(seed)
    if source == "existing":
        selected = [
            record
            for record in records
            if record.get("role") == "ground_truth"
        ]
        if not selected:
            if simulation.get("target_source", "configured") == "configured":
                return ActiveEndowments.from_endowment_list(records)
            raise ValueError(
                "ground-truth target_source needs at least one existing "
                "ground-truth agent"
            )
        weights = [
            float(record.get("weight") or 0.0)
            for record in selected
        ]
        if any(weight < 0.0 for weight in weights):
            raise ValueError(
                "existing ground-truth weights must be nonnegative"
            )
        total = sum(weights)
        if total <= 0.0:
            raise ValueError(
                "existing ground-truth weights must have positive mass"
            )
        for record, weight in zip(selected, weights, strict=True):
            record["weight"] = weight / total
        return ActiveEndowments.from_endowment_list(records)

    candidates = records
    if source == "modes":
        raw_modes = settings.get("modes")
        entropy_rows = _mode_entropy(data)
        if raw_modes is None:
            minimum = settings.get("minimum_entropy")
            maximum = settings.get("maximum_entropy")
            if minimum is None and maximum is None:
                raise ValueError(
                    "mode ground truth needs modes, minimum_entropy, "
                    "or maximum_entropy"
                )
            selected_modes = {
                mode
                for mode, entropy, _ in entropy_rows
                if (
                    (minimum is None or entropy >= float(minimum))
                    and (maximum is None or entropy <= float(maximum))
                )
            }
        else:
            if (
                not isinstance(raw_modes, Sequence)
                or isinstance(raw_modes, (str, bytes, bytearray))
            ):
                raise ValueError(
                    "simulation.ground_truth.modes must be a sequence"
                )
            selected_modes = {_mode_name(mode) for mode in raw_modes}
        candidates = [
            record
            for record in records
            if _mode_name(record.get("mode", ())) in selected_modes
        ]

    raw_count = settings.get("count")
    if raw_count is None:
        proxy_ratio = settings.get("proxy_ratio")
        if (
            isinstance(proxy_ratio, bool)
            or not isinstance(proxy_ratio, (int, float))
            or not 0.0 <= float(proxy_ratio) < 1.0
        ):
            raise ValueError(
                "simulation.ground_truth needs a positive count or "
                "proxy_ratio in [0, 1)"
            )
        proxy_count = int(float(proxy_ratio) * len(candidates))
        count = len(candidates) - proxy_count
    else:
        count = _positive_int(
            raw_count,
            field="simulation.ground_truth.count",
        )
    if len(candidates) < count:
        raise ValueError(
            f"only {len(candidates)} ground-truth candidates for "
            f"requested count {count}"
        )
    selected_eids = {
        record["eid"]
        for record in rng.sample(candidates, count)
    }
    raw_weights = [rng.random() for _ in range(count)]
    total = sum(raw_weights)
    weight_by_eid = dict(
        zip(sorted(selected_eids), raw_weights, strict=True)
    )
    weight_by_eid = {
        eid: weight / total
        for eid, weight in weight_by_eid.items()
    }
    for record in records:
        eid = record["eid"]
        if eid in selected_eids:
            record["role"] = "ground_truth"
            record["weight"] = weight_by_eid[eid]
        else:
            record["role"] = "proxy"
            record["weight"] = None
    return ActiveEndowments.from_endowment_list(records)


def _stratified_fraction(
    endowments: ActiveEndowments,
    fraction: float,
    *,
    seed: int,
    proxy_only: bool,
) -> tuple[ActiveEndowments, set[str]]:
    rng = random.Random(seed)
    records = endowments.get_endowments()
    selected: list[dict[str, Any]] = []
    if proxy_only:
        selected.extend(
            deepcopy(
                [
                    record
                    for record in records
                    if record.get("role") == "ground_truth"
                ]
            )
        )
        groups = {
            "proxy": [
                record
                for record in records
                if record.get("role") == "proxy"
            ]
        }
    else:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record.get("role", "proxy"))].append(record)

    for role in sorted(groups):
        group = groups[role]
        count = max(1, int(len(group) * fraction)) if group else 0
        selected.extend(deepcopy(rng.sample(group, count)))

    cloned = ActiveEndowments.from_endowment_list(selected)
    ground_truth = cloned.get_endowments_by_role("ground_truth")
    if ground_truth:
        total = sum(float(record.get("weight") or 0.0) for record in ground_truth)
        if total <= 0:
            raise ValueError(
                "sampled ground-truth weights have zero total"
            )
        for record in ground_truth:
            record["weight"] = float(record.get("weight") or 0.0) / total
    return cloned, {record["eid"] for record in selected}


def _subset_agents(
    data: RegressionData,
    endowments: ActiveEndowments,
    simulation: Mapping[str, Any],
    *,
    kind: str,
    seed: int,
) -> tuple[ActiveEndowments, set[str], EntropySample | None, tuple[str, ...]]:
    all_eids = {
        record["eid"]
        for record in endowments.get_endowments()
    }
    entropy_sample = None
    included_modes: tuple[str, ...] = ()
    if kind == "endowment":
        return (*_stratified_fraction(
            endowments,
            _fraction(simulation.get("fraction")),
            seed=seed,
            proxy_only=False,
        ), None, included_modes)
    if kind == "proxy":
        return (*_stratified_fraction(
            endowments,
            _fraction(simulation.get("fraction")),
            seed=seed,
            proxy_only=True,
        ), None, included_modes)
    if kind == "target_entropy":
        proxy_records = [
            record
            for record in endowments.get_endowments()
            if record.get("role") == "proxy"
        ]
        proxy_pool = ActiveEndowments.from_endowment_list(proxy_records)
        entropy_sample = sample_to_target_mean_entropy(
            proxy_pool,
            data.answer_responses,
            data.survey,
            sample_size=_positive_int(
                simulation.get("sample_size"),
                field="simulation.sample_size",
            ),
            target_entropy=simulation.get("target_mean_entropy"),
            seed=seed,
            max_evaluations=simulation.get("max_evaluations", 5000),
            tolerance=simulation.get("tolerance"),
            require_tolerance=simulation.get(
                "require_tolerance",
                False,
            ),
        )
        selected_eids = set(entropy_sample.selected_eids)
        selected_eids.update(
            record["eid"]
            for record in endowments.get_endowments()
            if record.get("role") == "ground_truth"
        )
    elif kind == "entropy_modes":
        rows = _mode_entropy(data)
        direction = simulation.get("entropy_order", "ascending")
        if direction not in {"ascending", "descending"}:
            raise ValueError(
                "simulation.entropy_order must be ascending or descending"
            )
        if direction == "descending":
            rows = list(reversed(rows))
        count = _positive_int(
            simulation.get("included_mode_count"),
            field="simulation.included_mode_count",
        )
        if count > len(rows):
            raise ValueError(
                "included_mode_count exceeds available modes"
            )
        included_modes = tuple(row[0] for row in rows[:count])
        proxy_candidates = [
            record
            for record in endowments.get_endowments()
            if (
                record.get("role") == "proxy"
                and _mode_name(record.get("mode", ())) in included_modes
            )
        ]
        n_proxies = _positive_int(
            simulation.get("n_proxies"),
            field="simulation.n_proxies",
        )
        if len(proxy_candidates) < n_proxies:
            raise ValueError(
                f"the first {count} entropy-ordered modes contain "
                f"{len(proxy_candidates)} proxies, fewer than {n_proxies}"
            )
        rng = random.Random(seed)
        selected_eids = {
            record["eid"]
            for record in rng.sample(proxy_candidates, n_proxies)
        }
        selected_eids.update(
            record["eid"]
            for record in endowments.get_endowments()
            if record.get("role") == "ground_truth"
        )
    else:
        return endowments, all_eids, None, included_modes

    selected_records = deepcopy(
        [
            record
            for record in endowments.get_endowments()
            if record["eid"] in selected_eids
        ]
    )
    return (
        ActiveEndowments.from_endowment_list(selected_records),
        selected_eids,
        entropy_sample,
        included_modes,
    )


def _sample_split_qids(
    data: RegressionData,
    *,
    fraction: float,
    seed: int,
    train_only: bool,
) -> dict[str, set[str]]:
    rng = random.Random(seed)
    selected: dict[str, set[str]] = {}
    binary_to_original = data.binary_survey.binary_to_original_map
    original_to_binary = data.binary_survey.original_to_binary_map
    for name, split in data.splits.items():
        if train_only and name == data.test_split:
            selected[name] = set(split.qids)
            continue
        original_ids = sorted(
            {
                binary_to_original.get(qid, {}).get(
                    "original_id",
                    qid,
                )
                for qid in split.qids
            }
        )
        count = max(1, int(len(original_ids) * fraction))
        sampled_original = rng.sample(original_ids, count)
        qids: set[str] = set()
        for original_id in sampled_original:
            qids.update(
                original_to_binary.get(
                    original_id,
                    (original_id,),
                )
            )
        selected[name] = set(split.qids) & qids
    return selected


def _ground_truth_targets(
    data: RegressionData,
    qids: Sequence[str],
    endowments: ActiveEndowments,
) -> np.ndarray:
    weights = {
        record["eid"]: float(record.get("weight") or 0.0)
        for record in endowments.get_endowments_by_role("ground_truth")
    }
    if not weights or sum(weights.values()) <= 0:
        raise ValueError(
            "ground-truth targets require positive ground-truth weights"
        )
    values = []
    for qid in qids:
        numerator = 0.0
        denominator = 0.0
        for eid, weight in weights.items():
            response = data.binary_responses.responses.get(qid, {}).get(eid)
            if response is None:
                continue
            numerator += float(response) * weight
            denominator += weight
        if denominator <= 0:
            raise ValueError(
                f"no weighted ground-truth response for {qid!r}"
            )
        values.append(numerator / denominator)
    return np.asarray(values, dtype=float)


def _prepared_data(
    data: RegressionData,
    endowments: ActiveEndowments,
    selected_eids: set[str],
    simulation: Mapping[str, Any],
    *,
    kind: str,
    seed: int,
) -> RegressionData:
    question_selection = {
        name: set(split.qids)
        for name, split in data.splits.items()
    }
    if kind in {"response", "trainvalid"}:
        question_selection = _sample_split_qids(
            data,
            fraction=_fraction(simulation.get("fraction")),
            seed=seed,
            train_only=(kind == "trainvalid"),
        )

    features = tuple(
        eid for eid in data.feature_names if eid in selected_eids
    )
    if not features:
        raise ValueError(
            "simulation retained no model features; for fixed "
            "ground-truth recovery set split.use_proxy_only: false"
        )
    target_source = simulation.get("target_source", "configured")
    if target_source not in {"configured", "ground_truth"}:
        raise ValueError(
            "simulation.target_source must be configured or ground_truth"
        )

    splits: dict[str, RegressionSplit] = {}
    for name, split in data.splits.items():
        mask = [
            qid in question_selection[name]
            for qid in split.qids
        ]
        qids = tuple(
            qid for qid, keep in zip(split.qids, mask, strict=True) if keep
        )
        if not qids:
            raise ValueError(
                f"simulation retained no questions in split {name!r}"
            )
        X = split.X.loc[list(qids), list(features)].copy()
        y = (
            _ground_truth_targets(data, qids, endowments)
            if target_source == "ground_truth"
            else split.y[np.asarray(mask, dtype=bool)].copy()
        )
        splits[name] = RegressionSplit(
            name=name,
            qids=qids,
            X=X,
            y=y,
        )
    return replace(
        data,
        endowments=endowments,
        splits=MappingProxyType(splits),
        feature_names=features,
    )


def _safe_ratio(numerator: int | float, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _finite_or_none(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _stats(
    data: RegressionData,
    fit: FitResult,
    *,
    kind: str,
    seed: int,
    fraction: float | None,
    included_modes: tuple[str, ...],
    entropy_sample: EntropySample | None,
) -> dict[str, Any]:
    train_X = pd.concat(
        [data.get_split(name).X for name in data.train_splits]
    )
    train_y = np.concatenate(
        [data.get_split(name).y for name in data.train_splits]
    )
    test = data.get_split(data.test_split)
    model = fit.model
    # The reference respondent is a basis element, not an agent, so it is not
    # a candidate for recovering the ground-truth population.
    selected = {
        eid
        for eid, coefficient in model.coef_dict_.items()
        if coefficient != 0 and eid != REFERENCE_FEATURE
    }
    ground_truth = set(data.endowments.get_eids_by_role("ground_truth"))
    proxies = set(data.endowments.get_eids_by_role("proxy"))
    tp = len(selected & ground_truth)
    fp = len(selected - ground_truth)
    fn = len(ground_truth - selected)
    coefficients = model.coef_dict_
    gt_values = [
        coefficients[eid]
        for eid in ground_truth
        if eid in coefficients
    ]
    fp_values = [
        coefficients[eid]
        for eid in selected - ground_truth
    ]
    n_obs, n_agents = train_X.shape
    result: dict[str, Any] = {
        "kind": kind,
        "fraction": fraction,
        "seed": seed,
        "model_type": fit.method,
        "n_obs": n_obs,
        "n_agents": n_agents,
        "n_gt": len(ground_truth),
        "n_proxy": len(proxies),
        "obs_per_agent": _safe_ratio(n_obs, n_agents),
        "obs_per_gt": _safe_ratio(n_obs, len(ground_truth)),
        "obs_per_proxy": _safe_ratio(n_obs, len(proxies)),
        "proxy_to_gt_ratio": _safe_ratio(
            len(proxies),
            len(ground_truth),
        ),
        "mse": _finite_or_none(model.score(train_X, train_y)),
        "r2": _finite_or_none(model.r2(train_X, train_y)),
        "mse_test": _finite_or_none(model.score(test.X, test.y)),
        "r2_test": _finite_or_none(model.r2(test.X, test.y)),
        "selected": len(selected),
        "precision": _safe_ratio(tp, tp + fp) or 0.0,
        "recall": _safe_ratio(tp, tp + fn) or 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_weight_gt": (
            float(np.mean(gt_values)) if gt_values else 0.0
        ),
        "mean_weight_fp": (
            float(np.mean(fp_values)) if fp_values else 0.0
        ),
        "selected_eids": sorted(selected),
        "gt_eids": sorted(ground_truth),
        "gt_entropy": float(
            ActiveEndowments.compute_entropy(
                data.endowments.get_endowments_by_role("ground_truth"),
                data.answer_responses,
                data.survey,
            )
        ),
        "proxy_entropy": float(
            ActiveEndowments.compute_entropy(
                data.endowments.get_endowments_by_role("proxy"),
                data.answer_responses,
                data.survey,
            )
        ),
        "included_modes": list(included_modes),
        "n_modes": len(included_modes),
    }
    if entropy_sample is not None:
        result.update(
            {
                "target_mean_entropy": entropy_sample.target_entropy,
                "achieved_mean_entropy": entropy_sample.achieved_entropy,
                "entropy_absolute_error": entropy_sample.absolute_error,
                "entropy_search_method": entropy_sample.method,
                "entropy_candidates_evaluated": (
                    entropy_sample.candidates_evaluated
                ),
            }
        )
    return result


def _write_csv(path: Path, stats: Mapping[str, Any]) -> None:
    normalized = {
        key: (
            json.dumps(value, sort_keys=True)
            if isinstance(value, (list, dict, tuple))
            else value
        )
        for key, value in stats.items()
    }
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(normalized))
        writer.writeheader()
        writer.writerow(normalized)


def run_scientific_simulation(
    config: Config,
    *,
    run_dir: str | os.PathLike[str],
    solver: str = DEFAULT_SOLVER,
    verbose: bool = False,
) -> ScientificSimulationResult:
    """Run one cached-basis simulation child and write auditable results."""

    if not isinstance(config, Config):
        raise TypeError("config must be Config")
    config.validate_for("simulate")
    simulation = _mapping(
        config.data.get("simulation"),
        field="simulation",
    )
    if simulation.get("invokes_generation", False):
        raise ValueError(
            "scientific simulations operate on cached bases and cannot "
            "invoke generation"
        )
    kind = simulation.get("kind")
    if kind not in _KINDS:
        raise ValueError(
            f"simulation.kind must be one of {sorted(_KINDS)}"
        )
    run = _mapping(config.data.get("run"), field="run")
    seed = run.get("seed", 101)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("run.seed must be an integer")

    base = _load_base(config)
    endowments = _assign_ground_truth(
        base,
        simulation,
        seed=seed,
    )
    (
        selected_endowments,
        selected_eids,
        entropy_sample,
        included_modes,
    ) = _subset_agents(
        base,
        endowments,
        simulation,
        kind=kind,
        seed=seed,
    )
    prepared = _prepared_data(
        base,
        selected_endowments,
        selected_eids,
        simulation,
        kind=kind,
        seed=seed,
    )
    fit = fit_model(
        prepared,
        _mapping(config.data.get("model"), field="model"),
        solver=solver,
        seed=seed,
        verbose=verbose,
    )
    stats = _stats(
        prepared,
        fit,
        kind=kind,
        seed=seed,
        fraction=(
            _fraction(simulation.get("fraction"))
            if kind in {"endowment", "response", "proxy", "trainvalid"}
            else None
        ),
        included_modes=included_modes,
        entropy_sample=entropy_sample,
    )

    output = Path(run_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "simulation.json"
    csv_path = output / "simulation.csv"
    payload = {
        "schema_version": 1,
        "artifact_type": "scientific_simulation_result",
        "config_digest": config.digest,
        "run": {
            "name": run.get("name"),
            "repeat": run.get("repeat", 1),
            "seed": seed,
        },
        "simulation": {
            key: value
            for key, value in config.to_dict()["simulation"].items()
            if key != "sweep"
        },
        "stats": stats,
        "fit": {
            "method": fit.method,
            "solver": fit.solver,
            "status": fit.status,
            "best_alpha": fit.best_alpha,
            "best_l1_ratio": fit.best_l1_ratio,
            "coefficients": fit.model.coef_dict_,
        },
    }
    artifact_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, stats)
    if verbose:
        print(
            f"[simulate] {kind}: n={stats['n_agents']}, "
            f"train MSE={stats['mse']:.6g}, "
            f"test MSE={stats['mse_test']:.6g}"
        )
    return ScientificSimulationResult(
        config_digest=config.digest,
        run_dir=output,
        artifact_path=artifact_path,
        csv_path=csv_path,
        data=prepared,
        fit=fit,
        stats=MappingProxyType(stats),
        entropy_sample=entropy_sample,
    )


__all__ = [
    "ScientificSimulationResult",
    "run_scientific_simulation",
]
