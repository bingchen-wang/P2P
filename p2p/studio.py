"""Local-only HTTP server for the P2P Experiment Studio."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import csv
from copy import deepcopy
import gzip
import json
import mimetypes
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse
import uuid
import webbrowser

import yaml

from p2p.config import Config, ConfigError, expand_grid
from p2p.registry import BACKENDS, COMMANDS, MODELS

# Import the public component packages once so their idempotent registrations
# are reflected by the Studio capabilities endpoint.  Studio reads the live
# registries; it does not maintain a second list of model or backend names.
import p2p.backends  # noqa: F401, E402
import p2p.commands  # noqa: F401, E402
import p2p.models  # noqa: F401, E402
from p2p.models.constrained_model import (  # noqa: E402
    DEFAULT_SOLVER,
    DEFAULT_SOLVER_FALLBACKS,
)
from p2p.models.folds import DEFAULT_FOLD_METHOD, FOLD_METHODS  # noqa: E402
from p2p.stages.evaluate import (  # noqa: E402
    _SUPPORTED_DISTRIBUTION_POLICIES,
    SUPPORTED_METRICS,
)
from p2p.stages.fit import (  # noqa: E402
    DEFAULT_ALPHA_GRID,
    DEFAULT_L1_RATIO_GRID,
    DEFAULT_ZERO_THRESHOLD,
    REFERENCE_DISTRIBUTION_CONFLICTS,
    SIMPLE_AVERAGE,
    VALIDATION_STRATEGIES,
)


ALLOWED_COMMANDS = {
    "pipeline",
    "generate",
    "regress",
    "evaluate",
    "cluster",
}

# Backends that run on this machine and cost nothing.
LOCAL_BACKENDS = {"dummy", "ollama", "simulated"}

# Commands whose CLI accepts ``--solver`` (Stage 2 is fitted).
SOLVER_COMMANDS = {"pipeline", "regress"}
SOLVERS = ("CLARABEL", "OSQP", "SCS")

# How each metric P2P can score is presented. The set of metrics comes from the
# evaluator (SUPPORTED_METRICS); a test fails if one lacks a description here.
METRIC_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "mse": {
        "label": "Raw option-level MSE",
        "short": "MSE",
        "direction": "lower",
        "unit": "binary option",
        "description": "Mean squared error over (question, option) cells, on the raw "
        "predictions before any rescaling.",
    },
    "distribution_mse": {
        "label": "Rescaled MSE",
        "short": "Rescaled MSE",
        "direction": "lower",
        "unit": "binary option",
        "description": "The same cells as MSE, with each question's predictions clipped "
        "at zero and rescaled to sum to one.",
    },
    "one_minus_jsd": {
        "label": "1 − JSD",
        "short": "1 − JSD",
        "direction": "higher",
        "unit": "question",
        "description": "One minus the Jensen–Shannon distance between the predicted and "
        "target distributions, averaged over questions.",
    },
    "emd": {
        "label": "EMD / WD",
        "short": "EMD",
        "direction": "lower",
        "unit": "question",
        "description": "Normalised ordinal Wasserstein distance, averaged over questions.",
    },
    "mcpa": {
        "label": "MCPA",
        "short": "MCPA",
        "direction": "higher",
        "unit": "question",
        "description": "Share of questions whose most-chosen option is predicted "
        "correctly.",
    },
}
METRIC_ORDER = ("mse", "distribution_mse", "one_minus_jsd", "emd", "mcpa")

# The guided Stage-2 configurations: the mass penalty on or off, each with no
# uniform component, a reference respondent or an intercept. Every other
# combination remains available by editing the manifest.
STAGE2_CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "mass_penalty",
        "label": "Mass penalty (default)",
        "description": "Each question's predicted mass is pulled towards one by a "
        "soft penalty. No uniform component.",
        "options": {},
    },
    {
        "id": "mass_penalty_reference",
        "label": "Mass penalty + reference respondent",
        "description": "Adds a reference respondent who answers every option of a "
        "k-option question with 1/k, weighted like any agent.",
        "options": {"reference_respondent": True},
    },
    {
        "id": "mass_penalty_intercept",
        "label": "Mass penalty + intercept",
        "description": "Adds an unpenalised constant on every option row; the mass "
        "penalty sees the k × intercept it adds to each question.",
        "options": {"intercept": True},
    },
    {
        "id": "no_mass_penalty",
        "label": "No mass penalty",
        "description": "Removes the per-question mass penalty "
        "(group_penalty_strength: 0).",
        "options": {"group_penalty_strength": 0},
    },
    {
        "id": "no_mass_penalty_reference",
        "label": "No mass penalty + reference respondent",
        "description": "No mass penalty, with the reference respondent.",
        "options": {"group_penalty_strength": 0, "reference_respondent": True},
    },
    {
        "id": "no_mass_penalty_intercept",
        "label": "No mass penalty + intercept",
        "description": "No mass penalty, with an intercept: the closest Stage-2 "
        "configuration to a scikit-learn lasso with an intercept.",
        "options": {"group_penalty_strength": 0, "intercept": True},
    },
)

# Methods the guided builder offers, with the controls that apply to each.
GUIDED_METHODS: dict[str, dict[str, Any]] = {
    "lasso": {
        "controls": ["alpha", "validation", "configuration", "block_weighting"],
        "description": "Constrained lasso: selects a sparse, signed ensemble.",
    },
    "elasticnet": {
        "controls": ["alpha", "l1_ratio", "validation", "configuration", "block_weighting"],
        "description": "Constrained elastic net: lasso with an added ridge penalty.",
    },
    SIMPLE_AVERAGE: {
        "controls": [],
        "description": "Uniform weights over every agent: no fitting, no penalty.",
    },
}

# Guided methods that are a registered method with fixed options. A manifest
# stores the registered method, so reading one back matches these options.
GUIDED_ALIASES: dict[str, dict[str, Any]] = {
    "ridge": {
        "label": "Ridge",
        "method": "elasticnet",
        "options": {"l1_ratio": [0.0]},
        "controls": ["alpha", "validation", "configuration", "block_weighting"],
        "description": "Elastic net with l1_ratio 0: every agent keeps a small weight; "
        "nothing is selected out.",
        # Ridge weights are many and small; P2P's default threshold (1e-5)
        # would set some of them to zero after fitting.
        "zeroThreshold": 1e-10,
    },
}

# How each distribution policy is presented. legacy_signed_normalize, the
# earlier spelling of signed_sum_normalize, is accepted but not offered.
DISTRIBUTION_POLICY_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "clip_and_normalize": {
        "label": "Clip, then rescale (default)",
        "description": "Set negative values to zero, then rescale each question's "
        "prediction to sum to one.",
    },
    "project_to_simplex": {
        "label": "Nearest distribution",
        "description": "Replace each question's prediction by the closest probability "
        "distribution (Euclidean projection).",
    },
    "signed_sum_normalize": {
        "label": "Divide by the signed sum",
        "description": "Divide each question's prediction by its signed sum (first by 100 "
        "if that sum exceeds 1.5), keeping negative values; scored with SciPy's "
        "routines, which renormalise their inputs.",
    },
    "validate": {
        "label": "Score as predicted",
        "description": "Score predictions unchanged; a question whose prediction is not "
        "already a distribution is treated as invalid.",
    },
}
DISTRIBUTION_POLICY_ORDER = ("clip_and_normalize", "project_to_simplex", "signed_sum_normalize", "validate")
INVALID_DISTRIBUTION_OPTIONS: tuple[dict[str, str], ...] = (
    {
        "id": "error",
        "label": "Stop the run (default)",
        "description": "A question whose prediction is still not a distribution after "
        "the policy stops the evaluation with an error.",
    },
    {
        "id": "allow",
        "label": "Skip the question",
        "description": "Such a question is left out of the distributional metrics; "
        "the number left out is recorded with each metric.",
    },
)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


# libyaml's safe loader when available: about ten times faster than the pure
# Python one, which matters when indexing tens of thousands of run records.
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_YAML_LOADER)
    except (OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = []
            for index, row in enumerate(csv.DictReader(handle)):
                if index >= limit:
                    break
                rows.append(dict(row))
            return rows
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def _file_digest(path: Path) -> str:
    """Return the stable SHA256 digest used by local dataset descriptors."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_path(value: Any, *, root: Path, field: str) -> Path:
    """Resolve a user-supplied path without exposing files outside the repo."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a repository-relative path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not _inside(candidate, root):
        raise ValueError(f"{field} must stay inside the repository")
    if not candidate.is_file():
        raise FileNotFoundError(_relative(candidate, root))
    return candidate


def _parse_option_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        str(key): str(label)
        for key, label in parsed.items()
        if str(key).strip() and str(label).strip()
    }


def _number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric else None


def _markdown_title(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    except (OSError, UnicodeDecodeError):
        pass
    return path.stem.replace("_", " ").title()


def _report_kind(directory: Path) -> str:
    names = {path.name for path in directory.glob("summary*.md")}
    if "summary.md" in names and (
        "summary_endowments.md" in names
        or "summary_plugin_elicitation.md" in names
    ):
        return "pipeline"
    if "summary.md" in names:
        return "regression"
    if "summary_plugin_elicitation.md" in names:
        return "elicitation"
    return "generation"


def _report_summary(directory: Path, root: Path) -> dict[str, Any]:
    markdown_paths = sorted(directory.glob("summary*.md"))
    image_paths = sorted(directory.glob("*.png"))
    metadata = _read_yaml(directory / "metadata.yaml")
    experiment = metadata.get("experiment", {})
    if not isinstance(experiment, dict):
        experiment = {}
    metric_rows = _read_csv(directory / "summary_aggregate.csv", limit=8)
    results = _read_json(directory / "results.json", {}) or {}
    token_usage = _read_json(directory / "token_usage_summary.json", {}) or {}
    if not token_usage and isinstance(results.get("token_usage_summary"), dict):
        token_usage = results["token_usage_summary"]
    models = [str(row.get("model")) for row in metric_rows if row.get("model")]
    primary_markdown = next(
        (path for path in markdown_paths if path.name == "summary.md"),
        markdown_paths[0],
    )
    name = str(experiment.get("name") or directory.name)
    modified_at = max(path.stat().st_mtime for path in markdown_paths)
    return {
        "path": _relative(directory, root),
        "name": name,
        "title": _markdown_title(primary_markdown),
        "kind": _report_kind(directory),
        "models": models,
        "metrics": [
            {
                "model": row.get("model"),
                "testMse": _number(row.get("test_mse")),
                "testOneMinusJsd": _number(row.get("test_one_minus_jsd")),
                "testEmd": _number(row.get("test_emd")),
                "testAccuracy": _number(row.get("test_accuracy")),
                "selected": _number(row.get("num_selected")),
            }
            for row in metric_rows
        ],
        "reportCount": len(markdown_paths),
        "imageCount": len(image_paths),
        "totalCost": _number(token_usage.get("total_cost")),
        "numEndowments": _number(results.get("num_endowments")),
        "modifiedAt": _iso_timestamp(modified_at),
    }


def _artifact_category(name: str) -> str:
    lowered = name.lower()
    if "prediction" in lowered or lowered.endswith("_pred.png"):
        return "predictions"
    if any(term in lowered for term in ("diagnostic", "singular", "mca", "_diag")):
        return "diagnostics"
    if any(term in lowered for term in ("entropy", "theme", "trajectory")):
        return "generation"
    if "weight" in lowered:
        return "weights"
    return "other"


def _derived_mode_selection_rows(
    endowment_rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> list[dict[str, str]]:
    """Reconstruct the legacy per-mode summary from canonical P2P records.

    Native P2P runs store the fitted coefficients in ``diagnostics.json`` and
    the mode membership in ``endowments.csv``. Earlier-format runs usually
    carry the already-aggregated ``summary_by_mode.csv`` instead. Keeping the
    reconstruction here gives Studio one read-only compatibility contract
    without modifying either kind of run on disk.
    """

    model = diagnostics.get("model", {})
    if not isinstance(model, dict):
        model = {}
    raw_coefficients = model.get("coefficients")
    coefficients = (
        {
            str(eid): numeric
            for eid, value in raw_coefficients.items()
            if (numeric := _number(value)) is not None
        }
        if isinstance(raw_coefficients, dict)
        else {}
    )

    # Some imported records store the fitted value directly beside each
    # endowment. Use that only when canonical coefficients are absent.
    if not coefficients:
        coefficients = {
            str(row.get("eid")): numeric
            for row in endowment_rows
            if row.get("eid")
            and (numeric := _number(row.get("weight"))) is not None
        }
    if not coefficients:
        return []

    by_id = {
        str(row.get("eid")): row
        for row in endowment_rows
        if row.get("eid")
    }
    aggregates: dict[str, dict[str, float]] = {}

    def bucket(mode: str) -> dict[str, float]:
        return aggregates.setdefault(
            mode,
            {
                "generated": 0.0,
                "selected": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "weight": 0.0,
                "absolute_weight": 0.0,
            },
        )

    for row in endowment_rows:
        mode = str(row.get("mode") or "unclassified")
        bucket(mode)["generated"] += 1

    for eid, coefficient in coefficients.items():
        row = by_id.get(eid)
        mode = str((row or {}).get("mode") or "unclassified")
        aggregate = bucket(mode)
        # Coefficients without an endowment row still represent a generated
        # basis element; account for them in the fallback mode denominator.
        if row is None:
            aggregate["generated"] += 1
        if abs(coefficient) <= 1e-10:
            continue
        aggregate["selected"] += 1
        aggregate["positive" if coefficient > 0 else "negative"] += 1
        aggregate["weight"] += coefficient
        aggregate["absolute_weight"] += abs(coefficient)

    method = str(model.get("method") or "model")
    rows: list[dict[str, str]] = []
    for mode, aggregate in aggregates.items():
        generated = int(aggregate["generated"])
        selected = int(aggregate["selected"])
        total_weight = aggregate["weight"]
        rows.append(
            {
                "model": method,
                "mode": mode,
                "num_selected": str(selected),
                "num_positive": str(int(aggregate["positive"])),
                "num_negative": str(int(aggregate["negative"])),
                "total_weight": str(total_weight),
                "total_abs_weight": str(aggregate["absolute_weight"]),
                "fraction_selected": str(selected / generated if generated else 0.0),
                "avg_weight": str(total_weight / selected if selected else 0.0),
            }
        )
    return rows


def _generation_analysis(
    directory: Path,
    diagnostics: dict[str, Any] | None = None,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Read generation diagnostics shared by current and earlier-format runs."""

    source = _population_source_directory(directory, root) if root is not None else directory
    endowment_rows: list[dict[str, Any]] = []
    for name in ("updated_endowments.csv", "endowments.csv"):
        endowment_rows = _read_csv(source / name, limit=5_000)
        if endowment_rows:
            break

    mode_rows = _read_csv(directory / "summary_by_mode.csv", limit=500)
    mode_selection_source = "summary_by_mode" if mode_rows else "unavailable"
    if not mode_rows:
        diagnostic_payload = (
            diagnostics
            if isinstance(diagnostics, dict)
            else (_read_json(directory / "diagnostics.json", {}) or {})
        )
        mode_rows = _derived_mode_selection_rows(
            endowment_rows,
            diagnostic_payload,
        )
        if mode_rows:
            mode_selection_source = "derived_coefficients"

    return {
        "endowmentRows": endowment_rows,
        "modeRows": mode_rows,
        "modeSelectionSource": mode_selection_source,
        "modeSummaryRows": _read_csv(
            directory / "mode_summary.csv" if (directory / "mode_summary.csv").is_file() else source / "mode_summary.csv",
            limit=500,
        ),
        "modeEntropyRows": _read_csv(
            directory / "persistent_mode_entropy.csv" if (directory / "persistent_mode_entropy.csv").is_file() else source / "persistent_mode_entropy.csv",
            limit=500,
        ),
        "entropyRows": _read_csv(
            directory / "question_entropy_trajectories.csv" if (directory / "question_entropy_trajectories.csv").is_file() else source / "question_entropy_trajectories.csv",
            limit=250,
        ),
        "aggregateEntropy": _read_csv(
            directory / "aggregate_entropy_trajectory.csv" if (directory / "aggregate_entropy_trajectory.csv").is_file() else source / "aggregate_entropy_trajectory.csv",
            limit=500,
        ),
    }


def _population_source_directory(directory: Path, root: Path) -> Path:
    """Resolve the proxy basis used by a canonical Stage-2-only run.

    Pipeline and earlier-format runs usually keep their endowments beside the
    result. Regression-only runs instead point at the Stage-1 basis in the
    resolved config. The Studio follows that local, repository-contained
    reference read-only so population summaries describe the real basis.
    """

    if any((directory / name).is_file() for name in ("updated_endowments.csv", "endowments.csv")):
        return directory
    # run.json records the basis files a regression read; their folder is the
    # basis. This avoids parsing the resolved config for every indexed run.
    artifacts = (_read_json(directory / "run.json", {}) or {}).get("source_artifacts")
    endowments = artifacts.get("endowments") if isinstance(artifacts, dict) else None
    if isinstance(endowments, dict) and isinstance(endowments.get("path"), str):
        candidate = Path(endowments["path"]).resolve().parent
        if _inside(candidate, root) and candidate.is_dir():
            return candidate
    payload = _read_yaml(directory / "config.resolved.yaml")
    config = payload.get("config") if isinstance(payload.get("config"), dict) else payload
    data = config.get("data", {}) if isinstance(config, dict) else {}
    basis: Any = data.get("basis") if isinstance(data, dict) else None
    if isinstance(basis, dict):
        basis = basis.get("path") or basis.get("source")
    if not isinstance(basis, str) or not basis.strip():
        return directory
    candidate = Path(basis)
    if not candidate.is_absolute():
        candidate = (directory / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate if _inside(candidate, root) and candidate.is_dir() else directory


def _population_summary(
    directory: Path,
    root: Path,
    *,
    diagnostics: dict[str, Any] | None = None,
    method: str | None = None,
) -> dict[str, Any] | None:
    """Return compact, factual population counts for Studio list views."""

    source = _population_source_directory(directory, root)
    endowment_rows: list[dict[str, Any]] = []
    for name in ("updated_endowments.csv", "endowments.csv"):
        endowment_rows = _read_csv(source / name, limit=5_000)
        if endowment_rows:
            break

    diagnostic_payload = diagnostics if isinstance(diagnostics, dict) else {}
    model = diagnostic_payload.get("model", {})
    if not isinstance(model, dict):
        model = {}
    selected_method = str(method or model.get("method") or "")
    raw_coefficients = model.get("coefficients")
    coefficients = (
        {
            str(eid): numeric
            for eid, value in raw_coefficients.items()
            if (numeric := _number(value)) is not None
        }
        if isinstance(raw_coefficients, dict)
        else {}
    )

    by_id = {
        str(row.get("eid")): row
        for row in endowment_rows
        if row.get("eid")
    }
    modes: dict[str, dict[str, int]] = {}
    bank = _read_yaml(source / "attribute_bank.yaml")
    templates = bank.get("templates", {}) if isinstance(bank, dict) else {}
    templates = templates if isinstance(templates, dict) else {}
    question_modes = {str(value) for value in templates.get("question", []) or []}
    thematic_modes = {str(value) for value in templates.get("thematic", []) or []}
    theoretical_modes = {str(value) for value in templates.get("theoretical", []) or []}

    def mode_family(name: str) -> str:
        """Mirror the template grouping used by the entropy figures."""

        tokens = [token.strip() for token in name.split("+") if token.strip()]
        if tokens == ["core"]:
            return "Core"
        if tokens == ["survey"]:
            return "Survey"
        if len(tokens) == 1 and tokens[0] in question_modes:
            return "Question patch"
        if len(tokens) > 1 and any(token in question_modes for token in tokens):
            return "Mixed patch"
        if len(tokens) == 2 and "core" in tokens:
            other = next(token for token in tokens if token != "core")
            if other in thematic_modes:
                return "Thematic"
            if other in theoretical_modes:
                return "Theoretical"

        # Historical runs sometimes retain the mode names but not the bank.
        lowered = [token.lower() for token in tokens]
        looks_like_question = lambda token: bool(
            re.search(r"(?:^|_)(?:gap\d*q|p?q|rq|scm|f\d)", token, re.IGNORECASE)
            or re.search(r"_w\d+$", token, re.IGNORECASE)
        )
        if len(tokens) > 1 and any(looks_like_question(token) for token in tokens):
            return "Mixed patch"
        if len(tokens) == 1 and any(looks_like_question(token) for token in tokens):
            return "Question patch"
        if "survey" in lowered:
            return "Survey"
        if any(token in {"maslow", "big_five"} for token in lowered):
            return "Theoretical"
        if name == "unclassified":
            return "Thematic"
        return "Thematic"

    def mode_bucket(name: str) -> dict[str, int]:
        return modes.setdefault(name or "unclassified", {"generated": 0, "selected": 0, "antiTypes": 0})

    for row in endowment_rows:
        mode_bucket(str(row.get("mode") or "unclassified"))["generated"] += 1

    if coefficients:
        for eid, coefficient in coefficients.items():
            if abs(coefficient) <= 1e-10:
                continue
            bucket = mode_bucket(str((by_id.get(eid) or {}).get("mode") or "unclassified"))
            bucket["selected"] += 1
            if coefficient < 0:
                bucket["antiTypes"] += 1
    else:
        selection_rows = _read_csv(source / "summary_by_mode.csv", limit=500)
        if selected_method:
            matching = [
                row for row in selection_rows
                if str(row.get("model") or "").lower() == selected_method.lower()
            ]
            if matching:
                selection_rows = matching
        for row in selection_rows:
            bucket = mode_bucket(str(row.get("mode") or "unclassified"))
            bucket["selected"] += int(_number(row.get("num_selected")) or 0)
            bucket["antiTypes"] += int(_number(row.get("num_negative")) or 0)

        # Some earlier-format runs have no per-mode summary but do carry the selected
        # weight beside each endowment.
        if not selection_rows:
            for row in endowment_rows:
                weight = _number(row.get("weight"))
                if weight is None or abs(weight) <= 1e-10:
                    continue
                bucket = mode_bucket(str(row.get("mode") or "unclassified"))
                bucket["selected"] += 1
                if weight < 0:
                    bucket["antiTypes"] += 1

    generated = len(endowment_rows) or len(coefficients)
    selected = sum(item["selected"] for item in modes.values())
    anti_types = sum(item["antiTypes"] for item in modes.values())
    if not generated and not selected:
        return None
    return {
        "generated": generated,
        "selected": selected,
        "antiTypes": anti_types,
        "sourcePath": _relative(source, root),
        "modes": [
            {"mode": name, "family": mode_family(name), **counts}
            for name, counts in sorted(
                modes.items(),
                key=lambda item: (-item[1]["generated"], item[0]),
            )
        ],
    }


def _report_detail(directory: Path, root: Path) -> dict[str, Any]:
    markdown_paths = sorted(directory.glob("summary*.md"))
    image_paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".svg"}
    )
    documents = []
    for path in markdown_paths:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        documents.append(
            {
                "name": path.name,
                "title": _markdown_title(path),
                "content": content[:1_000_000],
            }
        )
    return {
        "summary": _report_summary(directory, root),
        "documents": documents,
        "images": [
            {
                "name": path.name,
                "path": _relative(path, root),
                "category": _artifact_category(path.name),
                "size": path.stat().st_size,
            }
            for path in image_paths
        ],
        "metadata": _read_yaml(directory / "metadata.yaml"),
        "results": _read_json(directory / "results.json", {}) or {},
        "tokenUsage": _read_json(directory / "token_usage_summary.json", {}) or {},
        "metricRows": _read_csv(directory / "summary_aggregate.csv", limit=50),
        **_generation_analysis(directory),
        "files": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "type": path.suffix.lower().lstrip(".") or "file",
            }
            for path in sorted(directory.iterdir())
            if path.is_file()
        ],
    }


def _csv_row_count(path: Path) -> int:
    """Count data rows without materialising wide legacy response tables."""

    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for _ in reader)
    except (OSError, csv.Error, UnicodeDecodeError):
        return 0


def _legacy_config(directory: Path) -> dict[str, Any]:
    for filename in ("regression_config.yaml", "metadata.yaml"):
        value = _read_yaml(directory / filename)
        if isinstance(value.get("paths"), dict):
            return value
    return {}


def _legacy_usage(directory: Path) -> dict[str, float]:
    payload = _read_json(directory / "token_usage_summary.json", {}) or {}
    modules = payload.get("per_model_module", {})
    input_tokens = 0.0
    output_tokens = 0.0
    if isinstance(modules, dict):
        for item in modules.values():
            if not isinstance(item, dict):
                continue
            input_tokens += _number(item.get("total_input_tokens")) or 0.0
            output_tokens += _number(item.get("total_output_tokens")) or 0.0
    usage: dict[str, float] = {}
    total_cost = _number(payload.get("total_cost"))
    if total_cost is not None:
        usage["cost_usd"] = total_cost
    if input_tokens:
        usage["input_tokens"] = input_tokens
    if output_tokens:
        usage["output_tokens"] = output_tokens
    return usage


def _legacy_run_summary(directory: Path, root: Path) -> dict[str, Any]:
    """Expose an earlier-format run through the current P2P run contract."""

    report = _report_summary(directory, root)
    config = _legacy_config(directory)
    experiment = config.get("experiment", {})
    if not isinstance(experiment, dict):
        experiment = {}
    method = str(
        config.get("weight_assignment_model")
        or next((item for item in report["models"] if item), "legacy")
    )
    aggregate_rows = _read_csv(directory / "summary_aggregate.csv", limit=20)
    selected_row = next(
        (row for row in aggregate_rows if str(row.get("model")) == method),
        aggregate_rows[0] if aggregate_rows else {},
    )
    metric_columns = (
        ("mse", "test_mse"),
        ("one_minus_jsd", "test_one_minus_jsd"),
        ("emd", "test_emd"),
        ("mcpa", "test_accuracy"),
    )
    metrics = [
        {
            "metric": name,
            "value": value,
            "split": "test",
            "unit": "not recorded",
            "predictionPolicy": "legacy / not recorded",
            "qidsRequested": None,
            "qidsUsed": None,
            "diagnostics": {},
        }
        for name, column in metric_columns
        if (value := _number(selected_row.get(column))) is not None
    ]
    responses_path = directory / "responses.csv"
    freeze = _read_yaml(directory / "_freeze.yaml")
    return {
        "path": _relative(directory, root),
        "runId": directory.name,
        "repeat": None,
        "seed": experiment.get("seed"),
        "conditions": {
            "method": method,
            "pipeline_generation": "pre-migration",
        },
        "method": method,
        "solverStatus": "archived",
        "metrics": metrics,
        "usage": _legacy_usage(directory),
        "questionCount": _csv_row_count(responses_path),
        "hasDiagnostics": responses_path.is_file(),
        "population": _population_summary(
            directory,
            root,
            method=method,
        ),
        "modifiedAt": report["modifiedAt"],
        "origin": "legacy",
        "sourceFormat": "earlier-format-v1",
        "sourceVersion": freeze.get("source_commit") or freeze.get("source_git_sha"),
        "capabilities": {
            "metrics": bool(metrics),
            "predictions": responses_path.is_file(),
            "weights": (directory / "updated_endowments.csv").is_file(),
            "generation": any(
                (directory / name).is_file()
                for name in (
                    "persistent_mode_entropy.csv",
                    "aggregate_entropy_trajectory.csv",
                    "question_entropy_trajectories.csv",
                )
            ),
            "folds": False,
            "usage": bool(_legacy_usage(directory)),
            "reports": bool(report["reportCount"]),
            "export": True,
            "reevaluate": False,
        },
        "compatibilityWarnings": [
            "This earlier-format record may not contain newer fold, "
            "coverage, or prediction-policy fields."
        ],
        "provenance": {
            "label": "Earlier-format run (read-only)",
            "sourceRepository": freeze.get("source_repository"),
            "sourcePath": freeze.get("source_path"),
            "cleaningContract": freeze.get("cleaning_contract"),
        },
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _iso_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc,
    ).isoformat()


def _backend_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    backend = config.get("backend", {})
    if not isinstance(backend, dict):
        return []
    entries: list[dict[str, Any]] = []
    for module in ("response", "endowment", "attribute_learner"):
        value = backend.get(module)
        if not isinstance(value, dict):
            continue
        entries.append(
            {
                "module": module,
                "type": str(value.get("type", "profile")),
                "model": str(value.get("model_name", "configured by profile")),
            }
        )
    return entries


def _manifest_summary(path: Path, root: Path) -> dict[str, Any]:
    try:
        config = Config.load(path).to_dict()
    except ConfigError:
        config = _read_yaml(path)
    run = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
    generation = (
        config.get("generation", {})
        if isinstance(config.get("generation"), dict)
        else {}
    )
    model = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    backend = config.get("backend", {}) if isinstance(config.get("backend"), dict) else {}
    profile = backend.get("profile") if isinstance(backend, dict) else None
    backend_entries = _backend_entries(config)
    paid = any(item["type"] not in LOCAL_BACKENDS for item in backend_entries)
    if isinstance(profile, str):
        profile_path = (path.parent / profile).resolve()
        profile_config = _read_yaml(profile_path)
        profile_entries = _backend_entries({"backend": profile_config})
        backend_entries.extend(profile_entries)
        paid = paid or any(
            item["type"] not in LOCAL_BACKENDS
            for item in profile_entries
        )
    return {
        "path": _relative(path, root),
        "name": str(run.get("name", path.stem)),
        "seed": run.get("seed"),
        "mode": generation.get("mode"),
        "targetN": generation.get("target_n"),
        "method": model.get("method"),
        "backend": backend_entries,
        "paid": paid,
        "modifiedAt": _iso_timestamp(path.stat().st_mtime),
    }


def _manifest_filename(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        raise ValueError("Experiment name must contain a letter or number")
    return f"{slug}.yaml"


def _manifest_reference(value: Any, *, root: Path, directory: Path, field: str) -> Any:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a repository-relative path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not _inside(candidate, root):
        raise ValueError(f"{field} must stay inside the repository")
    if not candidate.exists():
        raise ValueError(f"{field} does not exist: {_relative(candidate, root)}")
    return os.path.relpath(candidate, directory)


def _prepare_studio_manifest(
    payload: dict[str, Any],
    *,
    root: Path,
    destination: Path,
) -> dict[str, Any]:
    config = deepcopy(payload)
    data = config.get("data")
    if isinstance(data, dict):
        survey = data.get("survey")
        if isinstance(survey, dict):
            for key in ("source", "schema"):
                if key in survey:
                    survey[key] = _manifest_reference(
                        survey[key], root=root, directory=destination.parent,
                        field=f"data.survey.{key}",
                    )
        target = data.get("target")
        if isinstance(target, dict):
            # A grouped target: {path, subgroup}.
            if "path" in target:
                target["path"] = _manifest_reference(
                    target["path"], root=root, directory=destination.parent,
                    field="data.target.path",
                )
        for key in ("basis", "target", "reference_distribution", "predictions", "fitted"):
            if key in data and not (key == "target" and isinstance(data[key], dict)):
                data[key] = _manifest_reference(
                    data[key], root=root, directory=destination.parent,
                    field=f"data.{key}",
                )
    generation = config.get("generation")
    if isinstance(generation, dict):
        attribute_bank = generation.get("attribute_bank")
        if isinstance(attribute_bank, dict) and "source" in attribute_bank:
            attribute_bank["source"] = _manifest_reference(
                attribute_bank["source"], root=root,
                directory=destination.parent,
                field="generation.attribute_bank.source",
            )
    backend = config.get("backend")
    if isinstance(backend, dict) and "profile" in backend:
        backend["profile"] = _manifest_reference(
            backend["profile"], root=root, directory=destination.parent,
            field="backend.profile",
        )
    model = config.get("model")
    validation = model.get("validation") if isinstance(model, dict) else None
    folds = validation.get("folds") if isinstance(validation, dict) else None
    if isinstance(folds, dict) and "file" in folds:
        folds["file"] = _manifest_reference(
            folds["file"], root=root, directory=destination.parent,
            field="model.validation.folds.file",
        )
    return config


def _check_reference_combination(config: dict[str, Any]) -> None:
    """Refuse before saving what the fit refuses with a reference distribution."""

    data = config.get("data")
    if not isinstance(data, dict) or data.get("reference_distribution") is None:
        return
    model = config.get("model")
    if not isinstance(model, dict):
        return
    if model.get("method") == SIMPLE_AVERAGE:
        raise ValueError(
            "data.reference_distribution cannot be used with method "
            f"{SIMPLE_AVERAGE!r}, which fits no weights"
        )
    conflicts = [
        f"model.{name}" for name in REFERENCE_DISTRIBUTION_CONFLICTS if model.get(name)
    ]
    if conflicts:
        raise ValueError(
            "data.reference_distribution cannot be combined with " + ", ".join(conflicts)
        )


def _target_subgroups(path: Path) -> dict[str, Any] | None:
    """The subgroups a grouped target file offers, or None for a flat file."""

    payload = _read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("subgroups"), dict):
        return None
    default = payload.get("default_subgroup")
    return {
        "subgroups": sorted(str(name) for name in payload["subgroups"]),
        "default": default if isinstance(default, str) else None,
    }


def _run_summary(run_file: Path, root: Path) -> dict[str, Any]:
    run_dir = run_file.parent
    run = _read_json(run_file, {}) or {}
    metrics_payload = _read_json(run_dir / "metrics.json", {}) or {}
    usage_payload = _read_json(run_dir / "usage.json", {}) or {}
    diagnostics = _read_json(run_dir / "diagnostics.json", {}) or {}
    metrics = metrics_payload.get("metrics", [])
    usage = usage_payload.get("summary", {})
    model = diagnostics.get("model", {})
    if not isinstance(metrics, list):
        metrics = []
    return {
        "path": _relative(run_dir, root),
        "runId": str(run.get("run_id", run_dir.name)),
        "repeat": run.get("repeat"),
        "seed": run.get("seed"),
        "conditions": run.get("conditions", {}),
        "method": model.get("method") or (run.get("conditions", {}) or {}).get("method"),
        "solverStatus": model.get("status"),
        "metrics": [
            {
                "metric": item.get("metric"),
                "value": item.get("value"),
                "split": item.get("split"),
                "unit": item.get("unit"),
                "predictionPolicy": item.get("prediction_policy"),
                "qidsRequested": item.get("qids_requested"),
                "qidsUsed": item.get("qids_used"),
                "diagnostics": item.get("diagnostics", {}),
            }
            for item in metrics
            if isinstance(item, dict)
        ],
        "usage": usage if isinstance(usage, dict) else {},
        "questionCount": len(diagnostics.get("questions", []))
        if isinstance(diagnostics.get("questions"), list)
        else 0,
        "hasDiagnostics": bool(diagnostics),
        "population": _population_summary(
            run_dir,
            root,
            diagnostics=diagnostics,
            method=str(model.get("method") or (run.get("conditions", {}) or {}).get("method") or ""),
        ),
        "modifiedAt": _iso_timestamp(run_file.stat().st_mtime),
        "origin": "current",
        "sourceFormat": "p2p-canonical-v1",
        "sourceVersion": run.get("schema_version"),
        "capabilities": {
            "metrics": bool(metrics),
            "predictions": bool(diagnostics.get("questions")),
            "weights": bool(model.get("coefficients")),
            "generation": any(
                (run_dir / name).is_file()
                for name in (
                    "persistent_mode_entropy.csv",
                    "aggregate_entropy_trajectory.csv",
                    "question_entropy_trajectories.csv",
                )
            ),
            "folds": (run_dir / "folds.json").is_file(),
            "usage": bool(usage),
            "reports": False,
            "export": True,
            "reevaluate": False,
        },
        "compatibilityWarnings": [],
        "provenance": {"label": "Current P2P pipeline"},
    }


RUN_INDEX_VERSION = 3
_INDEX_FILES = {
    "current": ("run.json", "metrics.json", "usage.json", "diagnostics.json", "endowments.csv"),
    "legacy": ("summary.md", "summary_aggregate.csv", "token_usage_summary.json", "_freeze.yaml"),
}


def _library_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """The part of a run summary the library list shows, for the run index.

    Per-metric diagnostics and per-mode population counts are read from the run
    itself when it is opened; the list needs only metric values and the
    population's totals by template family.
    """

    compact = dict(summary)
    compact["metrics"] = [
        {key: value for key, value in metric.items() if key != "diagnostics"}
        for metric in summary.get("metrics", [])
    ]
    population = summary.get("population")
    if isinstance(population, dict) and isinstance(population.get("modes"), list):
        families: dict[str, dict[str, Any]] = {}
        for mode in population["modes"]:
            family = str(mode.get("family") or mode.get("mode") or "Other")
            bucket = families.setdefault(
                family, {"mode": family, "family": family, "generated": 0, "selected": 0, "antiTypes": 0}
            )
            for field in ("generated", "selected", "antiTypes"):
                bucket[field] += int(mode.get(field) or 0)
        compact["population"] = {**population, "modes": list(families.values())}
    return compact


def _index_stamp(directory: Path, origin: str) -> list[list[Any]]:
    """Timestamps and sizes of the files a run summary is read from."""

    stamp: list[list[Any]] = []
    for name in _INDEX_FILES[origin]:
        try:
            status = (directory / name).stat()
        except OSError:
            continue
        stamp.append([name, status.st_mtime_ns, status.st_size])
    return stamp


def _short(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{digits}g}"
    return str(value)


def _configuration_of(model_config: dict[str, Any]) -> str:
    """Name the Stage-2 configuration a model block expresses, or 'custom'."""

    chosen = {
        "group_penalty_strength": float(model_config.get("group_penalty_strength", 100.0)),
        "reference_respondent": bool(model_config.get("reference_respondent", False)),
        "intercept": bool(model_config.get("intercept", False)),
    }
    for configuration in STAGE2_CONFIGURATIONS:
        expected = {
            "group_penalty_strength": 100.0,
            "reference_respondent": False,
            "intercept": False,
            **{key: (float(value) if key == "group_penalty_strength" else value)
               for key, value in configuration["options"].items()},
        }
        if expected == chosen:
            return str(configuration["id"])
    return "custom"


def _error_answer_count(run_dir: Path, root: Path, run: dict[str, Any]) -> tuple[int | None, str]:
    """Count answers recorded as ``[ERROR] …`` (every retry failed)."""

    candidates = [run_dir / "responses.csv"]
    artifacts = run.get("source_artifacts") if isinstance(run, dict) else None
    responses = artifacts.get("responses") if isinstance(artifacts, dict) else None
    if isinstance(responses, dict) and isinstance(responses.get("path"), str):
        candidates.append(Path(responses["path"]))
    for path in candidates:
        path = path.resolve()
        if not path.is_file() or not _inside(path, root):
            continue
        count = 0
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.reader(handle):
                    count += sum(1 for cell in row if cell.startswith("[ERROR]"))
        except (OSError, UnicodeDecodeError, csv.Error):
            return None, _relative(path, root)
        return count, _relative(path, root)
    return None, ""


def _run_health(
    run_dir: Path,
    root: Path,
    *,
    run: dict[str, Any],
    diagnostics: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Plain-language checks and facts about one canonical run.

    Read-only: everything comes from the run's own records (diagnostics.json,
    metrics.json, folds.json, config.resolved.yaml, run.json).
    """

    model = diagnostics.get("model", {}) if isinstance(diagnostics, dict) else {}
    model = model if isinstance(model, dict) else {}
    resolved = _read_yaml(run_dir / "config.resolved.yaml")
    config = resolved.get("config") if isinstance(resolved.get("config"), dict) else {}
    model_config = config.get("model") if isinstance(config.get("model"), dict) else {}
    validation = (
        model_config.get("validation") if isinstance(model_config.get("validation"), dict) else {}
    )
    folds = _read_json(run_dir / "folds.json", {}) or {}
    method = model.get("method") or (run.get("conditions") or {}).get("method")
    fitted = bool(model) and method not in (None, SIMPLE_AVERAGE)
    checks: list[dict[str, str]] = []

    def add(level: str, label: str, detail: str) -> None:
        checks.append({"level": level, "label": label, "detail": detail})

    if fitted and model.get("status"):
        used = model.get("solver_used") or model.get("solver")
        if model.get("solver_fallback"):
            add("warning", "Solver fallback",
                f"The configured solver {model.get('solver')} failed; {used} produced "
                f"the final fit (status {model['status']}).")
        elif model["status"] != "optimal":
            add("warning", "Solver status", f"{used} finished with status {model['status']}.")
        else:
            add("ok", "Solver", f"{used} reached an optimal solution.")
    if model.get("cv_solver_fallbacks"):
        add("warning", "Cross-validation fallbacks",
            f"{model['cv_solver_fallbacks']} of {model.get('cv_fits')} cross-validation "
            "fits needed a fallback solver.")
    if model.get("alpha_at_grid_edge"):
        add("warning", "Penalty at grid edge",
            f"The selected penalty α = {_short(model.get('best_alpha'))} sits at the "
            f"{model['alpha_at_grid_edge']} of the searched grid, so the best value may lie "
            "beyond it. Widen model.alpha.")
    if model.get("l1_ratio_at_grid_edge"):
        add("warning", "Mixing ratio at grid edge",
            f"The selected l1_ratio = {_short(model.get('best_l1_ratio'))} sits at the "
            f"{model['l1_ratio_at_grid_edge']} of its grid.")
    within, count = model.get("cv_configs_within_one_se"), model.get("cv_config_count")
    if isinstance(within, int) and isinstance(count, int) and count > 1:
        if within > 1:
            add("info", "Flat validation curve",
                f"{within} of {count} penalty settings score within one standard error of "
                "the best, so the choice among them is not distinguishable from noise.")
        else:
            add("ok", "Penalty selection",
                f"The best of {count} settings is clear of the rest by more than one "
                "standard error.")
    if model.get("default_grids"):
        add("info", "Default penalty grid",
            f"No grid was configured for {model['default_grids']}; P2P's default grid "
            "was used.")
    if model.get("ignored_model_options"):
        add("warning", "Options with no effect",
            f"{model['ignored_model_options']}: accepted for compatibility with older "
            "manifests, but they change nothing.")
    anti = model.get("anti_type_count")
    if fitted and isinstance(anti, int) and anti > 0:
        add("info", "Anti-type agents",
            f"{anti} agents carry negative weight (total {_short(model.get('anti_type_mass'), 3)}). "
            "Signed weights are allowed by design.")
    for record in metrics:
        requested, used = record.get("qids_requested"), record.get("qids_used")
        if isinstance(requested, int) and isinstance(used, int) and used < requested:
            add("warning", "Incomplete coverage",
                f"{record.get('metric')} on the {record.get('split')} split used {used} of "
                f"{requested} questions.")
    distribution = next(
        (
            record for record in metrics
            if record.get("split") == "test"
            and isinstance(record.get("diagnostics"), dict)
            and "invalid_qids_before_transform" in record["diagnostics"]
        ),
        None,
    )
    if distribution is not None:
        detail = distribution["diagnostics"]
        policy = distribution.get("prediction_policy")
        if detail.get("invalid_qids_after_transform"):
            add("warning", "Invalid distributions remain",
                f"{detail['invalid_qids_after_transform']} test questions are still invalid "
                f"after {policy}.")
        if detail.get("negative_entry_count"):
            add("info", "Negative predicted mass",
                f"{detail['negative_entry_count']} predicted option values were negative "
                f"(total {_short(detail.get('negative_mass'), 3)}); {policy} set them to zero.")
        if detail.get("invalid_qids_before_transform"):
            scored = distribution.get("qids_used") or 1
            mean_gap = float(detail.get("raw_sum_deviation") or 0.0) / scored
            negligible = mean_gap < 1e-3
            add("ok" if negligible else "info", "Rescaled predictions",
                f"For {detail['invalid_qids_before_transform']} of "
                f"{distribution.get('qids_used')} test questions the predicted shares summed "
                f"to something other than one (mean gap {_short(mean_gap, 2)}"
                + (", negligible" if negligible else "")
                + f"); {policy} rescaled them for 1 − JSD, EMD and MCPA. Raw MSE is "
                "unaffected.")
        if detail.get("canonical_answers") is False:
            add("info", "Verbatim answer scoring",
                "Agents' answer text was scored verbatim (canonical_answers: false) "
                "rather than mapped to each option's own text.")
    errors, source = _error_answer_count(run_dir, root, run)
    if errors:
        add("warning", "Failed answers",
            f"{errors} answers in {source} are [ERROR] records: every retry failed, and "
            "the answer text names the provider's reason.")
    elif errors == 0:
        add("ok", "Answers", f"No failed answers in {source}.")

    facts: list[dict[str, Any]] = []

    def fact(label: str, value: Any, hint: str = "") -> None:
        if value not in (None, ""):
            facts.append({"label": label, "value": value if isinstance(value, str) else _short(value), "hint": hint})

    fact("Method", _method_label(str(method)) if method else None)
    if fitted:
        configuration = _configuration_of(model_config)
        label = next(
            (item["label"] for item in STAGE2_CONFIGURATIONS if item["id"] == configuration),
            "Custom combination",
        )
        fact("Stage-2 configuration", label, configuration)
        strategy = model.get("validation_strategy") or validation.get("strategy", "cv")
        if strategy == "holdout":
            fact("Penalty selection", f"Holdout on {model.get('holdout_split') or validation.get('holdout_split', 'valid')}")
        else:
            fact("Penalty selection", f"{validation.get('cv_folds', folds.get('n_splits', 5))}-fold cross-validation")
        fact("Selected α", model.get("best_alpha"))
        fact("Selected l1_ratio", model.get("best_l1_ratio"))
        fact("Solver", model.get("solver_used") or model.get("solver"))
        fact("Σβ (coefficient sum)", model.get("coefficient_sum"))
        fact("Reference respondent weight", model.get("reference_respondent_weight"))
        fact("Intercept", model.get("intercept"))
    fact("Train MSE", model.get("train_mse"), "In-sample, raw predictions, unweighted")
    if folds:
        method_name = folds.get("method")
        fact("Folds", f"{method_name}" + (f", seed {folds['seed']}" if folds.get("seed") is not None else "")
             + f", {folds.get('n_splits')} folds over {folds.get('question_count')} questions",
             "Every option of a question stays in one fold")
        fact("Fold record", str(folds.get("partition_digest", ""))[:23] + "…" if folds.get("partition_digest") else None,
             str(folds.get("source", "")))
    fact("Prediction route", diagnostics.get("distribution_route") if isinstance(diagnostics, dict) else None)
    fact("Cleaning policy", run.get("cleaning_policy"))
    fact("Config digest", (str(run.get("config_digest"))[:16] + "…") if run.get("config_digest") else None,
         str(run.get("config_digest") or ""))
    warnings = sum(item["level"] == "warning" for item in checks)
    return {
        "status": "attention" if warnings else "healthy",
        "warningCount": warnings,
        "checks": checks,
        "facts": facts,
    }


_AGGREGATE_ROW_FIELDS = {
    "metric", "unit", "prediction_policy", "center", "spread", "n", "missing_count", "estimand",
}


def _aggregate_name(directory: Path) -> str:
    return f"{directory.parent.name}/{directory.name}" if directory.name.startswith("_") else directory.name


def _aggregate_summary(directory: Path, root: Path) -> dict[str, Any]:
    validation = _read_json(directory / "aggregation_validation.json", {}) or {}
    summary = _read_json(directory / "summary.json", {}) or {}
    rows = summary.get("summary", []) if isinstance(summary, dict) else []
    rows = [row for row in rows if isinstance(row, dict)]
    spec = _read_yaml(directory / "aggregation.resolved.yaml").get("aggregate") or {}
    group_by = spec.get("group_by") if isinstance(spec, dict) else None
    keys = (
        [str(key) for key in group_by]
        if isinstance(group_by, list)
        else [key for key in (rows[0] if rows else {}) if key not in _AGGREGATE_ROW_FIELDS]
    )
    statistics = spec.get("statistics", {}) if isinstance(spec, dict) else {}
    marker = directory / "aggregation_validation.json"
    return {
        "path": _relative(directory, root),
        "name": _aggregate_name(directory),
        "status": validation.get("status", "unknown"),
        "runCount": validation.get("run_count"),
        "includedRunCount": validation.get("included_run_count"),
        "metricRows": validation.get("metric_row_count"),
        "groupKeys": keys,
        "groupCount": len({tuple(row.get(key) for key in keys) for row in rows}),
        "metrics": sorted({str(row.get("metric")) for row in rows if row.get("metric")}),
        "center": statistics.get("center") if isinstance(statistics, dict) else None,
        "spread": statistics.get("spread") if isinstance(statistics, dict) else None,
        "digest": validation.get("aggregation_digest"),
        "modifiedAt": _iso_timestamp(marker.stat().st_mtime),
    }


def _aggregate_detail(directory: Path, root: Path) -> dict[str, Any]:
    validation = _read_json(directory / "aggregation_validation.json", {}) or {}
    summary = _read_json(directory / "summary.json", {}) or {}
    rows = summary.get("summary", []) if isinstance(summary, dict) else []
    return {
        "summary": _aggregate_summary(directory, root),
        "rows": [row for row in rows if isinstance(row, dict)],
        "checks": validation.get("checks", {}) if isinstance(validation.get("checks"), dict) else {},
        "missingRows": validation.get("missing_rows", []) if isinstance(validation.get("missing_rows"), list) else [],
        "missingPolicy": validation.get("missing_policy"),
        "files": sorted(path.name for path in directory.iterdir() if path.is_file()),
    }


def _budget_summary(path: Path, root: Path) -> dict[str, Any]:
    payload = _read_json(path, {}) or {}
    scenarios = payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        scenarios = []
    return {
        "path": _relative(path.parent, root),
        "configSource": payload.get("config_source"),
        "confidence": payload.get("confidence"),
        "historicalRuns": payload.get("historical_runs"),
        "pilotRecords": payload.get("pilot_records"),
        "scenarios": [
            {
                "name": item.get("name"),
                "model": next(
                    (
                        module.get("model_name")
                        for module in item.get("modules", [])
                        if isinstance(module, dict)
                    ),
                    None,
                ),
                "tokens": item.get("total_tokens", {}),
                "cost": item.get("total_cost_usd", {}),
                "status": item.get("budget_status"),
                "budgetUsd": item.get("budget_usd"),
            }
            for item in scenarios
            if isinstance(item, dict)
        ],
        "modifiedAt": _iso_timestamp(path.stat().st_mtime),
    }


def _survey_summary(path: Path, root: Path) -> dict[str, Any]:
    schema = _read_yaml(path)
    return {
        "path": _relative(path, root),
        "name": path.stem.replace("_", " ").title(),
        "questionId": schema.get("question_id"),
        "questionText": schema.get("question_text"),
        "optionMapping": schema.get("code_to_answer"),
        "modifiedAt": _iso_timestamp(path.stat().st_mtime),
    }


def _backend_summary(path: Path, root: Path) -> dict[str, Any]:
    payload = _read_yaml(path)
    entries = _backend_entries({"backend": payload})
    backend_type = entries[0]["type"] if entries else "unknown"
    env_name = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }.get(backend_type)
    return {
        "path": _relative(path, root),
        "name": path.stem.replace("_", " ").title(),
        "type": backend_type,
        "local": backend_type in LOCAL_BACKENDS,
        "credentialName": env_name,
        "credentialAvailable": bool(env_name and os.environ.get(env_name)),
        "modules": entries,
        "modifiedAt": _iso_timestamp(path.stat().st_mtime),
    }


def _module_installed(name: str) -> bool:
    """Whether a module can be imported, without importing it.

    For a dotted name, find_spec imports the parent package and raises
    ModuleNotFoundError when that is missing (``google.genai`` without
    ``google``), so the missing parent is answered here as not installed.
    """

    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _backend_capability(name: str) -> dict[str, Any]:
    dependency = {
        "openai": "openai",
        "deepseek": "openai",
        "gemini": "google.genai",
    }.get(name)
    credential = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }.get(name)
    installed = dependency is None or _module_installed(dependency)
    local = name in LOCAL_BACKENDS
    # The adapter's own constructor is the source of its options, so a new
    # option appears here without a second list to maintain.
    options: list[dict[str, Any]] = []
    model_required = False
    try:
        parameters = inspect.signature(BACKENDS.get(name)).parameters
    except (TypeError, ValueError, KeyError):
        parameters = {}
    for parameter in parameters.values():
        if parameter.name == "model_name":
            model_required = parameter.default is inspect.Parameter.empty
            continue
        annotation = str(parameter.annotation)
        if (
            parameter.name in _BACKEND_INTERNAL_PARAMETERS
            or parameter.kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            # Injected callables (an HTTP transport, a config factory) are test
            # and integration hooks, not settings a manifest can express.
            or "Callable" in annotation
            or "Transport" in annotation
            or "TokenUsage" in annotation
        ):
            continue
        default = parameter.default
        options.append(
            {
                "name": parameter.name,
                "default": None
                if default is inspect.Parameter.empty
                else default
                if isinstance(default, (str, int, float, bool)) or default is None
                else repr(default),
                "required": default is inspect.Parameter.empty,
            }
        )
    return {
        "id": name,
        "local": local,
        "metered": not local,
        "dependency": dependency,
        "dependencyAvailable": installed,
        "credentialName": credential,
        "credentialAvailable": bool(credential and os.environ.get(credential)),
        "modelRequired": model_required,
        "options": options,
    }


# Constructor parameters that are wiring rather than configuration.
_BACKEND_INTERNAL_PARAMETERS = {"self", "client", "api_key"}


def _requested_solver(value: Any, command: str) -> str | None:
    """Validate an optional solver choice; None keeps P2P's default."""

    if value in (None, ""):
        return None
    if not isinstance(value, str) or value not in SOLVERS:
        raise ValueError(f"solver must be one of {', '.join(SOLVERS)}")
    if command not in SOLVER_COMMANDS:
        raise ValueError(f"p2p {command} does not fit Stage 2 and takes no solver")
    return value


def _method_label(name: str) -> str:
    special = {"elasticnet": "Elastic net", "weighted-elasticnet": "Weighted elastic net"}
    return special.get(name, name.replace("-", " ").replace("_", " ").capitalize())


def _dataset_summary(path: Path, root: Path) -> dict[str, Any]:
    payload = _read_yaml(path)
    source = payload.get("source")
    schema = payload.get("schema")
    target = payload.get("target")
    return {
        "id": path.stem,
        "name": str(payload.get("name") or path.stem.replace("_", " ").title()),
        "path": _relative(path, root),
        "source": source,
        "schema": schema,
        "target": target,
        "questionCount": payload.get("question_count"),
        "splitCounts": payload.get("split_counts", {}),
        "digests": payload.get("digests", {}),
        "registeredAt": payload.get("registered_at"),
        "portable": bool(payload.get("portable", True)),
    }


def _page(
    items: list[dict[str, Any]],
    *,
    cursor: str | None,
    limit: int,
) -> dict[str, Any]:
    try:
        offset = max(0, int(cursor or "0"))
    except ValueError as error:
        raise ValueError("cursor must be a nonnegative integer") from error
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    visible = items[offset : offset + limit]
    next_offset = offset + len(visible)
    return {
        "items": visible,
        "cursor": str(offset),
        "nextCursor": str(next_offset) if next_offset < len(items) else None,
        "total": len(items),
    }


@dataclass
class StudioJob:
    job_id: str
    command: str
    config: str
    output: str
    process: subprocess.Popen[str] | None = None
    pid: int | None = None
    created_at: float = field(default_factory=time.time)
    status: str = "running"
    log: list[str] = field(default_factory=list)
    recovery: str = "live"
    solver: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def as_dict(self) -> dict[str, Any]:
        code = self.process.poll() if self.process is not None else None
        if code is not None and self.status not in {"stopped", "failed", "complete"}:
            self.status = "complete" if code == 0 else "failed"
            self.recovery = "inspect" if code == 0 else "new-output"
        with self.lock:
            log = self.log[-400:]
        return {
            "id": self.job_id,
            "command": self.command,
            "config": self.config,
            "output": self.output,
            "status": self.status,
            "exitCode": code,
            "pid": self.pid,
            "createdAt": _iso_timestamp(self.created_at),
            "log": log,
            "recovery": self.recovery,
            "solver": self.solver,
            "durable": True,
        }


class StudioState:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.jobs: dict[str, StudioJob] = {}
        self.lock = threading.Lock()
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self.job_store = self.root / "outputs" / ".studio" / "jobs"
        self.annotation_store = self.root / "outputs" / ".studio" / "annotations.json"
        # Persistent run index: summaries keyed by the timestamps and sizes of
        # the files they are read from, so only new or changed runs are read.
        self.index_store = self.root / "outputs" / ".studio" / "run_index.json.gz"
        self._index_entries: dict[str, dict[str, Any]] | None = None
        self._index_lock = threading.Lock()
        self._index_ready = threading.Event()
        self._load_jobs()

    def annotations(self) -> dict[str, dict[str, Any]]:
        payload = _read_json(self.annotation_store, {}) or {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(path): value
            for path, value in payload.items()
            if isinstance(value, dict)
        }

    def update_annotation(self, body: dict[str, Any]) -> dict[str, Any]:
        relative_path = str(body.get("path", ""))
        path = (self.root / relative_path).resolve()
        output_root = (self.root / "outputs").resolve()
        if not _inside(path, output_root) or not path.is_dir():
            raise ValueError("Annotated records must be directories inside outputs/")
        label = str(body.get("label", "")).strip()
        if len(label) > 80:
            raise ValueError("Annotation labels must be 80 characters or fewer")
        pinned = bool(body.get("pinned", False))
        annotations = self.annotations()
        key = _relative(path, self.root)
        if not label and not pinned:
            annotations.pop(key, None)
        else:
            annotations[key] = {
                "label": label,
                "pinned": pinned,
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            }
        self.annotation_store.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.annotation_store.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(annotations, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.annotation_store)
        return {"path": key, "annotation": annotations.get(key, {})}

    def _load_jobs(self) -> None:
        if not self.job_store.is_dir():
            return
        for record_path in sorted(self.job_store.glob("*.json")):
            payload = _read_json(record_path, {}) or {}
            if not isinstance(payload, dict):
                continue
            try:
                created_at = float(payload.get("createdTimestamp", record_path.stat().st_mtime))
                job = StudioJob(
                    job_id=str(payload["id"]),
                    command=str(payload["command"]),
                    config=str(payload["config"]),
                    output=str(payload["output"]),
                    pid=payload.get("pid"),
                    created_at=created_at,
                    status=str(payload.get("status", "interrupted")),
                    log=[str(line) for line in payload.get("log", [])][-400:],
                    recovery=str(payload.get("recovery", "new-output")),
                    solver=payload.get("solver"),
                )
            except (KeyError, TypeError, ValueError):
                continue
            output = (self.root / job.output).resolve()
            if (output / "run.json").is_file():
                job.status = "complete"
                job.recovery = "inspect"
            elif job.status in {"running", "paused"}:
                job.status = "interrupted"
                job.recovery = "checkpoint" if any(output.glob("*checkpoint*")) else "new-output"
                job.log.append("Studio restarted; the original process is no longer attached.")
            self.jobs[job.job_id] = job
            self._save_job(job)

    def _save_job(self, job: StudioJob) -> None:
        self.job_store.mkdir(parents=True, exist_ok=True)
        payload = job.as_dict()
        payload["createdTimestamp"] = job.created_at
        destination = self.job_store / f"{job.job_id}.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    def _manifest_paths(self) -> list[Path]:
        if "manifest_paths" in self._cache:
            return self._cache["manifest_paths"]  # type: ignore[return-value]
        paths = [
            path
            for path in (self.root / "config").glob("**/*.yaml")
            if any(part in {"central", "experiments", "smoke"} for part in path.parts)
        ]
        self._cache["manifest_paths"] = paths  # type: ignore[assignment]
        return paths

    def _recent_manifest_summaries(self, limit: int) -> list[dict[str, Any]]:
        # Only counts and the most recent manifests are shown; parsing every
        # generated run manifest (tens of thousands) would dominate start-up.
        paths = sorted(self._manifest_paths(), key=lambda path: path.stat().st_mtime, reverse=True)
        return [_manifest_summary(path, self.root) for path in paths[:limit]]

    def _manifest_summaries(self) -> list[dict[str, Any]]:
        if "manifests" in self._cache:
            return self._cache["manifests"]
        manifest_paths = sorted((self.root / "config").glob("**/*.yaml"))
        summaries = [
            _manifest_summary(path, self.root)
            for path in manifest_paths
            if any(part in {"central", "experiments", "smoke"} for part in path.parts)
        ]
        self._cache["manifests"] = summaries
        return summaries

    def _report_summaries(self) -> list[dict[str, Any]]:
        if "reports" in self._cache:
            return self._cache["reports"]
        report_dirs = {
            path.parent for path in self._outputs("summary")
        }
        sorted_dirs = sorted(
            report_dirs,
            key=lambda path: max(item.stat().st_mtime for item in path.glob("summary*.md")),
            reverse=True,
        )
        summaries = [_report_summary(path, self.root) for path in sorted_dirs]
        self._cache["reports"] = summaries
        return summaries

    def _recent_report_summaries(self, limit: int) -> list[dict[str, Any]]:
        report_dirs = {
            path.parent for path in self._outputs("summary")
        }
        sorted_dirs = sorted(
            report_dirs,
            key=lambda path: max(item.stat().st_mtime for item in path.glob("summary*.md")),
            reverse=True,
        )
        return [_report_summary(path, self.root) for path in sorted_dirs[:limit]]

    def _outputs(self, kind: str) -> list[Path]:
        """Record files under outputs/, found in one walk and cached per refresh.

        kind: run (run.json), summary (summary*.md), paper_summary (summary*.md
        under outputs/paper), budget (budget.json), aggregate
        (aggregation_validation.json).
        """

        scan = self._cache.get("outputs_scan")
        if scan is None:
            found: dict[str, list[Path]] = {
                "run": [], "summary": [], "paper_summary": [], "budget": [], "aggregate": [],
            }
            outputs = self.root / "outputs"
            paper = outputs / "paper"
            for directory, _, filenames in os.walk(outputs):
                base = Path(directory)
                for name in filenames:
                    if name == "run.json":
                        found["run"].append(base / name)
                    elif name == "budget.json":
                        found["budget"].append(base / name)
                    elif name == "aggregation_validation.json":
                        found["aggregate"].append(base / name)
                    elif name.startswith("summary") and name.endswith(".md"):
                        found["summary"].append(base / name)
                        if base == paper or paper in base.parents:
                            found["paper_summary"].append(base / name)
            scan = found
            self._cache["outputs_scan"] = scan  # type: ignore[assignment]
        return list(scan[kind])  # type: ignore[index]

    def _legacy_run_dirs(self) -> list[Path]:
        return sorted(
            {path.parent for path in self._outputs("paper_summary")}
        )

    def _load_index_entries(self) -> dict[str, dict[str, Any]]:
        if self._index_entries is None:
            try:
                with gzip.open(self.index_store, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, ValueError, EOFError):
                payload = {}
            payload = payload if isinstance(payload, dict) else {}
            entries = payload.get("runs") if payload.get("version") == RUN_INDEX_VERSION else None
            self._index_entries = entries if isinstance(entries, dict) else {}
        return self._index_entries

    def _save_index_entries(self, entries: dict[str, dict[str, Any]]) -> None:
        try:
            self.index_store.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.index_store.with_suffix(".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=5) as handle:
                json.dump(
                    {"version": RUN_INDEX_VERSION, "runs": entries},
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            temporary.replace(self.index_store)
        except OSError:
            # A read-only workspace still works; it just re-reads next time.
            pass

    def _run_summaries(self) -> list[dict[str, Any]]:
        with self._index_lock:
            if "runs" in self._cache:
                return self._cache["runs"]
            entries = self._load_index_entries()
            fresh: dict[str, dict[str, Any]] = {}
            changed = False
            sources = [
                *((path.parent, "current") for path in self._outputs("run")),
                *((path, "legacy") for path in self._legacy_run_dirs()),
            ]
            for directory, origin in sources:
                key = _relative(directory, self.root)
                stamp = _index_stamp(directory, origin)
                cached = entries.get(key)
                if cached is not None and cached.get("stamp") == stamp:
                    fresh[key] = cached
                    continue
                try:
                    summary = (
                        _run_summary(directory / "run.json", self.root)
                        if origin == "current"
                        else _legacy_run_summary(directory, self.root)
                    )
                except (OSError, ValueError):
                    continue
                fresh[key] = {"stamp": stamp, "summary": _library_summary(summary)}
                changed = True
            if changed or len(fresh) != len(entries):
                self._index_entries = fresh
                self._save_index_entries(fresh)
            runs = [dict(entry["summary"]) for entry in fresh.values()]
            runs.sort(key=lambda item: item["modifiedAt"], reverse=True)
            self._cache["runs"] = runs
            self._index_ready.set()
            return runs

    def warm_index(self) -> None:
        """Build the run index in the background so the library opens quickly."""

        threading.Thread(target=self._run_summaries, daemon=True).start()

    def _newest_page(self, *, cursor: str | None, limit: int, origin: str) -> dict[str, Any]:
        """Newest-first page without the full index: sort by file time, read one page."""

        candidates: list[tuple[float, Path, str]] = []
        if origin in {"all", "current"}:
            candidates += [
                (path.stat().st_mtime, path.parent, "current")
                for path in self._outputs("run")
            ]
        if origin in {"all", "legacy"}:
            for directory in self._legacy_run_dirs():
                stamps = [item.stat().st_mtime for item in directory.glob("summary*.md")]
                candidates.append((max(stamps), directory, "legacy"))
        candidates.sort(key=lambda item: item[0], reverse=True)
        page = _page(candidates, cursor=cursor, limit=limit)  # type: ignore[arg-type]
        page["items"] = [
            _run_summary(directory / "run.json", self.root)
            if kind == "current"
            else _legacy_run_summary(directory, self.root)
            for _, directory, kind in page["items"]
        ]
        page["indexing"] = True
        return page

    def _recent_run_summaries(self, limit: int) -> list[dict[str, Any]]:
        run_files = sorted(
            self._outputs("run"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        report_dirs = {
            path.parent for path in self._outputs("paper_summary")
        }
        legacy_dirs = sorted(
            report_dirs,
            key=lambda path: max(item.stat().st_mtime for item in path.glob("summary*.md")),
            reverse=True,
        )
        candidates = [
            *(_run_summary(path, self.root) for path in run_files[:limit]),
            *(_legacy_run_summary(path, self.root) for path in legacy_dirs[:limit]),
        ]
        candidates.sort(key=lambda item: item["modifiedAt"], reverse=True)
        return candidates[:limit]

    def _aggregate_summaries(self) -> list[dict[str, Any]]:
        if "aggregates" in self._cache:
            return self._cache["aggregates"]
        markers = sorted(
            self._outputs("aggregate"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        summaries = [_aggregate_summary(marker.parent, self.root) for marker in markers]
        self._cache["aggregates"] = summaries
        return summaries

    def list_aggregates(
        self,
        *,
        cursor: str | None = None,
        limit: int = 30,
        query: str = "",
    ) -> dict[str, Any]:
        normalized_query = query.strip().lower()
        items = [
            item
            for item in self._aggregate_summaries()
            if not normalized_query
            or normalized_query in " ".join(
                [item["path"], item["name"], *item["groupKeys"], *item["metrics"]]
            ).lower()
        ]
        return _page(items, cursor=cursor, limit=limit)

    def aggregate_detail(self, relative_path: str) -> dict[str, Any]:
        path = (self.root / relative_path).resolve()
        if not _inside(path, self.root / "outputs"):
            raise ValueError("Aggregate paths must stay inside outputs/")
        if not (path / "aggregation_validation.json").is_file():
            raise FileNotFoundError(relative_path)
        return _aggregate_detail(path, self.root)

    def workspace(self) -> dict[str, Any]:
        if "runs" in self._cache:
            current_count = sum(item["origin"] == "current" for item in self._cache["runs"])
            historical_count = sum(item["origin"] == "legacy" for item in self._cache["runs"])
        else:
            current_count = sum(1 for _ in self._outputs("run"))
            historical_count = len(
                {path.parent for path in self._outputs("paper_summary")}
            )
        report_count = (
            len(self._cache["reports"])
            if "reports" in self._cache
            else len({path.parent for path in self._outputs("summary")})
        )
        return {
            "repository": self.root.name,
            "root": str(self.root),
            "localOnly": True,
            "counts": {
                "runs": current_count + historical_count,
                "currentRuns": current_count,
                "historicalRuns": historical_count,
                "reports": report_count,
                "aggregates": len(self._aggregate_summaries()),
                "manifests": len(self._manifest_paths()),
                "datasets": len(list((self.root / "config" / "datasets").glob("*.yaml"))),
            },
            "privacy": {
                "boundary": "repository",
                "network": "No workspace data is uploaded by Studio itself.",
                "providers": "Configured model backends may make explicit provider calls.",
            },
        }

    def capabilities(self) -> dict[str, Any]:
        model_names = MODELS.names()
        return {
            "commands": [
                {"id": name, "guided": name in {"pipeline", "generate", "regress", "evaluate"}}
                for name in COMMANDS.names()
            ],
            "workflows": [
                {"id": "pipeline", "label": "Full pipeline", "needsGeneration": True, "needsModel": True, "needsTarget": True},
                {"id": "generate", "label": "Generation only", "needsGeneration": True, "needsModel": False, "needsTarget": False},
                {"id": "regress", "label": "Fit an existing proxy basis", "needsGeneration": False, "needsModel": True, "needsTarget": True},
                {"id": "evaluate", "label": "Evaluate external predictions", "needsGeneration": False, "needsModel": False, "needsTarget": True},
            ],
            "generationModes": [
                {"id": "active", "label": "Active", "advanced": False},
                {"id": "vanilla", "label": "Vanilla", "advanced": False},
            ],
            "models": [
                {
                    "id": name,
                    "label": _method_label(name),
                    # Weighted variants are reached through model.block_weighting,
                    # not as methods of their own.
                    "guided": name in GUIDED_METHODS,
                    "controls": list(GUIDED_METHODS.get(name, {}).get("controls", [])),
                    "description": GUIDED_METHODS.get(name, {}).get("description", ""),
                }
                for name in model_names
            ]
            + [
                {
                    "id": name,
                    "label": alias["label"],
                    "guided": True,
                    "controls": list(alias["controls"]),
                    "description": alias["description"],
                    "method": alias["method"],
                    "options": dict(alias["options"]),
                    "zeroThreshold": alias["zeroThreshold"],
                }
                for name, alias in GUIDED_ALIASES.items()
                if alias["method"] in model_names
            ],
            "backends": [_backend_capability(name) for name in BACKENDS.names()],
            "metrics": [
                {"id": name, **METRIC_DESCRIPTIONS[name]}
                for name in (
                    *[item for item in METRIC_ORDER if item in SUPPORTED_METRICS],
                    *sorted(SUPPORTED_METRICS - set(METRIC_ORDER)),
                )
                if name in METRIC_DESCRIPTIONS
            ],
            "stage2": {
                "configurations": [dict(item) for item in STAGE2_CONFIGURATIONS],
                "defaultConfiguration": STAGE2_CONFIGURATIONS[0]["id"],
                "validationStrategies": list(VALIDATION_STRATEGIES),
                "foldMethods": list(FOLD_METHODS),
                "defaultFoldMethod": DEFAULT_FOLD_METHOD,
                "alphaGrid": dict(DEFAULT_ALPHA_GRID),
                "alphaGridValues": [min(expand_grid(DEFAULT_ALPHA_GRID)), max(expand_grid(DEFAULT_ALPHA_GRID))],
                "l1RatioGrid": dict(DEFAULT_L1_RATIO_GRID),
                "solvers": list(SOLVERS),
                "defaultSolver": DEFAULT_SOLVER,
                "solverFallbacks": list(DEFAULT_SOLVER_FALLBACKS),
                "solverCommands": sorted(SOLVER_COMMANDS),
                "zeroThreshold": DEFAULT_ZERO_THRESHOLD,
                # Re-anchoring to a reference distribution (data.reference_distribution).
                "reference": {
                    "conflicts": list(REFERENCE_DISTRIBUTION_CONFLICTS),
                    "excludedMethods": [SIMPLE_AVERAGE],
                },
            },
            "evaluation": {
                "distributionPolicies": [
                    {"id": name, **DISTRIBUTION_POLICY_DESCRIPTIONS[name]}
                    for name in DISTRIBUTION_POLICY_ORDER
                    if name in _SUPPORTED_DISTRIBUTION_POLICIES
                ],
                "invalidDistribution": [dict(item) for item in INVALID_DISTRIBUTION_OPTIONS],
            },
            "defaults": {
                "seed": 101,
                "generationMode": "active",
                "targetN": 300,
                "modelMethod": "lasso" if "lasso" in model_names else (model_names[0] if model_names else None),
                "cvFolds": 5,
                "holdoutSplit": "valid",
                "metrics": [item for item in METRIC_ORDER if item in SUPPORTED_METRICS],
                "distributionPredictionPolicy": "clip_and_normalize",
                "invalidDistribution": "error",
                "msePredictionPolicy": "raw",
            },
            "processControls": {
                "pause": os.name == "posix" and hasattr(signal, "SIGSTOP"),
                "resume": os.name == "posix" and hasattr(signal, "SIGCONT"),
                "stop": True,
                "platform": platform.system(),
            },
        }

    def bootstrap(self) -> dict[str, Any]:
        # A user-initiated workspace refresh rebuilds the disk index once.
        # Subsequent paginated requests reuse that immutable snapshot.
        self._cache.clear()
        manifests = self._recent_manifest_summaries(40)
        annotations = self.annotations()
        runs = self._recent_run_summaries(40)
        reports = self._recent_report_summaries(40)
        for item in [*runs, *reports]:
            item["annotation"] = annotations.get(item["path"], {})
        budget_files = list(self._outputs("budget"))
        budget_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        surveys_dir = self.root / "config" / "surveys"
        backends_dir = self.root / "config" / "backends"
        workspace = self.workspace()
        return {
            "repository": self.root.name,
            "root": str(self.root),
            "localOnly": True,
            "workspace": workspace,
            "capabilities": self.capabilities(),
            "manifests": manifests,
            "runs": runs[:40],
            "reports": reports[:40],
            "budgets": [_budget_summary(path, self.root) for path in budget_files[:100]],
            "surveys": [
                _survey_summary(path, self.root)
                for path in sorted(surveys_dir.glob("*.yaml"))
            ],
            "backends": [
                _backend_summary(path, self.root)
                for path in sorted(backends_dir.glob("*.yaml"))
            ],
            "datasets": self.datasets(),
            "jobs": [job.as_dict() for job in self.jobs.values()],
            "annotations": annotations,
        }

    def list_runs(
        self,
        *,
        cursor: str | None = None,
        limit: int = 30,
        origin: str = "all",
        query: str = "",
        sort: str = "newest",
    ) -> dict[str, Any]:
        if origin not in {"all", "current", "legacy"}:
            raise ValueError("origin must be all, current, or legacy")
        if sort not in {"newest", "pinned", "mse", "one_minus_jsd", "emd", "mcpa"}:
            raise ValueError("unsupported run sort")
        annotations = self.annotations()
        normalized_query = query.strip().lower()
        if (
            "runs" not in self._cache
            and not self._index_ready.is_set()
            and sort == "newest"
            and not normalized_query
        ):
            page = self._newest_page(cursor=cursor, limit=limit, origin=origin)
            for item in page["items"]:
                item["annotation"] = annotations.get(item["path"], {})
            return page
        items = [
            item
            for item in self._run_summaries()
            if (origin == "all" or item["origin"] == origin)
            and (
                not normalized_query
                or normalized_query in " ".join(
                    str(value)
                    for value in (
                        item.get("runId"), item.get("path"), item.get("method"),
                        (item.get("provenance") or {}).get("label"),
                        (annotations.get(item.get("path", "")) or {}).get("label"),
                    )
                ).lower()
            )
        ]
        if sort == "pinned":
            items.sort(key=lambda item: item.get("modifiedAt", ""), reverse=True)
            items.sort(
                key=lambda item: bool(
                    (annotations.get(item["path"]) or {}).get("pinned")
                ),
                reverse=True,
            )
        elif sort != "newest":
            higher_is_better = sort in {"one_minus_jsd", "mcpa"}

            def metric_value(item: dict[str, Any]) -> float | None:
                metrics = item.get("metrics", [])
                preferred = next(
                    (
                        metric.get("value")
                        for metric in metrics
                        if metric.get("metric") == sort and metric.get("split") == "test"
                    ),
                    None,
                )
                if preferred is None:
                    preferred = next(
                        (
                            metric.get("value")
                            for metric in reversed(metrics)
                            if metric.get("metric") == sort
                        ),
                        None,
                    )
                return _number(preferred)

            items.sort(
                key=lambda item: (
                    metric_value(item) is None,
                    -metric_value(item)
                    if higher_is_better and metric_value(item) is not None
                    else metric_value(item) or 0,
                    item.get("runId", ""),
                )
            )
        page = _page(items, cursor=cursor, limit=limit)
        for item in page["items"]:
            item["annotation"] = annotations.get(item["path"], {})
        return page

    def list_reports(
        self,
        *,
        cursor: str | None = None,
        limit: int = 30,
        query: str = "",
        sort: str = "newest",
    ) -> dict[str, Any]:
        if sort not in {"newest", "pinned"}:
            raise ValueError("unsupported report sort")
        normalized_query = query.strip().lower()
        annotations = self.annotations()
        items = [
            item
            for item in self._report_summaries()
            if not normalized_query
            or normalized_query in " ".join(
                [item["name"], item["path"], *item.get("models", []), str((annotations.get(item["path"]) or {}).get("label", ""))]
            ).lower()
        ]
        if sort == "pinned":
            items.sort(key=lambda item: item.get("modifiedAt", ""), reverse=True)
            items.sort(
                key=lambda item: bool(
                    (annotations.get(item["path"]) or {}).get("pinned")
                ),
                reverse=True,
            )
        page = _page(items, cursor=cursor, limit=limit)
        for item in page["items"]:
            item["annotation"] = annotations.get(item["path"], {})
        return page

    def datasets(self) -> list[dict[str, Any]]:
        directory = self.root / "config" / "datasets"
        return [_dataset_summary(path, self.root) for path in sorted(directory.glob("*.yaml"))]

    def check_backend(self, body: dict[str, Any]) -> dict[str, Any]:
        """Validate that a backend profile can be selected without exposing secrets.

        This is deliberately a readiness check rather than a paid inference call.
        Remote providers are checked for their SDK and credential boundary; local
        runtimes are checked when an experiment actually invokes them.
        """

        profile = _repository_path(
            body.get("path"), root=self.root, field="backend profile"
        )
        backend_root = (self.root / "config" / "backends").resolve()
        if not _inside(profile, backend_root) or profile.suffix not in {".yaml", ".yml"}:
            raise ValueError("Backend profile must be a YAML file inside config/backends/")
        summary = _backend_summary(profile, self.root)
        capability = _backend_capability(str(summary["type"]))
        checks = [
            {
                "label": "Profile",
                "ok": bool(summary["modules"]),
                "detail": f"{len(summary['modules'])} configured module(s)",
            },
            {
                "label": "Python dependency",
                "ok": bool(capability["dependencyAvailable"]),
                "detail": capability["dependency"] or "Built into P2P",
            },
        ]
        if capability["local"]:
            checks.append(
                {
                    "label": "Provider boundary",
                    "ok": True,
                    "detail": "Local backend; runtime availability is confirmed when a run starts",
                }
            )
        else:
            checks.append(
                {
                    "label": "Credential",
                    "ok": bool(capability["credentialAvailable"]),
                    "detail": capability["credentialName"] or "No credential variable declared",
                }
            )
        return {
            "path": summary["path"],
            "name": summary["name"],
            "type": summary["type"],
            "ready": all(bool(item["ok"]) for item in checks),
            "checks": checks,
            "networkCallMade": False,
        }

    def manifest_detail(self, relative_path: str) -> dict[str, Any]:
        path = (self.root / relative_path).resolve()
        config_root = (self.root / "config").resolve()
        if not _inside(path, config_root) or path.suffix not in {".yaml", ".yml"}:
            raise ValueError("Manifest must be a YAML file inside config/")
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        resolved = Config.load(path).to_dict()
        data = resolved.get("data", {})
        generation = resolved.get("generation", {})
        backend = resolved.get("backend", {})

        def relative(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            candidate = Path(value)
            if candidate.is_absolute() and _inside(candidate, self.root):
                return _relative(candidate, self.root)
            return value

        if isinstance(data, dict):
            survey = data.get("survey")
            if isinstance(survey, dict):
                for key in ("source", "schema"):
                    if key in survey:
                        survey[key] = relative(survey[key])
            for key in ("basis", "target", "reference_distribution", "predictions", "fitted"):
                if key in data:
                    data[key] = relative(data[key])
            target = data.get("target")
            if isinstance(target, dict) and "path" in target:
                target["path"] = relative(target["path"])
        if isinstance(generation, dict):
            bank = generation.get("attribute_bank")
            if isinstance(bank, dict) and "source" in bank:
                bank["source"] = relative(bank["source"])
        if isinstance(backend, dict) and "profile" in backend:
            backend["profile"] = relative(backend["profile"])
        model = resolved.get("model")
        validation = model.get("validation") if isinstance(model, dict) else None
        folds = validation.get("folds") if isinstance(validation, dict) else None
        if isinstance(folds, dict) and "file" in folds:
            folds["file"] = relative(folds["file"])

        if isinstance(data, dict) and data.get("predictions"):
            command = "evaluate"
        elif isinstance(data, dict) and data.get("basis"):
            command = "regress"
        elif isinstance(generation, dict) and resolved.get("model"):
            command = "pipeline"
        elif isinstance(generation, dict):
            command = "generate"
        else:
            command = "pipeline"
        return {
            "path": _relative(path, self.root),
            "command": command,
            "config": resolved,
            "summary": _manifest_summary(path, self.root),
        }

    def inspect_dataset(self, body: dict[str, Any]) -> dict[str, Any]:
        source = _repository_path(
            body.get("source"), root=self.root, field="source"
        )
        if source.suffix.lower() != ".csv":
            raise ValueError("source must be a CSV file")

        mapping = body.get("mapping")
        schema_path: Path | None = None
        if body.get("schema"):
            schema_path = _repository_path(
                body.get("schema"), root=self.root, field="schema"
            )
            mapping = _read_yaml(schema_path)
        if not isinstance(mapping, dict):
            mapping = {}

        rows = _read_csv(source, limit=10_000)
        with source.open(encoding="utf-8-sig", newline="") as handle:
            headers = list((csv.DictReader(handle).fieldnames or []))

        def infer(logical: str, candidates: tuple[str, ...]) -> str | None:
            configured = mapping.get(logical)
            if isinstance(configured, str) and configured in headers:
                return configured
            lowered = {header.lower(): header for header in headers}
            return next((lowered[name] for name in candidates if name in lowered), None)

        resolved_mapping = {
            "question_id": infer("question_id", ("qid", "question_id", "id")),
            "question_text": infer("question_text", ("question", "question_text", "text")),
            "code_to_answer": infer("code_to_answer", ("option_mapping", "options", "code_to_answer")),
            "split": infer("split", ("split", "partition", "set")),
        }
        errors: list[str] = []
        warnings: list[str] = []
        for logical in ("question_id", "question_text", "code_to_answer", "split"):
            if not resolved_mapping[logical]:
                errors.append(f"Could not map required field {logical!r} to a CSV column.")

        qids: list[str] = []
        split_counts: dict[str, int] = {}
        invalid_option_rows: list[int] = []
        empty_question_rows: list[int] = []
        preview: list[dict[str, Any]] = []
        option_rows = 0
        if not errors:
            for index, row in enumerate(rows, start=2):
                qid = str(row.get(str(resolved_mapping["question_id"]), "")).strip()
                question = str(row.get(str(resolved_mapping["question_text"]), "")).strip()
                split = str(row.get(str(resolved_mapping["split"]), "")).strip() or "unspecified"
                options = _parse_option_mapping(row.get(str(resolved_mapping["code_to_answer"])))
                qids.append(qid)
                split_counts[split] = split_counts.get(split, 0) + 1
                option_rows += len(options)
                if not qid or not question:
                    empty_question_rows.append(index)
                if not options:
                    invalid_option_rows.append(index)
                if len(preview) < 8:
                    preview.append(
                        {
                            "qid": qid,
                            "question": question,
                            "split": split,
                            "options": options,
                        }
                    )

        duplicates = sorted({qid for qid in qids if qid and qids.count(qid) > 1})
        if duplicates:
            errors.append(
                f"Question IDs must be unique; found {len(duplicates)} duplicate(s): "
                + ", ".join(duplicates[:8])
            )
        if empty_question_rows:
            errors.append(
                f"{len(empty_question_rows)} row(s) have an empty question ID or text."
            )
        if invalid_option_rows:
            errors.append(
                f"{len(invalid_option_rows)} row(s) have an empty or unparseable option mapping."
            )
        for required_split in ("train", "valid", "test"):
            if split_counts.get(required_split, 0) == 0:
                warnings.append(f"No questions are assigned to the {required_split!r} split.")

        target_path: Path | None = None
        target_qids: set[str] = set()
        if body.get("target"):
            target_path = _repository_path(
                body.get("target"), root=self.root, field="target"
            )
            target_payload = _read_json(target_path)
            if not isinstance(target_payload, dict):
                errors.append("Target file must contain a JSON object keyed by question ID.")
            else:
                # Subgroup targets are allowed; inspect the first mapping layer
                # that overlaps the survey QIDs.
                direct = set(target_payload) & set(qids)
                if direct:
                    target_qids = set(target_payload)
                else:
                    nested = next(
                        (value for value in target_payload.values() if isinstance(value, dict) and set(value) & set(qids)),
                        None,
                    )
                    if isinstance(nested, dict):
                        target_qids = set(nested)
                    else:
                        errors.append("Target JSON contains no question IDs from this survey.")
        else:
            warnings.append("No target JSON was supplied; fitting and evaluation workflows will need one.")

        survey_qids = {qid for qid in qids if qid}
        missing_target = sorted(survey_qids - target_qids) if target_path else []
        extra_target = sorted(target_qids - survey_qids) if target_path else []
        if missing_target:
            errors.append(f"Target JSON is missing {len(missing_target)} survey question(s).")
        if extra_target:
            warnings.append(f"Target JSON contains {len(extra_target)} extra question(s).")

        return {
            "valid": not errors,
            "source": _relative(source, self.root),
            "schema": _relative(schema_path, self.root) if schema_path else None,
            "target": _relative(target_path, self.root) if target_path else None,
            "headers": headers,
            "mapping": resolved_mapping,
            "questionCount": len(rows),
            "optionRowCount": option_rows,
            "splitCounts": split_counts,
            "targetCoverage": {
                "matched": len(survey_qids & target_qids),
                "missing": len(missing_target),
                "extra": len(extra_target),
            },
            "preview": preview,
            "errors": errors,
            "warnings": warnings,
            "digests": {
                "source": _file_digest(source),
                **({"schema": _file_digest(schema_path)} if schema_path else {}),
                **({"target": _file_digest(target_path)} if target_path else {}),
            },
        }

    def register_dataset(self, body: dict[str, Any]) -> dict[str, Any]:
        inspection = self.inspect_dataset(body)
        if not inspection["valid"]:
            raise ValueError("Dataset inspection failed: " + " ".join(inspection["errors"]))
        name = body.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name is required")
        slug = _manifest_filename(name).removesuffix(".yaml")

        schema = inspection["schema"]
        if schema is None:
            schema_path = self.root / "config" / "surveys" / f"{slug}.yaml"
            if schema_path.exists():
                raise ValueError(f"Schema already exists: {_relative(schema_path, self.root)}")
            schema_path.parent.mkdir(parents=True, exist_ok=True)
            schema_path.write_text(
                yaml.safe_dump(inspection["mapping"], sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            schema = _relative(schema_path, self.root)
            inspection["digests"]["schema"] = _file_digest(schema_path)

        destination = self.root / "config" / "datasets" / f"{slug}.yaml"
        if destination.exists():
            raise ValueError(f"Dataset already exists: {_relative(destination, self.root)}")
        descriptor = {
            "name": name.strip(),
            "source": inspection["source"],
            "schema": schema,
            "target": inspection["target"],
            "question_count": inspection["questionCount"],
            "split_counts": inspection["splitCounts"],
            "digests": inspection["digests"],
            "portable": True,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(descriptor, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return _dataset_summary(destination, self.root)

    def preview_manifest(self, body: dict[str, Any]) -> dict[str, Any]:
        command = str(body.get("command", "pipeline"))
        if command not in {"pipeline", "generate", "regress", "evaluate"}:
            raise ValueError(f"Unsupported guided workflow: {command}")
        config = body.get("config")
        if not isinstance(config, dict):
            raise ValueError("config must be a YAML mapping")
        run = config.get("run")
        if not isinstance(run, dict) or not isinstance(run.get("name"), str):
            raise ValueError("run.name is required")
        destination = (
            self.root / "config" / "experiments" / "studio" / _manifest_filename(run["name"])
        ).resolve()
        prepared = _prepare_studio_manifest(config, root=self.root, destination=destination)
        _check_reference_combination(prepared)
        resolved = Config.from_mapping(prepared, declared_in=destination).validate_for(command)
        target = (config.get("data") or {}).get("target")
        target_path = target.get("path") if isinstance(target, dict) else target
        target_groups = (
            _target_subgroups((self.root / target_path).resolve())
            if isinstance(target_path, str) and target_path.strip()
            else None
        )
        output_relative = str(body.get("output", ""))
        output = (self.root / output_relative).resolve() if output_relative else None
        if output is not None and not _inside(output, self.root / "outputs"):
            raise ValueError("Output must stay inside outputs/")
        solver = _requested_solver(body.get("solver"), command)
        summary = _manifest_summary(destination, self.root) if destination.exists() else None
        resolved_mapping = resolved.to_dict()
        entries = _backend_entries(resolved_mapping)
        backend_mapping = resolved_mapping.get("backend")
        if isinstance(backend_mapping, dict) and isinstance(backend_mapping.get("profile"), str):
            entries.extend(
                _backend_entries(
                    {"backend": _read_yaml(Path(backend_mapping["profile"]))}
                )
            )
        paid = any(item["type"] not in LOCAL_BACKENDS for item in entries)
        warnings: list[str] = []
        subgroup = target.get("subgroup") if isinstance(target, dict) else None
        if target_groups is not None:
            if subgroup is not None and subgroup not in target_groups["subgroups"]:
                raise ValueError(
                    f"unknown target subgroup {subgroup!r}; available: "
                    + ", ".join(target_groups["subgroups"])
                )
            if subgroup is None and target_groups["default"] is None and len(target_groups["subgroups"]) > 1:
                raise ValueError(
                    "The target file has several subgroups and no default; choose one: "
                    + ", ".join(target_groups["subgroups"])
                )
        elif subgroup is not None:
            raise ValueError("data.target.subgroup requires a grouped target file")
        if destination.exists():
            warnings.append("A manifest with this name already exists; choose another name before saving.")
        if output is not None and output.exists() and any(output.iterdir()):
            warnings.append("The output directory already exists and is not empty.")
        if paid:
            warnings.append("This configuration may call a metered provider; no cost forecast is attached to this draft.")
        return {
            "valid": not destination.exists() and not (
                output is not None and output.exists() and any(output.iterdir())
            ),
            "path": _relative(destination, self.root),
            "output": _relative(output, self.root) if output is not None else None,
            "content": yaml.safe_dump(prepared, sort_keys=False, allow_unicode=True),
            "configDigest": resolved.digest,
            "paid": paid,
            "costStatus": "unknown" if paid else "offline-or-local",
            "warnings": warnings,
            "solver": solver,
            "cli": (
                f"p2p {command} --config {_relative(destination, self.root)} "
                f"--output {output_relative}"
                + (f" --solver {solver}" if solver else "")
            ).strip(),
            "summary": summary,
            "targetSubgroups": target_groups,
        }

    def detail(self, relative_path: str, kind: str) -> dict[str, Any]:
        path = (self.root / relative_path).resolve()
        if not _inside(path, self.root / "outputs"):
            raise ValueError("Artifact paths must stay inside outputs/")
        filename = "diagnostics.json" if kind == "run" else "budget.json"
        payload = _read_json(path / filename)
        if kind == "run" and not isinstance(payload, dict):
            canonical_run = _read_json(path / "run.json")
            if isinstance(canonical_run, dict):
                conditions = canonical_run.get("conditions", {})
                method = (
                    conditions.get("method")
                    if isinstance(conditions, dict)
                    else None
                )
                # Standalone evaluate writes canonical run and metric records
                # without regression diagnostics.  That is a supported run
                # shape, not a missing-artifact error.
                payload = {
                    "model": {"method": method} if method else {},
                    "questions": [],
                    "diagnosticsAvailable": False,
                    "notes": [
                        "This evaluation-only run did not produce model or "
                        "per-question regression diagnostics."
                    ],
                }
            elif not (path / "_freeze.yaml").is_file():
                raise FileNotFoundError(filename)
            else:
                # Import lazily: indexing the Studio does not need pandas, while
                # opening an archive can use the established read-only adapter.
                from p2p.earlier_runs import load_earlier_run

                bundle = load_earlier_run(path)
                payload = dict(bundle.diagnostics or {})
                config = _legacy_config(path)
                method = str(
                    config.get("weight_assignment_model")
                    or payload.get("model", {}).get("method")
                    or "legacy"
                )
                payload.setdefault("model", {"method": method})
                payload.setdefault("questions", [])
                payload["metrics"] = bundle.metrics or {"metrics": []}
                payload["usage"] = {"summary": _legacy_usage(path)}
                payload["run"] = {
                    "run_id": path.name,
                    "origin": "legacy",
                    "provenance": _read_yaml(path / "_freeze.yaml"),
                }
                payload["legacy"] = {
                    "notes": list(bundle.notes),
                    "archivedImages": [
                        _relative(image, self.root)
                        for image in bundle.archived_images
                    ],
                }
                payload.update(_generation_analysis(path, payload, root=self.root))
                if any(path.glob("summary*.md")):
                    payload["archive"] = _report_detail(path, self.root)
                payload["summary"] = _legacy_run_summary(path, self.root)
                payload["annotation"] = self.annotations().get(relative_path, {})
                return payload
        if not isinstance(payload, dict):
            raise FileNotFoundError(filename)
        if kind == "run":
            payload["run"] = _read_json(path / "run.json", {})
            payload["metrics"] = _read_json(path / "metrics.json", {})
            payload["usage"] = _read_json(path / "usage.json", {})
            payload.update(_generation_analysis(path, payload, root=self.root))
            payload["summary"] = _run_summary(path / "run.json", self.root)
            metric_records = (payload["metrics"] or {}).get("metrics", [])
            payload["health"] = _run_health(
                path,
                self.root,
                run=payload["run"] if isinstance(payload["run"], dict) else {},
                diagnostics=payload,
                metrics=[item for item in metric_records if isinstance(item, dict)]
                if isinstance(metric_records, list)
                else [],
            )
            payload["annotation"] = self.annotations().get(relative_path, {})
            related_job = next(
                (
                    job.as_dict()
                    for job in self.jobs.values()
                    if job.output == relative_path
                ),
                None,
            )
            if related_job is not None:
                payload["job"] = related_job
        return payload

    def report_detail(self, relative_path: str) -> dict[str, Any]:
        path = (self.root / relative_path).resolve()
        if not _inside(path, self.root / "outputs"):
            raise ValueError("Report paths must stay inside outputs/")
        if not path.is_dir() or not any(path.glob("summary*.md")):
            raise FileNotFoundError(relative_path)
        return _report_detail(path, self.root)

    def artifact_path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        if not _inside(path, self.root / "outputs"):
            raise ValueError("Artifact paths must stay inside outputs/")
        if not path.is_file() or path.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".svg",
        }:
            raise FileNotFoundError(relative_path)
        return path

    def download_path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        output_root = (self.root / "outputs").resolve()
        if not _inside(path, output_root) or ".studio" in path.parts:
            raise ValueError("Download paths must stay inside public output records")
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        return path

    def create_manifest(self, body: dict[str, Any]) -> dict[str, Any]:
        command = str(body.get("command", "pipeline"))
        if command not in {"pipeline", "generate", "regress", "evaluate"}:
            raise ValueError(f"Unsupported guided workflow: {command}")
        config = body.get("config")
        if not isinstance(config, dict):
            raise ValueError("config must be a YAML mapping")
        run = config.get("run")
        if not isinstance(run, dict) or not isinstance(run.get("name"), str):
            raise ValueError("run.name is required")
        destination = (
            self.root
            / "config"
            / "experiments"
            / "studio"
            / _manifest_filename(run["name"])
        ).resolve()
        if not _inside(destination, self.root / "config" / "experiments" / "studio"):
            raise ValueError("Generated manifest must stay inside config/experiments/studio")
        if destination.exists():
            raise ValueError(
                f"Manifest already exists: {_relative(destination, self.root)}; "
                "choose a different experiment name"
            )
        prepared = _prepare_studio_manifest(
            config,
            root=self.root,
            destination=destination,
        )
        _check_reference_combination(prepared)
        Config.from_mapping(prepared, declared_in=destination).validate_for(command)
        content = yaml.safe_dump(
            prepared,
            sort_keys=False,
            allow_unicode=True,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        return {
            "path": _relative(destination, self.root),
            "content": content,
            "manifest": _manifest_summary(destination, self.root),
        }

    def start_job(self, body: dict[str, Any]) -> dict[str, Any]:
        command = str(body.get("command", "pipeline"))
        if command not in ALLOWED_COMMANDS:
            raise ValueError(f"Unsupported command: {command}")
        config_relative = str(body.get("config", ""))
        output_relative = str(body.get("output", ""))
        config = (self.root / config_relative).resolve()
        output = (self.root / output_relative).resolve()
        if not _inside(config, self.root / "config") or config.suffix not in {".yaml", ".yml"}:
            raise ValueError("Config must be a YAML file inside config/")
        if not config.is_file():
            raise FileNotFoundError(config_relative)
        if not _inside(output, self.root / "outputs"):
            raise ValueError("Output must stay inside outputs/")
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory already exists and is not empty")
        solver = _requested_solver(body.get("solver"), command)
        output.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            sys.executable,
            "-m",
            "p2p",
            command,
            "--config",
            str(config),
            "--output",
            str(output),
            "--verbose",
        ]
        if solver:
            argv += ["--solver", solver]
        process = subprocess.Popen(
            argv,
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job = StudioJob(
            job_id=uuid.uuid4().hex[:10],
            command=command,
            config=_relative(config, self.root),
            output=_relative(output, self.root),
            process=process,
            pid=process.pid,
            solver=solver,
        )
        with self.lock:
            self.jobs[job.job_id] = job
        self._save_job(job)
        threading.Thread(target=self._collect_log, args=(job,), daemon=True).start()
        return job.as_dict()

    def start_demo(self) -> dict[str, Any]:
        """Run the whole pipeline offline on the fictional demo survey.

        Stage 1 uses the rule-based ``simulated`` backend (not a language
        model): P2P's active generation creates residents of a fictional town
        and asks them the survey. Stage 2 fits a constrained lasso and the run
        is evaluated on the held-out questions.
        """

        config = self.root / "config" / "smoke" / "studio_demo_pipeline.yaml"
        if not config.is_file():
            raise FileNotFoundError(_relative(config, self.root))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return self.start_job(
            {
                "command": "pipeline",
                "config": _relative(config, self.root),
                "output": f"outputs/studio/demo-{stamp}",
            }
        )

    def _collect_log(self, job: StudioJob) -> None:
        if job.process is None or job.process.stdout is None:
            return
        for line in job.process.stdout:
            with job.lock:
                job.log.append(line.rstrip())
            self._save_job(job)
        job.process.wait()
        job.as_dict()
        self._save_job(job)

    def job_action(self, job_id: str, action: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise FileNotFoundError(job_id)
        if job.process is None:
            raise ValueError(
                "This durable job record is no longer attached to a live process; "
                "inspect its artifacts or create a new output directory."
            )
        if job.process.poll() is not None:
            self._save_job(job)
            return job.as_dict()
        if action == "pause":
            if os.name != "posix" or not hasattr(signal, "SIGSTOP"):
                raise ValueError("Pause is not supported on this platform")
            job.process.send_signal(signal.SIGSTOP)
            job.status = "paused"
        elif action == "resume":
            if os.name != "posix" or not hasattr(signal, "SIGCONT"):
                raise ValueError("Resume is not supported on this platform")
            job.process.send_signal(signal.SIGCONT)
            job.status = "running"
        elif action == "stop":
            job.process.terminate()
            job.status = "stopped"
            job.recovery = "checkpoint" if any(
                (self.root / job.output).glob("*checkpoint*")
            ) else "new-output"
        else:
            raise ValueError(f"Unsupported action: {action}")
        self._save_job(job)
        return job.as_dict()


def _handler_factory(state: StudioState, static_dir: Path) -> type[SimpleHTTPRequestHandler]:
    class StudioHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(static_dir), **kwargs)

        def end_headers(self) -> None:
            # Static files: the page itself is revalidated on every load, so an
            # upgraded Studio is never shown from a stale cache; the hashed
            # build assets it names never change and can be kept.
            if self.command in {"GET", "HEAD"} and not self.path.startswith("/api/") and "Cache-Control" not in self._headers_sent():
                if self.path.startswith("/assets/"):
                    self.send_header("Cache-Control", "public, max-age=31536000, immutable")
                else:
                    self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def _headers_sent(self) -> str:
            buffer = getattr(self, "_headers_buffer", [])
            return b"".join(buffer).decode("latin-1", "replace")

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _error(self, error: Exception, status: HTTPStatus) -> None:
            self._json({"error": str(error)}, status)

        def _artifact(self, path: Path) -> None:
            content_type, _ = mimetypes.guess_type(path.name)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Cache-Control", "private, max-age=300")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as handle:
                self.wfile.write(handle.read())

        def _download(self, path: Path) -> None:
            content_type, _ = mimetypes.guess_type(path.name)
            safe_name = path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{safe_name}"')
            self.send_header("Cache-Control", "private, no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as handle:
                self.wfile.write(handle.read())

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/bootstrap":
                self._json(state.bootstrap())
                return
            if parsed.path == "/api/workspace":
                self._json(state.workspace())
                return
            if parsed.path == "/api/capabilities":
                self._json(state.capabilities())
                return
            if parsed.path == "/api/datasets":
                self._json({"datasets": state.datasets()})
                return
            if parsed.path == "/api/annotations":
                self._json({"annotations": state.annotations()})
                return
            if parsed.path == "/api/manifest":
                query = parse_qs(parsed.query)
                try:
                    self._json(state.manifest_detail(query.get("path", [""])[0]))
                except FileNotFoundError as error:
                    self._error(error, HTTPStatus.NOT_FOUND)
                except (ValueError, ConfigError) as error:
                    self._error(error, HTTPStatus.BAD_REQUEST)
                return
            if parsed.path in {"/api/runs", "/api/reports"}:
                query = parse_qs(parsed.query)
                try:
                    limit = int(query.get("limit", ["30"])[0])
                    if parsed.path == "/api/runs":
                        payload = state.list_runs(
                            cursor=query.get("cursor", [None])[0],
                            limit=limit,
                            origin=query.get("origin", ["all"])[0],
                            query=query.get("query", [""])[0],
                            sort=query.get("sort", ["newest"])[0],
                        )
                    else:
                        payload = state.list_reports(
                            cursor=query.get("cursor", [None])[0],
                            limit=limit,
                            query=query.get("query", [""])[0],
                            sort=query.get("sort", ["newest"])[0],
                        )
                    self._json(payload)
                except ValueError as error:
                    self._error(error, HTTPStatus.BAD_REQUEST)
                return
            if parsed.path == "/api/aggregates":
                query = parse_qs(parsed.query)
                try:
                    self._json(
                        state.list_aggregates(
                            cursor=query.get("cursor", [None])[0],
                            limit=int(query.get("limit", ["30"])[0]),
                            query=query.get("query", [""])[0],
                        )
                    )
                except ValueError as error:
                    self._error(error, HTTPStatus.BAD_REQUEST)
                return
            if parsed.path == "/api/aggregate":
                query = parse_qs(parsed.query)
                try:
                    self._json(state.aggregate_detail(query.get("path", [""])[0]))
                except FileNotFoundError as error:
                    self._error(error, HTTPStatus.NOT_FOUND)
                except ValueError as error:
                    self._error(error, HTTPStatus.BAD_REQUEST)
                return
            if parsed.path in {"/api/run", "/api/budget"}:
                query = parse_qs(parsed.query)
                relative_path = query.get("path", [""])[0]
                try:
                    self._json(
                        state.detail(
                            relative_path,
                            "run" if parsed.path == "/api/run" else "budget",
                        )
                    )
                except FileNotFoundError as error:
                    self._error(error, HTTPStatus.NOT_FOUND)
                except ValueError as error:
                    self._error(error, HTTPStatus.BAD_REQUEST)
                return
            if parsed.path == "/api/report":
                query = parse_qs(parsed.query)
                try:
                    self._json(state.report_detail(query.get("path", [""])[0]))
                except FileNotFoundError as error:
                    self._error(error, HTTPStatus.NOT_FOUND)
                except ValueError as error:
                    self._error(error, HTTPStatus.BAD_REQUEST)
                return
            if parsed.path == "/api/artifact":
                query = parse_qs(parsed.query)
                try:
                    self._artifact(state.artifact_path(query.get("path", [""])[0]))
                except FileNotFoundError as error:
                    self._error(error, HTTPStatus.NOT_FOUND)
                except ValueError as error:
                    self._error(error, HTTPStatus.BAD_REQUEST)
                return
            if parsed.path == "/api/file":
                query = parse_qs(parsed.query)
                try:
                    self._download(state.download_path(query.get("path", [""])[0]))
                except FileNotFoundError as error:
                    self._error(error, HTTPStatus.NOT_FOUND)
                except ValueError as error:
                    self._error(error, HTTPStatus.BAD_REQUEST)
                return
            if parsed.path == "/api/jobs":
                self._json({"jobs": [job.as_dict() for job in state.jobs.values()]})
                return
            if parsed.path.startswith("/api/"):
                self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            if parsed.path != "/" and not (static_dir / parsed.path.lstrip("/")).exists():
                self.path = "/index.html"
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1_000_000:
                    raise ValueError("Request body is too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("Request body must be a JSON object")
                parsed = urlparse(self.path)
                if parsed.path == "/api/datasets/inspect":
                    self._json(state.inspect_dataset(body))
                    return
                if parsed.path == "/api/backends/check":
                    self._json(state.check_backend(body))
                    return
                if parsed.path == "/api/annotations":
                    self._json(state.update_annotation(body))
                    return
                if parsed.path == "/api/demo":
                    self._json(state.start_demo(), HTTPStatus.CREATED)
                    return
                if parsed.path == "/api/datasets":
                    self._json(state.register_dataset(body), HTTPStatus.CREATED)
                    return
                if parsed.path == "/api/manifests/preview":
                    self._json(state.preview_manifest(body))
                    return
                if parsed.path == "/api/manifests":
                    self._json(state.create_manifest(body), HTTPStatus.CREATED)
                    return
                if parsed.path == "/api/jobs":
                    self._json(state.start_job(body), HTTPStatus.CREATED)
                    return
                parts = parsed.path.strip("/").split("/")
                if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "action":
                    self._json(state.job_action(parts[2], str(body.get("action", ""))))
                    return
                self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            except FileNotFoundError as error:
                self._error(error, HTTPStatus.NOT_FOUND)
            except (ValueError, json.JSONDecodeError) as error:
                self._error(error, HTTPStatus.BAD_REQUEST)
            except Exception as error:  # pragma: no cover - defensive HTTP boundary
                self._error(error, HTTPStatus.INTERNAL_SERVER_ERROR)

    return StudioHandler


def run_studio(
    *,
    root: Path,
    host: str = "127.0.0.1",
    port: int = 5173,
    open_browser: bool = True,
) -> None:
    """Serve the built Studio and repository APIs on the loopback interface."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "P2P Experiment Studio is local-only; host must be "
            "127.0.0.1, localhost, or ::1."
        )
    root = root.resolve()
    static_dir = root / "studio" / "dist"
    if not (static_dir / "index.html").is_file():
        raise RuntimeError(
            "Studio assets are missing. Run `cd studio && npm run build` first."
        )
    state = StudioState(root)
    state.warm_index()
    server = ThreadingHTTPServer((host, port), _handler_factory(state, static_dir))
    url = f"http://{host}:{port}/"
    print(f"P2P Experiment Studio: {url}")
    print("Local-only mode: repository data is served from this process and is not uploaded.")
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["StudioState", "run_studio"]
