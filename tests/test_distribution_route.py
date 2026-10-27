"""Contracts for one prediction route and canonical answers.

The distributional metrics must score the
model's own prediction (so a non-agent feature such as the reference respondent
is included), and an answer that matches an option only after normalisation
must be read as that option.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p2p.responses.cleaner import ResponseCleaner
from p2p.stages.evaluate import (
    evaluate_model,
    answer_file_predictions,
    model_distribution_predictions,
)
from p2p.stages.fit import FitResult
from p2p.stages.split import REFERENCE_FEATURE, load_regression_data
from p2p.survey import Survey


OPTIONS = {"1": "yes", "2": "neither yes nor no", "3": "no"}
SPLITS = {"Q1": "train", "Q2": "train", "Q3": "train", "Q4": "train",
          "Q5": "valid", "Q6": "test", "Q7": "test"}
AGENTS = ["a1", "a2", "a3", "a4", "a5", "a6"]
# a6 writes the lowercase option with a capital N on test question Q6.
ANSWERS = {
    "Q1": ["yes", "no", "yes", "neither yes nor no", "no", "yes"],
    "Q2": ["no", "no", "yes", "yes", "neither yes nor no", "no"],
    "Q3": ["yes", "yes", "no", "neither yes nor no", "yes", "no"],
    "Q4": ["neither yes nor no", "no", "yes", "no", "yes", "yes"],
    "Q5": ["yes", "neither yes nor no", "no", "yes", "no", "no"],
    "Q6": ["no", "yes", "neither yes nor no", "yes", "no", "Neither yes nor no"],
    "Q7": ["yes", "no", "no", "neither yes nor no", "yes", "yes"],
}
TARGET = {"yes": 0.45, "neither yes nor no": 0.25, "no": 0.30}
WEIGHTS = np.array([0.30, 0.20, 0.10, 0.15, 0.10, 0.15])


@pytest.fixture()
def basis(tmp_path: Path) -> dict[str, Path]:
    schema = tmp_path / "schema.yaml"
    schema.write_text(
        "question_id: id\nquestion_text: question\n"
        "code_to_answer: option_mapping\nsplit: split\n",
        encoding="utf-8",
    )
    survey = tmp_path / "survey.csv"
    pd.DataFrame(
        [{"id": q, "question": "Choose.", "option_mapping": str(OPTIONS), "split": s}
         for q, s in SPLITS.items()]
    ).to_csv(survey, index=False)
    responses = tmp_path / "responses.csv"
    pd.DataFrame(
        [[q, *answers] for q, answers in ANSWERS.items()],
        columns=["qid", *AGENTS],
    ).to_csv(responses, index=False)
    endowments = tmp_path / "endowments.csv"
    pd.DataFrame(
        [{"eid": a, "endow_text": a, "role": "proxy", "weight": "", "mode": "vanilla",
          "attributes": ""} for a in AGENTS]
    ).to_csv(endowments, index=False)
    target = tmp_path / "target.json"
    target.write_text(json.dumps({q: TARGET for q in SPLITS}), encoding="utf-8")
    return {"survey": survey, "schema": schema, "responses": responses,
            "endowments": endowments, "target": target}


def _load(basis, **kwargs):
    return load_regression_data(
        survey_source=basis["survey"], survey_schema=basis["schema"],
        responses_path=basis["responses"], endowments_path=basis["endowments"],
        target_path=basis["target"], train_splits=("train", "valid"),
        test_split="test", use_proxy_only=True, **kwargs,
    )


class _Linear:
    """A fitted linear model with known weights, predicting on its own columns."""

    def __init__(self, names, beta):
        self.names = list(names)
        self.beta = np.asarray(beta, dtype=float)
        self.coef_dict_ = dict(zip(self.names, self.beta))

    def predict(self, X):
        return X[self.names].to_numpy(float) @ self.beta


def _fit(model) -> FitResult:
    return FitResult(method="lasso", model=model, solver="stub", status="optimal",
                     best_alpha=0.0, best_l1_ratio=None, diagnostics={})


def _evaluate(data, model, route, policy="clip_and_normalize"):
    return evaluate_model(
        data, _fit(model), split="test", metrics=("one_minus_jsd", "emd", "mcpa"),
        distribution_prediction_policy=policy, invalid_distribution="allow",
        distribution_route=route,
    )


# --- the cleaner (F7 root cause) ---------------------------------------------------


def test_cleaner_returns_the_options_own_text(basis) -> None:
    survey = Survey(csv_path=basis["survey"], config_path=basis["schema"])
    canonical = ResponseCleaner(survey)
    legacy = ResponseCleaner(survey, canonical=False)

    assert canonical.clean("Neither yes nor no", "Q6") == "neither yes nor no"
    assert canonical.clean("  NEITHER YES NOR NO. ", "Q6") == "neither yes nor no"
    assert legacy.clean("Neither yes nor no", "Q6") == "Neither yes nor no"
    # Later steps already returned canonical text, and still do.
    assert canonical.clean("2 - Neither yes nor no", "Q6") == "neither yes nor no"
    # An answer matching no option is still returned as written.
    assert canonical.clean("maybe", "Q6") == "maybe"


def test_answers_are_canonical_by_default_and_the_design_is_unchanged(basis) -> None:
    canonical = _load(basis)
    raw = _load(basis, canonical_answers=False)

    assert canonical.canonical_answers is True and raw.canonical_answers is False
    assert canonical.answer_responses.responses["Q6"]["a6"] == "neither yes nor no"
    assert raw.answer_responses.responses["Q6"]["a6"] == "Neither yes nor no"
    for split in ("train", "valid", "test"):
        pd.testing.assert_frame_equal(canonical.get_split(split).X, raw.get_split(split).X)
        assert np.array_equal(canonical.get_split(split).y, raw.get_split(split).y)


def test_canonical_answers_must_be_boolean(basis) -> None:
    with pytest.raises(ValueError, match="canonical_answers must be boolean"):
        _load(basis, canonical_answers="yes")


# --- one prediction route (F1) ---------------------------------------------------


@pytest.mark.parametrize("policy", ["clip_and_normalize", "legacy_signed_normalize"])
def test_model_route_equals_legacy_route_on_canonical_answers(basis, policy) -> None:
    """For pure-agent models the two routes are the same arithmetic."""

    data = _load(basis)
    model = _Linear(AGENTS, WEIGHTS)
    by_model = _evaluate(data, model, "model", policy)
    by_answers = _evaluate(data, model, "legacy_answer_file", policy)

    for metric in ("one_minus_jsd", "emd", "mcpa"):
        assert by_model.values[metric] == pytest.approx(by_answers.values[metric], abs=1e-12)
    assert by_model.diagnostics["distribution_route"] == "model"
    assert by_answers.diagnostics["distribution_route"] == "legacy_answer_file"


def test_a_case_variant_is_scored_on_the_real_option(basis) -> None:
    raw = _load(basis, canonical_answers=False)
    model = _Linear(AGENTS, WEIGHTS)

    legacy = answer_file_predictions(raw, model)["Q6"]
    assert legacy["Neither yes nor no"] == pytest.approx(0.15)       # a6's weight, stranded
    assert legacy["neither yes nor no"] == pytest.approx(0.10)       # a3 only

    fixed = model_distribution_predictions(raw, model, split="test")["Q6"]
    assert set(fixed) == set(OPTIONS.values())
    assert fixed["neither yes nor no"] == pytest.approx(0.25)        # a3 + a6

    assert (
        _evaluate(raw, model, "model").values["one_minus_jsd"]
        != pytest.approx(_evaluate(raw, model, "legacy_answer_file").values["one_minus_jsd"])
    )


def test_model_route_includes_the_reference_respondent(basis) -> None:
    data = _load(basis, reference_respondent=True)
    w_ref = 0.30
    model = _Linear([*AGENTS, REFERENCE_FEATURE], [*(WEIGHTS * 0.7), w_ref])

    predicted = model_distribution_predictions(data, model, split="test")["Q7"]
    chose = dict(zip(AGENTS, ANSWERS["Q7"]))
    for option in OPTIONS.values():
        agents_mass = sum(0.7 * w for a, w in zip(AGENTS, WEIGHTS) if chose[a] == option)
        assert predicted[option] == pytest.approx(agents_mass + w_ref / 3)
    assert sum(predicted.values()) == pytest.approx(1.0)


def test_legacy_route_refuses_a_non_agent_feature(basis) -> None:
    data = _load(basis, reference_respondent=True)
    model = _Linear([*AGENTS, REFERENCE_FEATURE], [*(WEIGHTS * 0.7), 0.3])

    with pytest.raises(ValueError, match="not all answering agents"):
        _evaluate(data, model, "legacy_answer_file")


def test_an_unknown_route_is_rejected(basis) -> None:
    with pytest.raises(ValueError, match="distribution_route must be one of"):
        _evaluate(_load(basis), _Linear(AGENTS, WEIGHTS), "answers")


# --- the regress command and its report ------------------------------------------------


def _config(basis, tmp_path, *, data=None, evaluation=None):
    from p2p.config import Config

    mapping = {
        "run": {"name": "route_contract", "seed": 101},
        "data": {"survey": {"source": str(basis["survey"]), "schema": str(basis["schema"])},
                 "basis": {"responses": str(basis["responses"]),
                           "endowments": str(basis["endowments"])},
                 "target": str(basis["target"]), **(data or {})},
        "model": {"method": "lasso", "alpha": [0.001, 0.01],
                  "validation": {"strategy": "cv", "cv_folds": 2}},
        "split": {"train_val": ["train", "valid"], "test": "test", "use_proxy_only": True},
        "metrics": ["mse", "one_minus_jsd", "emd", "mcpa"],
        "evaluation": {"distribution_prediction_policy": "clip_and_normalize",
                       "invalid_distribution": "allow", **(evaluation or {})},
    }
    return Config.from_mapping(mapping, declared_in=tmp_path / "regress.yaml")


def test_regress_records_the_route_and_scores_the_model_prediction(
    basis, tmp_path: Path, cvxpy_solver: str,
) -> None:
    from p2p.commands import regress

    result = regress.run(_config(basis, tmp_path), solver=cvxpy_solver,
                         run_dir=tmp_path / "run")
    diagnostics = json.loads(result.report.diagnostics_json.read_text(encoding="utf-8"))
    assert diagnostics["distribution_route"] == "model"
    assert diagnostics["canonical_answers"] is True

    expected = model_distribution_predictions(result.data, result.fit.model)
    shown = {q["qid"]: q["prediction"] for q in diagnostics["questions"]}
    for qid, prediction in expected.items():
        assert shown[qid] == pytest.approx(prediction)

    metrics = json.loads(result.report.metrics_json.read_text(encoding="utf-8"))
    distributional = [m for m in metrics["metrics"] if m["metric"] != "mse"]
    assert distributional and all(
        m["diagnostics"]["distribution_route"] == "model" for m in distributional
    )


def test_regress_passes_the_legacy_protocol_through(
    basis, tmp_path: Path, cvxpy_solver: str,
) -> None:
    from p2p.commands import regress

    result = regress.run(
        _config(basis, tmp_path, data={"canonical_answers": False},
                evaluation={"distribution_route": "legacy_answer_file"}),
        solver=cvxpy_solver,
    )
    assert result.data.canonical_answers is False
    assert result.evaluation.diagnostics["distribution_route"] == "legacy_answer_file"


@pytest.mark.parametrize(
    ("data", "evaluation", "match"),
    [
        ({"canonical_answers": "no"}, None, "data.canonical_answers must be boolean"),
        (None, {"distribution_route": "answers"}, "evaluation.distribution_route must be one of"),
    ],
)
def test_regress_rejects_invalid_settings(
    basis, tmp_path: Path, cvxpy_solver: str, data, evaluation, match,
) -> None:
    from p2p.commands import regress

    with pytest.raises(ValueError, match=match):
        regress.run(_config(basis, tmp_path, data=data, evaluation=evaluation),
                    solver=cvxpy_solver)
