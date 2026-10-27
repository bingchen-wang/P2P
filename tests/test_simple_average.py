"""Contracts for the simple-average baseline (Stage 2 without regression; A.7).

``model.method: simple_average`` gives every agent of the cleaned proxy panel the
same weight. It runs through ``p2p regress`` like a fitted model: the same loader
(proxy selection, column-drop cleaning), the same evaluation of the configured test
split, and the same run artefacts. The legacy runner it replaces averaged every raw
response and scored every question; these tests pin both defects shut.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p2p.commands import regress
from p2p.config import Config
from p2p.metrics import emd, mse
from p2p.models import SimpleAverage
from p2p.registry import MODELS
from p2p.stages import fit
from p2p.stages.split import RegressionData, RegressionSplit, load_regression_data

SCHEMA = Path(__file__).resolve().parents[1] / "config/surveys/american_trend_panel.yaml"

# key, options in survey order, split
QUESTIONS = [
    ("Q1", ["Yes", "No"], "train"),
    ("Q2", ["A", "B", "C"], "train"),
    ("Q3", ["Up", "Down"], "valid"),
    ("T1", ["Low", "Mid", "High"], "test"),
    ("T2", ["Agree", "Disagree"], "test"),
]

# p1-p4: complete proxies. o1: ground-truth agent (not a proxy). p5: a proxy with a
# blank answer on Q2, which column-drop cleaning removes from every question.
ANSWERS = {
    "Q1": {"p1": "Yes", "p2": "No", "p3": "Yes", "p4": "Yes", "o1": "No", "p5": "No"},
    "Q2": {"p1": "A", "p2": "B", "p3": "C", "p4": "A", "o1": "C", "p5": ""},
    "Q3": {"p1": "Up", "p2": "Up", "p3": "Down", "p4": "Down", "o1": "Down", "p5": "Up"},
    "T1": {"p1": "Low", "p2": "Mid", "p3": "Mid", "p4": "High", "o1": "High", "p5": "Low"},
    "T2": {"p1": "Agree", "p2": "Agree", "p3": "Disagree", "p4": "Agree", "o1": "Disagree",
           "p5": "Disagree"},
}
AGENTS = ["p1", "p2", "p3", "p4", "o1", "p5"]
ROLES = {"o1": "ground_truth"}

# Targets; T1's keys are deliberately not in survey order.
TARGET = {
    "Q1": {"Yes": 0.6, "No": 0.4},
    "Q2": {"A": 0.5, "B": 0.3, "C": 0.2},
    "Q3": {"Up": 0.45, "Down": 0.55},
    "T1": {"High": 0.3, "Low": 0.2, "Mid": 0.5},
    "T2": {"Agree": 0.6, "Disagree": 0.4},
}

# Hand-computed: the four complete proxies p1-p4.
EXPECTED_TEST = {
    "T1": {"Low": 0.25, "Mid": 0.50, "High": 0.25},
    "T2": {"Agree": 0.75, "Disagree": 0.25},
}


def _write_inputs(root: Path) -> dict[str, Path]:
    rows = []
    for key, options, split in QUESTIONS:
        mapping = {float(i + 1): label for i, label in enumerate(options)}
        rows.append({"key": key, "question": f"Question {key}", "option_mapping": repr(mapping),
                     "split": split})
    survey = root / "info.csv"
    pd.DataFrame(rows).to_csv(survey, index=False)

    basis = root / "basis"
    basis.mkdir()
    responses = pd.DataFrame([{"qid": key, **ANSWERS[key]} for key, _, _ in QUESTIONS])
    responses[["qid", *AGENTS]].to_csv(basis / "responses.csv", index=False)
    pd.DataFrame([{"eid": eid, "endow_text": f"agent {eid}", "role": ROLES.get(eid, "proxy"),
                   "weight": "", "mode": "core", "attributes": ""} for eid in AGENTS]
                 ).to_csv(basis / "endowments.csv", index=False)

    target = root / "target.json"
    target.write_text(json.dumps(TARGET), encoding="utf-8")
    return {"survey": survey, "basis": basis, "target": target}


def _config(root: Path, paths: dict[str, Path], **model_extra) -> Config:
    text = f"""\
run:
  name: simple_average_test
  seed: 101
model:
  method: simple_average
{''.join(f'  {k}: {json.dumps(v)}{chr(10)}' for k, v in model_extra.items())}data:
  survey:
    source: {paths['survey']}
    schema: {SCHEMA}
  target:
    path: {paths['target']}
  basis: {paths['basis']}
split:
  train_val: [train, valid]
  test: test
  use_proxy_only: true
metrics: [mse, distribution_mse, one_minus_jsd, emd, mcpa]
evaluation:
  mse_prediction_policy: raw
  distribution_prediction_policy: validate
  invalid_distribution: error
"""
    path = root / "regress.yaml"
    path.write_text(text, encoding="utf-8")
    return Config.load(path)


@pytest.fixture()
def inputs(tmp_path: Path) -> dict[str, Path]:
    return _write_inputs(tmp_path)


@pytest.fixture()
def result(tmp_path: Path, inputs):
    return regress.run(_config(tmp_path, inputs), solver="OSQP", run_dir=tmp_path / "run")


def _test_predictions(res) -> dict[str, dict[str, float]]:
    split = res.data.get_split("test")
    labels = {q["id"]: (q["base_id"], q["base_answer"]) for q in res.data.binary_survey.questions}
    out: dict[str, dict[str, float]] = {}
    for row, value in zip(split.X.index, res.fit.model.predict(split.X)):
        qid, label = labels[row]
        out.setdefault(qid, {})[label] = float(value)
    return out


# --- Registration and model -------------------------------------------------------

def test_registered_as_a_normal_model() -> None:
    assert "simple_average" in MODELS.names()
    assert MODELS.get("simple_average") is SimpleAverage


def test_model_gives_every_agent_the_same_weight() -> None:
    X = pd.DataFrame([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], index=["q_0", "q_1"],
                     columns=["a", "b", "c"])
    model = SimpleAverage().fit(X)
    np.testing.assert_array_equal(model.beta_, np.full(3, 1 / 3))
    np.testing.assert_allclose(model.predict(X), [2 / 3, 1 / 3])
    assert model.intercept_ == 0.0


# --- The contract's minimum tests ----------------------------------------------------

def test_exact_distribution_on_a_hand_computed_fixture(result) -> None:
    predicted = _test_predictions(result)
    assert predicted.keys() == EXPECTED_TEST.keys()
    for qid, expected in EXPECTED_TEST.items():
        assert predicted[qid] == pytest.approx(expected, abs=1e-15)


def test_non_proxy_agents_are_excluded(result) -> None:
    assert "o1" not in result.fit.model.coef_dict_


def test_agents_rejected_by_column_drop_are_excluded(result) -> None:
    assert "p5" not in result.fit.model.coef_dict_
    assert sorted(result.fit.model.coef_dict_) == ["p1", "p2", "p3", "p4"]


def test_the_denominator_is_fixed_across_questions(result) -> None:
    weights = np.array(list(result.fit.model.coef_dict_.values()))
    np.testing.assert_array_equal(weights, np.full(4, 0.25))
    # Every question's predicted masses sum to one: four agents, one answer each.
    for name in ("train", "valid", "test"):
        split = result.data.get_split(name)
        predicted = pd.Series(result.fit.model.predict(split.X), index=split.X.index)
        for members in result.data.binary_survey.original_to_binary_map.values():
            present = [m for m in members if m in predicted.index]
            if present:
                assert predicted[present].sum() == pytest.approx(1.0, abs=1e-15)


def test_a_design_without_one_answer_per_agent_and_question_is_refused() -> None:
    # Agent b has no answer on question q (a zero entry): the average would not
    # have a fixed denominator, so the fit refuses before predicting.
    X = pd.DataFrame([[1.0, 0.0], [0.0, 0.0]], index=["q_0", "q_1"], columns=["a", "b"])

    class Survey:
        original_to_binary_map = {"q": ["q_0", "q_1"]}

    split = RegressionSplit(name="train", qids=tuple(X.index), X=X, y=np.array([0.5, 0.5]))
    test = RegressionSplit(name="test", qids=tuple(X.index), X=X, y=np.array([0.5, 0.5]))
    data = RegressionData(survey=None, binary_survey=Survey(), answer_responses=None,
                          binary_responses=None, endowments=None, targets=None,
                          splits={"train": split, "test": test}, train_splits=("train",),
                          test_split="test", feature_names=("a", "b"),
                          cleaning_policy="column_drop_v2", source_artifacts={},
                          source_digests={})
    with pytest.raises(ValueError, match="exactly one answer"):
        fit.fit_model(data, {"method": "simple_average"}, solver="OSQP", seed=101)


def test_option_ordering_follows_the_survey(result) -> None:
    # T1's target keys are High, Low, Mid; the survey order is Low, Mid, High.
    # EMD depends on order, so the recorded value pins which order was used.
    observed = {"T1": [0.2, 0.5, 0.3], "T2": [0.6, 0.4]}
    predicted = {"T1": [0.25, 0.5, 0.25], "T2": [0.75, 0.25]}
    expected = np.mean([emd(np.array(observed[q]), np.array(predicted[q]), qid=q)
                        for q in ("T1", "T2")])
    in_target_key_order = np.mean([emd(np.array([0.3, 0.2, 0.5]), np.array([0.25, 0.25, 0.5]),
                                       qid="T1"),
                                   emd(np.array(observed["T2"]), np.array(predicted["T2"]),
                                       qid="T2")])
    assert expected != pytest.approx(in_target_key_order)
    assert result.evaluation.values["emd"] == pytest.approx(expected, abs=1e-15)


def test_missing_target_questions_fail_explicitly(tmp_path: Path, inputs) -> None:
    incomplete = dict(TARGET)
    del incomplete["T2"]
    inputs["target"].write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="T2"):
        regress.run(_config(tmp_path, inputs), solver="OSQP")


def test_a_missing_option_in_the_target_fails_explicitly(tmp_path: Path, inputs) -> None:
    # T1's target lacks the "High" option: a renormalisation would hide it, so the
    # target is refused as an invalid distribution rather than silently completed.
    bad = dict(TARGET)
    bad["T1"] = {"Low": 0.2, "Mid": 0.5}
    inputs["target"].write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="'T1': human target must sum to 1"):
        regress.run(_config(tmp_path, inputs), solver="OSQP")


def test_only_test_split_questions_are_scored(result) -> None:
    observed = np.array([0.2, 0.5, 0.3, 0.6, 0.4])
    predicted = np.array([0.25, 0.5, 0.25, 0.75, 0.25])
    assert result.evaluation.values["mse"] == pytest.approx(mse(observed, predicted), abs=1e-15)
    assert result.evaluation.values["mse"] == pytest.approx(0.01, abs=1e-15)


def test_requested_and_used_coverage_are_recorded(result, tmp_path: Path) -> None:
    records = json.loads((tmp_path / "run" / "metrics.json").read_text())["metrics"]
    assert {r["metric"] for r in records} == {"mse", "distribution_mse", "one_minus_jsd", "emd",
                                              "mcpa"}
    for record in records:
        assert record["model"] == "simple_average"
        assert record["split"] == "test"
        assert record["qids_requested"] == 2
        assert record["qids_used"] == 2


def test_legacy_all_questions_condition_is_not_what_p2p_scores(result) -> None:
    # The legacy runner scored every question it had predictions for. Recreate that
    # number from the same predictions and show P2P reports the test split only.
    model, data = result.fit.model, result.data
    rows = pd.concat([data.get_split(n).X for n in ("train", "valid", "test")])
    targets = np.concatenate([data.get_split(n).y for n in ("train", "valid", "test")])
    all_questions = mse(targets, model.predict(rows))
    assert all_questions != pytest.approx(0.01)
    assert result.evaluation.values["mse"] == pytest.approx(0.01, abs=1e-15)


# --- Strictness and run artefacts ------------------------------------------------------

@pytest.mark.parametrize("option, value", [
    ("alpha", [0.1, 1.0]),
    ("validation", {"strategy": "cv", "cv_folds": 2}),
    ("reference_respondent", True),
    ("intercept", True),
    ("block_weighting", False),
    ("group_penalty_strength", 0.0),
])
def test_inapplicable_options_are_refused(option, value) -> None:
    with pytest.raises(ValueError, match="do not apply to method 'simple_average'"):
        fit.validate_model_options({"method": "simple_average", option: value})


def test_legacy_no_effect_options_are_accepted_and_recorded(tmp_path: Path, inputs) -> None:
    res = regress.run(_config(tmp_path, inputs, max_iter=10000), solver="OSQP")
    assert res.fit.diagnostics["ignored_model_options"] == "max_iter"


def test_run_artefacts_match_a_fitted_model(result, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for name in ("run.json", "metrics.json", "diagnostics.json"):
        assert (run_dir / name).is_file(), name
    assert not (run_dir / "folds.json").exists()
    run = json.loads((run_dir / "run.json").read_text())
    assert run["cleaning_policy"] == "column_drop_v2"
    model = json.loads((run_dir / "diagnostics.json").read_text())["model"]
    assert model["method"] == "simple_average"
    assert model["solver"] == "none"
    assert model["best_alpha"] is None
    assert model["coefficients"] == {"p1": 0.25, "p2": 0.25, "p3": 0.25, "p4": 0.25}
    assert model["coefficient_sum"] == 1.0
    assert model["anti_type_count"] == 0


def test_same_panel_as_a_fitted_model_on_the_same_basis(tmp_path: Path, inputs) -> None:
    # The baseline must average exactly the agents a regression would weight.
    data = load_regression_data(survey_source=inputs["survey"], survey_schema=SCHEMA,
                                responses_path=inputs["basis"] / "responses.csv",
                                endowments_path=inputs["basis"] / "endowments.csv",
                                target_path=inputs["target"])
    res = regress.run(_config(tmp_path, inputs), solver="OSQP")
    assert tuple(res.fit.model.coef_dict_) == data.feature_names
