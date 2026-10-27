"""Contracts for the opt-in target renormalization.

``renormalize`` reproduces the historical alignment pipeline's
``experiments/utils.py::_to_prob_vec``, which divided every target vector by its
own sum after a percent test and returned an all-zero vector unchanged. It is
opt-in, so the default must still refuse a target that is not a distribution.

The repair is deliberately asymmetric: ``raw`` (which reaches the metrics) is
repaired and ``binary`` (which reaches the fit) is not, because the original
pipeline repaired inside ``_align_support_dicts`` at scoring time while
``AggregateResponses._process_item`` handed the unrepaired JSON value to the
regression. Tidying that up would stop reproducing the frozen runs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from p2p.stages.targets import load_targets
from p2p.survey import BinaryExtendedSurvey, Survey


def _binary_survey(tmp_path: Path, questions) -> BinaryExtendedSurvey:
    schema_path = tmp_path / "survey_schema.yaml"
    schema_path.write_text(
        "\n".join(
            [
                "question_id: id",
                "question_text: question",
                "code_to_answer: option_mapping",
                "split: split",
                "",
            ]
        ),
        encoding="utf-8",
    )

    survey_path = tmp_path / "survey.csv"
    with survey_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["id", "question", "option_mapping", "split"],
        )
        writer.writeheader()
        for qid, options in questions:
            writer.writerow(
                {
                    "id": qid,
                    "question": "Choose",
                    "option_mapping": options,
                    "split": "test",
                }
            )

    return BinaryExtendedSurvey.from_survey(
        Survey(csv_path=survey_path, config_path=schema_path)
    )


def _write(tmp_path: Path, payload) -> Path:
    path = tmp_path / "target.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture()
def one_question(tmp_path: Path) -> BinaryExtendedSurvey:
    return _binary_survey(
        tmp_path,
        [("Q1", {"A": "Alpha", "B": "Beta"})],
    )


def test_default_still_refuses_a_non_unit_target(
    tmp_path: Path,
    one_question: BinaryExtendedSurvey,
) -> None:
    path = _write(tmp_path, {"Q1": {"Alpha": 0.4, "Beta": 0.423}})

    with pytest.raises(ValueError, match="must sum to 1"):
        load_targets(path, one_question)


def test_renormalize_repairs_a_non_unit_target(
    tmp_path: Path,
    one_question: BinaryExtendedSurvey,
) -> None:
    # 0.823 is the total of ATP W26's GUNLOCKED2_W26, the wave the frozen
    # main-paper run could score and the unrepaired loader cannot.
    path = _write(tmp_path, {"Q1": {"Alpha": 0.223, "Beta": 0.600}})

    targets = load_targets(path, one_question, renormalize=True)

    assert targets.renormalized == ("Q1",)
    assert targets.degenerate == ()
    assert targets.raw["Q1"]["Alpha"] == pytest.approx(0.223 / 0.823)
    assert sum(targets.raw["Q1"].values()) == pytest.approx(1.0)
    # The fit target is NOT repaired: it stays exactly what the file says, and
    # its block still sums to 0.823, as it did in the frozen runs.
    assert targets.binary["Q1_0"] == pytest.approx(0.223)
    assert targets.binary["Q1_1"] == pytest.approx(0.600)
    assert (
        targets.binary["Q1_0"] + targets.binary["Q1_1"]
    ) == pytest.approx(0.823)


def test_renormalize_divides_percentages_by_one_hundred(
    tmp_path: Path,
    one_question: BinaryExtendedSurvey,
) -> None:
    path = _write(tmp_path, {"Q1": {"Alpha": 30.0, "Beta": 70.0}})

    targets = load_targets(path, one_question, renormalize=True)

    assert targets.renormalized == ()      # already unit once scaled
    assert targets.raw["Q1"]["Alpha"] == pytest.approx(0.30)
    # Percent scaling is a scoring-side repair too, so the fit target keeps the
    # file's units, matching AggregateResponses._process_item.
    assert targets.binary["Q1_0"] == pytest.approx(30.0)


def test_renormalize_leaves_a_unit_target_untouched(
    tmp_path: Path,
    one_question: BinaryExtendedSurvey,
) -> None:
    path = _write(tmp_path, {"Q1": {"Alpha": 0.25, "Beta": 0.75}})

    targets = load_targets(path, one_question, renormalize=True)

    assert targets.renormalized == ()
    assert targets.raw["Q1"] == {"Alpha": 0.25, "Beta": 0.75}


def test_renormalize_tolerates_and_records_a_zero_sum_target(
    tmp_path: Path,
    one_question: BinaryExtendedSurvey,
) -> None:
    path = _write(tmp_path, {"Q1": {"Alpha": 0.0, "Beta": 0.0}})

    targets = load_targets(path, one_question, renormalize=True)

    assert targets.degenerate == ("Q1",)
    assert targets.renormalized == ()
    assert targets.raw["Q1"] == {"Alpha": 0.0, "Beta": 0.0}


def test_default_refuses_a_zero_sum_target(
    tmp_path: Path,
    one_question: BinaryExtendedSurvey,
) -> None:
    path = _write(tmp_path, {"Q1": {"Alpha": 0.0, "Beta": 0.0}})

    with pytest.raises(ValueError, match="must sum to 1"):
        load_targets(path, one_question)


def test_renormalize_refuses_a_negative_total(
    tmp_path: Path,
    one_question: BinaryExtendedSurvey,
) -> None:
    path = _write(tmp_path, {"Q1": {"Alpha": -0.6, "Beta": 0.1}})

    with pytest.raises(ValueError, match="negative total"):
        load_targets(path, one_question, renormalize=True)


def test_renormalize_reports_every_repaired_question(
    tmp_path: Path,
) -> None:
    binary_survey = _binary_survey(
        tmp_path,
        [
            ("Q1", {"A": "Alpha", "B": "Beta"}),
            ("Q2", {"A": "Alpha", "B": "Beta"}),
        ],
    )
    path = _write(
        tmp_path,
        {
            "Q1": {"Alpha": 0.4, "Beta": 0.4},
            "Q2": {"Alpha": 0.5, "Beta": 0.5},
        },
    )

    targets = load_targets(path, binary_survey, renormalize=True)

    assert targets.renormalized == ("Q1",)
    assert targets.raw["Q1"]["Alpha"] == pytest.approx(0.5)
    assert targets.raw["Q2"]["Alpha"] == pytest.approx(0.5)
    assert targets.binary["Q1_0"] == pytest.approx(0.4)   # fit side untouched
    assert targets.binary["Q2_0"] == pytest.approx(0.5)


def test_target_renormalize_reads_the_config_entry() -> None:
    from p2p.stages.targets import target_renormalize

    assert target_renormalize("targets.json") is False
    assert target_renormalize({"path": "targets.json"}) is False
    assert (
        target_renormalize(
            {"path": "targets.json", "renormalize": True}
        )
        is True
    )

    with pytest.raises(ValueError, match="must be boolean"):
        target_renormalize({"path": "targets.json", "renormalize": "yes"})


def test_the_repair_reaches_scoring_but_not_the_fit(
    tmp_path: Path,
    one_question: BinaryExtendedSurvey,
) -> None:
    """The asymmetry itself, stated once as a contract.

    ``split.py`` builds the regression's ``aggregate`` column from ``binary`` and
    ``evaluate.py`` reads human distributions from ``raw``, so this is exactly the
    fit/score split the frozen runs had.
    """

    path = _write(tmp_path, {"Q1": {"Alpha": 0.223, "Beta": 0.600}})

    targets = load_targets(path, one_question, renormalize=True)

    fit_total = targets.binary["Q1_0"] + targets.binary["Q1_1"]
    score_total = sum(targets.raw["Q1"].values())
    assert fit_total == pytest.approx(0.823)     # unrepaired, as fitted
    assert score_total == pytest.approx(1.0)     # repaired, as scored
    assert fit_total != pytest.approx(score_total)
