"""Contracts for Stage-2 target and split preparation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from p2p.stages.split import COLUMN_DROP_V2, load_regression_data
from p2p.stages.targets import load_targets
from p2p.survey import BinaryExtendedSurvey, Survey


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_targets_preserves_and_binary_expands(
    tmp_path: Path,
) -> None:
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
    with survey_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "question",
                "option_mapping",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "Q1",
                "question": "Choose",
                "option_mapping": {
                    "A": "Alpha",
                    "B": "Beta",
                },
                "split": "test",
            }
        )

    survey = Survey(
        csv_path=survey_path,
        config_path=schema_path,
    )
    binary_survey = BinaryExtendedSurvey.from_survey(survey)

    target_path = tmp_path / "target.json"
    target_path.write_text(
        json.dumps(
            {
                "Q1": {
                    "Alpha": 0.25,
                    "Beta": 0.75,
                }
            }
        ),
        encoding="utf-8",
    )

    targets = load_targets(target_path, binary_survey)

    assert targets.raw["Q1"] == {
        "Alpha": 0.25,
        "Beta": 0.75,
    }
    assert targets.binary == {
        "Q1_0": 0.25,
        "Q1_1": 0.75,
    }
    assert len(targets.source_digest) == 64

    with pytest.raises(TypeError):
        targets.raw["Q1"]["Alpha"] = 1.0


@pytest.mark.parametrize(
    (
        "dataset",
        "survey_relative",
        "schema_relative",
        "target_relative",
        "expected_original_test_qids",
    ),
    [
        (
            "atp_w42",
            "data/w42/info.csv",
            "config/surveys/american_trend_panel.yaml",
            "data/w42/W42_aggregate_results.json",
            20,
        ),
        (
            "wvs_gb",
            "data/wvs_gb/info.csv",
            "config/surveys/wvs.yaml",
            "data/wvs_gb/aggregate_results.json",
            60,
        ),
    ],
)
def test_parity_fixture_uses_global_column_drop_before_splitting(
    dataset: str,
    survey_relative: str,
    schema_relative: str,
    target_relative: str,
    expected_original_test_qids: int,
) -> None:
    fixture_dir = PROJECT_ROOT / "outputs" / "fixtures" / dataset
    required = [
        PROJECT_ROOT / survey_relative,
        PROJECT_ROOT / target_relative,
        fixture_dir / "responses.csv",
        fixture_dir / "endowments.csv",
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()]
    if missing:
        pytest.skip(
            "Licensed external fixtures are not distributed with the repository: "
            + ", ".join(missing)
        )

    data = load_regression_data(
        survey_source=PROJECT_ROOT / survey_relative,
        survey_schema=PROJECT_ROOT / schema_relative,
        responses_path=fixture_dir / "responses.csv",
        endowments_path=fixture_dir / "endowments.csv",
        target_path=PROJECT_ROOT / target_relative,
        train_splits=("train", "valid"),
        test_split="test",
        use_proxy_only=True,
    )

    assert data.cleaning_policy == COLUMN_DROP_V2
    assert data.train_splits == ("train", "valid")
    assert data.test_split == "test"
    assert 0 < len(data.feature_names) <= 300
    assert len(data.survey.get_questions_by_split("test")) == (
        expected_original_test_qids
    )

    survey_qids = {
        question["id"]
        for question in data.survey.questions
    }
    assert survey_qids <= set(data.targets.raw)

    for split_name in ("train", "valid", "test"):
        split = data.get_split(split_name)

        assert not split.X.empty
        assert split.X.columns.tolist() == list(data.feature_names)
        assert not split.X.isna().any().any()
        assert split.y.shape == (len(split.qids),)
        assert len(split.qids) > 0

    assert set(data.source_artifacts) == {
        "survey",
        "survey_schema",
        "responses",
        "endowments",
        "target",
    }
    assert all(
        len(digest) == 64
        for digest in data.source_digests.values()
    )


def test_grouped_target_loader_selects_explicit_subgroup(
    tmp_path: Path,
) -> None:
    schema_path = tmp_path / "survey_schema.yaml"
    schema_path.write_text(
        "question_id: id\nquestion_text: question\n"
        "code_to_answer: option_mapping\nsplit: split\n",
        encoding="utf-8",
    )
    survey_path = tmp_path / "survey.csv"
    survey_path.write_text(
        'id,question,option_mapping,split\n'
        'Q1,Choose,"{\'A\': \'Alpha\', \'B\': \'Beta\'}",test\n',
        encoding="utf-8",
    )
    survey = Survey(
        csv_path=survey_path,
        config_path=schema_path,
    )
    binary_survey = BinaryExtendedSurvey.from_survey(survey)
    target_path = tmp_path / "grouped.json"
    target_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_subgroup": "overall",
                "subgroups": {
                    "overall": {
                        "Q1": {"Alpha": 0.25, "Beta": 0.75}
                    },
                    "party.democrat": {
                        "Q1": {"Alpha": 0.75, "Beta": 0.25}
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    selected = load_targets(
        target_path,
        binary_survey,
        subgroup="party.democrat",
    )
    default = load_targets(target_path, binary_survey)

    assert selected.subgroup == "party.democrat"
    assert default.subgroup == "overall"
    assert selected.raw["Q1"]["Alpha"] == pytest.approx(0.75)
    assert default.raw["Q1"]["Alpha"] == pytest.approx(0.25)
    assert selected.source_digest == default.source_digest
    assert selected.selection_digest != default.selection_digest


def test_flat_target_rejects_subgroup_selection(
    tmp_path: Path,
) -> None:
    schema_path = tmp_path / "survey_schema.yaml"
    schema_path.write_text(
        "question_id: id\nquestion_text: question\n"
        "code_to_answer: option_mapping\nsplit: split\n",
        encoding="utf-8",
    )
    survey_path = tmp_path / "survey.csv"
    survey_path.write_text(
        'id,question,option_mapping,split\n'
        'Q1,Choose,"{\'A\': \'Alpha\', \'B\': \'Beta\'}",test\n',
        encoding="utf-8",
    )
    survey = Survey(
        csv_path=survey_path,
        config_path=schema_path,
    )
    binary_survey = BinaryExtendedSurvey.from_survey(survey)
    target_path = tmp_path / "flat.json"
    target_path.write_text(
        json.dumps(
            {"Q1": {"Alpha": 0.25, "Beta": 0.75}}
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="requires a grouped target file",
    ):
        load_targets(
            target_path,
            binary_survey,
            subgroup="party.democrat",
        )
