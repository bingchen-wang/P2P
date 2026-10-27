"""Generation and elicitation stage contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from p2p.config import Config
from p2p.endowments import ActiveEndowments
from p2p.registry import Registry
from p2p.stages import elicit, generate


def write_pipeline_config(
    tmp_path: Path,
) -> Config:
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
        for qid, split in (
            ("Q1", "train"),
            ("Q2", "valid"),
            ("Q3", "test"),
        ):
            writer.writerow(
                {
                    "id": qid,
                    "question": f"Question {qid}",
                    "option_mapping": {
                        "A": "Alpha",
                        "B": "Beta",
                    },
                    "split": split,
                }
            )

    target_path = tmp_path / "target.json"
    target_path.write_text(
        json.dumps(
            {
                "Q1": {
                    "Alpha": 0.5,
                    "Beta": 0.5,
                },
                "Q2": {
                    "Alpha": 0.5,
                    "Beta": 0.5,
                },
                "Q3": {
                    "Alpha": 0.5,
                    "Beta": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )

    endowment_answer = json.dumps(
        [
            {
                "eid": "generated_agent",
                "endow_text": "Synthetic generated agent",
            }
        ]
    )

    return Config.from_mapping(
        {
            "run": {
                "name": "pipeline_stage_contract",
                "seed": 41,
            },
            "data": {
                "survey": {
                    "source": survey_path,
                    "schema": schema_path,
                },
                "target": target_path,
            },
            "generation": {
                "mode": "vanilla",
                "target_n": 1,
                "batch_size": 1,
                "delay": 0,
                "retry_failed": False,
                "max_retries": 1,
                "retry_base_delay": 0,
            },
            "backend": {
                "response": {
                    "type": "dummy",
                    "model_name": "dummy",
                    "formality": (
                        "Return only the selected answer."
                    ),
                    "kwargs": {
                        "answer_text": "Alpha",
                    },
                },
                "endowment": {
                    "type": "dummy",
                    "model_name": "dummy",
                    "kwargs": {
                        "answer_text": endowment_answer,
                    },
                },
            },
            "model": {
                "method": "lasso",
                "alpha": [0.01],
                "validation": {
                    "strategy": "cv",
                    "cv_folds": 2,
                },
            },
            "split": {
                "train_val": ["train", "valid"],
                "test": "test",
                "use_proxy_only": True,
            },
            "metrics": [
                "mse",
                "one_minus_jsd",
            ],
        },
        declared_in=tmp_path / "pipeline.yaml",
    )


def test_generate_dispatches_registered_strategy_and_writes_basis(
    tmp_path: Path,
) -> None:
    config = write_pipeline_config(tmp_path)
    run_dir = tmp_path / "run"

    registry = Registry("test generation stages")
    calls: dict[str, Any] = {}

    endowments = ActiveEndowments.from_endowment_list(
        [
            {
                "eid": "agent_a",
                "endow_text": "Synthetic agent A",
                "role": "proxy",
                "weight": None,
                "mode": ("vanilla",),
                "attributes": [],
            }
        ]
    )

    def strategy(
        *,
        config,
        survey,
        verbose: bool,
    ):
        calls["strategy"] = {
            "config": config,
            "survey": survey,
            "verbose": verbose,
        }
        return endowments

    registry.register(
        "generate.vanilla",
        strategy,
    )

    result = generate.run(
        config,
        run_dir=run_dir,
        verbose=True,
        strategy_registry=registry,
    )

    assert isinstance(result, generate.GenerationResult)
    assert result.mode == "vanilla"
    assert result.endowments is endowments
    assert result.endowments_path == (
        run_dir / "endowments.csv"
    ).resolve()
    assert result.endowments_path.is_file()

    assert calls["strategy"]["config"] is config
    assert calls["strategy"]["survey"] is result.survey
    assert calls["strategy"]["verbose"] is True

    with result.endowments_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["eid"] == "agent_a"
    assert rows[0]["role"] == "proxy"


def test_registered_vanilla_strategy_generates_offline_basis(
    tmp_path: Path,
) -> None:
    config = write_pipeline_config(tmp_path)
    run_dir = tmp_path / "run"

    result = generate.run(
        config,
        run_dir=run_dir,
        verbose=False,
    )

    assert result.mode == "vanilla"
    assert result.endowments_path.is_file()
    assert result.endowments.get_endowments() == [
        {
            "eid": "vanilla_0000",
            "endow_text": "Synthetic generated agent",
            "role": "proxy",
            "weight": None,
            "mode": ("vanilla",),
            "attributes": [],
        }
    ]


def test_elicit_writes_matrix_responses_for_generated_basis(
    tmp_path: Path,
) -> None:
    config = write_pipeline_config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    endowments = ActiveEndowments.from_endowment_list(
        [
            {
                "eid": "agent_a",
                "endow_text": "Synthetic agent A",
                "role": "proxy",
                "weight": None,
                "mode": ("vanilla",),
                "attributes": [],
            }
        ]
    )

    generated = generate.GenerationResult(
        mode="vanilla",
        survey=None,
        endowments=endowments,
        endowments_path=(
            run_dir / "endowments.csv"
        ).resolve(),
    )

    generated = SimpleNamespace(
        mode=generated.mode,
        survey=generate.load_survey(config),
        endowments=generated.endowments,
        endowments_path=generated.endowments_path,
    )

    calls: dict[str, Any] = {}

    class FakeConductor:
        def run(
            self,
            *,
            save: bool,
            **agent_kwargs,
        ):
            calls["run"] = {
                "save": save,
                "agent_kwargs": agent_kwargs,
            }
            return [
                {
                    "eid": "agent_a",
                    "qid": "Q1",
                    "answer": "Alpha",
                },
                {
                    "eid": "agent_a",
                    "qid": "Q2",
                    "answer": "Alpha",
                },
                {
                    "eid": "agent_a",
                    "qid": "Q3",
                    "answer": "Alpha",
                },
            ]

    def conductor_factory(**kwargs):
        calls["conductor"] = kwargs
        return FakeConductor()

    result = elicit.run(
        config,
        generation=generated,
        run_dir=run_dir,
        verbose=True,
        conductor_factory=conductor_factory,
    )

    assert isinstance(result, elicit.ElicitationResult)
    assert result.survey is generated.survey
    assert result.endowments is endowments
    assert result.responses_path == (
        run_dir / "responses.csv"
    ).resolve()
    assert result.responses_path.is_file()
    assert len(result.records) == 3

    assert calls["conductor"] == {
        "survey": generated.survey,
        "endowments": endowments,
        "agent_type": "dummy",
        "model_name": "dummy",
        "formality": (
            "Return only the selected answer."
        ),
        "clean": False,
        "verbose": True,
        "token_tracker": result.usage_tracker,
    }
    assert calls["run"] == {
        "save": False,
        "agent_kwargs": {
            "answer_text": "Alpha",
        },
    }

    with result.responses_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.reader(file))

    assert rows == [
        ["qid", "agent_a"],
        ["Q1", "Alpha"],
        ["Q2", "Alpha"],
        ["Q3", "Alpha"],
    ]
