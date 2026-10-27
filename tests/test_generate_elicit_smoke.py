"""Offline, cost-free smoke tests for generation and elicitation."""

from __future__ import annotations

import pytest

from p2p.agents import SurveyConductor, create_backend
from p2p.backends.base import TokenUsage
from p2p.endowments import Endowments
from p2p.generators import EndowmentGenerator
from p2p.survey import Survey
from p2p.usage import UsageTracker


@pytest.mark.integration
def test_dummy_backend_generate_and_elicit_smoke() -> None:
    tracker = UsageTracker()

    generator = EndowmentGenerator(
        backend_type="dummy",
        model_name="dummy-model",
        backend_factory=create_backend,
        answer_text=(
            '[{"eid": "teacher", '
            '"endow_text": "A concise high-school teacher."}]'
        ),
        usage=TokenUsage(
            input_tokens=10,
            output_tokens=4,
            cached_input_tokens=2,
        ),
        token_tracker=tracker,
        batch_size=1,
        delay=0,
        retry_failed=False,
        verbose=False,
    )

    personas = generator.generate(1)

    assert len(personas) == 1
    assert personas[0]["eid"].startswith("teacher_")
    assert personas[0]["endow_text"] == (
        "A concise high-school teacher."
    )

    survey = Survey.from_questions(
        [
            {
                "id": "Q1",
                "question": "Choose one",
                "code_to_answer": {
                    "A": "Alpha",
                    "B": "Beta",
                },
                "split": "train",
            }
        ]
    )
    endowments = Endowments.from_endowment_list(personas)

    conductor = SurveyConductor(
        survey=survey,
        endowments=endowments,
        agent_type="dummy",
        model_name="dummy-model",
        token_tracker=tracker,
        max_retries=1,
        retry_delay=0,
        verbose=False,
    )

    responses = conductor.run(
        save=False,
        answer_text="A",
        usage=TokenUsage(
            input_tokens=6,
            output_tokens=1,
        ),
    )

    assert len(responses) == 1
    assert responses[0]["answer"] == "A"
    assert conductor.to_agent_records() == [
        {
            "agent_id": personas[0]["eid"],
            "mode": (),
            "responses": ["A"],
        }
    ]

    assert tracker.summary() == {
        "input_tokens": 16,
        "output_tokens": 5,
        "cached_input_tokens": 2,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
    }
    assert [
        record.module_name for record in tracker.records
    ] == [
        "endowment_generator",
        "survey_conductor",
    ]
