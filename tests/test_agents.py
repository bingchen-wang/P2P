"""Tests for normalized agent configuration and responses."""

from __future__ import annotations

import pytest

import p2p.agents.factory as agent_factory_module
from p2p.agents import (
    Agent,
    AgentConfig,
    AgentResponse,
    SurveyConductor,
    create_agent,
)
from p2p.backends.base import BackendResponse, TokenUsage
from p2p.endowments import Endowments
from p2p.survey import Survey
from p2p.usage import UsageTracker


class RecordingBackend:
    model_name = "dummy-model"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def chat(self, *, system: str, user: str) -> BackendResponse:
        self.calls.append((system, user))
        return BackendResponse(
            answer_text="normalized answer",
            usage=TokenUsage(
                input_tokens=8,
                output_tokens=3,
                cached_input_tokens=2,
            ),
        )


def test_agent_returns_normalized_response() -> None:
    backend = RecordingBackend()
    agent = Agent(
        eid="agent-1",
        endowment="You are a teacher.",
        formality="Answer briefly.",
        backend=backend,
    )

    response = agent.answer("What is entropy?")

    assert response == AgentResponse(
        eid="agent-1",
        answer_text="normalized answer",
        usage=TokenUsage(
            input_tokens=8,
            output_tokens=3,
            cached_input_tokens=2,
        ),
        model_name="dummy-model",
    )
    assert backend.calls == [
        (
            "You are a teacher.\n\nAnswer briefly.",
            "What is entropy?",
        )
    ]


def test_agent_rejects_legacy_string_backend_response() -> None:
    class LegacyBackend:
        model_name = "legacy"

        def chat(self, *, system: str, user: str) -> str:
            return "legacy string"

    agent = Agent(eid="agent-1", backend=LegacyBackend())

    with pytest.raises(TypeError, match="BackendResponse"):
        agent.answer("question")


def test_agent_config_defensively_copies_backend_arguments() -> None:
    backend_arguments = {"answer_text": "configured"}
    config = AgentConfig(
        agent_type="dummy",
        model_name="dummy-model",
        agent_kwargs=backend_arguments,
    )

    backend_arguments["answer_text"] = "changed"

    assert dict(config.agent_kwargs) == {
        "answer_text": "configured"
    }


@pytest.mark.parametrize(
    "values",
    [
        {"agent_type": ""},
        {"model_name": ""},
    ],
)
def test_agent_config_requires_backend_and_model_names(
    values: dict[str, str],
) -> None:
    defaults = {
        "agent_type": "dummy",
        "model_name": "dummy-model",
    }
    defaults.update(values)

    with pytest.raises(ValueError):
        AgentConfig(**defaults)


def test_agent_factory_resolves_through_backend_registry() -> None:
    config = AgentConfig(
        agent_type="dummy",
        model_name="dummy-model",
        agent_kwargs={
            "answer_text": "factory answer",
            "usage": TokenUsage(
                input_tokens=3,
                output_tokens=1,
            ),
        },
    )

    agent = create_agent(
        config=config,
        eid="factory-agent",
        endowment="A test persona.",
    )

    response = agent.answer("question")

    assert response.answer_text == "factory answer"
    assert response.model_name == "dummy-model"
    assert response.usage == TokenUsage(
        input_tokens=3,
        output_tokens=1,
    )


def test_agent_factory_has_no_parallel_registry() -> None:
    assert not hasattr(agent_factory_module, "AGENT_REGISTRY")


def test_conductor_accounts_for_each_backend_call_once() -> None:
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
    endowments = Endowments.from_endowment_list(
        [
            {
                "eid": "agent-1",
                "endow_text": "A test respondent.",
                "role": "proxy",
            }
        ]
    )
    tracker = UsageTracker()

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
            output_tokens=2,
            cached_input_tokens=1,
        ),
    )

    assert len(responses) == 1
    assert responses[0]["answer"] == "A"
    assert conductor.to_agent_records() == [
        {
            "agent_id": "agent-1",
            "mode": (),
            "responses": ["A"],
        }
    ]
    assert tracker.summary() == {
        "input_tokens": 6,
        "output_tokens": 2,
        "cached_input_tokens": 1,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
    }

    assert len(tracker.records) == 1
    record = tracker.records[0]
    assert record.model_name == "dummy-model"
    assert record.module_name == "survey_conductor"
    assert record.agent_id == "agent-1"
