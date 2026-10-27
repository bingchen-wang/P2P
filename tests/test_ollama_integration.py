"""Opt-in integration smoke for a locally running Ollama server."""

from __future__ import annotations

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

from p2p.agents import (
    AgentConfig,
    SurveyConductor,
    create_backend,
)
from p2p.generators import ActiveEndowmentGenerator
from p2p.survey import Survey
from p2p.usage import UsageTracker


OLLAMA_BASE_URL = os.environ.get(
    "P2P_OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
).rstrip("/")
OLLAMA_MODEL = os.environ.get(
    "P2P_OLLAMA_MODEL",
    "gemma3:4b",
)


class OllamaEndowmentModel:
    """Single-call test model using the normalized Ollama backend."""

    def __init__(self, *, backend, tracker: UsageTracker) -> None:
        self.backend = backend
        self.tracker = tracker

    def generate_endowments(
        self,
        attributes: tuple[str, ...],
        n: int,
        mode: tuple[str, ...],
    ) -> list[dict[str, object]]:
        if n != 1:
            raise ValueError(
                "the tiny Ollama smoke generates exactly one persona"
            )

        response = self.backend.chat(
            system=(
                "You create short synthetic survey personas. "
                "Return only one concise persona description."
            ),
            user=(
                "Write one persona, in at most two short sentences, "
                "that varies along these dimensions: "
                f"{', '.join(attributes)}."
            ),
        )

        self.tracker.log(
            usage=response.usage,
            model_name=self.backend.model_name,
            module_name="endowment_model",
            agent_id="ollama_generation_1",
        )

        answer_text = response.answer_text.strip()
        if not answer_text:
            raise AssertionError(
                "Ollama returned an empty persona description"
            )

        return [
            {
                "eid": "ollama-local-1",
                "endow_text": answer_text,
                "mode": mode,
                "attributes": list(attributes),
            }
        ]


class UnusedAttributeLearner:
    def generate_attributes(self, *args, **kwargs):
        raise AssertionError(
            "attribute learning must remain disabled in this smoke test"
        )


def _ollama_model_is_available() -> bool:
    request = Request(
        f"{OLLAMA_BASE_URL}/api/tags",
        method="GET",
    )

    try:
        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (
        OSError,
        URLError,
        TimeoutError,
        json.JSONDecodeError,
    ):
        return False

    return any(
        model.get("name") == OLLAMA_MODEL
        or model.get("model") == OLLAMA_MODEL
        for model in payload.get("models", [])
    )


pytestmark = pytest.mark.skipif(
    os.environ.get("P2P_RUN_OLLAMA") != "1",
    reason="set P2P_RUN_OLLAMA=1 to enable the local test",
)


def test_local_ollama_tiny_active_generate_and_elicit() -> None:
    if not _ollama_model_is_available():
        pytest.skip(
            f"Ollama model {OLLAMA_MODEL!r} is not available at "
            f"{OLLAMA_BASE_URL}"
        )

    tracker = UsageTracker()

    generation_backend = create_backend(
        backend_type="ollama",
        model_name=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        max_tokens=48,
        timeout=120,
        keep_alive="1m",
        options={
            "temperature": 0,
            "num_ctx": 1024,
        },
    )

    agent_config = AgentConfig(
        agent_type="ollama",
        model_name=OLLAMA_MODEL,
        formality="Answer with only A or B.",
        agent_kwargs={
            "base_url": OLLAMA_BASE_URL,
            "max_tokens": 8,
            "timeout": 120,
            "keep_alive": 0,
            "options": {
                "temperature": 0,
                "num_ctx": 1024,
            },
        },
    )

    survey = Survey.from_questions(
        [
            {
                "id": "Q1",
                "question": (
                    "Which teaching style do you prefer?\n"
                    "A. Structured lessons\n"
                    "B. Open discussion\n"
                    "Answer with A or B."
                ),
                "code_to_answer": {
                    "A": "Structured lessons",
                    "B": "Open discussion",
                },
                "split": "train",
            }
        ]
    )

    generator = ActiveEndowmentGenerator(
        survey=survey,
        endowment_model=OllamaEndowmentModel(
            backend=generation_backend,
            tracker=tracker,
        ),
        attribute_learner=UnusedAttributeLearner(),
        agent_config=agent_config,
        attribute_bank={
            "templates": {},
            "attributes": {
                "Q1": [
                    "teaching-style preference",
                ],
            },
        },
        conductor_factory=SurveyConductor,
        target_n=1,
        initial_n=1,
        num_update_steps=0,
        enable_survey_attributes=False,
        parallel=False,
        max_workers=1,
        verbose=False,
        token_tracker=tracker,
    )

    assert generator.parallel is False
    assert generator.max_workers == 1

    # Exercise one sampling call and one survey-simulation call directly.
    # The full adaptive expansion loop is intentionally outside this smoke.
    generated_batch = generator._sample_mode(
        ("Q1",),
        1,
    )

    assert generated_batch is not None
    assert len(generated_batch) == 1
    assert generated_batch[0]["endow_text"]

    agent_records = generator._simulate_survey_batch(
        [generated_batch]
    )

    assert len(agent_records) == 1
    assert generator.response_log == agent_records
    assert len(agent_records[0]["responses"]) == 1
    assert agent_records[0]["responses"][0].strip()

    records = tracker.records

    assert [record.module_name for record in records] == [
        "endowment_model",
        "survey_conductor",
    ]
    assert all(
        record.model_name == OLLAMA_MODEL
        for record in records
    )
    assert sum(
        record.usage.input_tokens for record in records
    ) > 0
    assert sum(
        record.usage.output_tokens for record in records
    ) > 0
    assert tracker.summary()["cost_usd"] == 0.0