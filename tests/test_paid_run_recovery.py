"""Regression tests for paid active-generation failure recovery."""

from __future__ import annotations

from collections import Counter
from itertools import count
from pathlib import Path
import random
from threading import Lock
from types import SimpleNamespace

import pytest

from p2p.backends.base import BackendResponse, TokenUsage
from p2p.config import Config
from p2p.generators import (
    ActiveEndowmentGenerator,
    ActiveGenerationCheckpoint,
    AttributeLearner,
)
from p2p.stages.generate import load_survey
from p2p.stages import generate as generate_stage
from p2p.usage import UsageTracker


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SequenceBackend:
    """Return scripted attribute answers and retain the prompts."""

    model_name = "scripted-model"

    def __init__(self, answers: list[str]) -> None:
        self.answers = iter(answers)
        self.calls: list[tuple[str, str]] = []

    def chat(self, *, system: str, user: str) -> BackendResponse:
        self.calls.append((system, user))
        return BackendResponse(
            answer_text=next(self.answers),
            usage=TokenUsage(input_tokens=10, output_tokens=2),
        )


def _fixture_survey():
    config = Config.load(
        PROJECT_ROOT
        / "config"
        / "central"
        / "generate_active_fixture.yaml"
    )
    return load_survey(config)


def test_attribute_learner_retries_unstructured_output() -> None:
    backend = SequenceBackend(
        [
            "I cannot provide that.",
            '["age", "political ideology"]',
        ]
    )
    tracker = UsageTracker()
    learner = AttributeLearner(
        survey=_fixture_survey(),
        backend=backend,
        tracker=tracker,
        max_retries=1,
        retry_base_delay=0,
    )

    attributes = learner.generate_survey_attributes()

    assert attributes == ["age", "political ideology"]
    assert len(backend.calls) == 2
    assert "Python-style list" in backend.calls[0][0]
    assert "previous response could not be parsed" in backend.calls[1][1]
    assert len(tracker.records) == 2


CUT_OFF_LIST = (
    '["level of scientific education", "income level", "geographic'
)


def test_attribute_learner_retries_a_cut_off_list() -> None:
    backend = SequenceBackend(
        [
            CUT_OFF_LIST,
            '["income level", "geographic location (urban/rural)"]',
        ]
    )
    learner = AttributeLearner(
        survey=_fixture_survey(),
        backend=backend,
        max_retries=1,
        retry_base_delay=0,
    )

    assert learner.generate_survey_attributes() == [
        "income level",
        "geographic location (urban/rural)",
    ]
    assert len(backend.calls) == 2


def test_attribute_learner_never_keeps_part_of_a_cut_off_list() -> None:
    backend = SequenceBackend([CUT_OFF_LIST, CUT_OFF_LIST])
    learner = AttributeLearner(
        survey=_fixture_survey(),
        backend=backend,
        max_retries=1,
        retry_base_delay=0,
    )

    assert learner.generate_survey_attributes() == []
    assert len(backend.calls) == 2


def test_attribute_learner_still_accepts_an_unbracketed_list() -> None:
    backend = SequenceBackend(["age, income level; political ideology"])
    learner = AttributeLearner(
        survey=_fixture_survey(),
        backend=backend,
        max_retries=0,
        retry_base_delay=0,
    )

    assert learner.generate_survey_attributes() == [
        "age",
        "income level",
        "political ideology",
    ]


def test_attribute_learner_exhaustion_is_nonfatal() -> None:
    backend = SequenceBackend(["[]", "No attributes"])
    learner = AttributeLearner(
        survey=_fixture_survey(),
        backend=backend,
        max_retries=1,
        retry_base_delay=0,
    )

    assert learner.generate_survey_attributes() == []
    assert len(backend.calls) == 2


class DeterministicEndowmentModel:
    """Minimal endowment model with checkpoint-compatible RNG state."""

    def __init__(self) -> None:
        self._ids = count(1)
        self.random_state = random.Random(101)

    def generate_endowments(self, attributes, *, n, mode):
        records = []
        for _ in range(n):
            identifier = next(self._ids)
            records.append(
                {
                    "eid": f"active_{identifier:06d}",
                    "endow_text": f"Persona {identifier}",
                    "role": "proxy",
                    "weight": None,
                    "mode": tuple(mode),
                    "attributes": list(attributes),
                }
            )
        return records


class UnusedAttributeLearner:
    def generate_survey_attributes(self):
        raise AssertionError("survey attributes are disabled")

    def generate_question_attributes(self, qid, max_attributes=None):
        raise AssertionError("question attributes are not needed")


class FailOnceConductor:
    """Fail one agent once while allowing sibling chunks to complete."""

    calls = Counter()
    failed = False
    lock = Lock()

    def __init__(self, *, survey, endowments, **kwargs) -> None:
        self.survey = survey
        self.endowments = endowments.get_endowments()

    def run(self, **kwargs) -> None:
        eid = self.endowments[0]["eid"]
        with self.lock:
            self.calls[eid] += 1
            if eid == "active_000002" and not self.failed:
                type(self).failed = True
                raise RuntimeError("injected paid-call failure")

    def to_agent_records(self):
        return [
            {
                "agent_id": endowment["eid"],
                "mode": endowment["mode"],
                "responses": [
                    "Option A"
                    for _ in self.survey.questions
                ],
            }
            for endowment in self.endowments
        ]


def _recovery_generator(
    *,
    survey,
    checkpoint,
    resume_state=None,
) -> ActiveEndowmentGenerator:
    return ActiveEndowmentGenerator(
        survey=survey,
        endowment_model=DeterministicEndowmentModel(),
        attribute_learner=UnusedAttributeLearner(),
        agent_config=SimpleNamespace(
            agent_type="dummy",
            model_name="dummy",
            formality="",
            agent_kwargs={},
            delay=0,
        ),
        attribute_bank={
            "templates": {
                "core": [],
                "thematic": [],
                "theoretical": [],
                "survey": [],
                "question": [],
            },
            "attributes": {
                "survey.global": ["age"],
            },
        },
        conductor_factory=FailOnceConductor,
        target_n=2,
        initial_n=2,
        num_update_steps=1,
        seed=101,
        parallel=True,
        max_workers=2,
        verbose=False,
        enable_survey_attributes=False,
        checkpoint_callback=checkpoint.save,
        resume_state=resume_state,
    )


def test_interrupted_paid_batch_resumes_only_missing_agents(
    tmp_path: Path,
) -> None:
    FailOnceConductor.calls.clear()
    FailOnceConductor.failed = False
    survey = _fixture_survey()
    checkpoint = ActiveGenerationCheckpoint(
        run_dir=tmp_path,
        config_digest="same-config",
        survey=survey,
    )
    first = _recovery_generator(
        survey=survey,
        checkpoint=checkpoint,
    )

    with pytest.raises(
        RuntimeError,
        match="same output directory",
    ):
        first.generate()

    partial = checkpoint.load()
    assert partial is not None
    assert partial["inflight_kind"] == "initial"
    assert len(partial["endowments"]) == 2
    assert len(partial["response_log"]) == 1

    resumed = _recovery_generator(
        survey=survey,
        checkpoint=checkpoint,
        resume_state=partial,
    )
    result = resumed.generate()

    assert len(result) == 2
    assert len(resumed.response_log) == 2
    assert FailOnceConductor.calls == {
        "active_000001": 1,
        "active_000002": 2,
    }
    complete = checkpoint.load()
    assert complete is not None
    assert complete["generation_complete"] is True
    assert complete["committed_batch_sizes"] == [2]

    calls_after_completion = FailOnceConductor.calls.copy()
    completed_rerun = _recovery_generator(
        survey=survey,
        checkpoint=checkpoint,
        resume_state=complete,
    )
    assert len(completed_rerun.generate()) == 2
    assert FailOnceConductor.calls == calls_after_completion


def test_checkpoint_rejects_a_different_config(
    tmp_path: Path,
) -> None:
    survey = _fixture_survey()
    original = ActiveGenerationCheckpoint(
        run_dir=tmp_path,
        config_digest="original",
        survey=survey,
    )
    generator = _recovery_generator(
        survey=survey,
        checkpoint=original,
    )
    # Constructor writes the initial, pre-call checkpoint.
    assert generator.checkpoint_state()["response_log"] == []

    incompatible = ActiveGenerationCheckpoint(
        run_dir=tmp_path,
        config_digest="changed",
        survey=survey,
    )
    with pytest.raises(RuntimeError, match="different"):
        incompatible.load()


def test_paid_usage_without_checkpoint_fails_before_backend_creation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = Config.load(
        PROJECT_ROOT
        / "config"
        / "central"
        / "generate_active_fixture.yaml"
    )
    survey = load_survey(config)
    tracker = UsageTracker(
        stream_path=tmp_path / "usage.csv",
    )
    tracker.log(
        usage=TokenUsage(input_tokens=10, output_tokens=2),
        model_name="paid-model",
        module_name="survey_conductor",
        agent_id="already-paid",
    )

    def fail_backend_creation(*args, **kwargs):
        raise AssertionError("a backend must not be created")

    monkeypatch.setattr(
        generate_stage,
        "create_backend",
        fail_backend_creation,
    )
    with pytest.raises(
        RuntimeError,
        match="Refusing to repeat API work",
    ):
        generate_stage._generate_active(
            config=config,
            survey=survey,
            verbose=False,
            usage_tracker=tracker,
            run_dir=tmp_path,
        )
