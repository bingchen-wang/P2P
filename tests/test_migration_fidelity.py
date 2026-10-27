"""Contracts preserving migrated entropy and sampling behavior."""

from __future__ import annotations

import random
from types import MethodType

import numpy as np
import pytest

from p2p.backends.base import BackendResponse, TokenUsage
from p2p.endowments import ActiveEndowments
from p2p.generators import (
    ActiveEndowmentGenerator,
    AttributeLearner,
    EndowmentModel,
    ThemeVariabilityTracker,
)
from p2p.responses import Responses
from p2p.survey import Survey
from p2p.usage import UsageTracker


def _survey() -> Survey:
    return Survey.from_questions(
        [
            {
                "id": "Q1",
                "question": "First?",
                "code_to_answer": {
                    "A": "Alpha",
                    "B": "Beta",
                },
                "split": "train",
            },
            {
                "id": "Q2",
                "question": "Second?",
                "code_to_answer": {
                    "X": "Xray",
                    "Y": "Yankee",
                },
                "split": "train",
            },
        ]
    )


def _pool() -> ActiveEndowments:
    return ActiveEndowments.from_endowment_list(
        [
            {
                "eid": "m1_a",
                "endow_text": "one",
                "role": "proxy",
                "weight": None,
                "mode": ("m1",),
                "attributes": [],
            },
            {
                "eid": "m1_b",
                "endow_text": "two",
                "role": "proxy",
                "weight": None,
                "mode": ("m1",),
                "attributes": [],
            },
            {
                "eid": "m2_a",
                "endow_text": "three",
                "role": "proxy",
                "weight": None,
                "mode": ("m2",),
                "attributes": [],
            },
            {
                "eid": "m2_b",
                "endow_text": "four",
                "role": "proxy",
                "weight": None,
                "mode": ("m2",),
                "attributes": [],
            },
        ]
    )


def test_active_pool_inherits_proven_duplicate_fraction_methods() -> None:
    assert "clone_with_fraction" not in ActiveEndowments.__dict__
    assert (
        "clone_with_proxy_fraction"
        not in ActiveEndowments.__dict__
    )

    pool = _pool()
    first = pool.clone_with_proxy_fraction(
        fraction=0.5,
        seed=17,
    )
    second = pool.clone_with_proxy_fraction(
        fraction=0.5,
        seed=17,
    )
    assert [
        row["eid"]
        for row in first.get_endowments()
    ] == [
        row["eid"]
        for row in second.get_endowments()
    ]


def test_fraction_sampling_does_not_mutate_global_random_state() -> None:
    random.seed(55)
    expected = random.random()
    random.seed(55)

    _pool().clone_with_fraction(
        fraction=0.5,
        seed=99,
    )

    assert random.random() == expected


def test_entropy_measurement_and_mode_order_match_legacy_design() -> None:
    survey = _survey()
    records = [
        {
            "eid": "m1_a",
            "qid": "Q1",
            "answer": "Alpha",
        },
        {
            "eid": "m1_b",
            "qid": "Q1",
            "answer": "Alpha",
        },
        {
            "eid": "m2_a",
            "qid": "Q1",
            "answer": "Alpha",
        },
        {
            "eid": "m2_b",
            "qid": "Q1",
            "answer": "Beta",
        },
    ]
    responses = Responses(
        records,
        survey,
        output_format="answer",
        clean=False,
    )
    pool = _pool()

    by_mode = pool.get_entropy_by_mode(
        responses,
        survey,
    )

    assert by_mode[0] == (
        ("m2",),
        pytest.approx(1.0),
        2,
    )
    assert by_mode[1] == (
        ("m1",),
        pytest.approx(0.0),
        2,
    )


def test_adaptive_tracker_keeps_legacy_entropy_trajectory() -> None:
    tracker = ThemeVariabilityTracker(
        _survey().questions,
        smoothing=0,
    )
    mode = ("core",)
    tracker.update_from_records(
        [
            {
                "mode": mode,
                "responses": [
                    "Alpha",
                    "Xray",
                ],
            },
            {
                "mode": mode,
                "responses": [
                    "Beta",
                    "Xray",
                ],
            },
        ]
    )

    assert tracker.compute_global_question_entropy() == {
        "Q1": pytest.approx(1.0),
        "Q2": pytest.approx(0.0),
    }
    assert tracker.aggregate_entropy_trajectory == [
        pytest.approx(0.5)
    ]
    assert tracker.mode_entropy_history[mode] == [
        pytest.approx(0.5)
    ]
    assert tracker.get_low_entropy_questions(
        top_k=1
    ) == ["Q2"]
    assert tracker.get_softmax_weights() == {
        mode: pytest.approx(1.0)
    }


def test_active_generator_fails_fast_after_no_initial_progress() -> None:
    generator = object.__new__(
        ActiveEndowmentGenerator
    )
    generator.endowment_manager = (
        ActiveEndowments.from_endowment_list([])
    )
    generator.target_n = 1
    generator.num_update_steps = 1
    generator.verbose = False
    generator.token_tracker = None
    generator._log = MethodType(
        lambda self, message: None,
        generator,
    )
    generator._initial_sampling = MethodType(
        lambda self: None,
        generator,
    )

    with pytest.raises(
        RuntimeError,
        match="no progress during initial sampling",
    ):
        generator.generate()


def test_endowment_model_accepts_numpy_multinomial_counts() -> None:
    class Backend:
        model_name = "dummy"

        @staticmethod
        def chat(*, system: str, user: str) -> BackendResponse:
            del system, user
            return BackendResponse(
                '[{"eid": "a", "endow_text": "Synthetic persona"}]'
            )

    records = EndowmentModel(
        backend=Backend()
    ).generate_endowments(
        ["age"],
        n=np.int64(1),
        mode=("core",),
    )

    assert len(records) == 1


TRUNCATED_REPLY = (
    '```json\n[\n  {\n    "eid": "persona_001",\n'
    '    "endow_text": "Maria is a 68-year-old retired teacher who'
)


class _ScriptedBackend:
    """Fake endowment backend returning its replies in order."""

    model_name = "dummy"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(self, *, system: str, user: str) -> BackendResponse:
        del system, user
        self.calls += 1
        return BackendResponse(
            self.replies.pop(0),
            TokenUsage(input_tokens=5, output_tokens=4),
        )


def test_truncated_persona_reply_is_retried_not_stored() -> None:
    backend = _ScriptedBackend(
        TRUNCATED_REPLY,
        '```json\n[{"eid": "a", "endow_text": "Maria is a retired teacher."}]\n```',
    )
    tracker = UsageTracker()

    records = EndowmentModel(
        backend=backend,
        tracker=tracker,
        delay=0,
        max_retries=1,
    ).generate_endowments(["age"], n=1, mode=("core",))

    assert backend.calls == 2
    assert [r["endow_text"] for r in records] == ["Maria is a retired teacher."]
    assert len(tracker.records) == 2


def test_truncated_persona_reply_fails_once_retries_are_spent() -> None:
    backend = _ScriptedBackend(TRUNCATED_REPLY, TRUNCATED_REPLY)

    with pytest.raises(RuntimeError, match="failed after 2 attempt") as error:
        EndowmentModel(
            backend=backend,
            delay=0,
            max_retries=1,
        ).generate_endowments(["age"], n=1, mode=("core",))

    assert backend.calls == 2
    assert "no JSON persona array" in str(error.value.__cause__)


def test_plain_text_persona_reply_is_an_error() -> None:
    backend = _ScriptedBackend("A retired teacher from Ohio.")

    with pytest.raises(RuntimeError) as error:
        EndowmentModel(
            backend=backend,
            delay=0,
            retry_failed=False,
        ).generate_endowments(["age"], n=1, mode=("core",))

    assert "no JSON persona array" in str(error.value.__cause__)


def test_active_endowment_model_keeps_legacy_batch_boundary() -> None:
    class Backend:
        model_name = "dummy"

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, *, system: str, user: str) -> BackendResponse:
            del system, user
            self.calls += 1
            return BackendResponse(
                (
                    '[{"eid": "a", "endow_text": "Persona A"}, '
                    '{"eid": "b", "endow_text": "Persona B"}]'
                ),
                TokenUsage(input_tokens=5, output_tokens=4),
            )

    backend = Backend()
    tracker = UsageTracker()
    records = EndowmentModel(
        backend=backend,
        tracker=tracker,
        batch_size=2,
        delay=0,
    ).generate_endowments(
        ["age", "education"],
        n=2,
        mode=("core",),
    )

    assert backend.calls == 1
    assert [record["eid"] for record in records] == [
        "active_000001",
        "active_000002",
    ]
    assert len(tracker.records) == 1


def test_attribute_learner_uses_training_questions_and_list_format() -> None:
    class Backend:
        model_name = "dummy"

        def __init__(self) -> None:
            self.users: list[str] = []

        def chat(self, *, system: str, user: str) -> BackendResponse:
            del system
            self.users.append(user)
            return BackendResponse(
                '["age", "trust in science"]'
            )

    survey = Survey.from_questions(
        [
            {
                "id": "TRAIN",
                "question": "Training question?",
                "code_to_answer": {"A": "Yes", "B": "No"},
                "split": "train",
            },
            {
                "id": "TEST",
                "question": "Held-out question?",
                "code_to_answer": {"A": "Yes", "B": "No"},
                "split": "test",
            },
        ]
    )
    backend = Backend()
    learner = AttributeLearner(
        survey=survey,
        backend=backend,
    )

    assert learner.generate_survey_attributes() == [
        "age",
        "trust in science",
    ]
    assert "Training question?" in backend.users[0]
    assert "Held-out question?" not in backend.users[0]
