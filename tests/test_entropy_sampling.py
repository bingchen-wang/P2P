"""Contracts for the explicit target-mean entropy sampler."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from p2p.endowments import ActiveEndowments
from p2p.sampling import sample_to_target_mean_entropy


class TinyResponses:
    def __init__(self) -> None:
        self.values = {
            "e1": {"q1": "a", "q2": "x"},
            "e2": {"q1": "a", "q2": "x"},
            "e3": {"q1": "b", "q2": "y"},
            "e4": {"q1": "b", "q2": "y"},
        }

    def get_agent_vector(self, eid: str):
        return self.values[eid]


@pytest.fixture
def entropy_inputs():
    endowments = ActiveEndowments.from_endowment_list(
        [
            {"eid": f"e{index}", "role": "proxy"}
            for index in range(1, 5)
        ]
    )
    survey = SimpleNamespace(
        questions=[
            {"id": "q1", "code_to_answer": {0: "a", 1: "b"}},
            {"id": "q2", "code_to_answer": {0: "x", 1: "y"}},
        ]
    )
    return endowments, TinyResponses(), survey


def test_entropy_sampler_exactly_finds_requested_diversity(
    entropy_inputs,
) -> None:
    endowments, responses, survey = entropy_inputs

    result = sample_to_target_mean_entropy(
        endowments,
        responses,
        survey,
        sample_size=2,
        target_entropy=1.0,
        max_evaluations=100,
    )

    assert result.method == "exact"
    assert result.achieved_entropy == pytest.approx(1.0)
    assert result.absolute_error == pytest.approx(0.0)
    assert result.selected_eids == ("e1", "e3")
    assert result.candidates_evaluated == 6


def test_entropy_sampler_is_seeded_without_global_rng_mutation(
    entropy_inputs,
) -> None:
    endowments, responses, survey = entropy_inputs
    random.seed(700)
    expected = random.random()
    random.seed(700)

    first = sample_to_target_mean_entropy(
        endowments,
        responses,
        survey,
        sample_size=2,
        target_entropy=0.4,
        seed=9,
        max_evaluations=3,
    )
    second = sample_to_target_mean_entropy(
        endowments,
        responses,
        survey,
        sample_size=2,
        target_entropy=0.4,
        seed=9,
        max_evaluations=3,
    )

    assert first == second
    assert random.random() == expected


def test_entropy_sampler_can_require_tolerance(
    entropy_inputs,
) -> None:
    endowments, responses, survey = entropy_inputs

    with pytest.raises(ValueError, match="requested tolerance"):
        sample_to_target_mean_entropy(
            endowments,
            responses,
            survey,
            sample_size=1,
            target_entropy=1.0,
            tolerance=0.01,
            require_tolerance=True,
        )
