"""Backend-neutral token usage and cost accounting tests."""

from __future__ import annotations

import csv
import json

import pytest

from p2p.backends.base import TokenUsage
from p2p.usage import TokenPricing, UsageTracker, write_usage_artifacts


def test_unknown_model_defaults_to_zero_cost() -> None:
    tracker = UsageTracker()
    usage = TokenUsage(
        input_tokens=20,
        output_tokens=5,
        cached_input_tokens=4,
        reasoning_tokens=2,
    )

    record = tracker.log(
        usage=usage,
        model_name="gemma3:4b",
        module_name="survey_conductor",
        agent_id="agent-1",
    )

    assert record.model_name == "gemma3:4b"
    assert record.module_name == "survey_conductor"
    assert record.agent_id == "agent-1"
    assert record.usage == usage
    assert record.cost_usd == 0.0
    assert tracker.summary() == {
        "input_tokens": 20,
        "output_tokens": 5,
        "cached_input_tokens": 4,
        "reasoning_tokens": 2,
        "cost_usd": 0.0,
    }


def test_configured_pricing_separates_cached_input() -> None:
    tracker = UsageTracker(
        pricing={
            "priced-model": TokenPricing(
                input_per_million_usd=2.0,
                output_per_million_usd=8.0,
                cached_input_per_million_usd=1.0,
            )
        }
    )

    record = tracker.log(
        usage=TokenUsage(
            input_tokens=100,
            output_tokens=20,
            cached_input_tokens=40,
            reasoning_tokens=15,
        ),
        model_name="priced-model",
        module_name="endowment_generator",
        agent_id="batch-1",
    )

    expected_cost = ((60 * 2.0) + (20 * 8.0) + (40 * 1.0)) / 1_000_000
    assert record.cost_usd == pytest.approx(expected_cost)
    assert record.usage.reasoning_tokens == 15


@pytest.mark.parametrize(
    "metadata",
    [
        {"model_name": ""},
        {"module_name": ""},
        {"agent_id": ""},
    ],
)
def test_tracker_requires_nonempty_metadata(
    metadata: dict[str, str],
) -> None:
    values = {
        "model_name": "model",
        "module_name": "module",
        "agent_id": "agent",
    }
    values.update(metadata)

    with pytest.raises(ValueError):
        UsageTracker().log(
            usage=TokenUsage(),
            **values,
        )


def test_usage_artifacts_include_records_and_summary(tmp_path) -> None:
    tracker = UsageTracker()
    tracker.log(
        usage=TokenUsage(input_tokens=7, output_tokens=3),
        model_name="gemma3:4b",
        module_name="survey_conductor",
        agent_id="agent-1",
    )

    artifacts = write_usage_artifacts(tracker, tmp_path)

    assert artifacts.csv_path.is_file()
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["summary"] == {
        "input_tokens": 7,
        "output_tokens": 3,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
    }
    assert payload["records"][0]["module_name"] == "survey_conductor"


def test_streaming_usage_is_visible_before_final_report(tmp_path) -> None:
    stream_path = tmp_path / "run" / "usage.csv"
    tracker = UsageTracker(stream_path=stream_path)

    tracker.log(
        usage=TokenUsage(input_tokens=11, output_tokens=4),
        model_name="pilot-model",
        module_name="survey_conductor",
        agent_id="agent-1",
    )

    rows = list(
        csv.DictReader(
            stream_path.open(encoding="utf-8")
        )
    )
    assert len(rows) == 1
    assert rows[0]["input_tokens"] == "11"
    assert rows[0]["output_tokens"] == "4"
    assert rows[0]["reasoning_tokens"] == "0"


def test_streaming_usage_can_resume_without_truncation(tmp_path) -> None:
    stream_path = tmp_path / "run" / "usage.csv"
    first = UsageTracker(stream_path=stream_path)
    first.log(
        usage=TokenUsage(input_tokens=11, output_tokens=4),
        model_name="pilot-model",
        module_name="survey_conductor",
        agent_id="agent-1",
    )

    resumed = UsageTracker(
        stream_path=stream_path,
        resume_stream=True,
    )
    resumed.log(
        usage=TokenUsage(input_tokens=7, output_tokens=3),
        model_name="pilot-model",
        module_name="survey_conductor",
        agent_id="agent-2",
    )

    assert len(resumed.records) == 2
    assert resumed.summary()["input_tokens"] == 18
    rows = list(
        csv.DictReader(
            stream_path.open(encoding="utf-8")
        )
    )
    assert [row["agent_id"] for row in rows] == [
        "agent-1",
        "agent-2",
    ]
