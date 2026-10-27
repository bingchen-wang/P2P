"""The rule-based ``simulated`` backend and the offline pipeline demo it powers.

The backend is not a language model: it answers P2P's three Stage-1 prompts
(persona generation, attribute discovery, survey questions) from a declarative
world, deterministically, so Stage 1 can run end to end without a network.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import pytest
import yaml

from p2p.agents.factory import create_backend
from p2p.backends import SimulatedBackend
from p2p.backends.base import TokenUsage
from p2p.commands import pipeline as pipeline_command
from p2p.config import Config
from p2p.generators.attribute_learner import QUESTION_ATTRIBUTE_PROMPT, SURVEY_ATTRIBUTE_PROMPT
from p2p.studio import StudioState

ROOT = Path(__file__).resolve().parents[1]
WORLD = {
    "setting": "a test town",
    "attributes": {
        "priority": {
            "relevance": "survey",
            "values": {
                "green": {"weight": 5.0, "prefers": ["Garden"], "describe": "loves plants"},
                "night": {"weight": 5.0, "prefers": ["Market"]},
            },
        },
        "age": {"values": {"young": {}, "old": {}}},
    },
}
QUESTION = "Which feature first? ([A] Garden, [B] Reading room, [C] Market)"


def test_answers_follow_the_persona_and_are_deterministic() -> None:
    backend = SimulatedBackend(world=WORLD, noise=0.5)
    green = "A simulated resident. Priority: green (loves plants)."
    night = "A simulated resident. Priority: night."

    assert backend.chat(system=green, user=QUESTION).answer_text == "Garden"
    assert backend.chat(system=night, user=QUESTION).answer_text == "Market"
    again = SimulatedBackend(world=WORLD, noise=0.5)
    assert again.chat(system=green, user=QUESTION).answer_text == "Garden"
    assert backend.chat(system=green, user=QUESTION).usage == TokenUsage()


def test_any_survey_is_answered_with_one_of_its_options() -> None:
    backend = SimulatedBackend()
    answers = {
        backend.chat(system=f"Persona {index}", user=QUESTION).answer_text for index in range(30)
    }
    assert answers <= {"Garden", "Reading room", "Market"}
    assert len(answers) > 1  # noise alone still varies across personas


def test_personas_use_the_world_and_follow_the_requested_count() -> None:
    backend = SimulatedBackend(world=WORLD, seed=3)
    prompt = (
        "Generate 4 diverse persona(s) that vary meaningfully along these attributes: "
        "priority, age, income. Return a JSON array of objects with eid and endow_text."
    )

    personas = json.loads(backend.chat(system="", user=prompt).answer_text)

    assert len(personas) == 4
    for persona in personas:
        text = persona["endow_text"]
        assert "a test town" in text
        assert ("Priority: green (loves plants)" in text) or ("Priority: night" in text)
        assert "Age: young" in text or "Age: old" in text
        assert "income" not in text.lower()  # unknown attributes are not invented


def test_attribute_discovery_ranks_relevant_world_attributes() -> None:
    backend = SimulatedBackend(world=WORLD)
    survey = backend.chat(system=SURVEY_ATTRIBUTE_PROMPT, user="Survey questions:\n" + QUESTION)
    question = backend.chat(system=QUESTION_ATTRIBUTE_PROMPT, user="Question:\n" + QUESTION)
    capped = backend.chat(
        system=SURVEY_ATTRIBUTE_PROMPT + "\nReturn no more than 0 attributes.",
        user="Survey questions:\n" + QUESTION,
    )

    assert json.loads(survey.answer_text) == ["priority"]
    assert json.loads(question.answer_text) == ["priority"]  # "age" prefers nothing here
    assert json.loads(capped.answer_text) == []


@pytest.mark.parametrize(
    "world, message",
    [
        ({"attributes": {"a": {"values": {}}}}, "at least one value"),
        ({"attributes": {"a": {"values": {"x": {"prefers": "Garden"}}}}}, "list of option texts"),
        ({"attributes": {"a": {"values": {"x": {"weight": "high"}}}}}, "weight must be a number"),
    ],
)
def test_malformed_worlds_are_refused(world, message) -> None:
    with pytest.raises(ValueError, match=message):
        SimulatedBackend(world=world)


def test_demo_profile_matches_the_demo_survey() -> None:
    profile = yaml.safe_load((ROOT / "config" / "backends" / "simulated_demo.yaml").read_text())
    with (ROOT / "examples" / "studio_demo" / "survey.csv").open(newline="") as handle:
        options = {
            text
            for row in csv.DictReader(handle)
            for text in eval(row["option_mapping"]).values()  # noqa: S307 - fixture CSV
        }
    for role in ("response", "endowment", "attribute_learner"):
        spec = profile[role]
        backend = create_backend(backend_type=spec["type"], model_name=spec["model_name"], **spec["kwargs"])
        assert isinstance(backend, SimulatedBackend)
    preferred = {
        option
        for attribute in profile["response"]["kwargs"]["world"]["attributes"].values()
        for value in attribute["values"].values()
        for option in value["prefers"]
    }
    assert preferred <= options  # every rule refers to an option the survey offers


def test_offline_pipeline_demo_runs_stage_one_and_fits(tmp_path: Path) -> None:
    config = Config.load(ROOT / "config" / "smoke" / "studio_demo_pipeline.yaml")
    pipeline_command.run(config, solver="CLARABEL", run_dir=tmp_path / "demo")
    run_dir = tmp_path / "demo"

    with (run_dir / "endowments.csv").open(newline="") as handle:
        agents = list(csv.DictReader(handle))
    modes = {agent["mode"] for agent in agents}
    with (run_dir / "responses.csv").open(newline="") as handle:
        cells = [cell for row in list(csv.reader(handle))[1:] for cell in row[1:]]
    metrics = json.loads((run_dir / "metrics.json").read_text())["metrics"]
    trajectory = (run_dir / "aggregate_entropy_trajectory.csv").read_text().splitlines()

    assert len(agents) == 60
    assert any(mode.startswith("question.") for mode in modes)  # question-targeted patches ran
    assert not any(cell.startswith("[ERROR]") or not cell for cell in cells)
    assert len(trajectory) == 1 + 5  # header + steps 0..4
    assert all(row["qids_used"] == row["qids_requested"] == 4 for row in metrics if row["split"] == "test")


def test_studio_demo_runs_the_pipeline_as_a_job(tmp_path: Path) -> None:
    for relative in ("config/smoke", "config/backends", "config/surveys", "examples"):
        shutil.copytree(ROOT / relative, tmp_path / relative)
    state = StudioState(tmp_path)

    job = state.start_demo()
    deadline = time.time() + 180
    while time.time() < deadline:
        current = state.jobs[job["id"]].as_dict()
        if current["status"] != "running":
            break
        time.sleep(0.5)

    assert current["status"] == "complete", "\n".join(current["log"][-20:])
    assert current["command"] == "pipeline"
    detail = state.detail(job["output"], "run")
    assert detail["health"]["warningCount"] == 0 or all(
        check["label"] != "Failed answers" for check in detail["health"]["checks"]
    )
    assert detail["summary"]["capabilities"]["generation"] is True
    assert detail["summary"]["population"]["generated"] == 60
