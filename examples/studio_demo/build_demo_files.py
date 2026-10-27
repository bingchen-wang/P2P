#!/usr/bin/env python3
"""Write the license-free Studio demo files: survey, synthetic agent basis, targets.

Everything here is fictional and deterministic. Fifteen synthetic agents fall
into three archetypes; each agent mostly gives its archetype's favourite answer
and departs from it on three questions. The "human" targets are a known mixture,
50% / 30% / 20% of the three archetypes plus a 10% uniform share, so a fitted
Stage 2 has something real to recover:

- lasso should weight the agents roughly in proportion to the mixture;
- the reference-respondent configuration should find the uniform share;
- the simple average weights every agent equally.

It also writes the inputs for a simulated Stage 1, so the Studio demo can run
the whole pipeline offline:

- config/backends/simulated_demo.yaml: a backend profile for the rule-based
  ``simulated`` backend (not a language model), whose world gives each
  archetype ("everyday priority") the same favourite answers, plus weaker
  traits that matter for a few questions;
- examples/studio_demo/attribute_bank.yaml: the attributes active generation
  varies.

Run from the repository root: python examples/studio_demo/build_demo_files.py
"""

from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

# qid, question, options (A, B, C), split, favourite option per archetype
# (green, night, practical).
QUESTIONS = [
    ("AMENITY_01", "Which fictional neighborhood feature should receive priority?",
     ("A shared garden", "A quiet reading room", "A night market"), "train", "ACB"),
    ("AMENITY_02", "Which transit improvement would you choose first?",
     ("More frequent trams", "Protected cycle lanes", "Wider footpaths"), "train", "BAC"),
    ("AMENITY_03", "Which community event sounds most useful?",
     ("Repair cafe", "Street festival", "Public lecture"), "valid", "ABC"),
    ("AMENITY_04", "Which library service should expand?",
     ("Tool lending", "Longer opening hours", "Language exchange"), "valid", "ACB"),
    ("AMENITY_05", "Which park addition would improve everyday use?",
     ("More shade trees", "Outdoor tables", "A small stage"), "test", "ACB"),
    ("AMENITY_06", "Which local information format is easiest to use?",
     ("Printed noticeboard", "Weekly email", "Mobile bulletin"), "test", "ABC"),
    ("AMENITY_07", "Which weekend activity space should open first?",
     ("Community sports court", "Indoor climbing wall", "Board-game lounge"), "train", "ACB"),
    ("AMENITY_08", "Which recycling service is most useful?",
     ("Kerbside compost bins", "Electronics drop-off", "Clothing swap shelf"), "train", "ACB"),
    ("AMENITY_09", "Which public art project should be funded?",
     ("Mural walk", "Light installation", "Sculpture garden"), "train", "CBA"),
    ("AMENITY_10", "Which street safety measure should come first?",
     ("Better lighting", "Raised crossings", "Speed cameras"), "train", "BAC"),
    ("AMENITY_11", "Which food option should the market add?",
     ("Local produce stall", "Hot food vans", "Bakery corner"), "train", "ABC"),
    ("AMENITY_12", "Which youth programme should expand?",
     ("Coding club", "Music lessons", "Outdoor skills"), "train", "CBA"),
    ("AMENITY_13", "Which energy upgrade should the town hall make?",
     ("Solar panels", "Heat pumps", "Better insulation"), "train", "ABC"),
    ("AMENITY_14", "Which way should residents be consulted?",
     ("Town meeting", "Online survey", "Street pop-up stand"), "valid", "ACB"),
    ("AMENITY_15", "Which health service should open locally?",
     ("Walk-in clinic", "Longer pharmacy hours", "Fitness classes"), "train", "CBA"),
    ("AMENITY_16", "Which waterfront change would you choose?",
     ("Swimming platform", "Boardwalk cafe", "Wildlife viewing deck"), "test", "CBA"),
    ("AMENITY_17", "Which shared resource should the library stock?",
     ("Camping gear", "Musical instruments", "Sewing machines"), "train", "ABC"),
    ("AMENITY_18", "Which late-night service matters most?",
     ("Night bus", "Late pharmacy", "Late study hall"), "test", "CAB"),
]

ARCHETYPES = ("green", "night", "practical")
MIXTURE = {"green": 0.5, "night": 0.3, "practical": 0.2}
UNIFORM_SHARE = 0.10

PERSONAS = {
    "green": [
        "Lives beside the river park, grows vegetables on an allotment and cycles everywhere.",
        "A retired science teacher who volunteers at the tree-planting group on Saturdays.",
        "Works from home, walks the dog twice a day and wants more shade and wildlife nearby.",
        "A student who organises the campus repair cafe and avoids buying anything new.",
        "Runs a small landscaping business and prefers outdoor, low-energy solutions.",
    ],
    "night": [
        "A nurse on late shifts who values evening services and lively public spaces.",
        "Plays in a local band and spends most evenings at venues and night markets.",
        "Recently moved to town and looks for festivals, clubs and ways to meet people.",
        "Runs a food truck and wants the town centre busy after dark.",
        "A graduate student who studies late and enjoys cultural events.",
    ],
    "practical": [
        "Commutes by car, manages family schedules on a phone and values convenience.",
        "A software developer who prefers digital services and clear online information.",
        "A parent of three who wants services that save time and work reliably.",
        "Works in logistics and judges every proposal by cost and efficiency.",
        "A small-business owner who wants straightforward, well-maintained facilities.",
    ],
}


# The simulated town. "everyday priority" carries the archetypes; the other
# traits pull a few answers, so question-targeted generation has something to
# find. Option texts must match QUESTIONS exactly.
PRIORITY_VALUES = {
    "green": "green and outdoors",
    "night": "evenings and culture",
    "practical": "convenience and efficiency",
}
DESCRIPTIONS = {
    "green and outdoors": "gardens, cycles and wants more shade and wildlife",
    "evenings and culture": "out most evenings at markets, gigs and festivals",
    "convenience and efficiency": "values services that save time and work reliably",
    "20s": "early in a career", "40s": "mid-career", "60s and over": "recently retired",
    "cycles": "rides to work most days", "takes the tram": "relies on public transport",
    "drives": "commutes by car",
    "lives alone": "a one-person household", "family with children": "two school-age children",
}
MINOR_TRAITS = {
    "age group": {
        "20s": ["Board-game lounge", "Music lessons", "Late study hall", "Street festival"],
        "40s": ["Outdoor tables", "Weekly email"],
        "60s and over": ["More shade trees", "Printed noticeboard", "Walk-in clinic", "Town meeting"],
    },
    "travel to work": {
        "cycles": ["Protected cycle lanes", "Raised crossings"],
        "takes the tram": ["More frequent trams", "Night bus"],
        "drives": ["Wider footpaths", "Speed cameras"],
    },
    "household": {
        "lives alone": ["Language exchange", "Late pharmacy"],
        "family with children": ["Outdoor skills", "Community sports court", "Swimming platform"],
    },
}


def simulated_world() -> dict:
    favourites = {
        archetype: [
            options["ABC".index(favourite[index])]
            for _, _, options, _, favourite in QUESTIONS
        ]
        for index, archetype in enumerate(ARCHETYPES)
    }
    attributes = {
        "everyday priority": {
            "relevance": "survey",
            "values": {
                PRIORITY_VALUES[archetype]: {
                    "weight": 2.5,
                    "prefers": favourites[archetype],
                    "describe": DESCRIPTIONS[PRIORITY_VALUES[archetype]],
                }
                for archetype in ARCHETYPES
            },
        }
    }
    for trait, values in MINOR_TRAITS.items():
        attributes[trait] = {
            "values": {
                value: {"weight": 0.8, "prefers": prefers, "describe": DESCRIPTIONS[value]}
                for value, prefers in values.items()
            }
        }
    return {"setting": "the fictional town of Linden", "attributes": attributes}


def agents() -> list[tuple[str, str, str]]:
    return [
        (f"{archetype}_{index + 1:02d}", archetype, text)
        for archetype in ARCHETYPES
        for index, text in enumerate(PERSONAS[archetype])
    ]


def answer(archetype: str, agent_index: int, question_index: int) -> int:
    """Option index (0, 1, 2) one agent gives to one question."""

    favourite = "ABC".index(QUESTIONS[question_index][4][ARCHETYPES.index(archetype)])
    shift = ARCHETYPES.index(archetype)
    departures = {(agent_index * 3 + shift * 2 + step * 5) % len(QUESTIONS) for step in range(3)}
    return (favourite + 1) % 3 if question_index in departures else favourite


def main() -> None:
    roster = agents()
    with (HERE / "survey.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["qid", "question", "option_mapping", "split"])
        for qid, question, options, split, _ in QUESTIONS:
            mapping = {code: text for code, text in zip("ABC", options)}
            writer.writerow([qid, question, repr(mapping), split])

    basis = HERE / "basis"
    basis.mkdir(exist_ok=True)
    with (basis / "endowments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["eid", "endow_text", "role", "weight", "mode", "attributes"])
        for eid, archetype, text in roster:
            writer.writerow([eid, text, "proxy", "", archetype, "everyday priorities"])
    with (basis / "responses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["qid", *(eid for eid, _, _ in roster)])
        for question_index, (qid, _, options, _, _) in enumerate(QUESTIONS):
            row = [qid]
            for eid, archetype, _ in roster:
                agent_index = int(eid.rsplit("_", 1)[1]) - 1
                row.append(options[answer(archetype, agent_index, question_index)])
            writer.writerow(row)

    targets: dict[str, dict[str, float]] = {}
    predictions: dict[str, dict[str, float]] = {}
    for question_index, (qid, _, options, _, _) in enumerate(QUESTIONS):
        shares = [UNIFORM_SHARE / 3] * 3
        for archetype, weight in MIXTURE.items():
            members = [eid for eid, kind, _ in roster if kind == archetype]
            for eid in members:
                agent_index = int(eid.rsplit("_", 1)[1]) - 1
                choice = answer(archetype, agent_index, question_index)
                shares[choice] += (1 - UNIFORM_SHARE) * weight / len(members)
        rounded = [round(value, 2) for value in shares]
        rounded[rounded.index(max(rounded))] += round(1 - sum(rounded), 2)
        targets[qid] = {option: round(value, 2) for option, value in zip(options, rounded)}
        # A fixed, slightly-off forecast for every question, for the
        # evaluation-only demo (evaluate needs a prediction per question).
        nudged = [rounded[0] - 0.04, rounded[1] + 0.03, rounded[2] + 0.01]
        predictions[qid] = {option: round(value, 2) for option, value in zip(options, nudged)}

    world = simulated_world()
    profile = {"type": "simulated", "model_name": "simulated-respondents",
               "kwargs": {"seed": 101, "noise": 0.6, "world": world}}
    header = (
        "# Simulated respondents for the offline Studio demo: deterministic rules,\n"
        "# not a language model. Written by examples/studio_demo/build_demo_files.py.\n"
    )
    (HERE.parents[1] / "config" / "backends" / "simulated_demo.yaml").write_text(
        header + yaml.safe_dump(
            {
                "response": {**deepcopy(profile), "formality": "Select one listed option. Return only its option text."},
                "endowment": deepcopy(profile),
                "attribute_learner": deepcopy(profile),
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (HERE / "attribute_bank.yaml").write_text(
        "# Attributes active generation varies in the simulated Studio demo.\n"
        + yaml.safe_dump(
            {
                "templates": {"core": ["core"], "thematic": ["household", "travel"]},
                "attributes": {
                    "core": ["everyday priority", "age group"],
                    "household": ["household"],
                    "travel": ["travel to work"],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    (HERE / "targets.json").write_text(json.dumps(targets, indent=2) + "\n", encoding="utf-8")
    (HERE / "predictions.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": "offline-demo",
                "cleaning_policy": "license-free synthetic example",
                "predictions": predictions,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
