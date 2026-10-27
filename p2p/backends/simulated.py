"""Simulated respondents: a deterministic, rule-based backend. Not a language model.

It lets Stage 1 run end to end offline (persona generation, attribute
discovery, survey answers), for demonstrations and tests. Behaviour comes from
a declarative ``world``:

.. code-block:: yaml

   world:
     setting: a fictional town              # used in persona text (optional)
     attributes:
       everyday priority:                   # attribute name
         relevance: survey                  # offered by survey-level discovery
         values:
           green and outdoors:              # value; appears verbatim in personas
             weight: 2.0                    # how strongly it pulls answers
             prefers: [A shared garden, Protected cycle lanes]
             describe: grows vegetables and cycles everywhere   # optional
           evenings and culture:
             prefers: [A night market]
       age group:
         values: {20s: {}, 40s: {}, 60s: {}}

Answers. For a question offered as ``text ([A] option, [B] option, ...)``, each
option scores the summed weight of the persona's values that prefer it, plus
deterministic noise (Gumbel, scale ``noise``) from a hash of the persona, the
question and the option. The highest score is answered. With no matching
preferences the answer is the noise alone, so any survey can be answered.

Personas. A request to vary named attributes gets one value per attribute that
the world knows, drawn with a seeded generator; unknown attributes are named but
left unvalued. Attribute discovery returns the world's attributes, those whose
values prefer an option of the question first.

Usage is reported as zero tokens: nothing is metered.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping
from typing import Any

from p2p.backends.base import BackendResponse, TokenUsage
from p2p.registry import BACKENDS

_OPTION = re.compile(r"\[([^\]]+)\]\s(.*?)(?=,\s\[[^\]]+\]\s|\)\s*$)", re.DOTALL)
_PERSONA_REQUEST = re.compile(r"vary meaningfully along these attributes:\s*(.*?)\.\s*Return", re.DOTALL)
_REQUESTED = re.compile(r"Generate\s+(\d+)\s+diverse persona")
_MAXIMUM = re.compile(r"Return no more than (\d+) attributes")


class SimulatedBackend:
    """Answer P2P's Stage-1 prompts with deterministic rules (see module docstring)."""

    def __init__(
        self,
        *,
        model_name: str = "simulated-respondents",
        world: Mapping[str, Any] | None = None,
        seed: int = 0,
        noise: float = 1.0,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a nonempty string")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if isinstance(noise, bool) or not isinstance(noise, (int, float)) or noise < 0:
            raise ValueError("noise must be a nonnegative number")
        self.model_name = model_name
        self.seed = seed
        self.noise = float(noise)
        self.setting = ""
        self.attributes: dict[str, dict[str, dict[str, Any]]] = {}
        self.relevance: dict[str, str] = {}
        self._load_world(world or {})
        self._persona_counter = 0

    # -- world --------------------------------------------------------------

    def _load_world(self, world: Mapping[str, Any]) -> None:
        if not isinstance(world, Mapping):
            raise ValueError("world must be a mapping")
        setting = world.get("setting", "")
        self.setting = setting if isinstance(setting, str) else ""
        attributes = world.get("attributes", {})
        if not isinstance(attributes, Mapping):
            raise ValueError("world.attributes must be a mapping")
        for name, spec in attributes.items():
            spec = spec or {}
            values = spec.get("values", {}) if isinstance(spec, Mapping) else {}
            if not isinstance(values, Mapping) or not values:
                raise ValueError(f"world attribute {name!r} needs at least one value")
            parsed: dict[str, dict[str, Any]] = {}
            for value, rule in values.items():
                rule = rule or {}
                if not isinstance(rule, Mapping):
                    raise ValueError(f"world value {name}: {value} must be a mapping")
                weight = rule.get("weight", 1.0)
                prefers = rule.get("prefers", [])
                if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                    raise ValueError(f"world value {name}: {value} weight must be a number")
                if not isinstance(prefers, list) or not all(isinstance(item, str) for item in prefers):
                    raise ValueError(f"world value {name}: {value} prefers must be a list of option texts")
                describe = rule.get("describe", "")
                parsed[str(value)] = {
                    "weight": float(weight),
                    "prefers": {item.strip().lower() for item in prefers},
                    "describe": describe if isinstance(describe, str) else "",
                }
            self.attributes[str(name)] = parsed
            relevance = spec.get("relevance", "question") if isinstance(spec, Mapping) else "question"
            self.relevance[str(name)] = str(relevance)

    # -- helpers ------------------------------------------------------------

    def _uniform(self, *parts: str) -> float:
        digest = hashlib.sha256("\x1f".join((str(self.seed), *parts)).encode("utf-8")).digest()
        return (int.from_bytes(digest[:8], "big") + 0.5) / 2**64

    def _persona_values(self, persona: str) -> list[tuple[str, str]]:
        lowered = persona.lower()
        return [
            (attribute, value)
            for attribute, values in self.attributes.items()
            for value in values
            if f"{attribute.lower()}: {value.lower()}" in lowered
        ]

    # -- the three kinds of request -----------------------------------------

    def _answer(self, persona: str, question: str) -> str:
        options = [(code.strip(), text.strip()) for code, text in _OPTION.findall(question)]
        if not options:
            return ""
        values = self._persona_values(persona)
        best_text, best_score = options[0][1], -math.inf
        for _, text in options:
            score = sum(
                self.attributes[attribute][value]["weight"]
                for attribute, value in values
                if text.lower() in self.attributes[attribute][value]["prefers"]
            )
            gumbel = -math.log(-math.log(self._uniform(persona, question, text)))
            score += self.noise * gumbel
            if score > best_score:
                best_text, best_score = text, score
        return best_text

    def _personas(self, request: str) -> str:
        names_match = _PERSONA_REQUEST.search(request)
        count_match = _REQUESTED.search(request)
        names = [item.strip() for item in names_match.group(1).split(",")] if names_match else []
        count = int(count_match.group(1)) if count_match else 1
        personas = []
        for _ in range(count):
            self._persona_counter += 1
            rng = random.Random(f"{self.seed}:{self._persona_counter}:{request}")
            facts = []
            for name in names:
                values = self.attributes.get(name) or next(
                    (spec for key, spec in self.attributes.items() if key.lower() == name.lower()),
                    None,
                )
                if values:
                    key = next(key for key in self.attributes if key.lower() == name.lower())
                    value = rng.choice(sorted(values))
                    describe = values[value]["describe"]
                    facts.append(
                        f"{key[:1].upper()}{key[1:]}: {value}" + (f" ({describe})" if describe else "")
                    )
            where = f" in {self.setting}" if self.setting else ""
            text = f"A simulated resident{where}." + (" " + ". ".join(facts) + "." if facts else "")
            personas.append({"eid": f"simulated_{self._persona_counter:05d}", "endow_text": text})
        return json.dumps(personas)

    def _attributes(self, system: str, request: str) -> str:
        maximum_match = _MAXIMUM.search(system)
        maximum = int(maximum_match.group(1)) if maximum_match else None
        survey_level = request.lstrip().startswith("Survey questions:")
        options = {text.strip().lower() for _, text in _OPTION.findall(request)}
        ranked = []
        for name, values in self.attributes.items():
            relevant = any(rule["prefers"] & options for rule in values.values())
            if survey_level and self.relevance[name] != "survey":
                continue
            if not survey_level and not relevant:
                continue
            ranked.append(name)
        if maximum is not None:
            ranked = ranked[:maximum]
        return json.dumps(ranked)

    def chat(self, *, system: str, user: str) -> BackendResponse:
        """Answer one prompt by rule; usage is always zero."""

        stripped = user.lstrip()
        if _REQUESTED.search(stripped) and _PERSONA_REQUEST.search(stripped):
            text = self._personas(stripped)
        elif stripped.startswith("Survey questions:") or stripped.startswith("Question:"):
            text = self._attributes(system, stripped)
        else:
            text = self._answer(system, user)
        return BackendResponse(answer_text=text, usage=TokenUsage())


BACKENDS.register("simulated", SimulatedBackend)


__all__ = ["SimulatedBackend"]
