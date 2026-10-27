"""Backend-neutral attribute discovery for active generation."""

from __future__ import annotations

import ast
from collections.abc import Sequence
import logging
import re
import time


logger = logging.getLogger(__name__)


SURVEY_ATTRIBUTE_PROMPT = """
You are an intelligent research assistant trained to analyze survey questions
and infer which human attributes might influence how individuals respond.

You are given a set of training-only survey questions. Propose relevant human
attributes—such as demographics, beliefs, values, personality traits, or
ideological leanings—that are likely to affect responses. Focus on attributes
that are salient, discriminative, and variable across respondents.

Respond only with a Python-style list of double-quoted strings. Do not include
an explanation, header, or prose before or after the list.

Example:
["attribute1", "attribute2", "attribute3"]
""".strip()


QUESTION_ATTRIBUTE_PROMPT = """
You are an intelligent research assistant trained to analyze an individual
survey question and infer which human attributes might influence how different
people respond.

Propose relevant attributes—such as demographics, beliefs, values, personality
traits, or ideological leanings—that are likely to cause meaningful variation
in answers. Avoid generic or overly broad attributes.

Respond only with a Python-style list of double-quoted strings. Do not include
an explanation, header, or prose before or after the list.

Example:
["religious affiliation", "political ideology", "trust in government"]
""".strip()


class AttributeParseError(ValueError):
    """Raised when an attribute response contains no usable values."""


def _parse_attributes(
    value: str,
    *,
    maximum: int | None = None,
) -> list[str]:
    if not isinstance(value, str):
        raise TypeError("attribute response must be a string")

    match = re.search(r"\[.*?\]", value, re.DOTALL)
    if match is not None:
        try:
            parsed = ast.literal_eval(match.group())
        except (SyntaxError, ValueError):
            parsed = None
        if (
            isinstance(parsed, list)
            and all(isinstance(item, str) for item in parsed)
        ):
            normalized = list(
                dict.fromkeys(
                    item.strip()
                    for item in parsed
                    if item.strip()
                )
            )
            if normalized:
                return (
                    normalized[:maximum]
                    if maximum is not None
                    else normalized
                )

    # A list that opens but never closes was cut off (e.g. at the token limit);
    # splitting it would keep a partial last attribute, so it is unreadable and
    # is retried rather than used.
    if match is None and "[" in value:
        raise AttributeParseError(
            "attribute list was opened but never closed"
        )

    looks_structured = (
        match is not None
        or any(separator in value for separator in (",", ";", "\n"))
        or re.match(r"^\s*(?:[-*]|\d+[.)])\s+", value) is not None
    )
    if not looks_structured:
        raise AttributeParseError(
            "attribute backend returned no structured attribute list"
        )

    pieces = re.split(r"[,;\n]+", value)
    normalized = []
    for piece in pieces:
        attribute = (
            piece.strip()
            .lstrip("-*0123456789. ")
            .strip("[]'\"` ")
        )
        if attribute and attribute not in normalized:
            normalized.append(attribute)
    if maximum is not None:
        normalized = normalized[:maximum]
    if not normalized:
        raise AttributeParseError(
            "attribute backend returned no usable attributes"
        )
    return normalized


class AttributeLearner:
    """Discover concise survey attributes through one backend."""

    def __init__(
        self,
        *,
        survey,
        backend,
        tracker=None,
        max_attributes: int | None = None,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
    ) -> None:
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError(
                "max_retries must be a nonnegative integer"
            )
        if (
            isinstance(retry_base_delay, bool)
            or not isinstance(retry_base_delay, (int, float))
            or retry_base_delay < 0
        ):
            raise ValueError(
                "retry_base_delay must be nonnegative"
            )
        self.survey = survey
        self.backend = backend
        self.tracker = tracker
        self.max_attributes = max_attributes
        self.max_retries = max_retries
        self.retry_base_delay = float(retry_base_delay)
        self.qid_to_text = {
            question["id"]: survey.get_prompt_text(question)
            for question in survey.get_questions_by_split("train")
        }

    def _request(
        self,
        prompt: str,
        *,
        system: str,
        maximum: int | None = None,
    ) -> list[str]:
        if maximum is not None:
            system = (
                system
                + f"\nReturn no more than {maximum} attributes."
            )

        attempts = self.max_retries + 1
        for attempt in range(attempts):
            if attempt and self.retry_base_delay:
                time.sleep(
                    self.retry_base_delay
                    * (2 ** (attempt - 1))
                )
            user = prompt
            if attempt:
                user += (
                    "\n\nYour previous response could not be parsed. "
                    "Return only a nonempty Python list of quoted "
                    "attribute strings."
                )
            response = self.backend.chat(
                system=system,
                user=user,
            )
            if self.tracker is not None:
                suffix = (
                    f"_retry{attempt}"
                    if attempt
                    else ""
                )
                self.tracker.log(
                    usage=response.usage,
                    model_name=self.backend.model_name,
                    module_name="attribute_learner",
                    agent_id="attribute_learner" + suffix,
                )
            try:
                return _parse_attributes(
                    response.answer_text,
                    maximum=maximum,
                )
            except AttributeParseError:
                logger.warning(
                    "Attribute response was unusable on attempt %s/%s: %r",
                    attempt + 1,
                    attempts,
                    response.answer_text[:500],
                )

        # The legacy implementation treated malformed attribute output as an
        # empty discovery result. Preserve that scientific behavior after the
        # bounded formatting retries instead of aborting a paid pipeline.
        logger.error(
            "Attribute discovery exhausted %s attempt(s); "
            "continuing with no new attributes",
            attempts,
        )
        return []

    def generate_survey_attributes(
        self,
    ) -> list[str]:
        return self._request(
            "Survey questions:\n"
            + "\n".join(self.qid_to_text.values()),
            system=SURVEY_ATTRIBUTE_PROMPT,
            maximum=self.max_attributes,
        )

    def generate_question_attributes(
        self,
        qid: str,
        max_attributes: int | None = None,
    ) -> list[str]:
        question_text = self.qid_to_text.get(qid)
        if question_text is None:
            raise ValueError(f"unknown survey qid {qid!r}")
        return self._request(
            "Question:\n" + question_text,
            system=QUESTION_ATTRIBUTE_PROMPT,
            maximum=(
                max_attributes
                if max_attributes is not None
                else self.max_attributes
            ),
        )


__all__ = [
    "AttributeParseError",
    "AttributeLearner",
    "QUESTION_ATTRIBUTE_PROMPT",
    "SURVEY_ATTRIBUTE_PROMPT",
]
