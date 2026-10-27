"""Backend-neutral persona generation for active endowments."""

from __future__ import annotations

from itertools import count
import json
from numbers import Integral
import random
import re
import time

from p2p.backends.base import BackendResponse


SYSTEM_PROMPT = """You are an expert assistant trained to generate realistic, diverse, and demographically plausible personas for social science surveys.

Each persona should include:
- `eid`: a short, lowercase, variable-safe identifier.
- `endow_text`: a natural-language description of the persona (1-5 sentences).

Return only a JSON array. Avoid repeated phrasing or demographic combinations.
"""


class EndowmentModel:
    """Generate normalized persona records through one backend."""

    def __init__(
        self,
        *,
        backend,
        tracker=None,
        batch_size: int = 10,
        delay: float = 0.5,
        retry_failed: bool = True,
        randomize_attributes: bool = True,
        max_attributes: int = 12,
        seed: int | None = None,
        max_retries: int = 1,
        retry_base_delay: float = 0.0,
    ) -> None:
        for name, value in (
            ("batch_size", batch_size),
            ("max_attributes", max_attributes),
            ("max_retries", max_retries),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(delay, bool)
            or not isinstance(delay, (int, float))
            or delay < 0
        ):
            raise ValueError("delay must be nonnegative")
        if (
            isinstance(retry_base_delay, bool)
            or not isinstance(retry_base_delay, (int, float))
            or retry_base_delay < 0
        ):
            raise ValueError("retry_base_delay must be nonnegative")

        self.backend = backend
        self.tracker = tracker
        self.batch_size = batch_size
        self.delay = float(delay)
        self.retry_failed = retry_failed
        self.randomize_attributes = randomize_attributes
        self.max_attributes = max_attributes
        self.max_retries = max_retries
        self.retry_base_delay = float(retry_base_delay)
        self.random_state = random.Random(seed)
        self._ids = count(1)

    @staticmethod
    def _extract_personas(
        raw_text: str,
        *,
        requested: int,
    ) -> list[dict[str, str]]:
        match = re.search(
            r"\[\s*{.*?}\s*\]",
            raw_text,
            re.DOTALL,
        )
        # No complete array, e.g. a reply cut off at the token limit, is an
        # error, so the batch is retried rather than stored as persona text.
        if match is None:
            raise ValueError(
                "endowment backend returned no JSON persona array"
            )

        loaded = json.loads(match.group())
        if not isinstance(loaded, list):
            raise ValueError(
                "endowment backend persona payload must be a list"
            )

        personas = []
        for value in loaded[:requested]:
            if not isinstance(value, dict):
                raise ValueError(
                    "endowment backend personas must be objects"
                )
            text = value.get("endow_text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "endowment backend returned an empty persona"
                )
            personas.append(
                {
                    "endow_text": text.strip(),
                }
            )
        if not personas:
            raise ValueError(
                "endowment backend returned no usable personas"
            )
        return personas

    def _request_batch(
        self,
        *,
        attributes,
        requested: int,
        mode: tuple[str, ...],
        batch_number: int,
        retry: int = 0,
    ) -> list[dict[str, str]]:
        prompt = (
            f"Generate {requested} diverse persona(s) that vary "
            "meaningfully along these attributes: "
            + ", ".join(str(item) for item in attributes)
            + ". Return a JSON array of objects with eid and endow_text."
        )
        response = self.backend.chat(
            system=SYSTEM_PROMPT,
            user=prompt,
        )
        if not isinstance(response, BackendResponse):
            raise TypeError(
                "backend.chat(...) must return BackendResponse"
            )
        if self.tracker is not None:
            suffix = (
                f"_retry{retry}"
                if retry
                else ""
            )
            self.tracker.log(
                usage=response.usage,
                model_name=self.backend.model_name,
                module_name="endowment_model",
                agent_id=(
                    "endowment_model_mode_"
                    + "+".join(mode)
                    + f"_batch{batch_number}{suffix}"
                ),
            )
        return self._extract_personas(
            response.answer_text,
            requested=requested,
        )

    def generate_endowments(
        self,
        attributes,
        n: int,
        mode,
    ) -> list[dict[str, object]]:
        if (
            isinstance(n, bool)
            or not isinstance(n, Integral)
            or n <= 0
        ):
            raise ValueError("n must be a positive integer")
        n = int(n)
        normalized_mode = (
            (mode,)
            if isinstance(mode, str)
            else tuple(mode)
        )
        selected_attributes = list(attributes)
        if (
            self.randomize_attributes
            and len(selected_attributes) > self.max_attributes
        ):
            selected_attributes = self.random_state.sample(
                selected_attributes,
                k=self.max_attributes,
            )

        records = []
        for offset in range(0, n, self.batch_size):
            requested = min(self.batch_size, n - offset)
            batch_number = offset // self.batch_size + 1
            last_error: Exception | None = None
            attempts = (
                self.max_retries + 1
                if self.retry_failed
                else 1
            )
            personas = None
            for attempt in range(attempts):
                if attempt and self.retry_base_delay:
                    time.sleep(
                        self.retry_base_delay
                        * (2 ** (attempt - 1))
                    )
                try:
                    personas = self._request_batch(
                        attributes=selected_attributes,
                        requested=requested,
                        mode=normalized_mode,
                        batch_number=batch_number,
                        retry=attempt,
                    )
                    break
                except Exception as error:
                    last_error = error
            if personas is None:
                assert last_error is not None
                raise RuntimeError(
                    "endowment generation batch failed after "
                    f"{attempts} attempt(s)"
                ) from last_error

            for persona in personas:
                identifier = next(self._ids)
                records.append(
                    {
                        "eid": f"active_{identifier:06d}",
                        "endow_text": persona["endow_text"],
                        "role": "proxy",
                        "weight": None,
                        "mode": normalized_mode,
                        "attributes": list(
                            selected_attributes
                        ),
                    }
                )
            if self.delay:
                time.sleep(self.delay)
        return records


__all__ = [
    "EndowmentModel",
    "SYSTEM_PROMPT",
]
