import json
import logging
import re
import time
import uuid
from typing import Any, Callable, List, Optional

from p2p.backends.base import BackendResponse


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert assistant trained to generate realistic, diverse, and demographically plausible personas for social science surveys.

Each persona should include:
- `eid`: a short, lowercase, variable-safe identifier that encodes key traits (e.g., urban_liberal_30s_female). No punctuation or spaces.
- `endow_text`: a natural language description of the persona (1–5 sentences), written as if describing a survey respondent.

Instructions:
- Represent a wide range of age, gender, race/ethnicity, education, region, and political ideology.
- Avoid repetition of phrasing or demographic combinations across personas.
- Do not include explanations or formatting outside of the JSON array.
"""


class EndowmentGenerator:
    """
    Generate baseline personas without entropy guidance or learned attributes.

    Backend resolution is injected so this module remains independent of
    provider registries and agent integrations. Callers may provide either an
    initialized backend or a factory that resolves one. Backend calls must
    return the normalized ``BackendResponse`` contract.

    Attributes:
        backend_type: Provider identifier passed to an injected backend factory.
        model_name: Model identifier used for generation and usage reporting.
        temperature: Sampling temperature.
        max_tokens: Maximum output-token setting passed to a backend factory.
        batch_size: Number of personas requested per backend call.
        delay: Delay between generation batches.
        system_prompt: System instruction for persona generation.
        retry_failed: Whether failed prompts should be retried.
        clear_after_run: Whether accumulated state is cleared after generation.
        token_tracker: Optional injected object exposing normalized ``log(...)``.
        backend: Injected or factory-created object exposing ``chat(...)``.
    """

    def __init__(
        self,
        backend_type: str = "openai",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.9,
        max_tokens: int = 1024,
        batch_size: int = 10,
        delay: float = 0.5,
        system_prompt: str = SYSTEM_PROMPT,
        retry_failed: bool = True,
        clear_after_run: bool = False,
        token_tracker: Any | None = None,
        verbose: bool = True,
        max_retries: int = 10,
        retry_base_delay: float = 10.0,
        backend: Any | None = None,
        backend_factory: Callable[..., Any] | None = None,
        **backend_kwargs,
    ):
        self.backend_type = backend_type
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.delay = delay
        self.system_prompt = system_prompt
        self.retry_failed = retry_failed
        self.clear_after_run = clear_after_run
        self.token_tracker = token_tracker
        self.verbose = verbose
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

        if backend is not None and backend_factory is not None:
            raise ValueError(
                "Provide either backend or backend_factory, not both"
            )

        if backend is not None:
            self.backend = backend
        elif backend_factory is not None:
            self.backend = backend_factory(
                backend_type=self.backend_type,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **backend_kwargs,
            )
        else:
            raise ValueError(
                "backend or backend_factory must be provided"
            )

        self.personas: List[dict] = []
        self.failed_prompts: List[str] = []

    def _log(self, msg: str, level: str = "info"):
        if self.verbose:
            log_fn = getattr(
                logger,
                level.lower(),
                logger.info,
            )
            log_fn(msg)

    def _extract_json_array(self, text: str) -> List[dict]:
        match = re.search(
            r"\[\s*{.*?}\s*\]",
            text,
            re.DOTALL,
        )
        if not match:
            raise ValueError(
                "No JSON array found in response."
            )
        return json.loads(match.group())

    def _build_prompt(
        self,
        n: int,
        topic: Optional[str] = None,
    ) -> str:
        topic_line = (
            f"The survey topic is: {topic}\n"
            if topic
            else ""
        )
        return (
            f"Generate {n} diverse personas for a survey experiment. "
            "Each persona must include:\n"
            "- eid: short, lowercase identifier\n"
            "- endow_text: a short natural-language description\n\n"
            f"{topic_line}Ensure diversity across demographics. "
            "Return the result as a JSON array."
        )

    def _chat_and_account(
        self,
        *,
        prompt: str,
        agent_id: str,
    ) -> str:
        """Call the backend and account for its normalized usage once."""

        response = self.backend.chat(
            system=self.system_prompt,
            user=prompt,
        )
        if not isinstance(response, BackendResponse):
            raise TypeError(
                "backend.chat(...) must return BackendResponse"
            )

        if self.token_tracker is not None:
            self.token_tracker.log(
                usage=response.usage,
                model_name=self.backend.model_name,
                module_name="endowment_generator",
                agent_id=agent_id,
            )

        return response.answer_text

    def generate(
        self,
        n: int,
        topic_description: Optional[str] = None,
    ) -> List[dict]:
        full_batch = []

        for i in range(0, n, self.batch_size):
            curr_n = min(self.batch_size, n - i)
            prompt = self._build_prompt(
                curr_n,
                topic_description,
            )
            batch_number = i // self.batch_size + 1

            try:
                raw_output = self._chat_and_account(
                    prompt=prompt,
                    agent_id=(
                        f"endowment_generator_batch"
                        f"{batch_number}"
                    ),
                )

                batch = self._extract_json_array(
                    raw_output
                )
                suffix = uuid.uuid4().hex[:6]

                for persona in batch:
                    if "eid" in persona:
                        persona["eid"] = (
                            f"{persona['eid']}_{suffix}"
                        )

                self.personas.extend(batch)
                full_batch.extend(batch)
                self._log(
                    f"✓ Batch {batch_number}: "
                    f"{len(batch)} personas generated."
                )

            except Exception as error:
                self._log(
                    f"✗ Batch {batch_number} failed: "
                    f"{error}",
                    level="warning",
                )
                self.failed_prompts.append(prompt)

            time.sleep(self.delay)

        if self.retry_failed and self.failed_prompts:
            recovered = self.retry_failed_batches()
            full_batch.extend(recovered)

        if self.clear_after_run:
            self.clear()

        return full_batch

    def retry_failed_batches(self) -> List[dict]:
        self._log("\nRetrying failed prompts...")
        recovered = []

        for i, prompt in enumerate(
            self.failed_prompts[:]
        ):
            for attempt in range(self.max_retries):
                wait_time = (
                    self.retry_base_delay
                    * (2**attempt)
                )
                self._log(
                    f"Retry {i + 1}, attempt "
                    f"{attempt + 1}/{self.max_retries}: "
                    f"waiting {wait_time:.0f}s..."
                )
                time.sleep(wait_time)

                try:
                    raw_output = self._chat_and_account(
                        prompt=prompt,
                        agent_id=(
                            f"endowment_generator_retry"
                            f"{i + 1}"
                        ),
                    )

                    batch = self._extract_json_array(
                        raw_output
                    )
                    suffix = uuid.uuid4().hex[:6]

                    for persona in batch:
                        if "eid" in persona:
                            persona["eid"] = (
                                f"{persona['eid']}_"
                                f"{suffix}"
                            )

                    self.personas.extend(batch)
                    recovered.extend(batch)
                    self.failed_prompts.remove(prompt)
                    self._log(
                        f"✓ Retry {i + 1}: "
                        f"{len(batch)} personas recovered."
                    )
                    break

                except Exception as error:
                    self._log(
                        f"✗ Retry {i + 1}, attempt "
                        f"{attempt + 1} failed: {error}",
                        level="warning",
                    )
                    if (
                        attempt
                        == self.max_retries - 1
                    ):
                        self._log(
                            f"✗ Giving up on prompt "
                            f"{i + 1} after "
                            f"{self.max_retries} attempts",
                            level="warning",
                        )

        return recovered

    def save(self, filepath: str):
        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as output_file:
            for persona in self.personas:
                output_file.write(
                    json.dumps(persona) + "\n"
                )

    def clear(self):
        self.personas.clear()
        self.failed_prompts.clear()