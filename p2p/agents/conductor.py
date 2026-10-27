"""Survey conduction over backend-neutral agents."""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path
import time
from typing import Any

from tqdm import tqdm

from p2p.agents.base import AgentResponse
from p2p.agents.config import AgentConfig
from p2p.agents.factory import create_agent
from p2p.backends.base import TokenUsage


logger = logging.getLogger(__name__)


class SurveyConductor:
    """Elicit survey answers and account for normalized usage once."""

    def __init__(
        self,
        *,
        survey,
        endowments,
        output_path: str | None = None,
        agent_type: str = "dummy",
        model_name: str = "dummy",
        formality: str = "",
        cleaner=None,
        clean: bool = False,
        verbose: bool = True,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        token_tracker=None,
        agent_factory=create_agent,
    ) -> None:
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 1
        ):
            raise ValueError("max_retries must be a positive integer")
        if (
            isinstance(retry_delay, bool)
            or not isinstance(retry_delay, (int, float))
            or retry_delay < 0
        ):
            raise ValueError("retry_delay must be nonnegative")

        self.survey = survey
        self.endowments = endowments
        self.output_path = output_path
        self.agent_type = agent_type
        self.model_name = model_name
        self.formality = formality
        self.cleaner = cleaner
        self.clean = clean
        self.verbose = verbose
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.token_tracker = token_tracker
        self.agent_factory = agent_factory
        self.responses: list[dict[str, Any]] = []

    def _log(self, message: str, level: str = "info") -> None:
        if self.verbose:
            getattr(logger, level, logger.info)(message)

    @staticmethod
    def _is_error_response(answer: str) -> bool:
        return answer.strip().startswith("[ERROR]")

    def _account(self, response: AgentResponse) -> None:
        if self.token_tracker is not None:
            self.token_tracker.log(
                usage=response.usage,
                model_name=response.model_name,
                module_name="survey_conductor",
                agent_id=response.eid,
            )

    def _get_valid_response(
        self,
        agent,
        question_prompt: str,
        qid: str,
    ) -> tuple[str, AgentResponse | None]:
        last_answer = "[ERROR] No response"
        last_response: AgentResponse | None = None

        for attempt in range(self.max_retries):
            try:
                response = agent.answer(question_prompt)
                self._account(response)
                last_response = response
                answer = response.answer_text

                if self.clean:
                    if self.cleaner is None:
                        raise RuntimeError(
                            "cleaner is required when clean=True"
                        )
                    answer = self.cleaner.clean(answer, qid=qid)

                last_answer = answer
                if not self._is_error_response(answer):
                    return answer, response

            except Exception as error:
                last_answer = f"[ERROR] {error}"
                self._log(last_answer, level="warning")

            if attempt + 1 < self.max_retries and self.retry_delay:
                time.sleep(self.retry_delay * (2**attempt))

        return last_answer, last_response

    def run(
        self,
        *,
        save: bool = False,
        parallel_worker: bool = False,
        **agent_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run the survey sequentially over all endowments."""

        del parallel_worker
        self.responses = []

        config = AgentConfig(
            agent_type=self.agent_type,
            model_name=self.model_name,
            formality=self.formality,
            agent_kwargs=agent_kwargs,
        )

        endowments = self.endowments.endowments
        self._log(
            "Starting survey with "
            f"{len(endowments)} endowments × "
            f"{len(self.survey.questions)} questions"
        )
        iterator = (
            tqdm(
                endowments,
                desc="Survey agents",
                unit="agent",
            )
            if self.verbose
            else endowments
        )

        for endowment in iterator:
            eid = endowment["eid"]
            agent = self.agent_factory(
                config=config,
                eid=eid,
                endowment=endowment["endow_text"],
            )

            for question in self.survey.questions:
                qid = question["id"]
                prompt = self.survey.get_prompt_text(question)
                answer, response = self._get_valid_response(
                    agent,
                    prompt,
                    qid,
                )

                usage = (
                    response.usage
                    if response is not None
                    else TokenUsage()
                )
                response_model = (
                    response.model_name
                    if response is not None
                    else self.model_name
                )

                self.responses.append(
                    {
                        "eid": eid,
                        "qid": qid,
                        "question": prompt,
                        "answer": answer,
                        "model_name": response_model,
                        "usage": asdict(usage),
                    }
                )

        if save:
            self.save()

        self._log("Survey completed")
        return self.responses

    def to_agent_records(self) -> list[dict[str, Any]]:
        """Convert flat answers to vectors ordered like the survey."""

        qids = [question["id"] for question in self.survey.questions]
        answers_by_eid: dict[str, dict[str, str]] = {}

        for response in self.responses:
            answers_by_eid.setdefault(response["eid"], {})[
                response["qid"]
            ] = response["answer"]

        records = []
        for endowment in self.endowments.endowments:
            eid = endowment["eid"]
            mode = ()
            get_mode = getattr(
                self.endowments,
                "get_mode_by_eid",
                None,
            )
            if callable(get_mode):
                try:
                    mode = get_mode(eid)
                except KeyError:
                    mode = ()

            records.append(
                {
                    "agent_id": eid,
                    "mode": mode,
                    "responses": [
                        answers_by_eid.get(eid, {}).get(qid)
                        for qid in qids
                    ],
                }
            )

        return records

    def save(self) -> None:
        """Save normalized flat responses as JSON."""

        if self.output_path is None:
            raise ValueError(
                "output_path is required when save=True"
            )

        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.responses, indent=2),
            encoding="utf-8",
        )


__all__ = ["SurveyConductor"]
