"""Durable, config-locked checkpoints for paid active generation."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any, Callable

import yaml

from p2p.endowments import ActiveEndowments
from p2p.responses import Responses


SCHEMA_VERSION = 1
CHECKPOINT_DIRNAME = "active_generation.checkpoint"
STATE_FILENAME = "state.json"


def _atomic_write(
    path: Path,
    writer: Callable[[Path], None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class ActiveGenerationCheckpoint:
    """Persist enough state to resume without repeating completed LLM work."""

    def __init__(
        self,
        *,
        run_dir: str | os.PathLike[str],
        config_digest: str,
        survey,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.directory = (
            self.run_dir / CHECKPOINT_DIRNAME
        ).resolve()
        self.state_path = (
            self.directory / STATE_FILENAME
        ).resolve()
        self.config_digest = config_digest
        self.survey = survey

    def load(self) -> dict[str, Any] | None:
        """Load a compatible checkpoint or fail closed on a mismatch."""

        if not self.state_path.is_file():
            return None
        try:
            payload = json.loads(
                self.state_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "active-generation checkpoint is unreadable: "
                f"{self.state_path}"
            ) from error

        if not isinstance(payload, Mapping):
            raise RuntimeError(
                "active-generation checkpoint root must be a mapping"
            )
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                "unsupported active-generation checkpoint schema"
            )
        if payload.get("config_digest") != self.config_digest:
            raise RuntimeError(
                "active-generation checkpoint belongs to a different "
                "resolved configuration; use a fresh output directory"
            )
        state = payload.get("state")
        if not isinstance(state, Mapping):
            raise RuntimeError(
                "active-generation checkpoint state must be a mapping"
            )
        return dict(state)

    def save(self, state: Mapping[str, Any]) -> Path:
        """Atomically commit resumable state and human-readable partial basis."""

        if not isinstance(state, Mapping):
            raise TypeError("checkpoint state must be a mapping")
        state = dict(state)
        endowments = state.get("endowments")
        responses = state.get("response_log")
        attribute_bank = state.get("attribute_bank")
        if not isinstance(endowments, list):
            raise ValueError(
                "checkpoint endowments must be a list"
            )
        if not isinstance(responses, list):
            raise ValueError(
                "checkpoint response_log must be a list"
            )
        if not isinstance(attribute_bank, Mapping):
            raise ValueError(
                "checkpoint attribute_bank must be a mapping"
            )

        self.directory.mkdir(parents=True, exist_ok=True)
        endowments_path = self.directory / "endowments.csv"
        responses_path = self.directory / "responses.csv"
        bank_path = self.directory / "attribute_bank.yaml"

        manager = ActiveEndowments.from_endowment_list(
            endowments
        )
        _atomic_write(
            endowments_path,
            lambda path: manager.save(path),
        )
        normalized_responses = [
            {
                **record,
                "mode": tuple(record.get("mode", ())),
            }
            for record in responses
        ]
        response_matrix = Responses.from_agent_records(
            normalized_responses,
            self.survey,
            output_format="answer",
            clean=True,
        )
        _atomic_write(
            responses_path,
            lambda path: response_matrix.save(path),
        )
        _atomic_write(
            bank_path,
            lambda path: path.write_text(
                yaml.safe_dump(
                    dict(attribute_bank),
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            ),
        )

        payload = {
            "schema_version": SCHEMA_VERSION,
            "config_digest": self.config_digest,
            "status": (
                "generation_complete"
                if state.get("generation_complete")
                else "in_progress"
            ),
            "completed_agents": len(responses),
            "generated_agents": len(endowments),
            "state": state,
        }
        _atomic_write(
            self.state_path,
            lambda path: path.write_text(
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            ),
        )
        return self.state_path


__all__ = [
    "ActiveGenerationCheckpoint",
    "CHECKPOINT_DIRNAME",
    "SCHEMA_VERSION",
    "STATE_FILENAME",
]
