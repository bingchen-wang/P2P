"""Contracts for resolved attribute-bank resources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from p2p.config import Config, ConfigValidationError


DUMMY_BACKEND = {
    "response": {
        "type": "dummy",
        "model_name": "dummy",
    },
    "endowment": {
        "type": "dummy",
        "model_name": "dummy",
    },
    "attribute_learner": {
        "type": "dummy",
        "model_name": "dummy",
    },
}


def write_yaml(path: Path, value: Any) -> Path:
    """Write one YAML fixture and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )
    return path


def write_active_manifest(
    root: Path,
    *,
    bank_source: str | None = None,
) -> Path:
    """Create a minimal active-generation manifest."""
    write_yaml(
        root / "survey.yaml",
        {
            "question_id": "id",
            "question_text": "question",
            "code_to_answer": "option_mapping",
        },
    )

    generation: dict[str, Any] = {
        "mode": "active",
    }
    if bank_source is not None:
        generation["attribute_bank"] = {
            "source": bank_source,
        }

    return write_yaml(
        root / "manifest.yaml",
        {
            "run": {
                "name": "active-bank-test",
                "seed": 101,
            },
            "data": {
                "survey": "survey.yaml",
            },
            "generation": generation,
            "backend": DUMMY_BACKEND,
        },
    )


def valid_bank() -> dict[str, Any]:
    return {
        "templates": {
            "core": ["core"],
        },
        "attributes": {
            "core": [
                "age",
                "education",
            ],
        },
    }


def test_active_generation_omitting_bank_resolves_builtin_default(
    tmp_path: Path,
) -> None:
    manifest = write_active_manifest(tmp_path)

    config = Config.load(manifest)

    assert config.validate_for("generate") is config
    assert config.attribute_bank is not None
    assert config.attribute_bank.source == "builtin:default"
    assert len(config.attribute_bank.source_digest) == 64
    assert config.attribute_bank.data["templates"]
    assert config.attribute_bank.data["attributes"]


def test_relative_bank_is_resolved_from_declaring_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_dir = tmp_path / "manifests"
    bank_path = write_yaml(
        manifest_dir / "resources" / "custom.yaml",
        valid_bank(),
    )
    manifest = write_active_manifest(
        manifest_dir,
        bank_source="resources/custom.yaml",
    )

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    config = Config.load(manifest)

    assert config.attribute_bank is not None
    assert config.attribute_bank.source == str(bank_path.resolve())
    assert config.attribute_bank.data["attributes"]["core"] == (
        "age",
        "education",
    )


def test_mutable_bank_copy_is_detached_from_resolved_source(
    tmp_path: Path,
) -> None:
    bank_path = write_yaml(
        tmp_path / "bank.yaml",
        valid_bank(),
    )
    manifest = write_active_manifest(
        tmp_path,
        bank_source="bank.yaml",
    )

    resolved = Config.load(manifest).attribute_bank
    assert resolved is not None

    mutable = resolved.to_dict()
    mutable["attributes"]["core"].append("new attribute")
    mutable["templates"]["survey"] = ["survey.global"]
    mutable["attributes"]["survey.global"] = ["survey attribute"]

    assert resolved.data["attributes"]["core"] == (
        "age",
        "education",
    )
    assert "survey" not in resolved.data["templates"]
    assert "survey.global" not in resolved.data["attributes"]


def test_unknown_builtin_is_rejected(tmp_path: Path) -> None:
    manifest = write_active_manifest(
        tmp_path,
        bank_source="builtin:missing",
    )

    with pytest.raises(
        ConfigValidationError,
        match="unknown built-in",
    ):
        Config.load(manifest)


@pytest.mark.parametrize(
    ("bank", "message"),
    [
        (
            {
                "templates": {
                    "core": ["core"],
                },
            },
            "attributes",
        ),
        (
            {
                "templates": {
                    "core": ["missing"],
                },
                "attributes": {},
            },
            "missing",
        ),
        (
            {
                "templates": {
                    "core": ["core"],
                },
                "attributes": {
                    "core": "age",
                },
            },
            "list",
        ),
        (
            {
                "templates": {
                    "core": ["core"],
                },
                "attributes": {
                    "core": ["age", ""],
                },
            },
            "nonempty string",
        ),
    ],
)
def test_malformed_bank_is_rejected(
    tmp_path: Path,
    bank: dict[str, Any],
    message: str,
) -> None:
    write_yaml(tmp_path / "bank.yaml", bank)
    manifest = write_active_manifest(
        tmp_path,
        bank_source="bank.yaml",
    )

    with pytest.raises(
        ConfigValidationError,
        match=message,
    ):
        Config.load(manifest)


@pytest.mark.parametrize(
    "reserved_mode",
    [
        "survey.static",
        "question.Q1",
    ],
)
def test_static_reserved_modes_are_rejected(
    tmp_path: Path,
    reserved_mode: str,
) -> None:
    write_yaml(
        tmp_path / "bank.yaml",
        {
            "templates": {
                "core": [reserved_mode],
            },
            "attributes": {
                reserved_mode: ["reserved"],
            },
        },
    )
    manifest = write_active_manifest(
        tmp_path,
        bank_source="bank.yaml",
    )

    with pytest.raises(
        ConfigValidationError,
        match="reserved",
    ):
        Config.load(manifest)


def test_snapshot_can_capture_generator_owned_dynamic_bank(
    tmp_path: Path,
) -> None:
    write_yaml(
        tmp_path / "bank.yaml",
        valid_bank(),
    )
    manifest = write_active_manifest(
        tmp_path,
        bank_source="bank.yaml",
    )
    resolved = Config.load(manifest).attribute_bank
    assert resolved is not None

    generator_bank = resolved.to_dict()
    generator_bank["templates"]["survey"] = ["survey.global"]
    generator_bank["attributes"]["survey.global"] = [
        "survey attribute",
    ]

    snapshot_path = resolved.write_snapshot(
        tmp_path / "run",
        attribute_bank=generator_bank,
    )

    assert snapshot_path == (
        tmp_path / "run" / "attribute_bank.resolved.yaml"
    )

    snapshot = yaml.safe_load(
        snapshot_path.read_text(encoding="utf-8")
    )
    assert snapshot["schema_version"] == 1
    assert snapshot["source"] == str(
        (tmp_path / "bank.yaml").resolve()
    )
    assert snapshot["source_digest"] == resolved.source_digest
    assert snapshot["attribute_bank"]["attributes"][
        "survey.global"
    ] == ["survey attribute"]