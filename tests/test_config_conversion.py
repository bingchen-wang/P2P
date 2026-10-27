"""Contracts for converting earlier-format configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from p2p.config import Config
from p2p.config_conversion import (
    ConfigConversionError,
    convert_earlier_config,
)
from p2p.sweep import SweepSpec, expand_sweep


LEGACY_FIXTURES = (
    Path(__file__).parent
    / "fixtures"
    / "earlier_configs"
)


def write_file(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def write_yaml(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )
    return path


def create_legacy_resources(root: Path) -> None:
    write_file(root / "data" / "survey.csv")
    write_yaml(
        root / "data" / "survey_schema.yaml",
        {
            "question_id": "id",
            "question_text": "question",
            "code_to_answer": "option_mapping",
        },
    )
    write_file(root / "data" / "responses.csv")
    write_file(root / "data" / "endowments.csv")
    write_file(root / "data" / "target.json", "{}\n")
    write_yaml(
        root / "data" / "attribute_bank.yaml",
        {
            "templates": {
                "core": ["core"],
            },
            "attributes": {
                "core": [
                    "age",
                    "education",
                ],
            },
        },
    )


def create_panel_resources(root: Path) -> None:
    create_legacy_resources(root)
    for wave in ("W1", "W2"):
        write_file(
            root / "data" / wave / "survey.csv"
        )
        write_file(
            root / "data" / wave / "target.json",
            "{}\n",
        )


def test_source_root_and_its_earlier_name_give_identical_manifests(
    tmp_path: Path,
) -> None:
    create_legacy_resources(tmp_path)
    by_new_name = convert_earlier_config(
        LEGACY_FIXTURES / "regress.yaml",
        tmp_path / "central" / "new.yaml",
        family="regress",
        source_root=tmp_path,
        method="elasticnet",
    )
    by_earlier_name = convert_earlier_config(
        LEGACY_FIXTURES / "regress.yaml",
        tmp_path / "central" / "earlier.yaml",
        family="regress",
        legacy_root=tmp_path,
        method="elasticnet",
    )

    assert by_new_name.read_bytes() == by_earlier_name.read_bytes()
    with pytest.raises(ConfigConversionError, match="not both"):
        convert_earlier_config(
            LEGACY_FIXTURES / "regress.yaml",
            tmp_path / "central" / "both.yaml",
            family="regress",
            source_root=tmp_path,
            legacy_root=tmp_path,
        )
    with pytest.raises(ConfigConversionError, match="source_root is required"):
        convert_earlier_config(
            LEGACY_FIXTURES / "regress.yaml",
            tmp_path / "central" / "none.yaml",
            family="regress",
        )


def test_migrate_regress_produces_valid_cwd_independent_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_legacy_resources(tmp_path)
    destination = tmp_path / "central" / "regress.yaml"

    result = convert_earlier_config(
        LEGACY_FIXTURES / "regress.yaml",
        destination,
        family="regress",
        legacy_root=tmp_path,
        method="elasticnet",
    )

    assert result == destination.resolve()

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    config = Config.load(destination)

    assert config.validate_for("regress") is config
    assert config.data["run"]["name"] == "legacy_regress"
    assert config.data["run"]["seed"] == 101
    assert config.data["data"]["survey"] == {
        "source": str(
            (tmp_path / "data" / "survey.csv").resolve()
        ),
        "schema": str(
            (
                tmp_path
                / "data"
                / "survey_schema.yaml"
            ).resolve()
        ),
    }
    assert config.data["data"]["basis"] == {
        "responses": str(
            (tmp_path / "data" / "responses.csv").resolve()
        ),
        "endowments": str(
            (tmp_path / "data" / "endowments.csv").resolve()
        ),
    }
    assert config.data["model"]["method"] == "elasticnet"
    assert config.data["model"]["alpha"][0] == pytest.approx(1e-5)
    assert config.data["model"]["alpha"][-1] == pytest.approx(10.0)
    assert len(config.data["model"]["alpha"]) == 20
    assert config.data["model"]["l1_ratio"][0] == pytest.approx(0.1)
    assert config.data["model"]["l1_ratio"][-1] == pytest.approx(1.0)
    assert len(config.data["model"]["l1_ratio"]) == 10
    assert config.data["metrics"] == (
        "mse",
        "one_minus_jsd",
        "emd",
        "mcpa",
    )


def test_migrate_active_pipeline_preserves_backend_and_bank_boundaries(
    tmp_path: Path,
) -> None:
    create_legacy_resources(tmp_path)
    destination = tmp_path / "central" / "pipeline.yaml"

    convert_earlier_config(
        LEGACY_FIXTURES / "pipeline.yaml",
        destination,
        family="pipeline",
        legacy_root=tmp_path,
        method="elasticnet",
    )

    config = Config.load(destination)

    assert config.validate_for("pipeline") is config
    assert config.data["generation"]["mode"] == "active"
    assert config.data["generation"]["target_n"] == 10
    assert config.data["backend"]["response"] == {
        "type": "dummy",
        "model_name": "dummy",
        "formality": "Return only the selected answer.",
        "kwargs": {},
    }
    assert config.data["backend"]["endowment"]["type"] == "dummy"
    assert config.data["backend"]["attribute_learner"]["type"] == "dummy"
    assert config.data["backend"]["endowment"]["kwargs"] == {
        "temperature": 0,
    }
    assert config.data["backend"]["attribute_learner"]["kwargs"] == {
        "max_tokens": 32,
    }
    assert config.data["generation"]["endowment_model"] == {
        "batch_size": 2,
        "delay": 0,
        "retry_failed": False,
        "randomize_attributes": False,
        "max_attributes": 4,
    }

    assert config.attribute_bank is not None
    assert config.attribute_bank.source == str(
        (
            tmp_path
            / "data"
            / "attribute_bank.yaml"
        ).resolve()
    )


def test_multiple_legacy_models_require_explicit_method(
    tmp_path: Path,
) -> None:
    create_legacy_resources(tmp_path)

    with pytest.raises(
        ConfigConversionError,
        match="method",
    ):
        convert_earlier_config(
            LEGACY_FIXTURES / "regress.yaml",
            tmp_path / "central" / "regress.yaml",
            family="regress",
            legacy_root=tmp_path,
        )


def test_unsafe_legacy_grid_expression_is_rejected(
    tmp_path: Path,
) -> None:
    create_legacy_resources(tmp_path)
    legacy = yaml.safe_load(
        (
            LEGACY_FIXTURES
            / "regress.yaml"
        ).read_text(encoding="utf-8")
    )
    legacy["elasticnet"]["alpha_expr"] = (
        "__import__('os').system('echo unsafe')"
    )
    source = write_yaml(
        tmp_path / "unsafe.yaml",
        legacy,
    )

    with pytest.raises(
        ConfigConversionError,
        match="expression",
    ):
        convert_earlier_config(
            source,
            tmp_path / "central" / "unsafe.yaml",
            family="regress",
            legacy_root=tmp_path,
            method="elasticnet",
        )


def test_migrate_ablation_expands_two_valid_pipeline_configs(
    tmp_path: Path,
) -> None:
    create_legacy_resources(tmp_path)
    destination = tmp_path / "central" / "ablate.yaml"

    convert_earlier_config(
        LEGACY_FIXTURES / "ablate.yaml",
        destination,
        family="ablate",
        legacy_root=tmp_path,
    )

    base = Config.load(destination)
    assert base.validate_for("ablate") is base

    expanded = expand_sweep(
        base,
        SweepSpec.from_config(base, "ablate"),
    )

    assert len(expanded) == 2
    assert [
        config.data["generation"]["target_n"]
        for config in expanded
    ] == [
        10,
        20,
    ]
    assert all(
        config.validate_for("pipeline") is config
        for config in expanded
    )


def test_migrate_panel_expands_waves_and_repeats_deterministically(
    tmp_path: Path,
) -> None:
    create_panel_resources(tmp_path)
    destination = tmp_path / "central" / "panel.yaml"

    convert_earlier_config(
        LEGACY_FIXTURES / "panel.yaml",
        destination,
        family="panel",
        legacy_root=tmp_path,
    )

    base = Config.load(destination)
    assert base.validate_for("panel") is base

    expanded = expand_sweep(
        base,
        SweepSpec.from_config(base, "panel"),
    )

    assert len(expanded) == 4
    assert [
        config.data["run"]["wave"]
        for config in expanded
    ] == [
        "W1",
        "W2",
        "W1",
        "W2",
    ]
    assert [
        config.data["run"]["seed"]
        for config in expanded
    ] == [
        300,
        300,
        301,
        301,
    ]
    assert all(
        config.validate_for("pipeline") is config
        for config in expanded
    )
    assert expanded[1].data["data"]["survey"]["source"] == str(
        (
            tmp_path
            / "data"
            / "W2"
            / "survey.csv"
        ).resolve()
    )


def test_migrate_simulation_expands_structured_fraction_grid(
    tmp_path: Path,
) -> None:
    create_legacy_resources(tmp_path)
    destination = tmp_path / "central" / "simulate.yaml"

    convert_earlier_config(
        LEGACY_FIXTURES / "simulate.yaml",
        destination,
        family="simulate",
        legacy_root=tmp_path,
    )

    base = Config.load(destination)
    expanded = expand_sweep(
        base,
        SweepSpec.from_config(base, "simulate"),
    )

    assert len(expanded) == 2
    assert [
        config.data["simulation"]["fraction"]
        for config in expanded
    ] == pytest.approx([
        0.25,
        0.5,
    ])
    assert base.data["simulation"]["executor"] == "scientific"
    assert base.data["simulation"]["kind"] == "endowment"
    assert base.data["simulation"]["target_source"] == "ground_truth"
    assert base.data["simulation"]["ground_truth"] == {
        "source": "random",
        "proxy_ratio": 0.7,
    }
    assert all(
        config.validate_for("simulate") is config
        for config in expanded
    )
