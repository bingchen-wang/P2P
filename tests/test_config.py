"""Contracts for central manifest loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest
import yaml

from p2p.config import (
    Config,
    ConfigLoadError,
    ConfigValidationError,
    expand_grid,
)


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


def write_file(path: Path, content: str = "") -> Path:
    """Create a fixture file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip(), encoding="utf-8")
    return path


def write_yaml(path: Path, value: Any) -> Path:
    """Write one YAML fixture and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )
    return path


def make_resources(tmp_path: Path) -> dict[str, str]:
    survey = write_file(tmp_path / "resources" / "survey.yaml")
    basis = tmp_path / "resources" / "basis"
    basis.mkdir(parents=True)
    target = write_file(tmp_path / "resources" / "target.json", "{}\n")
    predictions = write_file(
        tmp_path / "resources" / "predictions.json",
        "{}\n",
    )
    fitted = write_file(
        tmp_path / "resources" / "fitted.joblib",
    )

    return {
        "survey": str(survey),
        "basis": str(basis),
        "target": str(target),
        "predictions": str(predictions),
        "fitted": str(fitted),
    }


def model_config() -> dict[str, Any]:
    return {
        "method": "elasticnet",
        "alpha": [0.1],
        "l1_ratio": [0.5],
    }


def split_config() -> dict[str, Any]:
    return {
        "train_val": [
            "train",
            "valid",
        ],
        "test": "test",
    }


def valid_manifest(
    verb: str,
    resources: dict[str, str],
) -> dict[str, Any]:
    if verb == "generate":
        return {
            "data": {
                "survey": resources["survey"],
            },
            "generation": {
                "mode": "active",
            },
            "backend": deepcopy(DUMMY_BACKEND),
        }

    if verb == "pipeline":
        manifest = valid_manifest("generate", resources)
        manifest["data"]["target"] = resources["target"]
        manifest["split"] = split_config()
        manifest["model"] = model_config()
        manifest["metrics"] = [
            "mse",
            "one_minus_jsd",
        ]
        return manifest

    if verb == "regress":
        return {
            "data": {
                "survey": resources["survey"],
                "basis": resources["basis"],
                "target": resources["target"],
            },
            "split": split_config(),
            "model": model_config(),
            "metrics": [
                "mse",
                "one_minus_jsd",
            ],
        }

    if verb == "evaluate":
        return {
            "data": {
                "survey": resources["survey"],
                "predictions": resources["predictions"],
                "target": resources["target"],
            },
            "split": split_config(),
            "metrics": [
                "mse",
                "one_minus_jsd",
            ],
        }

    if verb == "simulate":
        return {
            "simulation": {
                "kind": "fraction",
                "invokes_generation": False,
            },
        }

    raise AssertionError(f"unsupported fixture verb: {verb}")


def delete_dotted(
    mapping: dict[str, Any],
    dotted_field: str,
) -> None:
    parts = dotted_field.split(".")
    parent: dict[str, Any] = mapping
    for part in parts[:-1]:
        parent = parent[part]
    parent.pop(parts[-1])


def test_extends_deep_merges_and_preserves_declaring_manifest_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_dir = tmp_path / "family" / "base"
    child_dir = tmp_path / "family" / "child"

    survey_path = write_file(base_dir / "resources" / "survey.yaml")
    base_target = write_file(base_dir / "resources" / "base-target.json")
    child_target = write_file(child_dir / "child-target.json")

    write_file(
        base_dir / "base.yaml",
        """
        run:
          name: base
          seed: 7
        data:
          survey: resources/survey.yaml
          target: resources/base-target.json
        model:
          method: elasticnet
          alpha:
            logspace: [-2, 0, 3]
          l1_ratio:
            linspace: [0, 1, 3]
          options:
            max_iter: 10000
            cv_folds: 5
        evaluation:
          invalid_distribution: allow
        """,
    )
    child_manifest = write_file(
        child_dir / "child.yaml",
        """
        extends: ../base/base.yaml
        run:
          name: child
        data:
          target: child-target.json
        model:
          options:
            max_iter: 2000
        evaluation:
          distribution_prediction_policy: validate
        """,
    )

    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    config = Config.load(child_manifest)

    assert config.source_path == child_manifest.resolve()
    assert config.data["run"] == {
        "name": "child",
        "seed": 7,
    }
    assert config.data["data"]["survey"] == str(survey_path.resolve())
    assert config.data["data"]["target"] == str(child_target.resolve())
    assert config.data["data"]["target"] != str(base_target.resolve())

    assert config.data["model"]["method"] == "elasticnet"
    assert config.data["model"]["alpha"] == pytest.approx(
        (0.01, 0.1, 1.0)
    )
    assert config.data["model"]["l1_ratio"] == pytest.approx(
        (0.0, 0.5, 1.0)
    )
    assert config.data["model"]["options"] == {
        "max_iter": 2000,
        "cv_folds": 5,
    }

    assert config.data["evaluation"] == {
        "mse_prediction_policy": "raw",
        "distribution_prediction_policy": "validate",
        "invalid_distribution": "allow",
    }


def test_config_data_is_immutable_and_to_dict_is_detached(
    tmp_path: Path,
) -> None:
    manifest = write_file(
        tmp_path / "config.yaml",
        """
        run:
          name: immutable
          seed: 11
        """,
    )

    config = Config.load(manifest)

    with pytest.raises(TypeError):
        config.data["run"]["seed"] = 99

    mutable_copy = config.to_dict()
    mutable_copy["run"]["seed"] = 99

    assert config.data["run"]["seed"] == 11


def test_extends_cycle_reports_the_manifest_chain(
    tmp_path: Path,
) -> None:
    first = write_file(
        tmp_path / "first.yaml",
        "extends: second.yaml\n",
    )
    write_file(
        tmp_path / "second.yaml",
        "extends: first.yaml\n",
    )

    with pytest.raises(ConfigLoadError) as exc_info:
        Config.load(first)

    message = str(exc_info.value)
    assert "cycle" in message.lower()
    assert "first.yaml" in message
    assert "second.yaml" in message


def test_missing_referenced_path_identifies_its_field(
    tmp_path: Path,
) -> None:
    manifest = write_file(
        tmp_path / "config.yaml",
        """
        run:
          name: missing-resource
        data:
          survey: resources/missing.yaml
        """,
    )

    with pytest.raises(ConfigValidationError) as exc_info:
        Config.load(manifest)

    message = str(exc_info.value)
    assert "data.survey" in message
    assert "does not exist" in message


@pytest.mark.parametrize(
    "spec",
    [
        "np.logspace(-5, 1, 20)",
        {"logspace": [0, 1, 0]},
        {"linspace": [0, 1, 2.5]},
        {
            "logspace": [0, 1, 2],
            "linspace": [0, 1, 2],
        },
        {"unknown": [0, 1, 2]},
    ],
)
def test_structured_grid_rejects_unsafe_or_malformed_specs(
    spec: object,
) -> None:
    with pytest.raises(ConfigValidationError):
        expand_grid(spec)


def test_structured_grid_accepts_literal_and_structured_values() -> None:
    assert expand_grid(["active", "vanilla"]) == (
        "active",
        "vanilla",
    )
    assert expand_grid(
        {"linspace": [0, 1, 3]}
    ) == pytest.approx((0.0, 0.5, 1.0))
    assert expand_grid(
        {"logspace": [-2, 0, 3]}
    ) == pytest.approx((0.01, 0.1, 1.0))


@pytest.mark.parametrize(
    "verb",
    [
        "generate",
        "pipeline",
        "regress",
        "evaluate",
        "simulate",
    ],
)
def test_validate_for_accepts_authoritative_minimal_manifests(
    tmp_path: Path,
    verb: str,
) -> None:
    resources = make_resources(tmp_path)
    config = Config.from_mapping(
        valid_manifest(verb, resources),
        declared_in=tmp_path / f"{verb}.yaml",
    )

    assert config.validate_for(verb) is config

    if verb in {"generate", "pipeline"}:
        assert config.attribute_bank is not None
        assert config.attribute_bank.source == "builtin:default"


def test_evaluate_accepts_fitted_output_instead_of_predictions(
    tmp_path: Path,
) -> None:
    resources = make_resources(tmp_path)
    manifest = valid_manifest("evaluate", resources)
    manifest["data"].pop("predictions")
    manifest["data"]["fitted"] = resources["fitted"]

    config = Config.from_mapping(
        manifest,
        declared_in=tmp_path / "evaluate.yaml",
    )

    assert config.validate_for("evaluate") is config


@pytest.mark.parametrize(
    ("verb", "missing_field", "message"),
    [
        ("generate", "data.survey", "data.survey"),
        ("generate", "generation", "generation"),
        ("generate", "backend", "backend"),
        ("pipeline", "data.target", "data.target"),
        ("pipeline", "split", "split"),
        ("pipeline", "model", "model"),
        ("pipeline", "metrics", "metrics"),
        ("regress", "data.survey", "data.survey"),
        ("regress", "data.basis", "data.basis"),
        ("regress", "data.target", "data.target"),
        ("regress", "split", "split"),
        ("regress", "model", "model"),
        ("regress", "metrics", "metrics"),
        ("evaluate", "data.survey", "data.survey"),
        ("evaluate", "data.predictions", "predictions"),
        ("evaluate", "data.target", "data.target"),
        ("evaluate", "split", "split"),
        ("evaluate", "metrics", "metrics"),
        ("simulate", "simulation", "simulation"),
    ],
)
def test_validate_for_rejects_missing_required_fields(
    tmp_path: Path,
    verb: str,
    missing_field: str,
    message: str,
) -> None:
    resources = make_resources(tmp_path)
    manifest = valid_manifest(verb, resources)
    delete_dotted(manifest, missing_field)

    config = Config.from_mapping(
        manifest,
        declared_in=tmp_path / f"{verb}.yaml",
    )

    with pytest.raises(
        ConfigValidationError,
        match=message,
    ):
        config.validate_for(verb)


def test_simulation_invoking_generation_requires_generation_and_backend(
    tmp_path: Path,
) -> None:
    resources = make_resources(tmp_path)
    manifest = valid_manifest("simulate", resources)
    manifest["simulation"]["invokes_generation"] = True

    config = Config.from_mapping(
        manifest,
        declared_in=tmp_path / "simulate.yaml",
    )
    with pytest.raises(
        ConfigValidationError,
        match="generation",
    ):
        config.validate_for("simulate")

    manifest["generation"] = {
        "mode": "active",
    }
    config = Config.from_mapping(
        manifest,
        declared_in=tmp_path / "simulate.yaml",
    )
    with pytest.raises(
        ConfigValidationError,
        match="backend",
    ):
        config.validate_for("simulate")

    manifest["backend"] = deepcopy(DUMMY_BACKEND)
    config = Config.from_mapping(
        manifest,
        declared_in=tmp_path / "simulate.yaml",
    )

    assert config.validate_for("simulate") is config
    assert config.attribute_bank is not None


def test_generate_accepts_referenced_backend_profile(
    tmp_path: Path,
) -> None:
    resources = make_resources(tmp_path)
    profile = write_yaml(
        tmp_path / "backend.yaml",
        DUMMY_BACKEND,
    )
    manifest = valid_manifest("generate", resources)
    manifest["backend"] = str(profile)

    config = Config.from_mapping(
        manifest,
        declared_in=tmp_path / "generate.yaml",
    )

    assert config.validate_for("generate") is config
    assert config.data["backend"] == str(profile.resolve())


def test_unknown_verb_is_rejected(tmp_path: Path) -> None:
    config = Config.from_mapping(
        {},
        declared_in=tmp_path / "unknown.yaml",
    )

    with pytest.raises(
        ConfigValidationError,
        match="unsupported verb",
    ):
        config.validate_for("unknown")


def test_resolved_config_snapshot_has_digest_and_resolved_values(
    tmp_path: Path,
) -> None:
    resources = make_resources(tmp_path)
    manifest_path = write_yaml(
        tmp_path / "regress.yaml",
        valid_manifest("regress", resources),
    )
    config = Config.load(manifest_path)

    snapshot_path = config.write_snapshot(tmp_path / "run")

    assert snapshot_path == (
        tmp_path / "run" / "config.resolved.yaml"
    )
    assert len(config.digest) == 64
    assert config.digest == Config.load(manifest_path).digest

    snapshot = yaml.safe_load(
        snapshot_path.read_text(encoding="utf-8")
    )
    assert snapshot["schema_version"] == 1
    assert snapshot["source"] == str(manifest_path.resolve())
    assert snapshot["config_digest"] == config.digest
    assert snapshot["config"]["data"]["survey"] == str(
        Path(resources["survey"]).resolve()
    )
    assert snapshot["config"]["model"]["alpha"] == [0.1]
    assert snapshot["config"]["evaluation"][
        "distribution_prediction_policy"
    ] == "clip_and_normalize"


def test_grouped_target_mapping_resolves_path_and_preserves_subgroup(
    tmp_path: Path,
) -> None:
    target = write_file(
        tmp_path / "targets" / "grouped.json",
        "{}\n",
    )
    manifest = write_file(
        tmp_path / "grouped.yaml",
        """
        data:
          target:
            path: targets/grouped.json
            subgroup: party.democrat
        """,
    )

    config = Config.load(manifest)

    assert config.data["data"]["target"] == {
        "path": str(target.resolve()),
        "subgroup": "party.democrat",
    }
