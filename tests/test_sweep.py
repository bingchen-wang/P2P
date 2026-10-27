"""Contracts for generic sweep expansion."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from p2p.config import Config, ConfigValidationError
from p2p.sweep import SweepSpec, expand_sweep


def write_file(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def regress_manifest(tmp_path: Path) -> dict[str, Any]:
    survey = write_file(tmp_path / "survey.yaml")
    basis = tmp_path / "basis"
    basis.mkdir()
    target = write_file(tmp_path / "target.json", "{}\n")

    return {
        "run": {
            "name": "sweep-base",
            "seed": 200,
        },
        "data": {
            "survey": str(survey),
            "basis": str(basis),
            "target": str(target),
        },
        "split": {
            "train_val": [
                "train",
                "valid",
            ],
            "test": "test",
        },
        "model": {
            "method": "elasticnet",
            "alpha": [0.1],
            "l1_ratio": [0.5],
        },
        "metrics": [
            "mse",
            "one_minus_jsd",
        ],
    }


def test_two_point_ablate_expands_to_two_valid_configs(
    tmp_path: Path,
) -> None:
    manifest = regress_manifest(tmp_path)
    manifest["ablate"] = {
        "target": "regress",
        "over": "model.alpha",
        "values": {
            "logspace": [
                -2,
                -1,
                2,
            ],
        },
    }
    base = Config.from_mapping(
        manifest,
        declared_in=tmp_path / "ablate.yaml",
    )

    spec = SweepSpec.from_config(base, "ablate")
    expanded = expand_sweep(base, spec)

    assert len(expanded) == 2
    assert [
        config.data["model"]["alpha"][0]
        for config in expanded
    ] == pytest.approx([
        0.01,
        0.1,
    ])

    for config in expanded:
        assert config.validate_for("regress") is config
        assert "ablate" not in config.data

    assert "ablate" in base.data
    assert base.data["model"]["alpha"] == pytest.approx((0.1,))


def test_panel_expansion_is_repeat_major_and_seeded(
    tmp_path: Path,
) -> None:
    manifest = regress_manifest(tmp_path)
    manifest["panel"] = {
        "target": "regress",
        "overrides": [
            {
                "run": {
                    "wave": "W1",
                },
            },
            {
                "run": {
                    "wave": "W2",
                },
            },
        ],
        "repeats": 2,
    }
    base = Config.from_mapping(
        manifest,
        declared_in=tmp_path / "panel.yaml",
    )

    expanded = expand_sweep(
        base,
        SweepSpec.from_config(base, "panel"),
    )

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
        config.data["run"]["repeat"]
        for config in expanded
    ] == [
        1,
        1,
        2,
        2,
    ]
    assert [
        config.data["run"]["seed"]
        for config in expanded
    ] == [
        200,
        200,
        201,
        201,
    ]

    for config in expanded:
        assert config.validate_for("regress") is config
        assert "panel" not in config.data


def test_simulation_sweep_keeps_simulation_and_removes_meta_layer(
    tmp_path: Path,
) -> None:
    base = Config.from_mapping(
        {
            "run": {
                "name": "simulation-sweep",
                "seed": 10,
            },
            "simulation": {
                "kind": "fraction",
                "invokes_generation": False,
                "sweep": {
                    "target": "simulate",
                    "overrides": [
                        {
                            "simulation": {
                                "fraction": 0.25,
                            },
                        },
                        {
                            "simulation": {
                                "fraction": 0.5,
                            },
                        },
                    ],
                },
            },
        },
        declared_in=tmp_path / "simulate.yaml",
    )

    expanded = expand_sweep(
        base,
        SweepSpec.from_config(base, "simulate"),
    )

    assert len(expanded) == 2
    assert [
        config.data["simulation"]["fraction"]
        for config in expanded
    ] == [
        0.25,
        0.5,
    ]
    assert all(
        "sweep" not in config.data["simulation"]
        for config in expanded
    )
    assert all(
        config.validate_for("simulate") is config
        for config in expanded
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_target", "target"),
        ("both_forms", "exactly one"),
        ("zero_repeats", "repeats"),
        ("unknown_path", "model.missing"),
        ("unsafe_values", "grid"),
        ("empty_overrides", "overrides"),
    ],
)
def test_invalid_sweep_specs_are_rejected(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    manifest = regress_manifest(tmp_path)
    manifest["ablate"] = {
        "target": "regress",
        "over": "model.alpha",
        "values": [
            0.01,
            0.1,
        ],
    }

    if case == "missing_target":
        manifest["ablate"].pop("target")
    elif case == "both_forms":
        manifest["ablate"]["overrides"] = [
            {
                "model": {
                    "alpha": [0.2],
                },
            },
        ]
    elif case == "zero_repeats":
        manifest["ablate"]["repeats"] = 0
    elif case == "unknown_path":
        manifest["ablate"]["over"] = "model.missing"
    elif case == "unsafe_values":
        manifest["ablate"]["values"] = "np.linspace(0, 1, 2)"
    elif case == "empty_overrides":
        manifest["ablate"].pop("over")
        manifest["ablate"].pop("values")
        manifest["ablate"]["overrides"] = []
    else:
        raise AssertionError(f"unhandled case: {case}")

    base = Config.from_mapping(
        manifest,
        declared_in=tmp_path / "ablate.yaml",
    )

    with pytest.raises(
        ConfigValidationError,
        match=message,
    ):
        SweepSpec.from_config(base, "ablate")


def test_unknown_sweep_kind_is_rejected(
    tmp_path: Path,
) -> None:
    base = Config.from_mapping(
        regress_manifest(tmp_path),
        declared_in=tmp_path / "base.yaml",
    )

    with pytest.raises(
        ConfigValidationError,
        match="kind",
    ):
        SweepSpec.from_config(base, "unknown")


def test_expansion_validates_target_verb(
    tmp_path: Path,
) -> None:
    base = Config.from_mapping(
        {
            "simulation": {
                "kind": "fraction",
                "invokes_generation": False,
                "sweep": {
                    "target": "regress",
                    "overrides": [
                        {
                            "run": {
                                "condition": "invalid",
                            },
                        },
                    ],
                },
            },
        },
        declared_in=tmp_path / "invalid-target.yaml",
    )
    spec = SweepSpec.from_config(base, "simulate")

    with pytest.raises(
        ConfigValidationError,
        match="data",
    ):
        expand_sweep(base, spec)