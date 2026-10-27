"""Repository-wide validation gate for central manifests."""

from __future__ import annotations

from pathlib import Path

import pytest

from p2p.config import Config
from p2p.commands import cluster
from p2p.sweep import SweepSpec, expand_sweep


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CENTRAL_DIR = PROJECT_ROOT / "config" / "central"

EXPECTED_MANIFESTS = {
    "ablate_fixture.yaml": "ablate",
    "aggregate_fixture.yaml": "aggregate",
    "cluster_fixture.yaml": "cluster",
    "elicit_fixture.yaml": "generate",
    "generate_active_fixture.yaml": "generate",
    "panel_fixture.yaml": "panel",
    "pipeline_fixture.yaml": "pipeline",
    "regress_fixture.yaml": "regress",
    "evaluate_fixture.yaml": "evaluate",
    "simulate_fixture.yaml": "simulate",
    "simulate_scientific_fixture.yaml": "simulate",
}


def _has_content(path: Path) -> bool:
    """Return whether a planned manifest stub has been fleshed."""

    return bool(
        path.read_text(encoding="utf-8").strip()
    )


CENTRAL_MANIFESTS = tuple(
    path
    for path in sorted(CENTRAL_DIR.glob("*.yaml"))
    if _has_content(path)
)


def test_central_manifest_set_is_explicit() -> None:
    assert {
        path.name
        for path in CENTRAL_MANIFESTS
    } == set(EXPECTED_MANIFESTS)


@pytest.mark.parametrize(
    "manifest_path",
    CENTRAL_MANIFESTS,
    ids=lambda path: path.name,
)
def test_every_central_manifest_validates_without_project_cwd(
    manifest_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    verb = EXPECTED_MANIFESTS[manifest_path.name]
    config = Config.load(manifest_path)

    if verb == "cluster":
        result = cluster.run(
            config,
            run_dir=tmp_path / "cluster",
        )
        assert result.config_digest == config.digest
    else:
        assert config.validate_for(verb) is config

    if manifest_path.name == "generate_active_fixture.yaml":
        assert config.data["generation"]["mode"] == "active"
        assert config.attribute_bank is not None
        assert config.attribute_bank.source == "builtin:default"

    if verb == "ablate":
        expanded = expand_sweep(
            config,
            SweepSpec.from_config(config, "ablate"),
        )
        assert len(expanded) == 2
        assert all(
            child.validate_for("regress") is child
            for child in expanded
        )

    if verb in {"panel", "simulate"}:
        expanded = expand_sweep(
            config,
            SweepSpec.from_config(config, verb),
        )
        assert len(expanded) == 2
