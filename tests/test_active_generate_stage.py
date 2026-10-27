"""Offline active and injected-plugin generation contracts."""

from __future__ import annotations

from pathlib import Path

from p2p.commands import generate as generate_command
from p2p.config import Config
from p2p.endowments import ActiveEndowments
from p2p.registry import Registry
from p2p.stages import elicit as elicit_stage
from p2p.stages import generate as generate_stage


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_active_generate_runs_offline_and_snapshots_dynamic_bank(
    tmp_path: Path,
) -> None:
    config = Config.load(
        PROJECT_ROOT
        / "config"
        / "central"
        / "generate_active_fixture.yaml"
    )
    result = generate_command.run(
        config,
        run_dir=tmp_path / "active",
    )

    assert result.mode == "active"
    assert len(result.endowments) == 22
    assert result.endowments_path.is_file()
    assert result.responses is not None
    assert result.responses_path is not None
    assert result.responses_path.is_file()
    assert set(result.diagnostic_paths) == {
        "aggregate_entropy",
        "mode_entropy",
        "question_entropy",
        "mode_summary",
    }
    assert all(
        path.is_file()
        for path in result.diagnostic_paths.values()
    )
    assert result.usage_tracker is not None
    assert result.attribute_bank_snapshot is not None
    assert result.attribute_bank_snapshot.is_file()
    assert (
        tmp_path
        / "active"
        / "config.resolved.yaml"
    ).is_file()

    def fail_if_reelicited(**kwargs):
        raise AssertionError(
            "active responses must be reused"
        )

    elicited = elicit_stage.run(
        config,
        generation=result,
        run_dir=tmp_path / "active",
        conductor_factory=fail_if_reelicited,
    )
    assert elicited.reused_generation_responses is True
    assert elicited.responses_path == result.responses_path
    assert len(elicited.records) == 66


def test_plugin_mode_executes_injected_registered_strategy(
    tmp_path: Path,
) -> None:
    base = Config.load(
        PROJECT_ROOT
        / "config"
        / "central"
        / "pipeline_fixture.yaml"
    ).to_dict()
    base["generation"]["mode"] = "plugin"
    config = Config.from_mapping(
        base,
        declared_in=tmp_path / "plugin.yaml",
    )
    registry = Registry("plugin-test")
    expected = ActiveEndowments.from_endowment_list(
        [
            {
                "eid": "plugin_1",
                "endow_text": "Injected plugin persona",
                "role": "proxy",
                "weight": None,
                "mode": ("plugin",),
                "attributes": [],
            }
        ]
    )
    registry.register(
        "generate.plugin",
        lambda **kwargs: expected,
    )

    result = generate_stage.run(
        config,
        run_dir=tmp_path / "plugin",
        strategy_registry=registry,
    )

    assert result.mode == "plugin"
    assert result.endowments is expected
    assert result.endowments_path.is_file()
