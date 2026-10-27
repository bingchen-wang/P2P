"""Aggregate command and CLI contracts."""

from __future__ import annotations

from pathlib import Path

from p2p import cli
from p2p.commands import aggregate
from p2p.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "central"
    / "aggregate_fixture.yaml"
)
RUN_ROOT = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "commands"
    / "aggregation"
)


def test_aggregate_command_writes_required_outputs(
    tmp_path: Path,
) -> None:
    config = Config.load(MANIFEST)
    result = aggregate.run(
        config,
        output_dir=tmp_path / "configured",
    )

    assert result.result.validation["status"] == "valid"
    assert all(
        path.is_file()
        for path in result.artifacts.required_files
    )


def test_aggregate_cli_supports_config_and_quick_runs(
    tmp_path: Path,
) -> None:
    configured = tmp_path / "configured"
    assert cli.main(
        [
            "aggregate",
            "--config",
            str(MANIFEST),
            "--output",
            str(configured),
        ]
    ) == 0
    assert (configured / "summary.csv").is_file()

    quick = tmp_path / "quick"
    assert cli.main(
        [
            "aggregate",
            "--runs",
            str(RUN_ROOT / "r1"),
            str(RUN_ROOT / "r2"),
            str(RUN_ROOT / "r3"),
            "--group-by",
            "model",
            "split",
            "--metrics",
            "mse",
            "one_minus_jsd",
            "--output",
            str(quick),
        ]
    ) == 0
    assert (quick / "summary.json").is_file()
