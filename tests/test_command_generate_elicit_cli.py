"""User-facing generation and standalone elicitation contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from p2p.commands import elicit as elicit_command
from p2p.commands import generate as generate_command
from p2p.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_FIXTURE = (
    PROJECT_ROOT
    / "config"
    / "central"
    / "pipeline_fixture.yaml"
)


def test_generate_command_runs_vanilla_offline(
    tmp_path: Path,
) -> None:
    config = Config.load(PIPELINE_FIXTURE)
    run_dir = tmp_path / "generate"

    result = generate_command.run(
        config,
        run_dir=run_dir,
        verbose=False,
    )

    assert result.mode == "vanilla"
    assert len(result.endowments) == 2
    assert result.endowments_path == (
        run_dir / "endowments.csv"
    ).resolve()
    assert result.endowments_path.is_file()


def test_elicit_command_loads_existing_endowments_without_regenerating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config.load(PIPELINE_FIXTURE)
    basis_dir = tmp_path / "basis"
    generated = generate_command.run(
        config,
        run_dir=basis_dir,
        verbose=False,
    )

    def fail_if_regenerated(*args, **kwargs):
        raise AssertionError(
            "standalone elicit must not regenerate endowments"
        )

    monkeypatch.setattr(
        elicit_command.generate_stage,
        "run",
        fail_if_regenerated,
    )

    output_dir = tmp_path / "elicit"
    result = elicit_command.run(
        config,
        endowments_path=generated.endowments_path,
        run_dir=output_dir,
        verbose=False,
    )

    assert result.survey.csv_path == generated.survey.csv_path
    assert result.endowments.get_endowments() == (
        generated.endowments.get_endowments()
    )
    assert len(result.records) == 6
    assert result.responses_path == (
        output_dir / "responses.csv"
    ).resolve()
    assert result.responses_path.is_file()


def test_elicit_command_rejects_missing_endowments(
    tmp_path: Path,
) -> None:
    config = Config.load(PIPELINE_FIXTURE)

    with pytest.raises(
        ValueError,
        match="not a file",
    ):
        elicit_command.run(
            config,
            endowments_path=tmp_path / "missing.csv",
            run_dir=tmp_path / "elicit",
        )
