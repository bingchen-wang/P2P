"""Session 0 bootstrap checks."""

from __future__ import annotations

import cvxpy as cp
import pytest

import p2p
from p2p.cli import main


def test_package_version() -> None:
    assert p2p.__version__ == "0.1.0"


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "Prompts to Proxies" in capsys.readouterr().out


def test_pinned_solver_is_available(cvxpy_solver: str) -> None:
    assert cvxpy_solver == "OSQP"
    assert cvxpy_solver in cp.installed_solvers()