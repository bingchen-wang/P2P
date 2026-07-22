"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import cvxpy as cp
import pytest


PINNED_CVXPY_SOLVER = "OSQP"


@pytest.fixture(scope="session")
def cvxpy_solver() -> str:
    """Return the solver used by numerical parity tests."""
    if PINNED_CVXPY_SOLVER not in cp.installed_solvers():
        pytest.fail(
            f"Required solver {PINNED_CVXPY_SOLVER!r} is not installed. "
            f"Available solvers: {cp.installed_solvers()}"
        )

    return PINNED_CVXPY_SOLVER