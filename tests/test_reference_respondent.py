"""Contracts for the uniform reference respondent.

The reference respondent answers every option of every question with equal
probability. It is a basis element, not an intercept: its column sums to one on
every question block, so it preserves the mass constraints, it carries a fitted
weight like any other agent, and it is dropped when the basis already spans the
target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p2p.models.constrained_lasso import ConstrainedLasso
from p2p.stages.split import REFERENCE_FEATURE


N_AGENTS = 40
N_QUESTIONS = 30
N_OPTIONS = 4
TRUE_AGENTS = 10


def _synthetic_basis(
    *, seed: int = 0, with_reference: bool = True
) -> tuple[pd.DataFrame, dict[str, list[str]], np.ndarray]:
    """One-hot answers from N_AGENTS proxies, optionally plus the reference."""

    rng = np.random.default_rng(seed)
    index: list[str] = []
    groups: dict[str, list[str]] = {}
    blocks: list[np.ndarray] = []

    for question in range(N_QUESTIONS):
        qid = f"q{question}"
        members = [f"{qid}__{option}" for option in range(N_OPTIONS)]
        groups[qid] = members
        index.extend(members)

        block = np.zeros((N_OPTIONS, N_AGENTS))
        block[rng.integers(0, N_OPTIONS, N_AGENTS), np.arange(N_AGENTS)] = 1.0
        if with_reference:
            block = np.hstack(
                [block, np.full((N_OPTIONS, 1), 1.0 / N_OPTIONS)]
            )
        blocks.append(block)

    columns = [f"agent_{i}" for i in range(N_AGENTS)]
    if with_reference:
        columns.append(REFERENCE_FEATURE)

    frame = pd.DataFrame(np.vstack(blocks), index=index, columns=columns)
    true_agents = rng.choice(N_AGENTS, TRUE_AGENTS, replace=False)
    return frame, groups, true_agents


def _fit(frame: pd.DataFrame, target: np.ndarray, alpha: float = 0.01):
    model = ConstrainedLasso(
        alpha=alpha,
        constraint_groups=None,
        nonneg=False,
        sum_to_one=True,
        zero_threshold=1e-5,
    )
    model.fit(frame, target)
    return np.asarray(model.beta_, dtype=float)


def test_reference_column_sums_to_one_on_every_block() -> None:
    frame, groups, _ = _synthetic_basis()

    for members in groups.values():
        assert frame.loc[members, REFERENCE_FEATURE].sum() == pytest.approx(1.0)


def test_every_column_sums_to_one_on_every_block() -> None:
    """Agents and the reference respondent alike put mass one on each question."""

    for with_reference in (False, True):
        frame, groups, _ = _synthetic_basis(with_reference=with_reference)
        for members in groups.values():
            assert np.allclose(frame.loc[members].sum(axis=0).to_numpy(), 1.0)


def test_weight_is_zero_when_the_basis_already_spans_the_target() -> None:
    """Nothing protects it: an unneeded reference respondent is dropped."""

    frame, _, true_agents = _synthetic_basis()
    proxies = frame.to_numpy(dtype=float)[:, :N_AGENTS]
    target = proxies[:, true_agents].mean(axis=1)

    beta = _fit(frame, target)

    assert beta[-1] == pytest.approx(0.0, abs=1e-6)
    assert beta[true_agents].sum() == pytest.approx(1.0, abs=1e-4)
    assert int((beta != 0).sum()) == TRUE_AGENTS


@pytest.mark.parametrize("planted", [0.2, 0.5])
def test_weight_recovers_a_planted_uniform_share(planted: float) -> None:
    frame, _, true_agents = _synthetic_basis()
    matrix = frame.to_numpy(dtype=float)
    proxies = matrix[:, :N_AGENTS]
    reference = matrix[:, -1]
    target = (
        (1.0 - planted) * proxies[:, true_agents].mean(axis=1)
        + planted * reference
    )

    beta = _fit(frame, target)

    assert beta[-1] == pytest.approx(planted, abs=1e-4)
    assert beta[true_agents].sum() == pytest.approx(1.0 - planted, abs=1e-4)


def test_weights_still_sum_to_one_with_the_reference_respondent() -> None:
    frame, _, true_agents = _synthetic_basis()
    matrix = frame.to_numpy(dtype=float)
    target = (
        0.7 * matrix[:, :N_AGENTS][:, true_agents].mean(axis=1)
        + 0.3 * matrix[:, -1]
    )

    beta = _fit(frame, target)

    assert beta.sum() == pytest.approx(1.0, abs=1e-6)


def test_anti_type_weights_remain_available() -> None:
    """The reference respondent must not quietly force non-negative weights."""

    frame, _, _ = _synthetic_basis(seed=3)
    matrix = frame.to_numpy(dtype=float)
    # A target pulled away from the basis, forcing cancellation to fit it.
    target = 2.0 * matrix[:, 0] - 1.0 * matrix[:, 1]

    beta = _fit(frame, target, alpha=1e-6)

    assert (beta < 0).any()
    assert beta.sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parity
def test_fixture_gains_exactly_one_feature_that_sums_to_one_per_block() -> None:
    """End-to-end on a real basis: off is unchanged, on appends one column."""

    from tests import parity_runner

    fixture = parity_runner._FIXTURES["atp_w42"]
    missing = [
        key
        for key in (
            "survey_source",
            "survey_schema",
            "responses",
            "endowments",
            "target",
        )
        if not parity_runner._fixture_path(fixture[key]).is_file()
    ]
    if missing:
        pytest.skip(
            "licensed external fixtures are not distributed: "
            + ", ".join(missing)
        )

    def load(reference_respondent: bool):
        return parity_runner.load_regression_data(
            survey_source=parity_runner._fixture_path(
                fixture["survey_source"]
            ),
            survey_schema=parity_runner._fixture_path(
                fixture["survey_schema"]
            ),
            responses_path=parity_runner._fixture_path(fixture["responses"]),
            endowments_path=parity_runner._fixture_path(fixture["endowments"]),
            target_path=parity_runner._fixture_path(fixture["target"]),
            train_splits=("train", "valid"),
            test_split="test",
            use_proxy_only=True,
            reference_respondent=reference_respondent,
        )

    off = load(False)
    on = load(True)

    assert on.feature_names[:-1] == off.feature_names
    assert on.feature_names[-1] == REFERENCE_FEATURE

    design = on.get_split("train").X
    groups = on.binary_survey.original_to_binary_map
    for members in groups.values():
        present = [row for row in members if row in design.index]
        if not present:
            continue
        assert design.loc[
            present, REFERENCE_FEATURE
        ].sum() == pytest.approx(1.0)
