"""The penalty search should say when it did not actually decide anything.

Two failure modes, both seen in the frozen paper runs: the optimum sitting at an
end of the searched range, and many penalties scoring indistinguishably because
the penalty has stopped discriminating. Both are reported, not corrected --
they are facts about the grid and the data.
"""

from __future__ import annotations

import warnings

import pytest

from p2p.stages.fit import (
    PenaltySelectionWarning,
    penalty_selection_diagnostics,
)


class _FakeCV:
    """Stands in for a fitted ConstrainedModelCV."""

    def __init__(self, scores, deviations, best, k=5):
        self.cv_mean_mse_ = scores
        self.cv_std_mse_ = deviations    # population SD of the k fold errors
        self.best_params_ = best
        self.k = k


def _diagnose(scores, deviations, best, alphas, l1_ratios=None, k=5):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagnostics = penalty_selection_diagnostics(
            _FakeCV(scores, deviations, best, k),
            alphas=alphas,
            l1_ratios=l1_ratios,
        )
    messages = [
        str(entry.message)
        for entry in caught
        if issubclass(entry.category, PenaltySelectionWarning)
    ]
    return diagnostics, messages


def test_a_clear_interior_optimum_passes_quietly() -> None:
    alphas = [0.001, 0.01, 0.1]
    scores = {(0.001, None): 0.05, (0.01, None): 0.01, (0.1, None): 0.06}
    deviations = {key: 0.001 for key in scores}

    diagnostics, messages = _diagnose(
        scores, deviations, (0.01, None), alphas
    )

    assert diagnostics["alpha_at_grid_edge"] is None
    assert diagnostics["cv_configs_within_one_se"] == 1
    assert messages == []


def test_selection_at_the_ceiling_is_flagged() -> None:
    alphas = [0.001, 0.01, 0.1]
    scores = {(0.001, None): 0.06, (0.01, None): 0.05, (0.1, None): 0.01}
    deviations = {key: 0.001 for key in scores}

    diagnostics, messages = _diagnose(
        scores, deviations, (0.1, None), alphas
    )

    assert diagnostics["alpha_at_grid_edge"] == "ceiling"
    assert any("ceiling of the searched range" in m for m in messages)


def test_selection_at_the_floor_is_flagged() -> None:
    alphas = [0.001, 0.01, 0.1]
    scores = {(0.001, None): 0.01, (0.01, None): 0.05, (0.1, None): 0.06}
    deviations = {key: 0.001 for key in scores}

    diagnostics, messages = _diagnose(
        scores, deviations, (0.001, None), alphas
    )

    assert diagnostics["alpha_at_grid_edge"] == "floor"
    assert any("floor of the searched range" in m for m in messages)


def test_a_flat_curve_is_flagged_as_indistinguishable() -> None:
    """The frozen-run pattern: the penalty has stopped discriminating."""

    alphas = [0.001, 0.01, 0.1, 1.0]
    scores = {
        (0.001, None): 0.0200,
        (0.01, None): 0.0199,
        (0.1, None): 0.0198,
        (1.0, None): 0.0197,
    }
    deviations = {key: 0.005 for key in scores}

    diagnostics, messages = _diagnose(
        scores, deviations, (1.0, None), alphas
    )

    assert diagnostics["cv_configs_within_one_se"] == 4
    assert diagnostics["cv_config_count"] == 4
    assert any("not distinguishable from noise" in m for m in messages)


def test_l1_ratio_at_its_edge_is_flagged() -> None:
    """l1_ratio = 1 collapses ElasticNet to pure L1; worth knowing."""

    alphas = [0.01, 0.1]
    l1_ratios = [0.1, 0.5, 1.0]
    scores = {
        (a, r): (0.01 if (a, r) == (0.01, 1.0) else 0.05)
        for a in alphas
        for r in l1_ratios
    }
    deviations = {key: 0.0001 for key in scores}

    diagnostics, messages = _diagnose(
        scores, deviations, (0.01, 1.0), alphas, l1_ratios
    )

    assert diagnostics["l1_ratio_at_grid_edge"] == "ceiling"
    assert any("l1_ratio" in m for m in messages)


def test_a_single_point_grid_has_no_edge_to_report() -> None:
    scores = {(0.01, None): 0.02}
    deviations = {(0.01, None): 0.001}

    diagnostics, messages = _diagnose(
        scores, deviations, (0.01, None), [0.01]
    )

    assert "alpha_at_grid_edge" not in diagnostics
    assert messages == []


@pytest.mark.parity
def test_a_real_fit_reports_an_undetermined_penalty(cvxpy_solver: str) -> None:
    """On a real basis, the warning says the penalty was not determined.

    The folds are pinned to this run's recorded paper folds, so the result does
    not depend on the default fold rule. Which grid edge wins is *not*
    asserted: on this basis every grid value scores within one standard error
    of the best, and the winning edge flips with the fold assignment (floor
    under the recorded folds and under the hashed default, ceiling under the
    pre-S.2 name-ordered default). What holds under every assignment is
    asserted: an edge is flagged, and no penalty is distinguishable.
    """

    from p2p.models.folds import read_fold_record

    import numpy as np

    from p2p.stages.fit import fit_model
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
        pytest.skip("licensed external fixtures are not distributed")

    data = parity_runner.load_regression_data(
        survey_source=parity_runner._fixture_path(fixture["survey_source"]),
        survey_schema=parity_runner._fixture_path(fixture["survey_schema"]),
        responses_path=parity_runner._fixture_path(fixture["responses"]),
        endowments_path=parity_runner._fixture_path(fixture["endowments"]),
        target_path=parity_runner._fixture_path(fixture["target"]),
        train_splits=("train", "valid"),
        test_split="test",
        use_proxy_only=True,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_model(
            data,
            {
                "method": "lasso",
                "alpha": list(np.logspace(-3, 1, 6)),
                "validation": {"strategy": "cv", "cv_folds": 5},
            },
            solver=cvxpy_solver,
            seed=101,
            verbose=False,
            fold_assignments=read_fold_record(
                parity_runner._fixture_path(fixture["cv_folds"])
            ),
        )

    diagnostics = result.diagnostics
    assert diagnostics["alpha_at_grid_edge"] in {"floor", "ceiling"}
    assert (
        diagnostics["cv_configs_within_one_se"]
        == diagnostics["cv_config_count"]
    )
    assert any(
        issubclass(entry.category, PenaltySelectionWarning)
        for entry in caught
    )


def test_within_one_se_uses_the_standard_error_not_the_fold_sd() -> None:
    """S.6-F3: one standard error of the mean over k folds, not one fold SD.

    With k = 5 and a fold SD of 0.01 the standard error is 0.01 / sqrt(4) =
    0.005. A configuration 0.006 above the best is within one fold SD but not
    within one standard error; one 0.004 above is within both.
    """

    alphas = [0.001, 0.01, 0.1]
    scores = {(0.001, None): 0.106, (0.01, None): 0.100, (0.1, None): 0.104}
    deviations = {key: 0.01 for key in scores}

    diagnostics, _ = _diagnose(scores, deviations, (0.01, None), alphas, k=5)

    assert diagnostics["cv_configs_within_one_se"] == 2      # the best and 0.104


def test_penalty_diagnostics_require_the_fold_count() -> None:
    import pytest

    alphas = [0.001, 0.01]
    scores = {(0.001, None): 0.1, (0.01, None): 0.2}
    fake = _FakeCV(scores, {key: 0.01 for key in scores}, (0.001, None))
    del fake.k
    with pytest.raises(ValueError, match="number of CV folds"):
        penalty_selection_diagnostics(fake, alphas=alphas, l1_ratios=None)
