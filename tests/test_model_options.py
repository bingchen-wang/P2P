"""Unknown Stage-2 model options must be rejected, not ignored.

A misspelled option used to pass through in silence, leaving the feature off
while the run looked correctly configured -- which would quietly turn an
on/off experiment into off/off.
"""

from __future__ import annotations

import pytest

from p2p.stages.fit import MODEL_OPTIONS, validate_model_options


def test_recognised_options_are_accepted() -> None:
    validate_model_options(
        {
            "method": "lasso",
            "alpha": [0.01],
            "validation": {"strategy": "cv", "cv_folds": 5},
            "reference_respondent": True,
            "soft_group_sum": False,
        }
    )


def test_options_carried_by_shipped_configs_stay_legal() -> None:
    """`max_iter` and `post_selection_refit` appear in config/ but are unread."""

    validate_model_options(
        {
            "method": "lasso",
            "alpha": [0.01],
            "max_iter": 10_000,
            "post_selection_refit": False,
        }
    )


@pytest.mark.parametrize(
    ("typo", "intended"),
    [
        ("reference_respondant", "reference_respondent"),
        ("group_penalty_strenght", "group_penalty_strength"),
        ("sum_to_1", "sum_to_one"),
        ("nonneg_", "nonneg"),
    ],
)
def test_misspellings_are_rejected_with_a_suggestion(
    typo: str,
    intended: str,
) -> None:
    with pytest.raises(ValueError) as error:
        validate_model_options({"method": "lasso", typo: True})

    message = str(error.value)
    assert typo in message
    assert f"did you mean {intended!r}" in message


def test_unrecognisable_option_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model option"):
        validate_model_options({"method": "lasso", "qqqq": 1})


def test_every_option_read_by_the_stage_is_declared() -> None:
    """Guards against adding a config read without declaring the option."""

    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "p2p" / "stages" / "fit.py"
    ).read_text(encoding="utf-8")
    read = set(
        re.findall(r'model_config\.get\(\s*"([a-z_]+)"', source)
    ) | set(
        re.findall(r'model_config\.get\(\s*\n\s*"([a-z_]+)"', source)
    )

    assert read
    assert read <= set(MODEL_OPTIONS), sorted(read - set(MODEL_OPTIONS))


# --- S.6-F6: legacy options are accepted but never silent ----------------------------


def _fit_with(**options):
    import warnings

    from p2p.stages.fit import fit_model
    from tests.test_stage_fit import make_regression_data

    config = {
        "method": "lasso",
        "alpha": (0.001, 0.01),
        "validation": {"strategy": "cv", "cv_folds": 2},
        **options,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_model(make_regression_data(), config, solver="OSQP", seed=101)
    return result, caught


def _ignored_warnings(caught):
    from p2p.stages.fit import IgnoredModelOptionWarning

    return [w for w in caught if issubclass(w.category, IgnoredModelOptionWarning)]


def test_post_selection_refit_true_warns_and_is_recorded() -> None:
    result, caught = _fit_with(post_selection_refit=True, max_iter=10_000)

    assert result.diagnostics["ignored_model_options"] == "max_iter, post_selection_refit"
    warned = _ignored_warnings(caught)
    assert len(warned) == 1 and "post_selection_refit: true has no effect" in str(warned[0].message)


@pytest.mark.parametrize(
    ("options", "recorded"),
    [
        ({"post_selection_refit": False}, "post_selection_refit"),
        ({"max_iter": 10_000}, "max_iter"),
    ],
)
def test_legacy_options_are_recorded_without_warning(options, recorded) -> None:
    result, caught = _fit_with(**options)

    assert result.diagnostics["ignored_model_options"] == recorded
    assert _ignored_warnings(caught) == []


def test_absent_legacy_options_leave_no_record() -> None:
    result, caught = _fit_with()

    assert "ignored_model_options" not in result.diagnostics
    assert _ignored_warnings(caught) == []


def test_a_non_boolean_post_selection_refit_is_rejected() -> None:
    with pytest.raises(ValueError, match="post_selection_refit must be boolean"):
        _fit_with(post_selection_refit="yes")
