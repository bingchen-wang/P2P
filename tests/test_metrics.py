import math

import numpy as np
import pytest

from p2p.metrics import (
    clip_and_normalize,
    emd,
    mcpa,
    mse,
    one_minus_jsd,
    project_to_simplex,
    validate_distribution,
)


def test_mse_preserves_raw_prediction_contract():
    observed = np.array([0.0, 1.0])
    predicted = np.array([-0.2, 1.2])

    assert mse(observed, predicted) == pytest.approx(0.04)


@pytest.mark.parametrize(
    ("observed", "predicted"),
    [
        ([], []),
        ([0.0, 1.0], [[0.0, 1.0]]),
        ([0.0, np.nan], [0.0, 1.0]),
        ([0.0, 1.0], [0.0, np.inf]),
    ],
)
def test_mse_rejects_invalid_numeric_inputs(observed, predicted):
    with pytest.raises(ValueError):
        mse(observed, predicted)


def test_validate_distribution_returns_copy_without_transforming():
    original = np.array([0.25, 0.75])

    validated = validate_distribution(original)
    validated[0] = 1.0

    np.testing.assert_allclose(original, [0.25, 0.75])


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([-0.1, 1.1], "negative"),
        ([0.2, 1.1], "greater than 1"),
        ([0.2, 0.2], "sum to 1"),
        ([0.5, np.nan], "finite"),
        ([], "empty"),
        ([[0.5, 0.5]], "one-dimensional"),
    ],
)
def test_validate_distribution_rejects_invalid_values(values, message):
    with pytest.raises(ValueError, match=message):
        validate_distribution(values)


def test_validation_error_includes_qid():
    with pytest.raises(ValueError, match="Q7"):
        validate_distribution([-0.1, 1.1], qid="Q7")


def test_clip_and_normalize_clips_then_rescales_without_mutation():
    original = np.array([-0.2, 0.3, 0.9])

    transformed = clip_and_normalize(original, qid="Q1")

    np.testing.assert_allclose(transformed, [0.0, 0.25, 0.75])
    np.testing.assert_allclose(original, [-0.2, 0.3, 0.9])


def test_clip_and_normalize_normalizes_positive_mass():
    transformed = clip_and_normalize([2.0, 3.0])

    np.testing.assert_allclose(transformed, [0.4, 0.6])


@pytest.mark.parametrize("values", [[0.0, 0.0], [-2.0, -1.0]])
def test_clip_and_normalize_rejects_no_positive_mass(values):
    with pytest.raises(ValueError, match="positive mass"):
        clip_and_normalize(values, qid="Q2")


def test_clip_and_normalize_error_includes_qid():
    with pytest.raises(ValueError, match="Q2"):
        clip_and_normalize([0.0, -1.0], qid="Q2")


def test_project_to_simplex_has_known_solution():
    transformed = project_to_simplex([-0.1, 0.2, 0.9])

    np.testing.assert_allclose(transformed, [0.0, 0.15, 0.85])
    assert transformed.sum() == pytest.approx(1.0)
    assert np.all(transformed >= 0.0)


def test_project_to_simplex_is_identity_on_valid_distribution():
    original = np.array([0.1, 0.3, 0.6])

    transformed = project_to_simplex(original)

    np.testing.assert_allclose(transformed, original)


def test_project_to_simplex_handles_all_negative_vector():
    transformed = project_to_simplex([-2.0, -2.0, -2.0])

    np.testing.assert_allclose(transformed, [1 / 3, 1 / 3, 1 / 3])


def test_project_to_simplex_does_not_mutate_input():
    original = np.array([-0.1, 0.2, 0.9])

    project_to_simplex(original)

    np.testing.assert_allclose(original, [-0.1, 0.2, 0.9])


@pytest.mark.parametrize("values", [[], [0.0, np.nan], [[0.1, 0.9]]])
def test_project_to_simplex_rejects_invalid_vectors(values):
    with pytest.raises(ValueError):
        project_to_simplex(values)


def test_one_minus_jsd_is_one_for_identical_distributions():
    assert one_minus_jsd([0.25, 0.75], [0.25, 0.75]) == pytest.approx(1.0)


def test_one_minus_jsd_matches_legacy_natural_log_definition():
    expected = 1.0 - math.sqrt(math.log(2.0))

    assert one_minus_jsd([1.0, 0.0], [0.0, 1.0]) == pytest.approx(expected)


def test_one_minus_jsd_is_symmetric():
    first = one_minus_jsd([0.8, 0.2], [0.3, 0.7])
    second = one_minus_jsd([0.3, 0.7], [0.8, 0.2])

    assert first == pytest.approx(second)


def test_emd_is_zero_for_identical_distributions():
    assert emd([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0)


def test_emd_is_one_between_opposite_endpoints():
    assert emd([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]) == pytest.approx(1.0)


def test_emd_uses_normalized_canonical_option_distance():
    assert emd([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(0.5)


def test_emd_handles_single_category():
    assert emd([1.0], [1.0]) == pytest.approx(0.0)


def test_mcpa_reports_matching_and_nonmatching_majorities():
    assert mcpa([0.7, 0.3], [0.6, 0.4]) == 1.0
    assert mcpa([0.7, 0.3], [0.4, 0.6]) == 0.0


def test_mcpa_preserves_legacy_first_index_tie_behavior():
    assert mcpa([0.5, 0.5], [0.5, 0.5]) == 1.0
    assert mcpa([0.5, 0.5], [0.4, 0.6]) == 0.0


@pytest.mark.parametrize("metric", [one_minus_jsd, emd, mcpa])
def test_distribution_metrics_never_transform_invalid_predictions(metric):
    with pytest.raises(ValueError, match="Q-invalid"):
        metric(
            [0.5, 0.5],
            [-0.1, 1.1],
            qid="Q-invalid",
        )


@pytest.mark.parametrize("metric", [one_minus_jsd, emd, mcpa])
def test_distribution_metrics_require_matching_support(metric):
    with pytest.raises(ValueError, match="same shape"):
        metric([0.5, 0.5], [0.2, 0.3, 0.5], qid="Q-shape")