# python3 -m pytest tests/ -v
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from conformal import compute_scores, conformal_quantile, conformal_interval, evaluate_coverage


def test_compute_scores_basic():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 35])
    scores = compute_scores(y_true, y_pred)
    np.testing.assert_array_almost_equal(scores, [2, 2, 5])


def test_conformal_quantile_known_example():
    """Exam-score worked example: n=14, alpha=0.15 -> rank 13 -> q_hat=15."""
    scores = [1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 11, 15, 22]
    q_hat = conformal_quantile(scores, alpha=0.15)
    assert q_hat == 15, f"expected 15, got {q_hat}"


def test_rank_index_matches_formula():
    """Checks the ceil((n+1)(1-alpha)) index directly."""
    n = 199
    alpha = 0.1
    scores = np.arange(1, n + 1)  # rank == value
    expected_rank = int(np.ceil((n + 1) * (1 - alpha)))
    assert expected_rank == 180, f"sanity check on the formula itself failed: got {expected_rank}"

    q_hat = conformal_quantile(scores, alpha)
    assert q_hat == expected_rank, f"q_hat ({q_hat}) should equal the rank position ({expected_rank}) for this sequence"


def test_car_example():
    """The 10-car worked example: n=10, alpha=0.1 -> rank 10 -> q_hat=12000."""
    scores = [90, 100, 250, 300, 600, 800, 1500, 2200, 4000, 12000]
    q_hat = conformal_quantile(scores, alpha=0.1)
    assert q_hat == 12000


def test_coverage_at_zero_width_is_strict():
    """With q_hat = 0 only exact matches count."""
    y_true = np.array([100, 105, 110])
    y_pred = np.array([100, 100, 100])
    lower, upper = conformal_interval(y_pred, q_hat=0.0)
    coverage = evaluate_coverage(y_true, lower, upper)
    assert coverage == pytest.approx(1 / 3)


def test_coverage_at_infinite_width_is_always_one():
    y_true = np.array([100, -500, 99999])
    y_pred = np.array([0, 0, 0])
    lower, upper = conformal_interval(y_pred, q_hat=np.inf)
    coverage = evaluate_coverage(y_true, lower, upper)
    assert coverage == 1.0


def test_empty_calibration_set_raises():
    with pytest.raises(ValueError):
        conformal_quantile([], alpha=0.1)


def test_rank_exceeding_n_returns_infinite_quantile():
    """Should return inf, not quietly fall back to the max score."""
    scores = [1, 2, 3]
    # rank = ceil(4 * 0.99) = 4 > n = 3
    q_hat = conformal_quantile(scores, alpha=0.01)
    assert q_hat == np.inf
