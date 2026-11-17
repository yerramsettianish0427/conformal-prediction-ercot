import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from cqr import cqr_scores, cqr_quantile, cqr_interval


def test_cqr_score_positive_when_outside():
    y_true = np.array([100])
    q_lo = np.array([50])
    q_hi = np.array([90])
    score = cqr_scores(y_true, q_lo, q_hi)
    assert score[0] == 10


def test_cqr_score_negative_when_inside():
    y_true = np.array([70])
    q_lo = np.array([50])
    q_hi = np.array([90])
    score = cqr_scores(y_true, q_lo, q_hi)
    assert score[0] < 0


def test_cqr_quantile_matches_split_conformal_formula():
    # same exam-score example as test_conformal.py
    scores = [1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 11, 15, 22]
    Q = cqr_quantile(scores, alpha=0.15)
    assert Q == 15


def test_cqr_interval_asymmetric():
    q_lo = np.array([40])
    q_hi = np.array([60])
    Q = 5
    lower, upper = cqr_interval(q_lo, q_hi, Q)
    assert lower[0] == 35
    assert upper[0] == 65
