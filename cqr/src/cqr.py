"""
Conformalized quantile regression (Romano et al. 2019): fit lower and upper
quantile models, then conformalize the gap between them.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def fit_quantile_models(X_train, y_train, alpha, random_state=0):
    """Gradient boosting at the alpha/2 and 1 - alpha/2 quantiles."""
    lo_q = alpha / 2
    hi_q = 1 - alpha / 2

    model_lo = GradientBoostingRegressor(
        loss="quantile", alpha=lo_q, n_estimators=100, max_depth=3, random_state=random_state
    )
    model_hi = GradientBoostingRegressor(
        loss="quantile", alpha=hi_q, n_estimators=100, max_depth=3, random_state=random_state
    )
    model_lo.fit(X_train, y_train)
    model_hi.fit(X_train, y_train)
    return model_lo, model_hi


def cqr_scores(y_true, q_lo_pred, q_hi_pred):
    """
    E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
    Positive if y fell outside the raw band, negative if inside.
    """
    y_true = np.asarray(y_true)
    return np.maximum(q_lo_pred - y_true, y_true - q_hi_pred)


def cqr_quantile(scores, alpha):
    """Same rank formula as split conformal's conformal_quantile."""
    sorted_scores = np.sort(np.asarray(scores))
    n = len(sorted_scores)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        return np.inf
    return sorted_scores[rank - 1]


def cqr_interval(q_lo_pred, q_hi_pred, Q):
    """[q_lo(x) - Q, q_hi(x) + Q]. Width varies per point."""
    return q_lo_pred - Q, q_hi_pred + Q
