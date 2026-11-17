"""
Split conformal on real electricity demand, random splits.

Random shuffling treats the hours as exchangeable. That assumption is
wrong for hourly data (neighboring hours are correlated), but shuffling
hides the time structure well enough that coverage should land near the
target. experiment_chronological.py removes the shuffle.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "split_conformal", "src"))

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from conformal import compute_scores, conformal_quantile, conformal_interval, evaluate_coverage
from load_data import load_ercot

ALPHA = 0.1
N_TRIALS = 50  # GBM on 43k rows is slow

X, y, _ = load_ercot()


def run_one_trial(seed):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=0)
    model.fit(X_train, y_train)

    q_hat = conformal_quantile(compute_scores(y_cal, model.predict(X_cal)), ALPHA)
    lower, upper = conformal_interval(model.predict(X_test), q_hat)
    return evaluate_coverage(y_test, lower, upper), (upper - lower).mean()


if __name__ == "__main__":
    results = np.array([run_one_trial(s) for s in range(N_TRIALS)])
    cov, width = results.T
    print(f"=== Real ERCOT demand, RANDOM splits ({N_TRIALS} trials) ===")
    print(f"Target coverage: {1 - ALPHA:.2f}")
    print(f"Coverage: {cov.mean():.4f} +/- {cov.std():.4f}")
    print(f"Avg interval width: {width.mean():.0f} +/- {width.std():.0f} MW")
