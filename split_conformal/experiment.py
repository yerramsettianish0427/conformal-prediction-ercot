"""
Split conformal vs. naive Gaussian interval on the diabetes data.
200 random splits so the spread is visible, coverage and width reported together.
"""
import sys
import os
sys.path.insert(0, "src")

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from conformal import compute_scores, conformal_quantile, conformal_interval, evaluate_coverage, naive_gaussian_interval

ALPHA = 0.1
N_TRIALS = 200
Z_SCORE = 1.645  # two-sided 90% for a Gaussian

data = load_diabetes()
X, y = data.data, data.target


def run_one_trial(model_class, seed):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    model = model_class()
    model.fit(X_train, y_train)

    cal_pred = model.predict(X_cal)
    cal_scores = compute_scores(y_cal, cal_pred)
    q_hat = conformal_quantile(cal_scores, ALPHA)

    test_pred = model.predict(X_test)

    cf_lower, cf_upper = conformal_interval(test_pred, q_hat)
    cf_coverage = evaluate_coverage(y_test, cf_lower, cf_upper)
    cf_width = (cf_upper - cf_lower).mean()

    # naive baseline uses the training residuals, which is what makes it overconfident
    train_pred = model.predict(X_train)
    residual_std = np.std(y_train - train_pred)
    g_lower, g_upper = naive_gaussian_interval(test_pred, residual_std, z=Z_SCORE)
    g_coverage = evaluate_coverage(y_test, g_lower, g_upper)
    g_width = (g_upper - g_lower).mean()

    return cf_coverage, cf_width, g_coverage, g_width


def run_experiment(model_class, model_name):
    results = np.array([run_one_trial(model_class, seed) for seed in range(N_TRIALS)])
    cf_cov, cf_w, g_cov, g_w = results.T

    print(f"\n=== {model_name} ===")
    print(f"Target coverage: {1-ALPHA:.2f}")
    print()
    print(f"{'Method':<20} {'Coverage (mean +/- std)':<28} {'Width (mean +/- std)'}")
    print(f"{'Split Conformal':<20} {cf_cov.mean():.4f} +/- {cf_cov.std():.4f}      {cf_w.mean():8.2f} +/- {cf_w.std():.2f}")
    print(f"{'Naive Gaussian':<20} {g_cov.mean():.4f} +/- {g_cov.std():.4f}      {g_w.mean():8.2f} +/- {g_w.std():.2f}")

    return {
        "model": model_name,
        "cf_coverage_mean": cf_cov.mean(), "cf_coverage_std": cf_cov.std(),
        "cf_width_mean": cf_w.mean(), "cf_width_std": cf_w.std(),
        "g_coverage_mean": g_cov.mean(), "g_coverage_std": g_cov.std(),
        "g_width_mean": g_w.mean(), "g_width_std": g_w.std(),
    }


if __name__ == "__main__":
    print(f"Running {N_TRIALS} random splits per model...")
    linear_results = run_experiment(LinearRegression, "Linear Regression")
    gbm_results = run_experiment(
        lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0),
        "Gradient Boosting"
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in [linear_results, gbm_results]:
        print(f"\n{r['model']}:")
        print(f"  Split conformal: {r['cf_coverage_mean']*100:.1f}% coverage (target 90%), avg width {r['cf_width_mean']:.1f}")
        print(f"  Naive Gaussian:  {r['g_coverage_mean']*100:.1f}% coverage (target 90%), avg width {r['g_width_mean']:.1f}")
