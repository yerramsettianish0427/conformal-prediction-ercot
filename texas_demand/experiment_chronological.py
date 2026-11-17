"""
Chronological split: train Jan-Sep 2020, calibrate Oct-Dec 2020, test on
2021-2024 without recalibrating.

The test period has Winter Storm Uri (Feb 2021) and several years of demand
growth, neither of which the calibration window saw.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "split_conformal", "src"))

import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor

from conformal import compute_scores, conformal_quantile, conformal_interval
from load_data import load_ercot

ALPHA = 0.1

X, y, ts = load_ercot()

train_mask = ts < datetime(2020, 10, 1)
cal_mask = (ts >= datetime(2020, 10, 1)) & (ts < datetime(2021, 1, 1))
test_mask = ts >= datetime(2021, 1, 1)

model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=0)
model.fit(X[train_mask], y[train_mask])

q_hat = conformal_quantile(
    compute_scores(y[cal_mask], model.predict(X[cal_mask])), ALPHA
)

test_pred = model.predict(X[test_mask])
lower, upper = conformal_interval(test_pred, q_hat)
y_test = y[test_mask]
ts_test = ts[test_mask]
covered = (y_test >= lower) & (y_test <= upper)

print("=== Real ERCOT demand, CHRONOLOGICAL split (deploy-style) ===")
print("Train: Jan-Sep 2020 | Calibrate: Oct-Dec 2020 | Test: 2021-2024, never recalibrated")
print(f"Target coverage: {1 - ALPHA:.2f}")
print(f"q_hat: {q_hat:.0f} MW")
print(f"\nOverall test coverage: {covered.mean():.4f}")

print("\nCoverage by year:")
for year in (2021, 2022, 2023, 2024):
    mask = np.array([t.year == year for t in ts_test])
    print(f"  {year}: {covered[mask].mean():.4f}   (n={mask.sum()})")

uri = np.array([t.year == 2021 and t.month == 2 and 10 <= t.day <= 20 for t in ts_test])
print(f"\nWinter Storm Uri (Feb 10-20, 2021): {covered[uri].mean():.4f}   (n={uri.sum()})")

summer24 = np.array([t.year == 2024 and t.month in (7, 8) for t in ts_test])
print(f"Summer 2024 (Jul-Aug):               {covered[summer24].mean():.4f}   (n={summer24.sum()})")

print("\nRandom splits on the same data give ~90% (experiment_random_split.py).")
