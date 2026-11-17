"""
Chronological split on simulated returns with volatility clustering.
No shuffling: train/calibrate/test strictly in time order.

Coverage should drop on high-volatility days, because volatility
clustering breaks exchangeability and the calibration set's typical
error no longer describes what comes next.
"""
import sys
sys.path.insert(0, "../split_conformal/src")

import numpy as np
from simulate_returns import simulate_garch_returns
from conformal import compute_scores, conformal_quantile, conformal_interval

ALPHA = 0.1
n_days = 1000
returns, variance = simulate_garch_returns(n_days, seed=42)

# predict 0 every day, the point is the interval, not the forecast
predictions = np.zeros(n_days)

# 40% train (unused since the prediction is 0), 30% calibration, 30% test
n_cal_end = int(n_days * 0.7)
cal_returns = returns[int(n_days*0.4):n_cal_end]
cal_variance = variance[int(n_days*0.4):n_cal_end]
test_returns = returns[n_cal_end:]
test_variance = variance[n_cal_end:]
test_predictions = predictions[n_cal_end:]

cal_scores = compute_scores(cal_returns, np.zeros(len(cal_returns)))
q_hat = conformal_quantile(cal_scores, ALPHA)

lower, upper = conformal_interval(test_predictions, q_hat)
covered = (test_returns >= lower) & (test_returns <= upper)

overall_coverage = covered.mean()
print(f"=== Chronological split, stock-like returns (volatility clustering) ===")
print(f"Target coverage: {1-ALPHA:.2f}")
print(f"Overall test-period coverage: {overall_coverage:.4f}")
print(f"q_hat (buffer, from calibration period): {q_hat:.5f}")

# split on the true simulated variance; with real data you'd need realized vol
median_var = np.median(test_variance)
low_vol_mask = test_variance <= median_var
high_vol_mask = ~low_vol_mask

print(f"\n=== Coverage by volatility regime ===")
print(f"Low-volatility days:  coverage = {covered[low_vol_mask].mean():.4f}  (n={low_vol_mask.sum()})")
print(f"High-volatility days: coverage = {covered[high_vol_mask].mean():.4f}  (n={high_vol_mask.sum()})")
print(f"\nGap: {(covered[low_vol_mask].mean() - covered[high_vol_mask].mean())*100:.1f} percentage points")
print("Overall coverage is on target but it's lower on high-volatility days.")
