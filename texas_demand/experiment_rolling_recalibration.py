"""
Does monthly recalibration rescue coverage?

Same model and test period as experiment_chronological.py, but q_hat is
recomputed at the start of each month from the previous 90 days of errors.
Those errors are already known when the month starts, so no lookahead.

No guarantee here, a trailing window doesn't restore exchangeability.
Adaptive conformal inference (Gibbs & Candes 2021) is the proper version.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "split_conformal", "src"))

import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor

from conformal import compute_scores, conformal_quantile
from load_data import load_ercot

ALPHA = 0.1
WINDOW = 90 * 24  # hours of trailing errors used for each recalibration

X, y, ts = load_ercot()
train_mask = ts < datetime(2020, 10, 1)
cal_mask = (ts >= datetime(2020, 10, 1)) & (ts < datetime(2021, 1, 1))
test_mask = ts >= datetime(2021, 1, 1)

model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=0)
model.fit(X[train_mask], y[train_mask])

cal_err = compute_scores(y[cal_mask], model.predict(X[cal_mask]))
test_err = compute_scores(y[test_mask], model.predict(X[test_mask]))
ts_test = ts[test_mask]
all_err = np.concatenate([cal_err, test_err])
offset = len(cal_err)

q_static = conformal_quantile(cal_err, ALPHA)

covered_static = np.zeros(len(test_err), dtype=bool)
covered_roll = np.zeros(len(test_err), dtype=bool)
width_static, width_roll = [], []

months = sorted({(t.year, t.month) for t in ts_test})
for (yr, mo) in months:
    idx = np.array([i for i, t in enumerate(ts_test) if t.year == yr and t.month == mo])
    start = idx[0]
    window = all_err[offset + start - WINDOW: offset + start]
    q_roll = conformal_quantile(window, ALPHA)
    covered_static[idx] = test_err[idx] <= q_static
    covered_roll[idx] = test_err[idx] <= q_roll
    width_static.append((2 * q_static, len(idx)))
    width_roll.append((2 * q_roll, len(idx)))

print(f"=== Frozen vs. monthly recalibration (90% target, same model) ===")
print(f"{'':<28}{'Frozen (2020)':>15}{'Monthly recal.':>16}")
print(f"{'Overall coverage':<28}{covered_static.mean():>15.4f}{covered_roll.mean():>16.4f}")
for yr in (2021, 2022, 2023, 2024):
    m = np.array([t.year == yr for t in ts_test])
    print(f"{yr + 0:<28}{covered_static[m].mean():>15.4f}{covered_roll[m].mean():>16.4f}")
uri = np.array([t.year == 2021 and t.month == 2 and 10 <= t.day <= 20 for t in ts_test])
print(f"{'Winter Storm Uri':<28}{covered_static[uri].mean():>15.4f}{covered_roll[uri].mean():>16.4f}")
mon_cov_s = [covered_static[np.array([(t.year, t.month) == ym for t in ts_test])].mean() for ym in months]
mon_cov_r = [covered_roll[np.array([(t.year, t.month) == ym for t in ts_test])].mean() for ym in months]
print(f"{'Worst month':<28}{min(mon_cov_s):>15.4f}{min(mon_cov_r):>16.4f}")
def hour_weighted(pairs):
    return sum(w * n for w, n in pairs) / sum(n for _, n in pairs)

print(f"{'Avg interval width (MW)':<28}{hour_weighted(width_static):>15.0f}{hour_weighted(width_roll):>16.0f}")
print(f"\nq_hat range with recalibration: {min(w/2 for w, _ in width_roll):.0f} to {max(w/2 for w, _ in width_roll):.0f} MW (frozen: {q_static:.0f})")
