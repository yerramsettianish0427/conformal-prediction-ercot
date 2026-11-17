"""
CQR vs split conformal.
Shows CQR intervals adapt in width; measures conditional coverage in bins.
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "../split_conformal/src")

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from conformal import compute_scores, conformal_quantile, conformal_interval
from cqr import fit_quantile_models, cqr_scores, cqr_quantile, cqr_interval

ALPHA = 0.1
data = load_diabetes()
X, y = data.data, data.target

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

# Split conformal: point model + constant buffer
point_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
point_model.fit(X_train, y_train)
cal_scores = compute_scores(y_cal, point_model.predict(X_cal))
q_hat = conformal_quantile(cal_scores, ALPHA)
sc_lower, sc_upper = conformal_interval(point_model.predict(X_test), q_hat)
sc_width = sc_upper - sc_lower

# CQR: two quantile models + conformalized gap
model_lo, model_hi = fit_quantile_models(X_train, y_train, ALPHA)
cal_lo = model_lo.predict(X_cal)
cal_hi = model_hi.predict(X_cal)
cal_cqr_scores = cqr_scores(y_cal, cal_lo, cal_hi)
Q = cqr_quantile(cal_cqr_scores, ALPHA)

test_lo = model_lo.predict(X_test)
test_hi = model_hi.predict(X_test)
cqr_lower, cqr_upper = cqr_interval(test_lo, test_hi, Q)
cqr_width = cqr_upper - cqr_lower

print("=== Width comparison ===")
print(f"Split conformal width: constant at {sc_width[0]:.1f} for every point (std={sc_width.std():.4f})")
print(f"CQR width: varies, mean={cqr_width.mean():.1f}, std={cqr_width.std():.1f}, "
      f"min={cqr_width.min():.1f}, max={cqr_width.max():.1f}")

print("\n=== Overall marginal coverage ===")
sc_covered = (y_test >= sc_lower) & (y_test <= sc_upper)
cqr_covered = (y_test >= cqr_lower) & (y_test <= cqr_upper)
print(f"Split conformal: {sc_covered.mean()*100:.1f}%")
print(f"CQR:             {cqr_covered.mean()*100:.1f}%")

# Bin test points by the raw quantile width (before conformalizing) as a
# proxy for how uncertain the model is. ~30 points per bin, so this is noisy.
raw_width = test_hi - test_lo
bin_edges = np.quantile(raw_width, [0, 0.33, 0.66, 1.0])
bin_labels = ["Low uncertainty", "Medium uncertainty", "High uncertainty"]

print("\n=== Conditional coverage by predicted-uncertainty bin ===")
print(f"{'Bin':<20} {'n':<5} {'Split Conformal':<18} {'CQR'}")
for i in range(3):
    mask = (raw_width >= bin_edges[i]) & (raw_width <= bin_edges[i+1])
    n_bin = mask.sum()
    sc_bin_cov = sc_covered[mask].mean()
    cqr_bin_cov = cqr_covered[mask].mean()
    print(f"{bin_labels[i]:<20} {n_bin:<5} {sc_bin_cov*100:>6.1f}%          {cqr_bin_cov*100:>6.1f}%")
