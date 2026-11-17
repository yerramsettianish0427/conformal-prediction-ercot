"""
Coverage conditional on one calibration set should follow a Beta distribution.

simulate.py draws a new calibration set and one test point per trial. Here
each calibration set gets 500 test points, and the fraction covered is
recorded. Across many calibration sets those fractions should look like
Beta(k, n+1-k).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(seed=7)

n = 100
alpha = 0.1
n_calibration_sets = 2000
n_test_points_per_set = 500


def coverage_for_one_calibration_set(n, alpha, n_test_points, rng):
    calibration_scores = rng.exponential(scale=1.0, size=n)
    rank_position = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    q_hat = np.sort(calibration_scores)[rank_position - 1]

    test_scores = rng.exponential(scale=1.0, size=n_test_points)
    covered_fraction = (test_scores <= q_hat).mean()
    return covered_fraction


coverage_fractions = np.array([
    coverage_for_one_calibration_set(n, alpha, n_test_points_per_set, rng)
    for _ in range(n_calibration_sets)
])

print(f"Mean of coverage fractions:  {coverage_fractions.mean():.4f}")
print(f"Std of coverage fractions:   {coverage_fractions.std():.4f}")

# Angelopoulos & Bates write this as Beta(n+1-l, l) with l = n - k + 1
rank_position = min(int(np.ceil((n + 1) * (1 - alpha))), n)
beta_a = rank_position
beta_b = n + 1 - rank_position
theoretical_mean = beta_a / (beta_a + beta_b)
theoretical_std = np.sqrt(beta_a * beta_b / ((beta_a + beta_b) ** 2 * (beta_a + beta_b + 1)))

print(f"Theoretical Beta({beta_a},{beta_b}) mean: {theoretical_mean:.4f}")
print(f"Theoretical Beta({beta_a},{beta_b}) std:  {theoretical_std:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(coverage_fractions, bins=40, density=True, alpha=0.6,
        label="Empirical coverage fractions\n(2000 calibration sets)")

x = np.linspace(0.01, 0.99, 500)
beta_pdf = stats.beta.pdf(x, beta_a, beta_b)
ax.plot(x, beta_pdf, "r-", linewidth=2,
        label=f"Theoretical Beta({beta_a}, {beta_b})")

ax.axvline(1 - alpha, color="black", linestyle="--", alpha=0.7,
           label=f"Target coverage = {1-alpha}")
ax.set_xlabel("Coverage fraction (per calibration set)")
ax.set_ylabel("Density")
ax.set_title("Coverage Follows a Beta Distribution, Conditional on Calibration Set")
ax.legend()
fig.tight_layout()
fig.savefig("beta_coverage_check.png", dpi=150)
print("\nSaved plot: beta_coverage_check.png")
