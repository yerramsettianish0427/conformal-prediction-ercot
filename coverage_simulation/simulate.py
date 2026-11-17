"""
Verify the split conformal coverage guarantee by simulation.

Scores are exponential (skewed, long tail) rather than Gaussian, since the
guarantee shouldn't depend on the score distribution.

Checks that average coverage over many trials is close to 1 - alpha.
The Beta spread check (Angelopoulos & Bates, Section 3) is in beta_check.py.
"""

import numpy as np

rng = np.random.default_rng(seed=42)

n = 100          # calibration set size
alpha = 0.1       # miss rate we're allowing -> target coverage = 90%
n_trials = 5000   # how many times we repeat the whole experiment


def one_trial(n, alpha, rng):
    """Draw n calibration scores + 1 test score, return 1 if test <= q_hat."""
    # i.i.d. draws, so exchangeable
    scores = rng.exponential(scale=1.0, size=n + 1)

    calibration_scores = scores[:n]
    test_score = scores[n]

    # q_hat = the ceil((n+1)(1-alpha))-th smallest calibration score
    rank_position = int(np.ceil((n + 1) * (1 - alpha)))
    rank_position = min(rank_position, n)
    q_hat = np.sort(calibration_scores)[rank_position - 1]

    covered = test_score <= q_hat
    return int(covered)


results = np.array([one_trial(n, alpha, rng) for _ in range(n_trials)])

empirical_coverage = results.mean()
target_coverage = 1 - alpha

# My first version checked empirical >= target and "failed" at 0.898.
# That's just Monte Carlo noise, so compare in standard errors instead.
standard_error = np.sqrt(target_coverage * (1 - target_coverage) / n_trials)
z_score = (empirical_coverage - target_coverage) / standard_error

print(f"Target coverage (1 - alpha):      {target_coverage:.4f}")
print(f"Empirical coverage ({n_trials} trials): {empirical_coverage:.4f}")
print(f"Theoretical lower bound:           {target_coverage:.4f}")
print(f"Theoretical upper bound:           {target_coverage + 1/(n+1):.4f}")
print(f"Monte Carlo standard error:        {standard_error:.4f}")
print(f"Deviation from target:             {z_score:+.2f} standard errors")
print()
if abs(z_score) <= 3:
    print(f"PASS -- within {abs(z_score):.2f} SE of target, consistent with the guarantee")
else:
    print(f"CHECK -- {abs(z_score):.2f} SE from target, worth investigating")
