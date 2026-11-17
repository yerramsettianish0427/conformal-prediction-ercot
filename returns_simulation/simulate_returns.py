"""
Simulated daily stock returns with volatility clustering.

These are not real market returns. I built this without internet access, so
the returns come from a GARCH(1,1) process, the standard model for equity
volatility: calm and turbulent periods cluster together, which breaks
exchangeability.
"""
import numpy as np

def simulate_garch_returns(n_days, seed=42):
    """GARCH(1,1): today's variance depends on yesterday's variance and squared return."""
    rng = np.random.default_rng(seed)

    omega = 0.00001
    alpha = 0.10
    beta = 0.85

    returns = np.zeros(n_days)
    variance = np.zeros(n_days)
    variance[0] = omega / (1 - alpha - beta)  # start at long-run variance

    for t in range(1, n_days):
        variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]
        returns[t] = rng.normal(0, np.sqrt(variance[t]))

    return returns, variance


if __name__ == "__main__":
    n_days = 504  # ~2 trading years
    returns, variance = simulate_garch_returns(n_days)

    print(f"Simulated {n_days} daily returns (GARCH(1,1))")
    print(f"Mean return: {returns.mean():.5f}, Std: {returns.std():.5f}")
    print(f"Min variance day: {variance.min():.6f}, Max variance day: {variance.max():.6f}")
    print(f"Volatility ratio (max/min): {np.sqrt(variance.max()/variance.min()):.1f}x")

    np.save("returns.npy", returns)
    np.save("variance.npy", variance)
    print("Saved returns.npy, variance.npy")
