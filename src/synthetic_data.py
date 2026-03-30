"""
synthetic_data.py — Realistic Synthetic Market Data Generator
==============================================================
Used when network access is unavailable (or for unit testing).
Generates price series with realistic correlations, drift, and volatility
calibrated to approximate real US equity behaviour (2018–2023).
"""

import numpy as np
import pandas as pd

# Approximate 5-year annualized parameters for major US equities (2018-2023)
ASSET_PARAMS = {
    "AAPL":  {"mu": 0.28, "sigma": 0.31},
    "MSFT":  {"mu": 0.30, "sigma": 0.27},
    "GOOGL": {"mu": 0.18, "sigma": 0.29},
    "AMZN":  {"mu": 0.14, "sigma": 0.34},
    "TSLA":  {"mu": 0.45, "sigma": 0.75},
    "JPM":   {"mu": 0.12, "sigma": 0.26},
    "JNJ":   {"mu": 0.06, "sigma": 0.18},
    "XOM":   {"mu": 0.08, "sigma": 0.30},
}

# Approximate correlation structure (lower triangle)
CORR_MATRIX = np.array([
    [1.00, 0.72, 0.68, 0.65, 0.40, 0.45, 0.25, 0.20],  # AAPL
    [0.72, 1.00, 0.71, 0.67, 0.42, 0.47, 0.27, 0.22],  # MSFT
    [0.68, 0.71, 1.00, 0.66, 0.40, 0.44, 0.24, 0.21],  # GOOGL
    [0.65, 0.67, 0.66, 1.00, 0.38, 0.42, 0.22, 0.19],  # AMZN
    [0.40, 0.42, 0.40, 0.38, 1.00, 0.28, 0.15, 0.18],  # TSLA
    [0.45, 0.47, 0.44, 0.42, 0.28, 1.00, 0.35, 0.38],  # JPM
    [0.25, 0.27, 0.24, 0.22, 0.15, 0.35, 1.00, 0.30],  # JNJ
    [0.20, 0.22, 0.21, 0.19, 0.18, 0.38, 0.30, 1.00],  # XOM
])

def generate_synthetic_prices(
    tickers: list,
    start: str = "2018-01-01",
    end:   str = "2023-12-31",
    seed:  int = 42,
) -> pd.DataFrame:
    """
    Simulate correlated GBM price paths for the given tickers.

    Uses:
        S_{t+1} = S_t · exp[(μ_i − ½σ_i²)Δt + σ_i √Δt · ε_i_t]
    where ε ~ N(0,1) with correlation structure ρ.
    """
    rng = np.random.default_rng(seed)
    dt  = 1.0 / 252.0

    date_range = pd.bdate_range(start=start, end=end)
    T = len(date_range)
    n = len(tickers)

    mu_vec    = np.array([ASSET_PARAMS[t]["mu"]    for t in tickers])
    sigma_vec = np.array([ASSET_PARAMS[t]["sigma"] for t in tickers])

    # Build covariance matrix from correlation & sigma
    idx = [list(ASSET_PARAMS.keys()).index(t) for t in tickers]
    corr_sub = CORR_MATRIX[np.ix_(idx, idx)]
    cov_daily = np.outer(sigma_vec, sigma_vec) * corr_sub * dt

    # Cholesky factorization for correlated draws
    L = np.linalg.cholesky(cov_daily + np.eye(n) * 1e-10)

    # Daily drift (Itô-corrected)
    drift = (mu_vec - 0.5 * sigma_vec**2) * dt   # shape (n,)

    # Simulate log-return increments
    z = rng.standard_normal((T, n))
    log_increments = drift[None, :] + (L @ z.T).T   # shape (T, n)

    # Cumulative product → price paths
    log_prices = np.cumsum(log_increments, axis=0)
    prices     = 100.0 * np.exp(log_prices)  # start at $100

    return pd.DataFrame(prices, index=date_range, columns=tickers)
