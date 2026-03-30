"""
run_with_synthetic_data.py
--------------------------
Runs the full pipeline using synthetically generated price data that
closely mimics real US large-cap equity statistics (2019-2024 period).

This script is used when yfinance is unavailable (e.g., no network).
The synthetic data is generated via correlated Geometric Brownian Motion
using historically calibrated parameters.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from portfolio_optimizer import MeanVarianceOptimizer
from monte_carlo         import simulate_random_portfolios, simulate_gbm_paths
from visualizer          import (plot_efficient_frontier, plot_risk_return_scatter,
                                  plot_portfolio_weights, plot_gbm_simulation,
                                  plot_correlation_heatmap, plot_rolling_sharpe)

# ── Configuration ──────────────────────────────────────────────────────────────
TICKERS  = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "JNJ", "XOM"]
RF       = 0.04
OUT_DIR  = "outputs/"
os.makedirs(OUT_DIR, exist_ok=True)
TRADING_DAYS = 252
N_DAYS       = TRADING_DAYS * 6   # ~2019-2024

# ── Calibrated Parameters (historically representative for these tickers) ──────
# Annualised expected returns (approximate)
MU_ANNUAL = np.array([0.28, 0.30, 0.22, 0.24, 0.38, 0.14, 0.08, 0.12])

# Approximate annualised volatilities
SIGMA_VEC = np.array([0.32, 0.30, 0.28, 0.33, 0.62, 0.26, 0.18, 0.28])

# Approximate correlation matrix (sector-based)
CORR = np.array([
    # AAPL  MSFT  GOOGL AMZN  TSLA  JPM   JNJ   XOM
    [ 1.00, 0.78, 0.72, 0.68, 0.48, 0.42, 0.28, 0.22],  # AAPL
    [ 0.78, 1.00, 0.76, 0.70, 0.45, 0.44, 0.30, 0.24],  # MSFT
    [ 0.72, 0.76, 1.00, 0.74, 0.44, 0.40, 0.26, 0.20],  # GOOGL
    [ 0.68, 0.70, 0.74, 1.00, 0.46, 0.38, 0.24, 0.18],  # AMZN
    [ 0.48, 0.45, 0.44, 0.46, 1.00, 0.32, 0.20, 0.28],  # TSLA
    [ 0.42, 0.44, 0.40, 0.38, 0.32, 1.00, 0.38, 0.42],  # JPM
    [ 0.28, 0.30, 0.26, 0.24, 0.20, 0.38, 1.00, 0.30],  # JNJ
    [ 0.22, 0.24, 0.20, 0.18, 0.28, 0.42, 0.30, 1.00],  # XOM
])

# Convert to covariance matrix: Sigma_ij = rho_ij * sigma_i * sigma_j
SIGMA_ANNUAL = np.outer(SIGMA_VEC, SIGMA_VEC) * CORR


# ── Generate synthetic price paths via GBM ─────────────────────────────────────
def generate_synthetic_prices(
    mu_annual:    np.ndarray,
    Sigma_annual: np.ndarray,
    tickers:      list,
    n_days:       int = N_DAYS,
    seed:         int = 2019,
) -> pd.DataFrame:
    """
    Simulate correlated asset price paths via Multivariate GBM.

    For each asset i, the daily log-return is:
        r_i(t) = (mu_i - 0.5*sigma_i^2)*dt + [L * Z_iid]_i

    where L = cholesky(Sigma_daily) injects the correlation structure.
    """
    np.random.seed(seed)
    N  = len(tickers)
    dt = 1.0 / TRADING_DAYS

    # Daily drift (with Ito correction)
    daily_vol   = np.sqrt(np.diag(Sigma_annual) * dt)
    daily_drift = mu_annual * dt - 0.5 * daily_vol**2

    # Cholesky factor of daily covariance
    L = np.linalg.cholesky(Sigma_annual * dt)

    # Correlated daily log-returns
    Z     = np.random.randn(n_days, N)
    shocks = Z @ L.T                             # (n_days, N)
    log_ret = daily_drift[None, :] + shocks      # broadcast drift

    # Cumulative sum -> price paths
    cum_ret = np.vstack([np.zeros((1, N)), np.cumsum(log_ret, axis=0)])
    prices  = 100.0 * np.exp(cum_ret)            # start at 100

    # Build DatetimeIndex
    dates = pd.bdate_range(start="2019-01-02", periods=n_days + 1)
    return pd.DataFrame(prices, index=dates, columns=tickers)


# ── Compute returns and portfolio statistics ────────────────────────────────────
def compute_stats(prices: pd.DataFrame):
    log_ret = np.log(prices / prices.shift(1)).dropna()
    mu      = log_ret.mean() * TRADING_DAYS
    Sigma   = log_ret.cov()  * TRADING_DAYS
    return log_ret, mu, Sigma


# ── Main pipeline ──────────────────────────────────────────────────────────────
def main():
    print("\n" + "#"*65)
    print("#  Portfolio Optimization System — Synthetic Data Run")
    print("#"*65 + "\n")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("[1/6] Generating synthetic price data ...")
    prices  = generate_synthetic_prices(MU_ANNUAL, SIGMA_ANNUAL, TICKERS)
    log_ret, mu, Sigma = compute_stats(prices)

    print(f"      Price matrix : {prices.shape[0]} days x {prices.shape[1]} assets")
    print(f"      Date range   : {prices.index[0].date()} → {prices.index[-1].date()}")

    # Per-asset stats
    vols   = np.sqrt(np.diag(Sigma.values))
    stats  = pd.DataFrame({
        "Ann. Return (%)":     (mu.values * 100).round(1),
        "Ann. Volatility (%)": (vols * 100).round(1),
        "Sharpe":              ((mu.values - RF) / vols).round(3),
    }, index=TICKERS)
    print("\n" + stats.to_string())

    # ── 2. Optimisation ───────────────────────────────────────────────────────
    print("\n[2/6] Running Markowitz MVO ...")
    opt = MeanVarianceOptimizer(mu, Sigma, rf=RF, allow_short=False)
    gmv = opt.minimum_variance_portfolio()
    msr = opt.maximum_sharpe_portfolio()
    ew  = opt.equal_weight_portfolio()
    ef  = opt.efficient_frontier(n_points=300)

    for port in (gmv, msr, ew):
        print(port.summary())

    # ── 3. Monte Carlo (random portfolio cloud) ───────────────────────────────
    print("[3/6] Monte Carlo portfolio simulation ...")
    mc = simulate_random_portfolios(mu, Sigma, n_portfolios=25_000, rf=RF, seed=42)

    # ── 4. GBM Forward Simulation ────────────────────────────────────────────
    print("[4/6] GBM forward simulation (Max Sharpe portfolio) ...")
    gbm = simulate_gbm_paths(
        weights       = msr.weights,
        mu_annual     = mu.values,
        Sigma_annual  = Sigma.values,
        n_sims        = 5_000,
        horizon_years = 1.0,
        tickers       = TICKERS,
        seed          = 42,
    )
    print(f"      Median terminal value : {np.median(gbm.final_values):.4f}")
    print(f"      VaR(5%)  1-year       : {gbm.var()*100:.1f}%")
    print(f"      CVaR(5%) 1-year       : {gbm.cvar()*100:.1f}%")

    # ── 5. Visualisation ─────────────────────────────────────────────────────
    print("\n[5/6] Generating charts ...")

    plot_efficient_frontier(
        ef_df=ef, gmv=gmv, msr=msr, ew=ew, mc_result=mc, rf=RF,
        save_path=os.path.join(OUT_DIR, "efficient_frontier.png"),
    )
    plot_risk_return_scatter(
        mc_result=mc, gmv=gmv, msr=msr,
        save_path=os.path.join(OUT_DIR, "risk_return_scatter.png"),
    )
    plot_portfolio_weights(
        portfolios={"Max Sharpe": msr, "Min Variance": gmv, "Equal Weight": ew},
        save_path=os.path.join(OUT_DIR, "portfolio_weights.png"),
    )
    plot_gbm_simulation(
        gbm_result=gbm, label="Max Sharpe Portfolio",
        save_path=os.path.join(OUT_DIR, "gbm_simulation.png"),
    )
    plot_correlation_heatmap(
        returns=log_ret,
        save_path=os.path.join(OUT_DIR, "correlation_heatmap.png"),
    )
    plot_rolling_sharpe(
        returns=log_ret, weights=msr.weights,
        window=63, rf_daily=RF/252,
        save_path=os.path.join(OUT_DIR, "rolling_sharpe.png"),
    )

    # ── 6. Export CSVs ───────────────────────────────────────────────────────
    print("[6/6] Exporting results ...")
    ef.to_csv(os.path.join(OUT_DIR, "efficient_frontier.csv"), index=False)
    pd.DataFrame({
        "Ticker":     TICKERS,
        "GMV_Weight": gmv.weights,
        "MSR_Weight": msr.weights,
        "EW_Weight":  ew.weights,
    }).to_csv(os.path.join(OUT_DIR, "portfolio_weights.csv"), index=False)
    pd.DataFrame([
        {"Portfolio": n, "Return%": round(p.ret*100,2),
         "Vol%": round(p.vol*100,2), "Sharpe": round(p.sharpe,4)}
        for n, p in [("GMV", gmv), ("MSR", msr), ("EW", ew)]
    ]).to_csv(os.path.join(OUT_DIR, "portfolio_metrics.csv"), index=False)

    print(f"\n{'#'*65}")
    print(f"#  DONE — all outputs saved to {OUT_DIR}")
    print(f"{'#'*65}\n")


if __name__ == "__main__":
    main()
