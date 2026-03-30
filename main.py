"""
main.py
-------
End-to-end pipeline for the Portfolio Optimization and Risk Analysis System.

Orchestrates:
  1. Data ingestion  (data_fetcher)
  2. Optimization    (portfolio_optimizer)
  3. Monte Carlo     (monte_carlo)
  4. Visualization   (visualizer)
  5. Results export  (CSV + PNG)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

# Make sure src/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_fetcher       import (fetch_price_data, compute_log_returns,
                                 annualise_returns, annualise_covariance,
                                 summary_statistics)
from portfolio_optimizer import MeanVarianceOptimizer
from monte_carlo         import simulate_random_portfolios, simulate_gbm_paths
from visualizer          import (plot_efficient_frontier, plot_risk_return_scatter,
                                  plot_portfolio_weights, plot_gbm_simulation,
                                  plot_correlation_heatmap, plot_rolling_sharpe)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# CONFIGURATION
# ===========================================================================

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "JNJ", "XOM"]

CONFIG = dict(
    start         = "2019-01-01",
    end           = "2024-12-31",
    risk_free_rate= 0.04,           # annualised, e.g. 4% US T-bill proxy
    allow_short   = False,          # True = allow short selling
    n_mc_portfolios = 25_000,       # random portfolios for cloud chart
    n_gbm_sims    = 5_000,          # GBM forward simulation paths
    horizon_years = 1.0,            # GBM simulation horizon
    ef_points     = 300,            # points on the efficient frontier
    cache_dir     = "data/",
    output_dir    = "outputs/",
)


# ===========================================================================
# PIPELINE STEPS
# ===========================================================================

def step1_data(config: dict) -> tuple:
    """Fetch prices and compute returns / statistics."""
    logger.info("=" * 60)
    logger.info("STEP 1 — Data Ingestion")
    logger.info("=" * 60)

    prices = fetch_price_data(
        TICKERS, start=config["start"], end=config["end"],
        cache_dir=config["cache_dir"],
    )
    log_ret = compute_log_returns(prices)
    mu      = annualise_returns(log_ret)
    Sigma   = annualise_covariance(log_ret)

    logger.info(f"Dataset: {prices.shape[0]} trading days, {prices.shape[1]} assets")
    logger.info(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print("\n--- Per-Asset Summary Statistics ---")
    print(summary_statistics(prices).to_string())
    return prices, log_ret, mu, Sigma


def step2_optimize(mu, Sigma, config: dict) -> dict:
    """Run Markowitz MVO and compute key portfolios."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2 — Mean-Variance Optimization")
    logger.info("=" * 60)

    opt = MeanVarianceOptimizer(mu, Sigma, rf=config["risk_free_rate"],
                                allow_short=config["allow_short"])

    gmv = opt.minimum_variance_portfolio()
    msr = opt.maximum_sharpe_portfolio()
    ew  = opt.equal_weight_portfolio()
    ef  = opt.efficient_frontier(n_points=config["ef_points"])

    print(gmv.summary())
    print(msr.summary())
    print(ew.summary())

    logger.info(f"Efficient frontier: {len(ef)} feasible portfolios")
    return dict(opt=opt, gmv=gmv, msr=msr, ew=ew, ef=ef)


def step3_monte_carlo(mu, Sigma, msr_weights, config: dict) -> dict:
    """Run Monte Carlo random portfolios + GBM forward simulation."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3 — Monte Carlo Simulation")
    logger.info("=" * 60)

    # --- Random portfolio allocation cloud ---
    mc = simulate_random_portfolios(
        mu, Sigma,
        n_portfolios = config["n_mc_portfolios"],
        rf           = config["risk_free_rate"],
    )
    logger.info(
        f"MC random portfolios: best Sharpe = {mc.sharpes.max():.4f}  "
        f"| best Return = {mc.returns.max():.4f}"
    )

    # --- GBM forward paths for max-Sharpe portfolio ---
    gbm_msr = simulate_gbm_paths(
        weights        = msr_weights,
        mu_annual      = mu.values,
        Sigma_annual   = Sigma.values,
        n_sims         = config["n_gbm_sims"],
        horizon_years  = config["horizon_years"],
        tickers        = list(mu.index),
    )
    logger.info(
        f"GBM (Max-Sharpe): median terminal value = {np.median(gbm_msr.final_values):.4f} "
        f"| VaR(5%) = {gbm_msr.var():.4f} | CVaR(5%) = {gbm_msr.cvar():.4f}"
    )
    return dict(mc=mc, gbm_msr=gbm_msr)


def step4_visualize(prices, log_ret, opt_results, mc_results, config: dict):
    """Generate and save all charts."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4 — Visualisation")
    logger.info("=" * 60)

    out = config["output_dir"]
    os.makedirs(out, exist_ok=True)

    gmv = opt_results["gmv"]
    msr = opt_results["msr"]
    ew  = opt_results["ew"]
    ef  = opt_results["ef"]
    mc  = mc_results["mc"]
    gbm = mc_results["gbm_msr"]

    # 1. Efficient Frontier + MC cloud
    plot_efficient_frontier(
        ef_df     = ef,
        gmv       = gmv,
        msr       = msr,
        ew        = ew,
        mc_result = mc,
        rf        = config["risk_free_rate"],
        save_path = os.path.join(out, "efficient_frontier.png"),
    )

    # 2. Risk vs Return scatter
    plot_risk_return_scatter(
        mc_result = mc,
        gmv       = gmv,
        msr       = msr,
        save_path = os.path.join(out, "risk_return_scatter.png"),
    )

    # 3. Portfolio weights comparison
    plot_portfolio_weights(
        portfolios = {"Max Sharpe": msr, "Min Variance": gmv, "Equal Weight": ew},
        save_path  = os.path.join(out, "portfolio_weights.png"),
    )

    # 4. GBM simulation fan chart
    plot_gbm_simulation(
        gbm_result = gbm,
        label      = "Max Sharpe Portfolio",
        save_path  = os.path.join(out, "gbm_simulation.png"),
    )

    # 5. Correlation heatmap
    plot_correlation_heatmap(
        returns   = log_ret,
        save_path = os.path.join(out, "correlation_heatmap.png"),
    )

    # 6. Rolling Sharpe ratio
    plot_rolling_sharpe(
        returns   = log_ret,
        weights   = msr.weights,
        window    = 63,
        rf_daily  = config["risk_free_rate"] / 252,
        save_path = os.path.join(out, "rolling_sharpe.png"),
    )

    logger.info(f"All charts saved to {out}")


def step5_export(opt_results, mc_results, config: dict):
    """Export numerical results to CSV for further analysis."""
    out = config["output_dir"]
    os.makedirs(out, exist_ok=True)

    # Efficient frontier data
    opt_results["ef"].to_csv(os.path.join(out, "efficient_frontier.csv"), index=False)

    # Key portfolio weights
    w_df = pd.DataFrame({
        "Ticker":     opt_results["gmv"].tickers,
        "GMV_Weight": opt_results["gmv"].weights,
        "MSR_Weight": opt_results["msr"].weights,
        "EW_Weight":  opt_results["ew"].weights,
    })
    w_df.to_csv(os.path.join(out, "portfolio_weights.csv"), index=False)

    # Monte Carlo summary
    mc_df = mc_results["mc"].to_dataframe()
    mc_df.to_csv(os.path.join(out, "monte_carlo_portfolios.csv"), index=False)

    # Risk metrics for key portfolios
    metrics = []
    for name, port in [("GMV", opt_results["gmv"]),
                       ("MSR", opt_results["msr"]),
                       ("EW",  opt_results["ew"])]:
        metrics.append({
            "Portfolio":        name,
            "Ann_Return (%)":   round(port.ret * 100, 2),
            "Ann_Volatility (%)": round(port.vol * 100, 2),
            "Sharpe_Ratio":     round(port.sharpe, 4),
            "VaR_5pct (1yr)":   round(mc_results["gbm_msr"].var(), 4),
            "CVaR_5pct (1yr)":  round(mc_results["gbm_msr"].cvar(), 4),
        })
    pd.DataFrame(metrics).to_csv(os.path.join(out, "portfolio_metrics.csv"), index=False)

    logger.info(f"Results exported to {out}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("\n" + "#" * 70)
    print("#  Portfolio Optimization and Risk Analysis System")
    print("#  Markowitz MVO + Monte Carlo + GBM")
    print("#" * 70 + "\n")

    prices, log_ret, mu, Sigma = step1_data(CONFIG)
    opt_results                = step2_optimize(mu, Sigma, CONFIG)
    mc_results                 = step3_monte_carlo(mu, Sigma,
                                                   opt_results["msr"].weights,
                                                   CONFIG)
    step4_visualize(prices, log_ret, opt_results, mc_results, CONFIG)
    step5_export(opt_results, mc_results, CONFIG)

    print("\n" + "#" * 70)
    print("#  DONE — check the outputs/ directory for charts and CSVs")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
