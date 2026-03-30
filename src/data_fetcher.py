"""
data_fetcher.py
---------------
Fetches, cleans, and prepares historical price data from Yahoo Finance.

Mathematical context:
    We work with daily adjusted closing prices P(t), then compute
    log-returns:  r(t) = ln(P(t) / P(t-1))

    Log-returns are preferred over simple returns because:
      1. They are time-additive:  r_annual = sum of daily log-returns
      2. They are approximately normally distributed for short intervals
      3. They prevent negative price simulations in Monte Carlo paths
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252   # standard annualisation factor for equity markets


def fetch_price_data(
    tickers: list,
    start: str = "2019-01-01",
    end:   str = "2024-12-31",
    cache_dir: str = "data/",
) -> pd.DataFrame:
    """
    Download (or load cached) adjusted closing prices for a list of tickers.

    Parameters
    ----------
    tickers   : list of Yahoo Finance ticker symbols, e.g. ["AAPL","MSFT"]
    start     : ISO date string for the start of the historical window
    end       : ISO date string for the end of the historical window
    cache_dir : directory where a CSV cache is saved to avoid redundant downloads

    Returns
    -------
    pd.DataFrame  shape (T, N) -- T trading days x N assets
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = "_".join(sorted(tickers))
    cache_file = os.path.join(cache_dir, f"prices_{safe_name}_{start}_{end}.csv")

    if os.path.exists(cache_file):
        logger.info(f"Loading prices from cache: {cache_file}")
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return prices

    logger.info(f"Downloading price data for {tickers} ...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex when >1 ticker; normalise to (T, N)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.dropna(how="all", inplace=True)
    prices.to_csv(cache_file)
    logger.info(f"Saved price cache -> {cache_file}")
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log-returns from a price DataFrame.

    r_i(t) = ln( P_i(t) / P_i(t-1) )

    The first row is dropped (NaN) because there is no previous price.
    Log-returns are approximately normally distributed for short intervals,
    which is a key assumption in Markowitz portfolio theory.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    logger.info(
        f"Log-returns computed: {log_returns.shape[0]} obs x {log_returns.shape[1]} assets"
    )
    return log_returns


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple (arithmetic) returns.
    R_i(t) = (P_i(t) - P_i(t-1)) / P_i(t-1)

    Used in some portfolio metrics (e.g., arithmetic Sharpe ratio).
    """
    return prices.pct_change().dropna()


def annualise_returns(daily_returns: pd.DataFrame) -> pd.Series:
    """
    Annualise expected daily log-returns.

    mu_annual = E[r_daily] * 252

    Using the mean of log-returns is consistent with the geometric mean
    of wealth processes, which is what long-run investors care about.
    """
    return daily_returns.mean() * TRADING_DAYS_PER_YEAR


def annualise_covariance(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Annualise the sample covariance matrix of daily log-returns.

    Sigma_annual = Sigma_daily * 252

    From the iid assumption:  Var(sum of T returns) = T * Var(single return)
    This scales both variances and covariances by 252.
    """
    return daily_returns.cov() * TRADING_DAYS_PER_YEAR


def summary_statistics(prices: pd.DataFrame) -> pd.DataFrame:
    """Produce a human-readable statistics table for a quick sanity check."""
    log_ret = compute_log_returns(prices)
    mu      = annualise_returns(log_ret)
    cov     = annualise_covariance(log_ret)
    vol     = np.sqrt(np.diag(cov))          # annualised volatility (std dev)
    sharpe  = mu / vol                        # naive Sharpe assuming rf = 0

    stats = pd.DataFrame({
        "Ann. Return (%)":     (mu * 100).round(2),
        "Ann. Volatility (%)": (vol * 100).round(2),
        "Approx. Sharpe":      sharpe.round(3),
    })
    return stats


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "JNJ", "XOM"]
    prices  = fetch_price_data(tickers, start="2019-01-01", end="2024-12-31")
    print("\n--- Price Data (last 3 rows) ---")
    print(prices.tail(3))
    print("\n--- Summary Statistics ---")
    print(summary_statistics(prices))
