"""
monte_carlo.py
--------------
Monte Carlo simulation for portfolio analysis.

TWO DISTINCT USES OF MONTE CARLO HERE:
=======================================

1. RANDOM PORTFOLIO SAMPLING (Allocation Search)
   ------------------------------------------------
   Generate thousands of random weight vectors w drawn from a
   Dirichlet(alpha=1) distribution (uniform over the N-simplex).
   For each w, compute (sigma_p, mu_p, SR).
   This produces a cloud of points that illustrates the feasible set
   and highlights the efficient frontier as the upper-left envelope.

   Mathematically, any random weight vector w with sum(w)=1, w>=0 is a
   valid long-only portfolio.  The Dirichlet distribution is the natural
   prior over the probability simplex.

2. FORWARD PATH SIMULATION (Geometric Brownian Motion)
   -----------------------------------------------------
   Simulate future portfolio value paths using the multivariate GBM model:

       dS_i = mu_i * S_i * dt + sigma_i * S_i * dW_i

   In discrete log-return form:
       ln(S_i(t+dt)) - ln(S_i(t)) = (mu_i - sigma_i^2/2)*dt + sigma_i*sqrt(dt)*epsilon_i

   Where epsilon ~ N(0, I) is decorrelated and then Cholesky-transformed
   to inject the empirical correlation structure:

       L = chol(Sigma)          (lower-triangular Cholesky factor)
       r_correlated = L * epsilon_uncorrelated

   The (mu - sigma^2/2) drift term is the Ito correction: it ensures
   E[S(T)] = S(0) * exp(mu*T), i.e., log-normal paths have the correct
   expected value in real-space.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Random portfolio sampling
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    weights:  np.ndarray    # (n_portfolios, N)
    returns:  np.ndarray    # (n_portfolios,)
    vols:     np.ndarray    # (n_portfolios,)
    sharpes:  np.ndarray    # (n_portfolios,)
    tickers:  list

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "return":   self.returns,
            "vol":      self.vols,
            "sharpe":   self.sharpes,
        })
        for i, t in enumerate(self.tickers):
            df[f"w_{t}"] = self.weights[:, i]
        return df

    def best_sharpe(self) -> pd.Series:
        """Return the weights corresponding to the highest Sharpe ratio."""
        idx = np.argmax(self.sharpes)
        return pd.Series(self.weights[idx], index=self.tickers)

    def best_return(self) -> pd.Series:
        idx = np.argmax(self.returns)
        return pd.Series(self.weights[idx], index=self.tickers)

    def min_vol(self) -> pd.Series:
        idx = np.argmin(self.vols)
        return pd.Series(self.weights[idx], index=self.tickers)


def simulate_random_portfolios(
    mu:     pd.Series,
    Sigma:  pd.DataFrame,
    n_portfolios: int = 20_000,
    rf:     float = 0.04,
    seed:   Optional[int] = 42,
) -> MonteCarloResult:
    """
    Generate n_portfolios random long-only weight vectors and compute
    their (return, volatility, Sharpe) statistics.

    Weights are sampled from Dirichlet(alpha=1), which is equivalent to
    sampling from a uniform distribution over the N-simplex.

    Parameters
    ----------
    mu            : pd.Series  -- annualised expected returns
    Sigma         : pd.DataFrame -- annualised covariance matrix
    n_portfolios  : int        -- number of random portfolios to simulate
    rf            : float      -- risk-free rate (for Sharpe calculation)
    seed          : int|None   -- random seed for reproducibility

    Returns
    -------
    MonteCarloResult
    """
    if seed is not None:
        np.random.seed(seed)

    mu_arr    = mu.values.astype(float)
    Sigma_arr = Sigma.values.astype(float)
    N         = len(mu_arr)
    tickers   = list(mu.index)

    # Sample N uniform [0,1], normalise to sum=1 (equivalent to Dirichlet(1))
    # This is the Krauss-McIntosh simplex sampling method.
    raw     = np.random.uniform(0, 1, size=(n_portfolios, N))
    weights = raw / raw.sum(axis=1, keepdims=True)   # shape (n_portfolios, N)

    # Vectorised portfolio statistics
    # mu_p = weights @ mu                       -- (n_portfolios,)
    # var_p = diag(weights @ Sigma @ weights^T) -- use einsum for efficiency
    port_returns = weights @ mu_arr                                      # (n,)
    port_vars    = np.einsum("ij,jk,ik->i", weights, Sigma_arr, weights) # (n,)
    port_vols    = np.sqrt(port_vars)                                    # (n,)
    port_sharpes = (port_returns - rf) / port_vols                       # (n,)

    logger.info(
        f"Simulated {n_portfolios} random portfolios. "
        f"Best Sharpe: {port_sharpes.max():.4f}"
    )

    return MonteCarloResult(
        weights = weights,
        returns = port_returns,
        vols    = port_vols,
        sharpes = port_sharpes,
        tickers = tickers,
    )


# ---------------------------------------------------------------------------
# Forward price path simulation (Geometric Brownian Motion)
# ---------------------------------------------------------------------------

@dataclass
class GBMSimResult:
    paths:         np.ndarray   # (n_sims, n_days+1) portfolio value paths
    final_values:  np.ndarray   # (n_sims,)  terminal values
    daily_returns: np.ndarray   # (n_sims, n_days)  daily portfolio returns
    weights:       np.ndarray   # (N,)  portfolio weights used
    tickers:       list

    def percentile_paths(self, quantiles=(5, 25, 50, 75, 95)) -> pd.DataFrame:
        return pd.DataFrame(
            np.percentile(self.paths, quantiles, axis=0).T,
            columns=[f"p{q}" for q in quantiles]
        )

    def cvar(self, alpha: float = 0.05) -> float:
        """
        Conditional Value at Risk (CVaR / Expected Shortfall) at level alpha.

        CVaR_alpha = E[ loss | loss > VaR_alpha ]
                   = - E[ R | R < q_alpha(R) ]

        Where R = (V_T - V_0) / V_0 is the portfolio return over the horizon.
        CVaR is a coherent risk measure and gives the expected loss in the
        worst (1-alpha) fraction of scenarios.
        """
        terminal_returns = self.final_values - 1.0   # V_0 = 1
        var_threshold = np.percentile(terminal_returns, alpha * 100)
        cvar = -np.mean(terminal_returns[terminal_returns <= var_threshold])
        return float(cvar)

    def var(self, alpha: float = 0.05) -> float:
        """
        Value at Risk at confidence level (1 - alpha).
        VaR_alpha = -q_alpha(R)
        """
        terminal_returns = self.final_values - 1.0
        return float(-np.percentile(terminal_returns, alpha * 100))


def simulate_gbm_paths(
    weights:    np.ndarray,
    mu_annual:  np.ndarray,
    Sigma_annual: np.ndarray,
    n_sims:     int = 5_000,
    horizon_years: float = 1.0,
    initial_value: float = 1.0,
    seed:       Optional[int] = 42,
    tickers:    Optional[list] = None,
) -> GBMSimResult:
    """
    Simulate portfolio value paths under multivariate Geometric Brownian Motion.

    Each asset i follows:
        r_i(t) = (mu_i - 0.5*sigma_i^2)*dt + sigma_i*sqrt(dt)*Z_i

    where Z_i are correlated standard normals with correlation matrix rho,
    and  sigma_i = sqrt(Sigma_{ii}).

    To generate correlated normals:
        L = cholesky(Sigma)        (Sigma = L * L^T)
        Z_correlated = L @ Z_iid   where Z_iid ~ N(0, I)

    Portfolio daily log-return:
        r_p(t) = w^T * r(t)

    Portfolio value at day t:
        V(t) = V(0) * exp( sum_{s=1}^{t} r_p(s) )

    Parameters
    ----------
    weights        : (N,)  portfolio weight vector
    mu_annual      : (N,)  annualised expected returns
    Sigma_annual   : (N,N) annualised covariance matrix
    n_sims         : number of independent simulation paths
    horizon_years  : simulation horizon in years
    initial_value  : starting portfolio value
    seed           : random seed

    Returns
    -------
    GBMSimResult
    """
    if seed is not None:
        np.random.seed(seed)

    N       = len(weights)
    n_days  = int(horizon_years * TRADING_DAYS_PER_YEAR)
    dt      = 1.0 / TRADING_DAYS_PER_YEAR

    # Daily drift with Ito correction:  mu_daily - 0.5 * sigma_daily^2
    # This ensures E[S(T)] = S(0) * exp(mu * T) under log-normal dynamics
    sigma_daily  = np.sqrt(np.diag(Sigma_annual) * dt)   # (N,)
    mu_daily     = mu_annual * dt - 0.5 * sigma_daily**2 # Ito drift, (N,)

    # Cholesky decomposition of daily covariance matrix
    # Sigma_daily = Sigma_annual * dt
    # L s.t.  L @ L^T = Sigma_daily
    try:
        L = np.linalg.cholesky(Sigma_annual * dt)
    except np.linalg.LinAlgError:
        # Add tiny jitter to make matrix positive definite
        jitter = 1e-8 * np.eye(N)
        L = np.linalg.cholesky(Sigma_annual * dt + jitter)

    # Generate iid standard normal shocks: shape (n_sims, n_days, N)
    Z_iid = np.random.randn(n_sims, n_days, N)

    # Transform to correlated normals: Z_corr[s,t,:] = L @ Z_iid[s,t,:]
    # L is (N, N); Z_iid is (n_sims, n_days, N)
    # Result: (n_sims, n_days, N)
    Z_corr = Z_iid @ L.T   # equivalent to (L @ z^T)^T for each vector z

    # Daily asset log-returns: (n_sims, n_days, N)
    asset_returns = mu_daily[None, None, :] + Z_corr   # broadcast drift

    # Portfolio daily log-returns: w^T * r_i  => (n_sims, n_days)
    port_log_returns = asset_returns @ weights   # (n_sims, n_days)

    # Cumulative log-returns => portfolio value paths
    # V(t) = V(0) * exp(sum of log-returns up to t)
    # port_log_returns: (n_sims, n_days)
    cum_log_ret = np.cumsum(port_log_returns, axis=1)          # (n_sims, n_days)
    zeros       = np.zeros((n_sims, 1))
    paths = initial_value * np.exp(
        np.hstack([zeros, cum_log_ret])                         # (n_sims, n_days+1)
    )

    paths_T = paths   # already (n_sims, n_days+1)

    logger.info(
        f"GBM simulation complete: {n_sims} paths x {n_days} days. "
        f"Median terminal value: {np.median(paths_T[:, -1]):.4f}"
    )

    return GBMSimResult(
        paths         = paths_T,
        final_values  = paths_T[:, -1],
        daily_returns = np.diff(paths_T, axis=1) / paths_T[:, :-1],
        weights       = weights,
        tickers       = tickers or [],
    )


if __name__ == "__main__":
    np.random.seed(0)
    N  = 4
    tickers = ["A", "B", "C", "D"]
    mu    = pd.Series([0.10, 0.12, 0.08, 0.15], index=tickers)
    A     = np.random.randn(N, N)
    Sigma = pd.DataFrame((A.T @ A) / N * 0.05, index=tickers, columns=tickers)

    # Random portfolios
    mc = simulate_random_portfolios(mu, Sigma, n_portfolios=5_000)
    df = mc.to_dataframe()
    print(df[["return", "vol", "sharpe"]].describe().round(4))

    # GBM paths
    w = np.array([0.3, 0.3, 0.2, 0.2])
    gbm = simulate_gbm_paths(w, mu.values, Sigma.values, n_sims=1_000)
    print(f"\nVaR(5%):  {gbm.var():.4f}")
    print(f"CVaR(5%): {gbm.cvar():.4f}")
