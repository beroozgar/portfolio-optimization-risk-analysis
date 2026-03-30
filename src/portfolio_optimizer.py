"""
portfolio_optimizer.py
----------------------
Implements Markowitz Mean-Variance Optimization (MVO) from scratch.

THEORY OVERVIEW
===============

Given N assets with:
  - Expected (annualised) return vector:  mu  in R^N
  - Annualised covariance matrix:         Sigma in R^{N x N}
  - Portfolio weight vector:              w   in R^N  (sum(w) = 1, w >= 0)

Portfolio statistics:
  - Portfolio return:    mu_p   = w^T * mu
  - Portfolio variance:  sigma2_p = w^T * Sigma * w
  - Portfolio std dev:   sigma_p  = sqrt(sigma2_p)
  - Sharpe Ratio:        SR = (mu_p - r_f) / sigma_p

The Efficient Frontier
----------------------
Markowitz (1952) showed that for any target return mu*, the minimum-variance
portfolio solves:

    min_{w}   w^T * Sigma * w
    s.t.      w^T * mu = mu*        (target return constraint)
              1^T * w  = 1          (full investment constraint)
              w >= 0                (no short-selling; optional)

Tracing mu* over a range of feasible returns traces the EFFICIENT FRONTIER
in (sigma, mu) space -- the set of portfolios that maximise return for each
level of risk.

Maximum Sharpe Ratio Portfolio
------------------------------
This is the tangency portfolio -- the point where a line from the risk-free
rate r_f is tangent to the efficient frontier.

We find it by maximising:
    SR(w) = (w^T * mu - r_f) / sqrt(w^T * Sigma * w)

Using the change of variables  y = w / (1^T * Sigma^{-1} * (mu - r_f * 1)):
    y* = Sigma^{-1} * (mu - r_f * 1)
    w* = y* / sum(y*)          (normalise to sum to 1)

When short-selling is disallowed, we use Sequential Least Squares
Programming (SLSQP) via scipy to solve the constrained problem numerically.

Minimum Variance Portfolio
--------------------------
The Global Minimum Variance (GMV) portfolio has the smallest possible
variance irrespective of return:
    w_gmv = Sigma^{-1} * 1 / (1^T * Sigma^{-1} * 1)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container for optimisation results
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    weights:      np.ndarray
    tickers:      list
    ret:          float          # annualised return
    vol:          float          # annualised volatility
    sharpe:       float          # Sharpe ratio
    label:        str = ""

    def to_series(self) -> pd.Series:
        return pd.Series(dict(zip(self.tickers, self.weights)))

    def summary(self) -> str:
        lines = [
            f"\n{'='*52}",
            f"  {self.label or 'Portfolio'}",
            f"{'='*52}",
            f"  Annualised Return   : {self.ret*100:>7.2f} %",
            f"  Annualised Volatility: {self.vol*100:>7.2f} %",
            f"  Sharpe Ratio        : {self.sharpe:>7.4f}",
            f"{'--'*26}",
            "  Weights:",
        ]
        for t, w in zip(self.tickers, self.weights):
            lines.append(f"    {t:<8s}  {w*100:>6.2f} %")
        lines.append("=" * 52)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core optimiser class
# ---------------------------------------------------------------------------

class MeanVarianceOptimizer:
    """
    Markowitz Mean-Variance Optimiser.

    All public methods accept and return annualised quantities.

    Parameters
    ----------
    mu    : pd.Series  -- annualised expected returns  (N,)
    Sigma : pd.DataFrame -- annualised covariance matrix (N x N)
    rf    : float      -- annualised risk-free rate (default 0.04 = 4 %)
    allow_short : bool -- whether to allow negative (short) weights
    """

    def __init__(
        self,
        mu:    pd.Series,
        Sigma: pd.DataFrame,
        rf:    float = 0.04,
        allow_short: bool = False,
    ):
        self.mu    = mu.values.astype(float)       # (N,)
        self.Sigma = Sigma.values.astype(float)    # (N, N)
        self.rf    = rf
        self.tickers = list(mu.index)
        self.N     = len(self.tickers)
        self.allow_short = allow_short

        # Pre-compute inverse (used in analytical solutions)
        try:
            self.Sigma_inv = np.linalg.inv(self.Sigma)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix is singular; using pseudo-inverse.")
            self.Sigma_inv = np.linalg.pinv(self.Sigma)

    # ------------------------------------------------------------------
    # Portfolio statistics (given weights)
    # ------------------------------------------------------------------

    def portfolio_return(self, w: np.ndarray) -> float:
        """
        mu_p = w^T * mu
        Linear combination of individual expected returns.
        """
        return float(w @ self.mu)

    def portfolio_variance(self, w: np.ndarray) -> float:
        """
        sigma2_p = w^T * Sigma * w
        Quadratic form: captures both individual variances and pairwise
        covariances.  Off-diagonal terms represent diversification effects.
        """
        return float(w @ self.Sigma @ w)

    def portfolio_volatility(self, w: np.ndarray) -> float:
        """sigma_p = sqrt(w^T * Sigma * w)"""
        return float(np.sqrt(self.portfolio_variance(w)))

    def sharpe_ratio(self, w: np.ndarray) -> float:
        """
        SR = (mu_p - r_f) / sigma_p

        Measures excess return per unit of total risk.
        Higher SR -> better risk-adjusted performance.
        """
        excess = self.portfolio_return(w) - self.rf
        vol    = self.portfolio_volatility(w)
        return float(excess / vol) if vol > 1e-10 else 0.0

    def _make_result(self, w: np.ndarray, label: str = "") -> PortfolioResult:
        return PortfolioResult(
            weights = w,
            tickers = self.tickers,
            ret     = self.portfolio_return(w),
            vol     = self.portfolio_volatility(w),
            sharpe  = self.sharpe_ratio(w),
            label   = label,
        )

    # ------------------------------------------------------------------
    # Constraints & bounds helpers
    # ------------------------------------------------------------------

    def _constraints(self, target_return: Optional[float] = None) -> list:
        """
        Standard constraints:
          1. Weights sum to 1  (full-investment / budget constraint)
          2. (Optional) Portfolio return equals target_return
        """
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        if target_return is not None:
            cons.append({
                "type": "eq",
                "fun": lambda w, r=target_return: self.portfolio_return(w) - r
            })
        return cons

    def _bounds(self):
        """
        Weight bounds:
          allow_short=False -> each weight in [0, 1]  (long-only)
          allow_short=True  -> each weight in [-1, 1] (allow shorts)
        """
        lo = -1.0 if self.allow_short else 0.0
        return [(lo, 1.0)] * self.N

    # ------------------------------------------------------------------
    # Core optimisation routines
    # ------------------------------------------------------------------

    def minimum_variance_portfolio(self) -> PortfolioResult:
        """
        Analytical solution (long-only uses numerical fallback):

        w_gmv = Sigma^{-1} * 1 / (1^T * Sigma^{-1} * 1)

        This is the Global Minimum Variance (GMV) portfolio -- the leftmost
        point on the efficient frontier.
        """
        if self.allow_short:
            # Closed-form analytical solution
            ones = np.ones(self.N)
            raw  = self.Sigma_inv @ ones
            w    = raw / (ones @ raw)
        else:
            # Numerical solution with non-negativity constraints
            w0   = np.ones(self.N) / self.N
            res  = minimize(
                fun         = self.portfolio_variance,
                x0          = w0,
                method      = "SLSQP",
                bounds      = self._bounds(),
                constraints = self._constraints(),
                options     = {"ftol": 1e-12, "maxiter": 10_000},
            )
            if not res.success:
                logger.warning(f"GMV optimisation did not converge: {res.message}")
            w = res.x

        return self._make_result(w, label="Global Minimum Variance Portfolio")

    def maximum_sharpe_portfolio(self) -> PortfolioResult:
        """
        Tangency Portfolio -- maximises the Sharpe Ratio.

        Analytical solution (short-selling allowed):
          y* = Sigma^{-1} * (mu - rf * 1)
          w* = y* / sum(y*)

        Numerical solution (long-only):
          min_w  -SR(w)   [we minimise negative SR]
          s.t.   sum(w)=1, w>=0
        """
        if self.allow_short:
            excess = self.mu - self.rf
            raw    = self.Sigma_inv @ excess
            # Guard against degenerate case (all excess returns <= 0)
            if raw.sum() <= 0:
                logger.warning("All excess returns non-positive; falling back to GMV.")
                return self.minimum_variance_portfolio()
            w = raw / raw.sum()
        else:
            def neg_sharpe(w):
                return -self.sharpe_ratio(w)

            w0  = np.ones(self.N) / self.N
            res = minimize(
                fun         = neg_sharpe,
                x0          = w0,
                method      = "SLSQP",
                bounds      = self._bounds(),
                constraints = self._constraints(),
                options     = {"ftol": 1e-12, "maxiter": 10_000},
            )
            if not res.success:
                logger.warning(f"Max-Sharpe optimisation did not converge: {res.message}")
            w = res.x

        return self._make_result(w, label="Maximum Sharpe Ratio (Tangency) Portfolio")

    def minimum_variance_for_return(self, target_return: float) -> Optional[PortfolioResult]:
        """
        Find the minimum-variance portfolio that achieves exactly `target_return`.

        This is one point on the efficient frontier.

        min_w   w^T * Sigma * w
        s.t.    w^T * mu = target_return
                sum(w)   = 1
                w >= 0   (if not allow_short)
        """
        w0  = np.ones(self.N) / self.N
        res = minimize(
            fun         = self.portfolio_variance,
            x0          = w0,
            method      = "SLSQP",
            bounds      = self._bounds(),
            constraints = self._constraints(target_return=target_return),
            options     = {"ftol": 1e-12, "maxiter": 10_000},
        )
        if not res.success:
            return None   # infeasible target return
        return self._make_result(res.x)

    def efficient_frontier(self, n_points: int = 200) -> pd.DataFrame:
        """
        Trace the efficient frontier by solving the minimum-variance problem
        for a grid of target returns between the GMV return and the maximum
        single-asset return.

        Returns a DataFrame with columns: [ret, vol, sharpe, w_ticker1, ...]
        """
        gmv        = self.minimum_variance_portfolio()
        ret_min    = gmv.ret
        ret_max    = float(self.mu.max())

        # Add a small margin so the endpoints are feasible
        targets = np.linspace(ret_min, ret_max * 0.99, n_points)

        records = []
        for r in targets:
            result = self.minimum_variance_for_return(r)
            if result is None:
                continue
            row = {"ret": result.ret, "vol": result.vol, "sharpe": result.sharpe}
            for t, w in zip(self.tickers, result.weights):
                row[f"w_{t}"] = w
            records.append(row)

        logger.info(f"Efficient frontier traced: {len(records)} feasible portfolios")
        return pd.DataFrame(records)

    def equal_weight_portfolio(self) -> PortfolioResult:
        """1/N (naive diversification) benchmark portfolio."""
        w = np.ones(self.N) / self.N
        return self._make_result(w, label="Equal-Weight (1/N) Portfolio")


if __name__ == "__main__":
    # Quick smoke-test with synthetic data
    np.random.seed(42)
    N = 5
    tickers = [f"ASSET_{i}" for i in range(N)]
    mu_vals = np.random.uniform(0.05, 0.20, N)
    A = np.random.randn(N, N)
    Sigma_vals = A.T @ A / N * 0.04   # random PSD matrix

    mu    = pd.Series(mu_vals, index=tickers)
    Sigma = pd.DataFrame(Sigma_vals, index=tickers, columns=tickers)

    opt = MeanVarianceOptimizer(mu, Sigma, rf=0.04)

    gmv = opt.minimum_variance_portfolio()
    print(gmv.summary())

    msr = opt.maximum_sharpe_portfolio()
    print(msr.summary())

    ef = opt.efficient_frontier(n_points=50)
    print(f"\nEfficient Frontier shape: {ef.shape}")
    print(ef[["ret", "vol", "sharpe"]].describe().round(4))
