# Portfolio Optimization & Risk Analysis System

A production-quality Python project implementing **Markowitz Mean-Variance Optimization**,
**Monte Carlo simulation**, and **Geometric Brownian Motion** — built from mathematical first 
principles using real equity data from Yahoo Finance.

## Project Structure

    portfolio_optimizer/
    ├── data/                    # Cached price CSVs (auto-generated)
    ├── src/
    │   ├── data_fetcher.py      # Yahoo Finance ingestion + return math
    │   ├── portfolio_optimizer.py  # Markowitz MVO from scratch
    │   ├── monte_carlo.py       # Random portfolios + GBM simulation
    │   └── visualizer.py        # All charts (dark-theme, pub-quality)
    ├── notebooks/
    │   └── portfolio_optimization_tutorial.ipynb
    ├── outputs/                 # Auto-generated charts + CSVs
    ├── main.py                  # End-to-end pipeline runner
    └── requirements.txt

## Quickstart

    pip install -r requirements.txt
    python main.py

## Key Mathematics

Log-Returns:      r(t) = ln(P(t)/P(t-1))
Portfolio Return: mu_p = w^T * mu
Portfolio Vol:    sigma_p = sqrt(w^T * Sigma * w)
Sharpe Ratio:     SR = (mu_p - rf) / sigma_p
GBM Drift (Ito):  mu_i - sigma_i^2/2

## Outputs Generated

    efficient_frontier.png     -- Main result with MC cloud + CML
    risk_return_scatter.png    -- Feasible set coloured by Sharpe
    portfolio_weights.png      -- Weight comparison chart
    gbm_simulation.png         -- Forward paths + VaR/CVaR
    correlation_heatmap.png    -- Asset correlation matrix
    rolling_sharpe.png         -- Time-varying Sharpe
    efficient_frontier.csv     -- Frontier numerical data
    portfolio_metrics.csv      -- Summary metrics table
