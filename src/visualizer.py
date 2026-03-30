"""
visualizer.py
-------------
All visualisation routines for the Portfolio Optimization System.

Produces publication-quality charts:
  1. Efficient Frontier with key portfolio markers
  2. Risk vs Return scatter (Monte Carlo cloud)
  3. Portfolio weights bar chart
  4. GBM simulation fan chart
  5. Asset correlation heatmap
  6. Rolling Sharpe ratio
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import os
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

PALETTE = {
    "bg":          "#0d1117",
    "panel":       "#161b22",
    "grid":        "#21262d",
    "text":        "#e6edf3",
    "subtext":     "#8b949e",
    "accent_blue": "#58a6ff",
    "accent_gold": "#f0c040",
    "accent_red":  "#f85149",
    "accent_green":"#3fb950",
    "accent_purple":"#bc8cff",
    "frontier":    "#58a6ff",
    "scatter_cmap":"plasma",
}


def _apply_dark_style(fig, ax_list):
    """Apply a dark-theme style to a figure and its axes."""
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in ax_list:
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["text"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])
        ax.grid(True, color=PALETTE["grid"], linewidth=0.5, alpha=0.7)


def _pct(x):
    """Convert fraction to percentage string."""
    return f"{x*100:.1f}%"


# ---------------------------------------------------------------------------
# 1. Efficient Frontier
# ---------------------------------------------------------------------------

def plot_efficient_frontier(
    ef_df:         pd.DataFrame,
    gmv:           object = None,
    msr:           object = None,
    ew:            object = None,
    mc_result:     object = None,
    rf:            float  = 0.04,
    save_path:     str    = None,
    figsize:       tuple  = (14, 9),
) -> plt.Figure:
    """
    Plot the efficient frontier together with:
      - Monte Carlo random portfolio cloud (coloured by Sharpe ratio)
      - Capital Market Line (CML) from the risk-free rate through the tangency portfolio
      - Key portfolios: GMV, MSR (tangency), Equal-Weight

    The Capital Market Line satisfies:
        mu_CML(sigma) = r_f + SR_max * sigma
    where SR_max = (mu_msr - r_f) / sigma_msr is the Sharpe slope.

    Parameters
    ----------
    ef_df    : DataFrame from MeanVarianceOptimizer.efficient_frontier()
    gmv      : PortfolioResult -- Global Minimum Variance
    msr      : PortfolioResult -- Maximum Sharpe Ratio (tangency)
    ew       : PortfolioResult -- Equal-Weight benchmark
    mc_result: MonteCarloResult -- random portfolio cloud (optional)
    rf       : float           -- risk-free rate (for CML)
    save_path: str             -- if given, saves the figure to this path
    """
    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_style(fig, [ax])

    # ---- Monte Carlo scatter (background) ----
    if mc_result is not None:
        norm    = Normalize(vmin=mc_result.sharpes.min(), vmax=mc_result.sharpes.max())
        cmap    = cm.get_cmap(PALETTE["scatter_cmap"])
        colors  = cmap(norm(mc_result.sharpes))

        # Downsample for performance if very large
        n_show  = min(len(mc_result.vols), 15_000)
        idx     = np.random.choice(len(mc_result.vols), n_show, replace=False)
        sc = ax.scatter(
            mc_result.vols[idx] * 100,
            mc_result.returns[idx] * 100,
            c=mc_result.sharpes[idx],
            cmap=PALETTE["scatter_cmap"],
            alpha=0.25,
            s=4,
            linewidths=0,
            zorder=2,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Sharpe Ratio", color=PALETTE["text"], fontsize=10)
        cbar.ax.yaxis.set_tick_params(color=PALETTE["text"])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["text"])

    # ---- Efficient Frontier line ----
    ax.plot(
        ef_df["vol"] * 100,
        ef_df["ret"] * 100,
        color=PALETTE["frontier"],
        linewidth=2.5,
        zorder=5,
        label="Efficient Frontier",
    )

    # ---- Capital Market Line ----
    if msr is not None:
        sr_slope = (msr.ret - rf) / msr.vol
        sigma_range = np.linspace(0, ef_df["vol"].max() * 1.3, 200)
        cml_ret     = rf + sr_slope * sigma_range
        ax.plot(
            sigma_range * 100,
            cml_ret * 100,
            color=PALETTE["accent_gold"],
            linewidth=1.5,
            linestyle="--",
            zorder=4,
            label=f"Capital Market Line (slope={sr_slope:.2f})",
        )
        # Mark the risk-free rate intercept
        ax.scatter(0, rf * 100, color=PALETTE["accent_gold"], s=80, zorder=6,
                   marker="D", label=f"Risk-Free Rate ({_pct(rf)})")

    # ---- Key Portfolio Markers ----
    marker_specs = []
    if gmv is not None:
        marker_specs.append((gmv,  "o", PALETTE["accent_green"], "GMV",  "Global Min. Variance"))
    if msr is not None:
        marker_specs.append((msr,  "*", PALETTE["accent_gold"],  "MSR",  "Max Sharpe (Tangency)"))
    if ew  is not None:
        marker_specs.append((ew,   "s", PALETTE["accent_purple"],"1/N",  "Equal Weight"))

    for port, marker, color, abbr, full_label in marker_specs:
        ax.scatter(
            port.vol * 100, port.ret * 100,
            color=color, s=200, marker=marker, zorder=7,
            edgecolors="white", linewidth=0.8,
            label=f"{full_label}  (SR={port.sharpe:.2f})",
        )
        ax.annotate(
            f"  {abbr}\n  σ={_pct(port.vol)}\n  μ={_pct(port.ret)}",
            xy=(port.vol * 100, port.ret * 100),
            xytext=(8, -6), textcoords="offset points",
            color=color, fontsize=8.5, fontweight="bold",
            zorder=8,
        )

    # ---- Labels & Legend ----
    ax.set_xlabel("Annualised Volatility (%)", fontsize=12)
    ax.set_ylabel("Annualised Expected Return (%)", fontsize=12)
    ax.set_title("Markowitz Efficient Frontier & Monte Carlo Portfolio Simulation",
                 fontsize=14, fontweight="bold", pad=15)

    legend = ax.legend(
        loc="upper left", fontsize=9,
        framealpha=0.4, facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"],
    )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        logger.info(f"Efficient frontier saved -> {save_path}")

    return fig


# ---------------------------------------------------------------------------
# 2. Risk vs Return Scatter (standalone; identical data, different framing)
# ---------------------------------------------------------------------------

def plot_risk_return_scatter(
    mc_result:  object,
    gmv:        object = None,
    msr:        object = None,
    save_path:  str    = None,
    figsize:    tuple  = (12, 8),
) -> plt.Figure:
    """
    Risk vs Return scatter plot coloured by Sharpe ratio.
    Highlights where diversification creates value.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_style(fig, [ax])

    n_show = min(len(mc_result.vols), 20_000)
    idx    = np.random.choice(len(mc_result.vols), n_show, replace=False)

    sc = ax.scatter(
        mc_result.vols[idx] * 100,
        mc_result.returns[idx] * 100,
        c=mc_result.sharpes[idx],
        cmap="RdYlGn",
        alpha=0.35,
        s=5,
        linewidths=0,
        zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Sharpe Ratio", color=PALETTE["text"])
    cbar.ax.yaxis.set_tick_params(color=PALETTE["text"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["text"])

    for port, color, label in [
        (gmv, PALETTE["accent_green"], "Global Min. Variance"),
        (msr, PALETTE["accent_gold"],  "Max Sharpe"),
    ]:
        if port:
            ax.scatter(port.vol*100, port.ret*100, color=color,
                       s=280, marker="*", zorder=7, edgecolors="white",
                       linewidth=0.8, label=label)

    ax.set_xlabel("Annualised Volatility (%)", fontsize=12)
    ax.set_ylabel("Annualised Expected Return (%)", fontsize=12)
    ax.set_title("Risk vs Return — Monte Carlo Portfolio Cloud", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.4, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        logger.info(f"Risk-return scatter saved -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 3. Portfolio Weights Comparison
# ---------------------------------------------------------------------------

def plot_portfolio_weights(
    portfolios: dict,
    save_path:  str   = None,
    figsize:    tuple = (13, 7),
) -> plt.Figure:
    """
    Grouped bar chart comparing weights across multiple portfolios.

    Parameters
    ----------
    portfolios : dict {label: PortfolioResult}
    """
    labels   = list(portfolios.keys())
    tickers  = portfolios[labels[0]].tickers
    n_ports  = len(labels)
    n_assets = len(tickers)

    x   = np.arange(n_assets)
    w   = 0.75 / n_ports
    colors = [PALETTE["accent_blue"], PALETTE["accent_gold"],
              PALETTE["accent_green"], PALETTE["accent_red"],
              PALETTE["accent_purple"]]

    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_style(fig, [ax])

    for i, (lbl, port) in enumerate(portfolios.items()):
        offset = (i - n_ports / 2 + 0.5) * w
        bars   = ax.bar(
            x + offset,
            port.weights * 100,
            width=w * 0.9,
            color=colors[i % len(colors)],
            label=f"{lbl}  (SR={port.sharpe:.2f})",
            alpha=0.85,
            zorder=3,
        )
        # Value labels
        for bar in bars:
            h = bar.get_height()
            if h > 2:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.1f}%", ha="center", va="bottom",
                        color=PALETTE["text"], fontsize=7.5, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, fontsize=10)
    ax.set_ylabel("Portfolio Weight (%)", fontsize=11)
    ax.set_title("Portfolio Weights Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.4, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
    ax.set_ylim(0, max(
        max(p.weights.max() * 100 for p in portfolios.values()) * 1.18, 10
    ))

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        logger.info(f"Weights chart saved -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 4. GBM Simulation Fan Chart
# ---------------------------------------------------------------------------

def plot_gbm_simulation(
    gbm_result:  object,
    label:       str   = "Optimal Portfolio",
    save_path:   str   = None,
    figsize:     tuple = (13, 7),
    n_paths_show:int   = 200,
) -> plt.Figure:
    """
    Fan chart of Monte Carlo GBM simulation paths.
    Shows percentile bands and a random sample of individual paths.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    _apply_dark_style(fig, [ax1, ax2])

    days = np.arange(gbm_result.paths.shape[1])
    pcts = gbm_result.percentile_paths(quantiles=[5, 25, 50, 75, 95])

    # ---- Fan bands ----
    ax1.fill_between(days, pcts["p5"]*100,  pcts["p95"]*100,
                     color=PALETTE["accent_blue"], alpha=0.12, label="5-95th pct")
    ax1.fill_between(days, pcts["p25"]*100, pcts["p75"]*100,
                     color=PALETTE["accent_blue"], alpha=0.25, label="25-75th pct")
    ax1.plot(days, pcts["p50"]*100, color=PALETTE["accent_gold"],
             linewidth=2, label="Median")

    # ---- Sample paths ----
    idx  = np.random.choice(len(gbm_result.paths), n_paths_show, replace=False)
    for path in gbm_result.paths[idx]:
        ax1.plot(days, path*100, color=PALETTE["accent_blue"],
                 linewidth=0.3, alpha=0.10, zorder=1)

    ax1.axhline(100, color=PALETTE["subtext"], linewidth=1, linestyle="--")
    ax1.set_xlabel("Trading Days", fontsize=11)
    ax1.set_ylabel("Portfolio Value (Initial = 100)", fontsize=11)
    ax1.set_title(f"GBM Price Paths — {label}", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, framealpha=0.4, facecolor=PALETTE["panel"],
               edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    # ---- Terminal value distribution ----
    final = (gbm_result.final_values - 1) * 100   # percentage return
    ax2.hist(final, bins=80, color=PALETTE["accent_blue"], alpha=0.6,
             edgecolor="none", zorder=2)

    var5  = gbm_result.var()  * 100
    cvar5 = gbm_result.cvar() * 100
    ax2.axvline(-var5, color=PALETTE["accent_red"], linewidth=2,
                label=f"VaR(5%) = {var5:.1f}%")
    ax2.axvline(-cvar5, color=PALETTE["accent_gold"], linewidth=2,
                linestyle="--", label=f"CVaR(5%) = {cvar5:.1f}%")
    ax2.axvline(np.median(final), color=PALETTE["accent_green"], linewidth=2,
                linestyle=":", label=f"Median = {np.median(final):.1f}%")

    ax2.set_xlabel("1-Year Portfolio Return (%)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("Terminal Return Distribution", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.4, facecolor=PALETTE["panel"],
               edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    fig.suptitle(f"Monte Carlo GBM Simulation — {label}",
                 fontsize=14, fontweight="bold", color=PALETTE["text"], y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        logger.info(f"GBM simulation chart saved -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 5. Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    returns:    pd.DataFrame,
    save_path:  str   = None,
    figsize:    tuple = (10, 8),
) -> plt.Figure:
    """
    Annotated heatmap of pairwise return correlations.

    rho_{ij} = Cov(r_i, r_j) / (sigma_i * sigma_j)

    Values near +1 indicate highly correlated assets (poor diversification).
    Values near -1 indicate negatively correlated assets (excellent hedge).
    Values near  0 indicate independent assets (maximum diversification benefit).
    """
    corr = returns.corr()
    N    = len(corr)

    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_style(fig, [ax])

    cmap = plt.cm.RdYlGn
    im   = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto", zorder=2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation", color=PALETTE["text"])
    cbar.ax.yaxis.set_tick_params(color=PALETTE["text"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["text"])

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(corr.index, fontsize=10)

    for i in range(N):
        for j in range(N):
            val   = corr.values[i, j]
            color = "black" if abs(val) < 0.5 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold", zorder=3)

    ax.set_title("Asset Return Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        logger.info(f"Correlation heatmap saved -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 6. Rolling Sharpe Ratio
# ---------------------------------------------------------------------------

def plot_rolling_sharpe(
    returns:     pd.DataFrame,
    weights:     np.ndarray,
    window:      int   = 63,   # approx. 1 quarter
    rf_daily:    float = 0.04 / 252,
    save_path:   str   = None,
    figsize:     tuple = (13, 5),
) -> plt.Figure:
    """
    Plot rolling Sharpe ratio of the portfolio to assess time-varying performance.

    Rolling Sharpe at time t (window size T):
        SR(t) = (mean(r_p[t-T:t]) - r_f) / std(r_p[t-T:t]) * sqrt(252)
    """
    port_ret = (returns @ weights).rename("portfolio")

    rolling_mean = port_ret.rolling(window).mean()
    rolling_std  = port_ret.rolling(window).std()
    rolling_sr   = (rolling_mean - rf_daily) / rolling_std * np.sqrt(252)

    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_style(fig, [ax])

    ax.plot(rolling_sr.index, rolling_sr.values,
            color=PALETTE["accent_blue"], linewidth=1.5, zorder=3)
    ax.fill_between(rolling_sr.index, rolling_sr.values, 0,
                    where=rolling_sr.values > 0,
                    color=PALETTE["accent_green"], alpha=0.15, zorder=2)
    ax.fill_between(rolling_sr.index, rolling_sr.values, 0,
                    where=rolling_sr.values <= 0,
                    color=PALETTE["accent_red"], alpha=0.20, zorder=2)
    ax.axhline(0, color=PALETTE["subtext"], linewidth=1, linestyle="--")
    ax.axhline(rolling_sr.mean(), color=PALETTE["accent_gold"],
               linewidth=1.5, linestyle=":", label=f"Mean SR = {rolling_sr.mean():.2f}")

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Annualised Sharpe Ratio", fontsize=11)
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio — Max Sharpe Portfolio",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.4, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        logger.info(f"Rolling Sharpe chart saved -> {save_path}")
    return fig
