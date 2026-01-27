# backtest/visualiser_v2.py
"""
Dashboard V2 - Audit-Proof Performance Visualization

CRITICAL: Uses format_frozen_config_text() for config box to enable drift checking.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from strategies.config import StrategyConfig

from backtest.analytics.monthly_metrics import (
    calculate_monthly_r,
    calculate_monthly_stats,
)
from backtest.analytics.yearly_metrics import calculate_yearly_breakdown
from backtest.config_formatter import format_frozen_config_text

logger = logging.getLogger(__name__)

NY_TZ = "America/New_York"


def generate_dashboard_v2(
    *,
    results_r: pd.Series,
    equity_curve: pd.Series,
    drawdown: pd.Series,
    daily_pnl_r: pd.Series,
    trades_df: pd.DataFrame,
    config: StrategyConfig,
    symbol: str,
    days: int,
    run_id: str,
    period_start: Optional[pd.Timestamp] = None,
    period_end: Optional[pd.Timestamp] = None,
    stats: Optional[Dict[str, Any]] = None,
    price_series: Optional[pd.Series] = None,
    output_dir: str = "backtest/backtest_png",
) -> Path:
    """
    Generate Dashboard V2 PNG with audit-proof config box.

    Args:
        results_r: Trade results series (R values)
        equity_curve: Cumulative equity (R)
        drawdown: Drawdown series (R, negative values)
        daily_pnl_r: Daily PnL in R (indexed by NY date)
        trades_df: DataFrame with all trades (must have 'exit_time', 'net_r')
        config: Strategy configuration
        symbol: Trading symbol
        days: Number of days in backtest
        run_id: Unique run identifier
        period_start: Start timestamp
        period_end: End timestamp
        stats: Additional stats dict
        price_series: Optional price series for context
        output_dir: Output directory

    Returns:
        Path to generated PNG
    """
    # Calculate monthly metrics
    monthly_df = calculate_monthly_r(
        trades_df,
        ny_tz=NY_TZ,
        costs_per_trade_r=config.costs_per_trade_r,
    )
    monthly_stats = calculate_monthly_stats(monthly_df)

    # Calculate yearly metrics (from daily equity)
    yearly_df = calculate_yearly_breakdown(
        daily_pnl_r,
        trades_df,
        ny_tz=NY_TZ,
    )

    # Get frozen config text (for drift checking)
    config_text = format_frozen_config_text(config)

    # Create figure
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(
        nrows=9,
        ncols=2,
        height_ratios=[1, 2, 1.5, 1.5, 1.5, 1.5, 2, 1.5, 2],
        width_ratios=[3, 1],
        hspace=0.4,
        wspace=0.3,
    )

    # === ROW 0: TITLE + CONFIG BOX ===
    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.axis("off")

    title = f"Brooks Backtest Dashboard V2: {symbol} ({days} days)"
    if period_start and period_end:
        title += f"\n[{period_start.date()} → {period_end.date()}]"

    ax_title.text(
        0.5,
        0.5,
        title,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    # Config box (right side)
    ax_config = fig.add_subplot(gs[0, 1])
    ax_config.axis("off")
    ax_config.text(
        0.05,
        0.95,
        config_text,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
    )

    # === ROW 1: EQUITY CURVE ===
    ax_equity = fig.add_subplot(gs[1, :])
    ax_equity.plot(
        equity_curve.index,
        equity_curve.values,
        color="#2ca02c",
        linewidth=2.5,
        label="Cumulative R",
    )
    ax_equity.fill_between(
        equity_curve.index,
        equity_curve.values,
        color="#2ca02c",
        alpha=0.1,
    )
    ax_equity.set_ylabel("Cumulative R")
    ax_equity.set_title("Equity Growth (R)", fontweight="bold")
    ax_equity.grid(True, alpha=0.3)
    ax_equity.legend(loc="upper left")
    _apply_time_axis(ax_equity)

    # === ROW 2: DRAWDOWN ===
    ax_dd = fig.add_subplot(gs[2, :])
    ax_dd.fill_between(
        drawdown.index,
        drawdown.values,
        0,
        color="#d62728",
        alpha=0.3,
    )
    ax_dd.plot(
        drawdown.index,
        drawdown.values,
        color="#d62728",
        linewidth=1,
        label="Drawdown (R)",
    )
    ax_dd.set_ylabel("Drawdown (R)")
    ax_dd.set_title("Drawdown (from Daily Equity)", fontweight="bold")
    ax_dd.grid(True, alpha=0.3)
    ax_dd.legend(loc="lower left")
    _apply_time_axis(ax_dd)

    # === ROW 3: MONTHLY HEATMAP ===
    ax_heatmap = fig.add_subplot(gs[3, :])
    _plot_monthly_heatmap(ax_heatmap, monthly_df)

    # === ROW 4: ROLLING EXPECTANCY ===
    ax_exp = fig.add_subplot(gs[4, 0])
    rolling_exp = results_r.rolling(window=30).mean()
    ax_exp.plot(
        rolling_exp.index,
        rolling_exp.values,
        color="#ff7f0e",
        linewidth=2,
        label="Rolling Expectancy (30 trades)",
    )
    ax_exp.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_exp.set_ylabel("R/trade")
    ax_exp.set_title("Rolling Expectancy", fontweight="bold")
    ax_exp.grid(True, alpha=0.3)
    ax_exp.legend(loc="upper left")
    _apply_time_axis(ax_exp)

    # === ROW 4: ROLLING WINRATE ===
    ax_wr = fig.add_subplot(gs[4, 1])
    rolling_wr = (results_r > 0).rolling(window=30).mean() * 100
    ax_wr.plot(
        rolling_wr.index,
        rolling_wr.values,
        color="#1f77b4",
        linewidth=2,
        label="Winrate % (30 trades)",
    )
    ax_wr.axhline(33.3, color="orange", linestyle="--", label="Breakeven (1:2 RR)")
    ax_wr.set_ylabel("Winrate %")
    ax_wr.set_ylim(0, 100)
    ax_wr.set_title("Rolling Winrate", fontweight="bold")
    ax_wr.grid(True, alpha=0.3)
    ax_wr.legend(loc="upper left")
    _apply_time_axis(ax_wr)

    # === ROW 5: TRADES PER MONTH ===
    ax_tpm = fig.add_subplot(gs[5, 0])
    _plot_trades_per_month(ax_tpm, monthly_df)

    # === ROW 5: REGIME PERFORMANCE ===
    ax_regime = fig.add_subplot(gs[5, 1])
    _plot_regime_performance(ax_regime, trades_df, config)

    # === ROW 6: PRICE (if available) ===
    if price_series is not None and not price_series.empty:
        ax_price = fig.add_subplot(gs[6, :])
        ax_price.plot(
            price_series.index,
            price_series.values,
            color="#111111",
            linewidth=1.2,
            label=f"{symbol} M15 close",
        )
        ax_price.set_ylabel("Price")
        ax_price.set_title("Instrument Price", fontweight="bold")
        ax_price.grid(True, alpha=0.25)
        ax_price.legend(loc="upper left")
        _apply_time_axis(ax_price)

    # === ROW 7: YEAR-BY-YEAR TABLE ===
    ax_yearly = fig.add_subplot(gs[7, :])
    _plot_yearly_table(ax_yearly, yearly_df)

    # === ROW 8: METRICS SUMMARY ===
    ax_metrics = fig.add_subplot(gs[8, :])
    _plot_metrics_summary(ax_metrics, monthly_stats, stats, config)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / f"dashboard_v2_{run_id}.png"

    plt.savefig(str(filename), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Dashboard V2 saved: {filename}")
    return filename


def _apply_time_axis(ax):
    """Apply consistent time axis formatting"""
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _plot_monthly_heatmap(ax, monthly_df: pd.DataFrame):
    """Plot monthly returns heatmap (matplotlib only, no seaborn)"""
    ax.set_title("Monthly Returns (R) - NY Timezone", fontweight="bold")

    if monthly_df.empty:
        ax.text(0.5, 0.5, "No monthly data", ha="center", va="center")
        ax.axis("off")
        return

    # Pivot to year x month grid
    monthly_reset = monthly_df.reset_index()
    pivot = monthly_reset.pivot(index="year", columns="month", values="net_r")

    # Plot using imshow
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdYlGn",
        interpolation="nearest",
    )

    # Set ticks
    ax.set_xticks(range(12))
    ax.set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Net R")

    # Annotate cells with values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    color="black" if abs(val) < pivot.values.std() else "white",
                    fontsize=8,
                )


def _plot_trades_per_month(ax, monthly_df: pd.DataFrame):
    """Bar chart of trades per month"""
    ax.set_title("Trades per Month", fontweight="bold")

    if monthly_df.empty or "trades" not in monthly_df.columns:
        ax.text(0.5, 0.5, "No trade count data", ha="center", va="center")
        ax.axis("off")
        return

    # Create x-axis labels (Year-Month)
    monthly_reset = monthly_df.reset_index()
    monthly_reset["label"] = (
        monthly_reset["year"].astype(str) + "-" + monthly_reset["month"].astype(str).str.zfill(2)
    )

    ax.bar(
        range(len(monthly_reset)),
        monthly_reset["trades"],
        color="steelblue",
        alpha=0.7,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Trades")
    ax.set_xticks(range(len(monthly_reset)))
    ax.set_xticklabels(monthly_reset["label"], rotation=45, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def _plot_regime_performance(ax, trades_df: pd.DataFrame, config: StrategyConfig):
    """Plot regime performance breakdown (with fallback)"""
    ax.set_title("Regime Performance", fontweight="bold")
    ax.axis("off")

    if not config.regime_filter or "regime_at_entry" not in trades_df.columns:
        ax.text(
            0.5,
            0.5,
            "No regime data\n(regime_filter: OFF)",
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )
        return

    # Aggregate by regime
    regime_stats = (
        trades_df.groupby("regime_at_entry").agg({"net_r": ["sum", "count", "mean"]}).round(2)
    )

    # Format as text table
    lines = ["Regime Breakdown:"]
    lines.append("-" * 40)

    for regime in regime_stats.index:
        net_r = regime_stats.loc[regime, ("net_r", "sum")]
        count = int(regime_stats.loc[regime, ("net_r", "count")])
        avg_r = regime_stats.loc[regime, ("net_r", "mean")]

        lines.append(f"{regime:12s}: {net_r:+7.2f}R ({count:3d} trades, avg {avg_r:+.3f}R)")

    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )


def _plot_yearly_table(ax, yearly_df: pd.DataFrame):
    """Display year-by-year table"""
    ax.set_title("Year-by-Year Performance (Max DD from Daily Equity)", fontweight="bold")
    ax.axis("off")

    if yearly_df.empty:
        ax.text(0.5, 0.5, "No yearly data", ha="center", va="center")
        return

    # Format table
    lines = [f"{'Year':<6} {'Net R':>10} {'Max DD':>10} {'Trades':>8} {'Sharpe':>8}"]
    lines.append("-" * 50)

    for year, row in yearly_df.iterrows():
        lines.append(
            f"{year:<6} {row['net_r']:>+10.2f} {row['max_dd_r']:>10.2f} "
            f"{int(row['trades']):>8} {row['sharpe']:>8.3f}"
        )

    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )


def _plot_metrics_summary(
    ax,
    monthly_stats: dict,
    stats: Optional[dict],
    config: StrategyConfig,
):
    """Display metrics summary block"""
    ax.axis("off")

    lines = ["METRICS SUMMARY"]
    lines.append("=" * 60)

    # Monthly metrics
    lines.append("\n📊 MONTHLY CONSISTENCY:")
    lines.append(f"  Median monthly R    : {monthly_stats['median_monthly_r']:+.2f}R")
    lines.append(f"  P25 monthly R       : {monthly_stats['p25_monthly_r']:+.2f}R")
    lines.append(f"  % positive months   : {monthly_stats['pct_positive_months']:.1f}%")
    lines.append(f"  Best month          : {monthly_stats['best_month_r']:+.2f}R")
    lines.append(f"  Worst month         : {monthly_stats['worst_month_r']:+.2f}R")
    lines.append(f"  Avg trades/month    : {monthly_stats['avg_trades_per_month']:.1f}")
    lines.append(f"  Avg costs/month     : {monthly_stats['avg_costs_per_month_r']:.3f}R")

    # Additional stats (if provided)
    if stats:
        lines.append("\n📈 OVERALL:")
        if "net_r" in stats:
            lines.append(f"  Total Net R         : {stats['net_r']:+.2f}R")
        if "daily_sharpe_r" in stats:
            lines.append(f"  Daily Sharpe        : {stats['daily_sharpe_r']:.3f}")
        if "max_dd_r_daily" in stats:
            lines.append(f"  Max DD (daily)      : {stats['max_dd_r_daily']:.2f}R")
        if "recovery_factor" in stats:
            lines.append(f"  Recovery Factor     : {stats['recovery_factor']:.2f}")

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )
