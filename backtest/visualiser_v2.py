# backtest/visualiser_v2.py
"""
Dashboard V2 - Clean, Audit-Friendly Backtest Visualization

FIXED ISSUES:
1. Correct import: from strategies.config import StrategyConfig
2. No emoji characters (prevents glyph warnings)
3. Clean layout hierarchy: Performance → Edge → Behaviour → Summary
4. All metrics preserved
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# FIXED: Correct import path
from strategies.config import StrategyConfig

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
    period_start: Optional[pd.Timestamp],
    period_end: Optional[pd.Timestamp],
    stats: Optional[dict],
    price_series: Optional[pd.Series] = None,
    output_dir: str = "backtest/backtest_png",
    **_kwargs,
) -> Path:
    """
    Dashboard V2 (PNG) - Clean Layout

    Layout Hierarchy:
    - Row 1: Equity Growth (main performance)
    - Row 2: Price Chart (instrument context)
    - Row 3: Drawdown (risk)
    - Row 4: Rolling Expectancy (edge quality)
    - Row 5: Rolling Winrate (consistency)
    - Row 6: Trades per Month (frequency)
    - Row 7: Daily PnL Distribution (daily behaviour)
    - Row 8: Monthly Heatmap (NY timezone)
    - Row 9: Regime Performance (if enabled)
    - Row 10: Yearly Summary Table
    - Row 11: Metrics Summary + Config
    """

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup figure
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(20, 28))

    # Grid: 11 rows, varying heights
    gs = fig.add_gridspec(
        nrows=11,
        ncols=1,
        height_ratios=[2.5, 1.5, 1.2, 1.2, 1.2, 1.2, 1.5, 2.0, 1.5, 1.5, 2.5],
        hspace=0.40,
    )

    # Title
    title_text = f"Brooks Backtest Dashboard V2: {symbol} ({days} days)"
    if period_start and period_end:
        title_text += f"\n{period_start.date()} to {period_end.date()}"

    fig.suptitle(title_text, fontsize=16, fontweight="bold")

    # === ROW 1: EQUITY CURVE ===
    ax_equity = fig.add_subplot(gs[0, 0])
    ax_equity.set_title(
        "Performance: Equity Growth (R-units)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )

    if isinstance(equity_curve.index, pd.DatetimeIndex):
        ax_equity.plot(
            equity_curve.index,
            equity_curve.values,
            label="Cumulative R",
            color="#2ca02c",
            linewidth=2.5,
        )
        ax_equity.fill_between(equity_curve.index, equity_curve.values, color="#2ca02c", alpha=0.15)
        _apply_time_axis(ax_equity)
        ax_equity.set_xlabel("Date")
    else:
        ax_equity.plot(equity_curve.values, label="Cumulative R", color="#2ca02c", linewidth=2.5)
        ax_equity.set_xlabel("Trades")

    ax_equity.set_ylabel("Cumulative R")
    ax_equity.grid(True, alpha=0.3)
    ax_equity.legend(loc="upper left")

    # === ROW 2: PRICE CHART ===
    ax_price = fig.add_subplot(gs[1, 0])
    ax_price.set_title("Instrument Price (M15 close)", fontsize=11, loc="left")

    if price_series is not None and not price_series.empty:
        ps = pd.Series(price_series)
        if period_start and period_end:
            try:
                ps = ps.loc[period_start:period_end]
            except:
                pass

        if isinstance(ps.index, pd.DatetimeIndex):
            ax_price.plot(
                ps.index,
                ps.values,
                color="#111111",
                linewidth=1.2,
                label=f"{symbol} M15",
            )
            _apply_time_axis(ax_price)
            ax_price.set_xlabel("Date")
        else:
            ax_price.plot(ps.values, color="#111111", linewidth=1.2)
            ax_price.set_xlabel("Bars")

        ax_price.set_ylabel("Price")
        ax_price.grid(True, alpha=0.25)
        ax_price.legend(loc="upper left")
    else:
        ax_price.text(0.5, 0.5, "No price data", ha="center", va="center")
        ax_price.axis("off")

    # === ROW 3: DRAWDOWN ===
    ax_dd = fig.add_subplot(gs[2, 0])
    ax_dd.set_title("Risk: Drawdown (R-units)", fontsize=11, loc="left")

    if isinstance(drawdown.index, pd.DatetimeIndex):
        ax_dd.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.30)
        ax_dd.plot(drawdown.index, drawdown.values, color="#d62728", linewidth=1.5)
        _apply_time_axis(ax_dd)
        ax_dd.set_xlabel("Date")
    else:
        ax_dd.fill_between(range(len(drawdown)), drawdown.values, 0, color="#d62728", alpha=0.30)
        ax_dd.plot(drawdown.values, color="#d62728", linewidth=1.5)
        ax_dd.set_xlabel("Trades")

    ax_dd.set_ylabel("Drawdown (R)")
    ax_dd.grid(True, alpha=0.3)

    # === ROW 4: ROLLING EXPECTANCY ===
    ax_exp = fig.add_subplot(gs[3, 0])
    ax_exp.set_title("Edge: Rolling Expectancy (30-trade window)", fontsize=11, loc="left")

    rolling_expectancy = results_r.rolling(window=30).mean()

    if isinstance(results_r.index, pd.DatetimeIndex):
        ax_exp.plot(
            results_r.index,
            rolling_expectancy.values,
            color="#1f77b4",
            linewidth=2,
            label="Rolling Avg R",
        )
        _apply_time_axis(ax_exp)
        ax_exp.set_xlabel("Date")
    else:
        ax_exp.plot(
            rolling_expectancy.values,
            color="#1f77b4",
            linewidth=2,
            label="Rolling Avg R",
        )
        ax_exp.set_xlabel("Trades")

    ax_exp.axhline(0, color="red", linestyle="--", alpha=0.5, label="Breakeven")
    ax_exp.set_ylabel("Avg R/trade")
    ax_exp.legend(loc="upper left")
    ax_exp.grid(True, alpha=0.3)

    # === ROW 5: ROLLING WINRATE ===
    ax_wr = fig.add_subplot(gs[4, 0])
    ax_wr.set_title("Consistency: Rolling Winrate (30-trade window)", fontsize=11, loc="left")

    rolling_winrate = (results_r > 0).rolling(window=30).mean() * 100.0

    if isinstance(results_r.index, pd.DatetimeIndex):
        ax_wr.plot(
            results_r.index,
            rolling_winrate.values,
            color="#2ca02c",
            linewidth=2,
            label="Winrate %",
        )
        _apply_time_axis(ax_wr)
        ax_wr.set_xlabel("Date")
    else:
        ax_wr.plot(rolling_winrate.values, color="#2ca02c", linewidth=2, label="Winrate %")
        ax_wr.set_xlabel("Trades")

    ax_wr.axhline(33.3, color="orange", linestyle="--", label="Breakeven (1:2 RR)", alpha=0.7)
    ax_wr.set_ylabel("Winrate %")
    ax_wr.set_ylim(0, 100)
    ax_wr.legend(loc="upper left")
    ax_wr.grid(True, alpha=0.3)

    # === ROW 6: TRADES PER MONTH ===
    ax_tpm = fig.add_subplot(gs[5, 0])
    ax_tpm.set_title("Frequency: Trades per Month (NY timezone)", fontsize=11, loc="left")

    if not trades_df.empty and "ny_day" in trades_df.columns:
        trades_df_copy = trades_df.copy()
        trades_df_copy["ny_month"] = pd.to_datetime(trades_df_copy["ny_day"]).dt.to_period("M")
        monthly_counts = trades_df_copy.groupby("ny_month").size()

        x = [str(m) for m in monthly_counts.index]
        ax_tpm.bar(x, monthly_counts.values, color="#1f77b4", alpha=0.6)
        ax_tpm.set_ylabel("Trades")
        ax_tpm.tick_params(axis="x", rotation=45)
        ax_tpm.grid(True, alpha=0.3, axis="y")
    else:
        ax_tpm.text(0.5, 0.5, "No monthly data", ha="center", va="center")
        ax_tpm.axis("off")

    # === ROW 7: DAILY PNL DISTRIBUTION ===
    ax_dpnl = fig.add_subplot(gs[6, 0])
    ax_dpnl.set_title("Behaviour: Daily PnL (R/day)", fontsize=11, loc="left")

    if not daily_pnl_r.empty:
        dp = pd.Series(daily_pnl_r).fillna(0.0)
        try:
            x = pd.to_datetime(dp.index.astype(str))
        except:
            x = np.arange(len(dp))

        colors = np.where(dp.values >= 0, "#2ca02c", "#d62728")
        ax_dpnl.bar(x, dp.values, color=colors, alpha=0.5, label="Daily PnL")

        cum = dp.cumsum()
        ax_dpnl.plot(x, cum.values, color="#111111", linewidth=1.5, label="Cumulative Daily PnL")

        if isinstance(x, pd.DatetimeIndex):
            _apply_time_axis(ax_dpnl)

        ax_dpnl.set_xlabel("Date (NY)")
        ax_dpnl.set_ylabel("R")
        ax_dpnl.legend(loc="upper left")
        ax_dpnl.grid(True, alpha=0.25)
    else:
        ax_dpnl.text(0.5, 0.5, "No daily PnL data", ha="center", va="center")
        ax_dpnl.axis("off")

    # === ROW 8: MONTHLY HEATMAP ===
    ax_heatmap = fig.add_subplot(gs[7, 0])
    ax_heatmap.set_title("Monthly Performance Heatmap (NY timezone)", fontsize=11, loc="left")

    if not trades_df.empty and "ny_day" in trades_df.columns:
        # Calculate monthly returns
        trades_df_copy = trades_df.copy()
        trades_df_copy["ny_date"] = pd.to_datetime(trades_df_copy["ny_day"])
        trades_df_copy["year"] = trades_df_copy["ny_date"].dt.year
        trades_df_copy["month"] = trades_df_copy["ny_date"].dt.month

        monthly = trades_df_copy.groupby(["year", "month"])["net_r"].sum().reset_index()

        if not monthly.empty:
            pivot = monthly.pivot(index="year", columns="month", values="net_r")

            # Plot heatmap
            im = ax_heatmap.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-20, vmax=20)

            ax_heatmap.set_xticks(range(12))
            ax_heatmap.set_xticklabels(
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
            ax_heatmap.set_yticks(range(len(pivot.index)))
            ax_heatmap.set_yticklabels(pivot.index)

            # Add values
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.iloc[i, j]
                    if not pd.isna(val):
                        text_color = "white" if abs(val) > 10 else "black"
                        ax_heatmap.text(
                            j,
                            i,
                            f"{val:.1f}R",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=9,
                        )

            plt.colorbar(im, ax=ax_heatmap, label="Net R")
        else:
            ax_heatmap.text(0.5, 0.5, "Insufficient monthly data", ha="center", va="center")
            ax_heatmap.axis("off")
    else:
        ax_heatmap.text(0.5, 0.5, "No monthly data", ha="center", va="center")
        ax_heatmap.axis("off")

    # === ROW 9: REGIME PERFORMANCE ===
    ax_regime = fig.add_subplot(gs[8, 0])
    ax_regime.set_title("Regime Performance", fontsize=11, loc="left")

    if config.regime_filter and not trades_df.empty and "regime_at_entry" in trades_df.columns:
        regime_stats = (
            trades_df.groupby("regime_at_entry").agg({"net_r": ["sum", "mean", "count"]}).round(2)
        )

        regimes = regime_stats.index.tolist()
        net_r = regime_stats[("net_r", "sum")].values

        colors = ["#2ca02c" if x > 0 else "#d62728" for x in net_r]
        ax_regime.bar(regimes, net_r, color=colors, alpha=0.6)
        ax_regime.set_ylabel("Net R")
        ax_regime.grid(True, alpha=0.3, axis="y")

        # Add annotations
        for i, (regime, r) in enumerate(zip(regimes, net_r)):
            count = regime_stats[("net_r", "count")].iloc[i]
            avg = regime_stats[("net_r", "mean")].iloc[i]
            ax_regime.text(
                i,
                r,
                f"{count} trades\nAvg: {avg:.2f}R",
                ha="center",
                va="bottom" if r > 0 else "top",
                fontsize=9,
            )
    else:
        ax_regime.text(0.5, 0.5, "Regime filter disabled", ha="center", va="center")
        ax_regime.axis("off")

    # === ROW 10: YEARLY SUMMARY TABLE ===
    ax_yearly = fig.add_subplot(gs[9, 0])
    ax_yearly.set_title("Yearly Summary", fontsize=11, loc="left")
    ax_yearly.axis("off")

    if not trades_df.empty and "ny_day" in trades_df.columns:
        trades_df_copy = trades_df.copy()
        trades_df_copy["year"] = pd.to_datetime(trades_df_copy["ny_day"]).dt.year

        yearly = (
            trades_df_copy.groupby("year")
            .agg({"net_r": ["sum", "mean"], "exit_time": "count"})
            .round(2)
        )

        yearly.columns = ["Net R", "Avg R", "Trades"]

        # Create table
        table_data = []
        for year, row in yearly.iterrows():
            table_data.append(
                [
                    str(year),
                    f"{row['Net R']:+.2f}R",
                    f"{row['Avg R']:+.4f}R",
                    f"{int(row['Trades'])}",
                ]
            )

        table = ax_yearly.table(
            cellText=table_data,
            colLabels=["Year", "Net R", "Avg R/trade", "Trades"],
            cellLoc="center",
            loc="center",
            bbox=[0.2, 0.2, 0.6, 0.6],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
    else:
        ax_yearly.text(0.5, 0.5, "No yearly data", ha="center", va="center")

    # === ROW 11: METRICS SUMMARY + CONFIG ===
    ax_summary = fig.add_subplot(gs[10, 0])
    ax_summary.axis("off")

    summary_lines = []
    summary_lines.append(f"RUN INFO")
    summary_lines.append(f"  Run ID: {run_id}")
    summary_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")

    if stats:
        summary_lines.append("KEY METRICS")
        summary_lines.append(f"  Trades: {stats.get('trades', 0)}")
        summary_lines.append(f"  Net R: {stats.get('net_r', 0):+.2f}R")
        summary_lines.append(f"  Avg R/trade: {stats.get('avg_r', 0):+.4f}R")
        summary_lines.append(f"  Winrate: {stats.get('winrate', 0) * 100:.1f}%")
        summary_lines.append(f"  Profit Factor: {stats.get('profit_factor', 0):.2f}")
        summary_lines.append(f"  Daily Sharpe (R/day): {stats.get('daily_sharpe_r', 0):.3f}")
        summary_lines.append(f"  Max DD (daily, R): {stats.get('max_dd_r_daily', 0):.2f}R")
        summary_lines.append("")

    summary_lines.append("STRATEGY CONFIG")
    summary_lines.append(f"  Symbol: {config.symbol}")
    summary_lines.append(f"  Regime Filter: {'ON' if config.regime_filter else 'OFF'}")
    if config.regime_filter:
        summary_lines.append(f"  Chop Threshold: {config.regime_params.chop_threshold}")
    summary_lines.append(f"  Risk/Trade: {config.risk_pct}%")
    summary_lines.append(f"  Max Trades/Day: {config.guardrails.max_trades_per_day}")
    summary_lines.append(f"  Costs: {config.costs_per_trade_r:.4f}R")

    text = "\n".join(summary_lines)
    ax_summary.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.2),
    )

    # Save
    filename = output_path / f"dashboard_v2_{run_id}.png"
    plt.savefig(str(filename), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Dashboard V2 saved: {filename}")
    return filename


def _apply_time_axis(ax):
    """Apply clean time axis formatting"""
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
