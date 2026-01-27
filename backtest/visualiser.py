# backtest/visualiser.py
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("BacktestVisualiser")


def _format_dt(ts) -> str:
    if ts is None:
        return "NA"
    try:
        return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def _wrap_text(s: str, width: int = 120) -> str:
    if not s:
        return ""
    words = s.split()
    out = []
    line = []
    count = 0
    for w in words:
        extra = len(w) + (1 if line else 0)
        if count + extra > width:
            out.append(" ".join(line))
            line = [w]
            count = len(w)
        else:
            line.append(w)
            count += extra
    if line:
        out.append(" ".join(line))
    return "\n".join(out)


def _apply_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def generate_performance_report(
    results_r,
    equity_curve,
    drawdown,
    symbol=None,
    days=None,
    run_id=None,
    command=None,
    period_start=None,
    period_end=None,
    stats=None,
    price_series=None,
    daily_pnl=None,
):
    """
    Full-width dashboard:
    1) Equity (R) over time (trade entry timestamps)
    2) Instrument price (M15 close) over test period
    3) Drawdown (R) over time
    4) Rolling winrate (30 trades)
    5) Daily PnL (R/day) + cumulative daily PnL (R)
    Bottom: metrics/command panel

    Saves to: backtest/backtest_png/
    """
    symbol = symbol or "UNKNOWN"
    days = days or "NA"

    plt.style.use("ggplot")

    res = pd.Series(results_r)
    eq = pd.Series(equity_curve)
    dd = pd.Series(drawdown)

    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(
        nrows=6,
        ncols=1,
        height_ratios=[2.2, 1.4, 1.2, 1.2, 1.4, 2.2],
        hspace=0.35,
    )

    ax1 = fig.add_subplot(gs[0, 0])  # Equity
    ax_price = fig.add_subplot(gs[1, 0])  # Price
    ax2 = fig.add_subplot(gs[2, 0])  # Drawdown
    ax3 = fig.add_subplot(gs[3, 0])  # Rolling winrate
    ax5 = fig.add_subplot(gs[4, 0])  # Daily pnl
    ax_info = fig.add_subplot(gs[5, 0])  # Metrics box

    title_run = f"{symbol} ({days} days)"
    if period_start is not None and period_end is not None:
        title_run += f"  [{_format_dt(period_start)} → {_format_dt(period_end)}]"
    ax1.set_title(f"Brooks Backtest Dashboard: {title_run}", fontsize=16, fontweight="bold")

    # 1) Equity curve
    if isinstance(eq.index, pd.DatetimeIndex):
        ax1.plot(eq.index, eq.values, label="Equity Growth (R)", color="#2ca02c", linewidth=2.5)
        ax1.fill_between(eq.index, eq.values, color="#2ca02c", alpha=0.10)
        _apply_time_axis(ax1)
        ax1.set_xlabel("Time")
    else:
        ax1.plot(eq.values, label="Equity Growth (R)", color="#2ca02c", linewidth=2.5)
        ax1.fill_between(range(len(eq)), eq.values, color="#2ca02c", alpha=0.10)
        ax1.set_xlabel("Trades")
    ax1.set_ylabel("Cumulative R")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # 2) Price chart
    ax_price.set_title("Instrument price over test period", fontsize=12, fontweight="bold")
    if price_series is not None:
        ps = pd.Series(price_series)
        if (
            isinstance(ps.index, pd.DatetimeIndex)
            and period_start is not None
            and period_end is not None
        ):
            try:
                ps = ps.loc[pd.to_datetime(period_start) : pd.to_datetime(period_end)]
            except Exception:
                pass

        if isinstance(ps.index, pd.DatetimeIndex):
            ax_price.plot(
                ps.index, ps.values, color="#111111", linewidth=1.2, label=f"{symbol} M15 close"
            )
            _apply_time_axis(ax_price)
            ax_price.set_xlabel("Time")
        else:
            ax_price.plot(ps.values, color="#111111", linewidth=1.2, label=f"{symbol} close")
            ax_price.set_xlabel("Bars")

        ax_price.set_ylabel("Price")
        ax_price.grid(True, alpha=0.25)
        ax_price.legend(loc="upper left")
    else:
        ax_price.text(0.5, 0.5, "No price series provided", ha="center", va="center")
        ax_price.axis("off")

    # 3) Drawdown
    if isinstance(dd.index, pd.DatetimeIndex):
        ax2.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.30)
        ax2.plot(dd.index, dd.values, color="#d62728", linewidth=1, label="Drawdown (R)")
        _apply_time_axis(ax2)
        ax2.set_xlabel("Time")
    else:
        ax2.fill_between(range(len(dd)), dd.values, 0, color="#d62728", alpha=0.30)
        ax2.plot(dd.values, color="#d62728", linewidth=1, label="Drawdown (R)")
        ax2.set_xlabel("Trades")
    ax2.set_ylabel("Drawdown (R)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left")

    # 4) Rolling winrate
    rolling_winrate = (res > 0).rolling(window=30).mean() * 100.0
    if isinstance(res.index, pd.DatetimeIndex):
        ax3.plot(
            res.index,
            rolling_winrate.values,
            color="#1f77b4",
            linewidth=2,
            label="Winrate % (Rolling 30)",
        )
        _apply_time_axis(ax3)
        ax3.set_xlabel("Time")
    else:
        ax3.plot(
            rolling_winrate.values, color="#1f77b4", linewidth=2, label="Winrate % (Rolling 30)"
        )
        ax3.set_xlabel("Trades")
    ax3.axhline(33.3, color="orange", linestyle="--", label="Breakeven (1:2 RR)")
    ax3.set_ylabel("Winrate %")
    ax3.set_ylim(0, 100)
    ax3.legend(loc="upper left")

    # 5) Daily PnL (R/day) + cumulative
    ax5.set_title("Daily PnL (R/day) and cumulative daily PnL", fontsize=12, fontweight="bold")
    if daily_pnl is not None and len(daily_pnl) > 0:
        dp = pd.Series(daily_pnl).astype("float64")
        # dp index is date objects; convert to datetime for plotting
        try:
            x = pd.to_datetime(dp.index.astype(str))
        except Exception:
            x = np.arange(len(dp))

        colors = np.where(dp.values >= 0, "#2ca02c", "#d62728")
        ax5.bar(x, dp.values, color=colors, alpha=0.45, label="Daily PnL (R/day)")
        cum = dp.cumsum()
        ax5.plot(x, cum.values, color="#111111", linewidth=1.4, label="Cumulative daily PnL (R)")
        if isinstance(x, pd.DatetimeIndex):
            _apply_time_axis(ax5)
        ax5.set_xlabel("Date (NY)")
        ax5.set_ylabel("R")
        ax5.legend(loc="upper left")
        ax5.grid(True, alpha=0.25)
    else:
        ax5.text(0.5, 0.5, "No daily PnL series provided", ha="center", va="center")
        ax5.axis("off")

    # Bottom info box
    ax_info.axis("off")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(f"Run time: {now}")
    if run_id:
        lines.append(f"Run ID: {run_id}")
    lines.append(f"Symbol: {symbol}")
    lines.append(f"Days: {days}")
    if period_start is not None and period_end is not None:
        lines.append(f"Period: {_format_dt(period_start)} → {_format_dt(period_end)}")

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if abs(v) >= 1000:
                return f"{v:.1f}"
            return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
        return str(v)

    if stats and isinstance(stats, dict):
        lines.append("")
        lines.append("Key metrics:")

        preferred = [
            ("trades", "Trades"),
            ("net_r", "Net R"),
            ("avg_r", "Avg R/trade"),
            ("winrate_pct", "Winrate %"),
            ("profit_factor", "Profit Factor"),
            ("max_dd_r_trade", "Max DD (trade, R)"),
            ("trade_sharpe", "Trade Sharpe (legacy)"),
            ("recovery_factor", "Recovery Factor"),
            ("mar_ratio", "MAR Ratio"),
            ("annual_r", "Annual R est."),
            ("avg_win_r", "Avg Win (R)"),
            ("avg_loss_r", "Avg Loss (R)"),
            ("payoff_ratio", "Payoff Ratio"),
            ("max_consec_losses", "Max consec losses"),
            ("long_trades", "Long trades"),
            ("pf_long", "PF (Long)"),
            ("net_r_long", "Net R (Long)"),
            ("short_trades", "Short trades"),
            ("pf_short", "PF (Short)"),
            ("net_r_short", "Net R (Short)"),
            # R-based daily manager metrics
            ("daily_sharpe_r", "Daily Sharpe (R/day)"),
            ("daily_sortino_r", "Daily Sortino (R/day)"),
            ("var_95_r", "VaR 95% (R/day)"),
            ("cvar_95_r", "CVaR 95% (R/day)"),
            ("best_day_r", "Best day (R)"),
            ("worst_day_r", "Worst day (R)"),
            ("pct_pos_days", "% positive days"),
            ("max_underwater_days", "Max underwater (days)"),
            ("max_dd_r_daily", "Max DD (daily, R)"),
            ("max_dd_pct_initial", "Max DD (% of initial)"),
            ("skew_r", "Skew (R/day)"),
            ("kurtosis_r", "Kurtosis (R/day)"),
            ("calendar_days", "Calendar days"),
            ("days_with_trades", "Days w/ trades"),
            ("trades_per_active_day", "Trades/day active"),
            ("trades_per_calendar_day", "Trades/day calendar"),
            # regime/costs
            ("regime_filter", "Regime filter"),
            ("choppy_segments_skipped", "Choppy segs skipped"),
            ("costs_per_trade_r", "Costs/trade (R)"),
            ("total_cost_r", "Total costs (R)"),
        ]

        for key, label in preferred:
            if key in stats:
                lines.append(f"{label}: {fmt(stats[key])}")

        # Regime breakdown lines
        for k in sorted([k for k in stats.keys() if k.startswith("net_r_reg_")]):
            reg = k.replace("net_r_reg_", "")
            tkey = f"trades_reg_{reg}"
            lines.append(f"Reg {reg}: trades={stats.get(tkey, 'NA')}, netR={fmt(stats[k])}")

    if command:
        lines.append("")
        lines.append("Command:")
        lines.append(_wrap_text(command, width=120))

    ax_info.text(
        0.01,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.05)

    # Output dir: backtest/backtest_png/
    out_dir = Path(__file__).resolve().parent / "backtest_png"
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_id:
        filename = out_dir / f"backtest_report_{run_id}.png"
    else:
        filename = out_dir / f"backtest_report_{symbol}_{days}d.png"

    plt.savefig(str(filename), dpi=150)
    plt.close()
    logger.info(f"Dashboard '{filename}' opgeslagen.")
    return str(filename)
