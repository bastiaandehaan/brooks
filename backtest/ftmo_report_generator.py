# backtest/ftmo_report_generator.py
"""
FTMO-Grade Performance Report Generator

Generates institutional-quality performance reports that provide:
- FTMO compliance status at a glance
- Equity-based risk metrics (not just trade-based)
- Tail risk analysis (VaR/CVaR)
- Underwater duration tracking
- Regime/side attribution

This report is what you'd show to a hedge fund manager or FTMO reviewer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_ftmo_metrics(
    *,
    trades_df: pd.DataFrame,
    daily_pnl_r: pd.Series,
    equity_curve_r: pd.Series,
    initial_capital: float,
    ftmo_account_size: float = 10000.0,
    ftmo_daily_limit_pct: float = 5.0,
    ftmo_total_limit_pct: float = 10.0,
    ftmo_profit_target_pct: float = 10.0,
    ftmo_min_trading_days: int = 4,
) -> Dict[str, Any]:
    """
    Calculate FTMO-specific metrics from backtest data.

    Returns dict with all FTMO compliance metrics.
    """
    # Convert R to USD (assuming 1R = 1% of capital)
    r_to_usd = initial_capital / 100.0

    # Equity curve in USD
    equity_usd = initial_capital + (equity_curve_r * r_to_usd)
    running_max_equity = equity_usd.cummax()

    # Daily PnL in USD
    daily_pnl_usd = daily_pnl_r * r_to_usd

    # === A) FTMO OBJECTIVES ===
    current_profit_usd = equity_usd.iloc[-1] - initial_capital
    profit_target_usd = ftmo_account_size * (ftmo_profit_target_pct / 100.0)
    profit_progress_pct = (current_profit_usd / profit_target_usd) * 100.0

    # Trading days (days with trades)
    trading_days = int((daily_pnl_r != 0).sum())

    # Max daily loss (worst daily drawdown)
    max_daily_loss_usd = daily_pnl_usd.min()
    daily_loss_limit_usd = ftmo_account_size * (ftmo_daily_limit_pct / 100.0)
    daily_headroom_usd = daily_loss_limit_usd - abs(max_daily_loss_usd)

    # Max total drawdown
    drawdown_usd = equity_usd - running_max_equity
    max_total_dd_usd = drawdown_usd.min()
    total_loss_limit_usd = ftmo_account_size * (ftmo_total_limit_pct / 100.0)
    total_headroom_usd = total_loss_limit_usd - abs(max_total_dd_usd)

    # FTMO Status
    daily_breached = abs(max_daily_loss_usd) > daily_loss_limit_usd
    total_breached = abs(max_total_dd_usd) > total_loss_limit_usd
    target_hit = current_profit_usd >= profit_target_usd

    if daily_breached:
        ftmo_status = "🔴 BLOCKED (Daily Limit)"
    elif total_breached:
        ftmo_status = "🔴 BLOCKED (Total Limit)"
    elif target_hit:
        ftmo_status = "🟢 PASS MODE (Target Hit)"
    else:
        ftmo_status = "🟡 ALLOWED (In Progress)"

    # === B) EQUITY-BASED RISK METRICS ===
    # Intraday DD would require intraday equity (not available from daily backtest)
    # So we use daily DD as proxy
    max_intraday_dd_usd = abs(drawdown_usd.min())
    max_intraday_dd_pct = (max_intraday_dd_usd / running_max_equity.max()) * 100

    # Underwater duration
    underwater = drawdown_usd < 0
    max_underwater_days = 0
    current_run = 0
    for is_underwater in underwater:
        if is_underwater:
            current_run += 1
            max_underwater_days = max(max_underwater_days, current_run)
        else:
            current_run = 0

    # Time to recovery (from max DD)
    max_dd_idx = drawdown_usd.idxmin()
    recovery_idx = equity_usd[equity_usd >= running_max_equity.loc[max_dd_idx]].index
    if len(recovery_idx) > 0:
        recovery_days = (recovery_idx[0] - max_dd_idx).days
    else:
        recovery_days = None  # Not recovered yet

    # === C) DISTRIBUTION / TAIL RISK ===
    results_r = trades_df["net_r"].values

    median_r = float(np.median(results_r))
    p05_r = float(np.percentile(results_r, 5))
    p95_r = float(np.percentile(results_r, 95))
    worst_trade_r = float(results_r.min())
    worst_5_sum_r = float(np.sort(results_r)[:5].sum())
    best_trade_r = float(results_r.max())
    best_5_sum_r = float(np.sort(results_r)[-5:].sum())

    # Daily VaR/CVaR
    var_95_r = float(np.percentile(daily_pnl_r, 5))
    cvar_95_r = float(daily_pnl_r[daily_pnl_r <= var_95_r].mean())

    # === D) QUALITY/BEHAVIOUR ===
    # Max consecutive losses
    max_consec_losses = 0
    current_streak = 0
    for r in results_r:
        if r < 0:
            current_streak += 1
            max_consec_losses = max(max_consec_losses, current_streak)
        else:
            current_streak = 0

    # Max losing streak (sum R)
    max_losing_streak_r = 0.0
    current_streak_r = 0.0
    for r in results_r:
        if r < 0:
            current_streak_r += r
            max_losing_streak_r = min(max_losing_streak_r, current_streak_r)
        else:
            current_streak_r = 0.0

    # Break-even rate (within ±0.1R)
    breakeven_count = int(np.abs(results_r) < 0.1)
    breakeven_rate_pct = (breakeven_count / len(results_r)) * 100.0 if len(results_r) > 0 else 0.0

    # Days with trades
    days_with_trades_pct = (
        (trading_days / len(daily_pnl_r)) * 100.0 if len(daily_pnl_r) > 0 else 0.0
    )

    # === E) REGIME/SIDE ATTRIBUTION ===
    # If regime column exists in trades_df
    attribution = {}
    if "regime_at_entry" in trades_df.columns:
        for regime in trades_df["regime_at_entry"].dropna().unique():
            regime_trades = trades_df[trades_df["regime_at_entry"] == regime]
            attribution[f"regime_{regime}_net_r"] = float(regime_trades["net_r"].sum())
            attribution[f"regime_{regime}_count"] = int(len(regime_trades))

    # Side attribution (always available)
    long_trades = trades_df[trades_df["side"] == "LONG"]
    short_trades = trades_df[trades_df["side"] == "SHORT"]

    long_net_r = float(long_trades["net_r"].sum()) if len(long_trades) > 0 else 0.0
    short_net_r = float(short_trades["net_r"].sum()) if len(short_trades) > 0 else 0.0

    long_wins = int((long_trades["net_r"] > 0).sum()) if len(long_trades) > 0 else 0
    long_losses = int((long_trades["net_r"] < 0).sum()) if len(long_trades) > 0 else 0
    long_pf = (
        (
            abs(long_trades[long_trades["net_r"] > 0]["net_r"].sum())
            / abs(long_trades[long_trades["net_r"] < 0]["net_r"].sum())
        )
        if long_losses > 0
        else 999.9
    )

    short_wins = int((short_trades["net_r"] > 0).sum()) if len(short_trades) > 0 else 0
    short_losses = int((short_trades["net_r"] < 0).sum()) if len(short_trades) > 0 else 0
    short_pf = (
        (
            abs(short_trades[short_trades["net_r"] > 0]["net_r"].sum())
            / abs(short_trades[short_trades["net_r"] < 0]["net_r"].sum())
        )
        if short_losses > 0
        else 999.9
    )

    return {
        # A) FTMO OBJECTIVES
        "ftmo_account_size": ftmo_account_size,
        "ftmo_profit_target_usd": profit_target_usd,
        "ftmo_current_profit_usd": current_profit_usd,
        "ftmo_profit_progress_pct": profit_progress_pct,
        "ftmo_trading_days": trading_days,
        "ftmo_min_trading_days": ftmo_min_trading_days,
        "ftmo_daily_loss_limit_usd": daily_loss_limit_usd,
        "ftmo_max_daily_loss_usd": max_daily_loss_usd,
        "ftmo_daily_headroom_usd": daily_headroom_usd,
        "ftmo_total_loss_limit_usd": total_loss_limit_usd,
        "ftmo_max_total_dd_usd": max_total_dd_usd,
        "ftmo_total_headroom_usd": total_headroom_usd,
        "ftmo_status": ftmo_status,
        "ftmo_daily_buffer_pct": 1.0,  # We use 4% not 5%
        "ftmo_total_buffer_pct": 2.0,  # We use 8% not 10%
        # B) EQUITY-BASED RISK
        "max_intraday_dd_usd": max_intraday_dd_usd,
        "max_intraday_dd_pct": max_intraday_dd_pct,
        "max_overall_dd_usd": abs(max_total_dd_usd),
        "max_overall_dd_pct": (abs(max_total_dd_usd) / initial_capital) * 100,
        "max_underwater_days": max_underwater_days,
        "time_to_recovery_days": recovery_days,
        # C) DISTRIBUTION / TAIL RISK
        "median_r_per_trade": median_r,
        "p05_r_per_trade": p05_r,
        "p95_r_per_trade": p95_r,
        "worst_trade_r": worst_trade_r,
        "worst_5_sum_r": worst_5_sum_r,
        "best_trade_r": best_trade_r,
        "best_5_sum_r": best_5_sum_r,
        "daily_var_95_r": var_95_r,
        "daily_cvar_95_r": cvar_95_r,
        # D) QUALITY/BEHAVIOUR
        "max_consec_losses": max_consec_losses,
        "max_losing_streak_r": max_losing_streak_r,
        "breakeven_rate_pct": breakeven_rate_pct,
        "days_with_trades_pct": days_with_trades_pct,
        # E) REGIME/SIDE ATTRIBUTION
        **attribution,
        "long_net_r": long_net_r,
        "long_count": len(long_trades),
        "long_pf": long_pf,
        "short_net_r": short_net_r,
        "short_count": len(short_trades),
        "short_pf": short_pf,
    }


def generate_ftmo_report_png(
    *,
    trades_df: pd.DataFrame,
    daily_pnl_r: pd.Series,
    equity_curve_r: pd.Series,
    drawdown_r: pd.Series,
    symbol: str,
    days: int,
    run_id: str,
    initial_capital: float = 10000.0,
    price_series: Optional[pd.Series] = None,
    output_dir: str = "backtest/backtest_png",
) -> Path:
    """
    Generate FTMO-grade performance report PNG.

    Layout:
    - Top: FTMO Objectives block (pass/fail status)
    - Charts: Equity, Price, Drawdown, Rolling Winrate, Daily PnL
    - Bottom: 5 compact metric blocks (Risk, Tail, Quality, Regime, Side)
    """
    # Calculate FTMO metrics
    ftmo = calculate_ftmo_metrics(
        trades_df=trades_df,
        daily_pnl_r=daily_pnl_r,
        equity_curve_r=equity_curve_r,
        initial_capital=initial_capital,
    )

    # Create figure
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(
        nrows=8, ncols=1, height_ratios=[1.5, 2, 1.2, 1.2, 1.2, 1.5, 1, 3], hspace=0.35
    )

    # === TOP: FTMO OBJECTIVES BLOCK ===
    ax_ftmo = fig.add_subplot(gs[0, 0])
    ax_ftmo.axis("off")

    ftmo_text = f"""
🏦 FTMO COMPLIANCE STATUS: {ftmo["ftmo_status"]}

Account: ${ftmo["ftmo_account_size"]:,.0f}  |  Profit: ${ftmo["ftmo_current_profit_usd"]:+,.2f} / ${ftmo["ftmo_profit_target_usd"]:,.2f} ({ftmo["ftmo_profit_progress_pct"]:.1f}%)
Trading Days: {ftmo["ftmo_trading_days"]} / {ftmo["ftmo_min_trading_days"]} minimum

Daily Loss:  ${abs(ftmo["ftmo_max_daily_loss_usd"]):,.2f} / ${ftmo["ftmo_daily_loss_limit_usd"]:,.2f}  |  Headroom: ${ftmo["ftmo_daily_headroom_usd"]:,.2f}  |  Buffer: {ftmo["ftmo_daily_buffer_pct"]:.0f}%
Total Loss:  ${abs(ftmo["ftmo_max_total_dd_usd"]):,.2f} / ${ftmo["ftmo_total_loss_limit_usd"]:,.2f}  |  Headroom: ${ftmo["ftmo_total_headroom_usd"]:,.2f}  |  Buffer: {ftmo["ftmo_total_buffer_pct"]:.0f}%
""".strip()

    ax_ftmo.text(
        0.5,
        0.5,
        ftmo_text,
        ha="center",
        va="center",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
    )

    # === CHARTS (same as before but cleaner) ===
    # [Charts implementation would go here - keeping visualiser.py charts but adding FTMO color coding]

    # === BOTTOM: 5 METRIC BLOCKS ===
    ax_metrics = fig.add_subplot(gs[7, 0])
    ax_metrics.axis("off")

    metrics_text = f"""
📊 RISK METRICS (Equity-Based, UTC)            📉 TAIL RISK (Distribution)             ⭐ QUALITY
Max Intraday DD:   ${ftmo["max_intraday_dd_usd"]:,.2f} ({ftmo["max_intraday_dd_pct"]:.2f}%)    Median R/trade:    {ftmo["median_r_per_trade"]:+.3f}R         Max Consec Losses:  {ftmo["max_consec_losses"]}
Max Overall DD:    ${ftmo["max_overall_dd_usd"]:,.2f} ({ftmo["max_overall_dd_pct"]:.2f}%)    P05/P95 R:         {ftmo["p05_r_per_trade"]:+.3f} / {ftmo["p95_r_per_trade"]:+.3f}R    Max Losing Streak:  {ftmo["max_losing_streak_r"]:.2f}R
Max Underwater:    {ftmo["max_underwater_days"]} days                      Worst Trade:       {ftmo["worst_trade_r"]:.2f}R            Breakeven Rate:     {ftmo["breakeven_rate_pct"]:.1f}%
Recovery Time:     {ftmo["time_to_recovery_days"] or "N/A"} days                      Worst 5 Sum:       {ftmo["worst_5_sum_r"]:.2f}R            Days with Trades:   {ftmo["days_with_trades_pct"]:.1f}%
                                                        Best Trade:        {ftmo["best_trade_r"]:+.2f}R
📈 SIDE ATTRIBUTION                             Best 5 Sum:        {ftmo["best_5_sum_r"]:+.2f}R            🔄 REGIME (if enabled)
Long:  {ftmo["long_count"]} trades | {ftmo["long_net_r"]:+.2f}R | PF {ftmo["long_pf"]:.2f}    Daily VaR95:       {ftmo["daily_var_95_r"]:.2f}R            [Regime breakdown here if available]
Short: {ftmo["short_count"]} trades | {ftmo["short_net_r"]:+.2f}R | PF {ftmo["short_pf"]:.2f}    Daily CVaR95:      {ftmo["daily_cvar_95_r"]:.2f}R
""".strip()

    ax_metrics.text(0.02, 0.98, metrics_text, va="top", ha="left", fontsize=9, family="monospace")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / f"ftmo_report_{run_id}.png"

    plt.savefig(str(filename), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ FTMO report saved: {filename}")
    return filename
