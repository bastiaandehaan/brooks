# backtest/analytics/yearly_metrics.py
"""
Yearly Metrics - From Daily Equity (NY Timezone)

Year-by-year breakdown with Max DD calculated from daily equity curve,
not from trade-level drawdowns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_yearly_breakdown(
    daily_equity_r: pd.Series,
    trades_df: pd.DataFrame,
    *,
    ny_tz: str = "America/New_York",
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    Calculate year-by-year metrics from daily equity.

    CRITICAL: Max DD is calculated from DAILY equity (NY timezone),
    not from individual trades. This matches FTMO MetriX behavior.

    Args:
        daily_equity_r: Daily PnL series indexed by NY date (date objects or DatetimeIndex)
        trades_df: Trades with 'exit_time' and 'net_r'
        ny_tz: Timezone for year boundaries
        trading_days_per_year: For Sharpe annualization

    Returns:
        DataFrame indexed by year with columns:
        - net_r: Total return for year
        - max_dd_r: Worst drawdown from daily equity
        - trades: Number of trades
        - sharpe: Daily Sharpe (annualized)
    """
    if daily_equity_r.empty:
        return pd.DataFrame(columns=["net_r", "max_dd_r", "trades", "sharpe"])

    # Ensure daily_equity_r has datetime index
    if not isinstance(daily_equity_r.index, pd.DatetimeIndex):
        # Convert date objects to datetime
        daily_equity_r.index = pd.to_datetime(daily_equity_r.index)

    # Extract year
    daily_equity_r = daily_equity_r.copy()
    years = daily_equity_r.index.year.unique()

    results = []

    for year in sorted(years):
        year_mask = daily_equity_r.index.year == year
        year_daily = daily_equity_r[year_mask]

        if year_daily.empty:
            continue

        # Net R for year
        net_r = float(year_daily.sum())

        # Max DD from daily equity (this is the KEY requirement)
        cumulative = year_daily.cumsum()
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        max_dd_r = float(drawdown.min())

        # Trade count (from trades_df)
        if not trades_df.empty and "exit_time" in trades_df.columns:
            trades_df_copy = trades_df.copy()

            # Ensure exit_time is tz-aware
            if trades_df_copy["exit_time"].dt.tz is None:
                trades_df_copy["exit_time"] = trades_df_copy["exit_time"].dt.tz_localize("UTC")

            # Convert to NY and filter by year
            trades_df_copy["ny_time"] = trades_df_copy["exit_time"].dt.tz_convert(ny_tz)
            year_trades = trades_df_copy[trades_df_copy["ny_time"].dt.year == year]
            n_trades = len(year_trades)
        else:
            n_trades = 0

        # Sharpe (from daily returns)
        mean_daily = float(year_daily.mean())
        std_daily = float(year_daily.std(ddof=1)) if len(year_daily) > 1 else 0.0
        sharpe = (mean_daily / std_daily) * np.sqrt(trading_days_per_year) if std_daily > 0 else 0.0

        results.append(
            {
                "year": year,
                "net_r": net_r,
                "max_dd_r": max_dd_r,
                "trades": n_trades,
                "sharpe": sharpe,
            }
        )

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.set_index("year")

    return df
