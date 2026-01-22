# utils/daily_sharpe_calculator.py
"""
Calculate daily Sharpe ratio from trade logs.
Converts trade-level P&L to daily returns and applies proper annualization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def calculate_daily_sharpe(
    trades_df: pd.DataFrame,
    initial_capital: float = 10000.0,
    trading_days_per_year: int = 252,
) -> dict[str, float]:
    """
    Calculate daily Sharpe ratio from trade history.

    Args:
        trades_df: DataFrame with columns ['exit_time', 'net_r']
        initial_capital: Starting capital in account currency
        trading_days_per_year: Typically 252 for equities

    Returns:
        Dict with 'daily_sharpe', 'annualized_return', 'annualized_vol'
    """
    if trades_df.empty:
        return {
            "daily_sharpe": 0.0,
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "total_trading_days": 0,
        }

    # Ensure exit_time is datetime
    trades_df = trades_df.copy()
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
    trades_df["date"] = trades_df["exit_time"].dt.date

    # Convert R to dollar P&L (assume 1R = 1% of capital per trade)
    risk_per_trade = initial_capital * 0.01  # 1% risk
    trades_df["pnl"] = trades_df["net_r"] * risk_per_trade

    # Aggregate to daily P&L
    daily_pnl = trades_df.groupby("date")["pnl"].sum().reset_index()
    daily_pnl = daily_pnl.sort_values("date")

    # Create complete date range (include non-trading days as 0)
    date_range = pd.date_range(start=daily_pnl["date"].min(), end=daily_pnl["date"].max(), freq="D")

    daily_series = pd.Series(0.0, index=date_range)
    for _, row in daily_pnl.iterrows():
        daily_series[pd.Timestamp(row["date"])] = row["pnl"]

    # Calculate daily returns
    capital_series = initial_capital + daily_series.cumsum()
    daily_returns = daily_series / capital_series.shift(1).fillna(initial_capital)

    # Calculate Sharpe
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()

    if std_daily_return == 0 or np.isnan(std_daily_return):
        daily_sharpe = 0.0
    else:
        daily_sharpe = mean_daily_return / std_daily_return * np.sqrt(trading_days_per_year)

    # Annualized metrics
    annualized_return = mean_daily_return * trading_days_per_year
    annualized_vol = std_daily_return * np.sqrt(trading_days_per_year)

    return {
        "daily_sharpe": round(daily_sharpe, 3),
        "annualized_return": round(annualized_return * 100, 2),  # as %
        "annualized_vol": round(annualized_vol * 100, 2),  # as %
        "total_trading_days": len(date_range),
        "days_with_trades": len(daily_pnl),
        "mean_daily_return": mean_daily_return,
        "std_daily_return": std_daily_return,
    }


def add_daily_sharpe_to_backtest(
    trades_log_path: str | Path, initial_capital: float = 10000.0
) -> None:
    """
    Read trade log and print daily Sharpe calculation.

    Usage:
        add_daily_sharpe_to_backtest('backtest_trades.csv')
    """
    trades_df = pd.read_csv(trades_log_path)

    if "net_r" not in trades_df.columns or "exit_time" not in trades_df.columns:
        print("❌ Trade log must have 'net_r' and 'exit_time' columns")
        return

    metrics = calculate_daily_sharpe(trades_df, initial_capital)

    print("\n" + "=" * 50)
    print("📊 DAILY SHARPE RATIO ANALYSIS")
    print("=" * 50)
    print(f"Daily Sharpe Ratio    : {metrics['daily_sharpe']}")
    print(f"Annualized Return     : {metrics['annualized_return']}%")
    print(f"Annualized Volatility : {metrics['annualized_vol']}%")
    print(f"Total Calendar Days   : {metrics['total_trading_days']}")
    print(f"Days with Trades      : {metrics['days_with_trades']}")
    print("=" * 50 + "\n")

    # Interpretation guide
    if metrics["daily_sharpe"] > 1.5:
        print("✅ Excellent Sharpe (>1.5) - Institutional grade")
    elif metrics["daily_sharpe"] > 1.0:
        print("✅ Good Sharpe (1.0-1.5) - Strong risk-adjusted returns")
    elif metrics["daily_sharpe"] > 0.5:
        print("⚠️  Moderate Sharpe (0.5-1.0) - Acceptable but room for improvement")
    else:
        print("❌ Low Sharpe (<0.5) - Needs optimization")

    return metrics


# Example usage in backtest/runner.py or scripts/live_tracker.py
if __name__ == "__main__":
    # Test with your backtest results
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m utils.daily_sharpe_calculator <trades_csv_path>")
        sys.exit(1)

    add_daily_sharpe_to_backtest(sys.argv[1])
