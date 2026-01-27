# backtest/analytics/monthly_metrics.py
"""
Monthly Metrics Calculation - NY Timezone Aware

Aggregates trades by NY calendar month for consistent reporting.
"""

from __future__ import annotations

import pandas as pd


def calculate_monthly_r(
    trades_df: pd.DataFrame,
    *,
    ny_tz: str = "America/New_York",
    costs_per_trade_r: float = 0.04,
) -> pd.DataFrame:
    """
    Aggregate trades by NY calendar month.

    Args:
        trades_df: Must have 'exit_time' (UTC aware) and 'net_r' columns
        ny_tz: Timezone for month boundaries
        costs_per_trade_r: Cost per trade in R units

    Returns:
        DataFrame indexed by (year, month) with columns:
        - net_r: Sum of net_r for that month
        - trades: Count of trades
        - costs_r: trades * costs_per_trade_r

    Raises:
        ValueError: If required columns missing or exit_time not tz-aware
    """
    if trades_df.empty:
        return pd.DataFrame(columns=["net_r", "trades", "costs_r"])

    required = {"exit_time", "net_r"}
    missing = required - set(trades_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = trades_df.copy()

    # Ensure exit_time is tz-aware
    if df["exit_time"].dt.tz is None:
        raise ValueError("exit_time must be tz-aware (UTC recommended)")

    # Convert to NY timezone and extract month
    df["ny_time"] = df["exit_time"].dt.tz_convert(ny_tz)
    df["year"] = df["ny_time"].dt.year
    df["month"] = df["ny_time"].dt.month

    # Group by (year, month)
    monthly = (
        df.groupby(["year", "month"])
        .agg(
            {
                "net_r": "sum",
                "exit_time": "count",  # trade count
            }
        )
        .rename(columns={"exit_time": "trades"})
    )

    # Calculate costs
    monthly["costs_r"] = monthly["trades"] * costs_per_trade_r

    return monthly


def calculate_monthly_stats(monthly_df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics from monthly aggregation.

    Args:
        monthly_df: Output from calculate_monthly_r()

    Returns:
        Dict with:
        - median_monthly_r: Median monthly return
        - p25_monthly_r: 25th percentile (downside)
        - pct_positive_months: % of months with net_r > 0
        - best_month_r: Best monthly return
        - worst_month_r: Worst monthly return
        - avg_trades_per_month: Average trades per month
        - avg_costs_per_month_r: Average costs per month
    """
    if monthly_df.empty:
        return {
            "median_monthly_r": 0.0,
            "p25_monthly_r": 0.0,
            "pct_positive_months": 0.0,
            "best_month_r": 0.0,
            "worst_month_r": 0.0,
            "avg_trades_per_month": 0.0,
            "avg_costs_per_month_r": 0.0,
        }

    net_r_series = monthly_df["net_r"]

    return {
        "median_monthly_r": float(net_r_series.median()),
        "p25_monthly_r": float(net_r_series.quantile(0.25)),
        "pct_positive_months": float((net_r_series > 0).mean() * 100),
        "best_month_r": float(net_r_series.max()),
        "worst_month_r": float(net_r_series.min()),
        "avg_trades_per_month": float(monthly_df["trades"].mean()),
        "avg_costs_per_month_r": float(monthly_df["costs_r"].mean()),
    }


def verify_monthly_sanity(
    trades_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    *,
    tolerance_r: float = 0.01,
) -> tuple[bool, str]:
    """
    Sanity check: monthly sum should equal trade sum.

    Args:
        trades_df: Original trades
        monthly_df: Monthly aggregation
        tolerance_r: Allowed difference in R units

    Returns:
        (is_valid, message) tuple
    """
    if trades_df.empty or monthly_df.empty:
        return True, "No data to verify"

    trade_sum = float(trades_df["net_r"].sum())
    monthly_sum = float(monthly_df["net_r"].sum())

    diff = abs(trade_sum - monthly_sum)

    if diff > tolerance_r:
        return (
            False,
            f"Monthly sum mismatch: trade_sum={trade_sum:.4f}R, monthly_sum={monthly_sum:.4f}R, diff={diff:.4f}R",
        )

    return True, f"Sanity check OK: diff={diff:.6f}R (within tolerance)"
