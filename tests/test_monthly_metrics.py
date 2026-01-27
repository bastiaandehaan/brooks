import numpy as np
import pandas as pd
import pytest
from backtest.analytics.monthly_metrics import (
    calculate_monthly_r,
    calculate_monthly_stats,
    verify_monthly_sanity,
)


def test_monthly_aggregation_synthetic():
    """Test with known synthetic data across 3 months"""
    trades = pd.DataFrame(
        {
            "exit_time": pd.date_range("2024-01-15", periods=15, freq="7D", tz="UTC"),
            "net_r": [
                1.0,
                -0.5,
                2.0,
                -1.0,
                1.5,
                0.5,
                -0.5,
                1.0,
                2.0,
                -1.0,
                0.5,
                1.0,
                -0.5,
                1.5,
                -1.0,
            ],
        }
    )

    monthly = calculate_monthly_r(trades, costs_per_trade_r=0.04)

    # Should have Jan, Feb, Mar
    assert len(monthly) == 4

    is_valid, msg = verify_monthly_sanity(trades, monthly)
    assert is_valid, msg


def test_monthly_stats():
    monthly = pd.DataFrame(
        {
            "net_r": [5.0, -2.0, 3.0, 1.0, -1.0],
            "trades": [10, 8, 12, 9, 7],
            "costs_r": [0.4, 0.32, 0.48, 0.36, 0.28],
        }
    )

    stats = calculate_monthly_stats(monthly)

    assert stats["median_monthly_r"] == 1.0
    assert stats["pct_positive_months"] == 60.0
    assert stats["best_month_r"] == 5.0
    assert stats["worst_month_r"] == -2.0
