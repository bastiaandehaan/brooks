import numpy as np
import pandas as pd
import pytest
from backtest.analytics.yearly_metrics import calculate_yearly_breakdown


def test_yearly_max_dd_from_daily():
    """Verify Max DD calculated from daily equity, not trades"""
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")

    # Build daily pnl with one big drawdown day in 2025
    values = np.full(len(dates), 0.1)
    values[400] = -0.5  # Force a big DD in year 2

    daily_pnl = pd.Series(values, index=dates)

    trades = pd.DataFrame(
        {
            "exit_time": pd.date_range("2024-01-01", periods=100, freq="W", tz="UTC"),
            "net_r": np.random.randn(100),
        }
    )

    yearly = calculate_yearly_breakdown(daily_pnl, trades)

    assert 2024 in yearly.index
    assert 2025 in yearly.index

    # 2025 should have larger DD
    assert yearly.loc[2025, "max_dd_r"] < yearly.loc[2024, "max_dd_r"]
