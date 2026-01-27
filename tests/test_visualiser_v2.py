import pytest
import pandas as pd
import numpy as np
from strategies.config import StrategyConfig
from backtest.visualiser_v2 import generate_dashboard_v2


def test_dashboard_generation():
    config = StrategyConfig.load("config/strategies/us500_sniper.yaml")

    results_r = pd.Series(np.random.randn(100), index=pd.date_range("2024-01-01", periods=100, tz="UTC"))
    equity_curve = results_r.cumsum()
    drawdown = equity_curve - equity_curve.cummax()

    daily_pnl = pd.Series(np.random.randn(100), index=pd.date_range("2024-01-01", periods=100))

    trades_df = pd.DataFrame({
        "exit_time": pd.date_range("2024-01-01", periods=100, tz="UTC"),
        "net_r": results_r.values,
    })

    output = generate_dashboard_v2(
        results_r=results_r,
        equity_curve=equity_curve,
        drawdown=drawdown,
        daily_pnl_r=daily_pnl,
        trades_df=trades_df,
        config=config,
        symbol="TEST",
        days=100,
        run_id="test_001",
        output_dir="tests/outputs",
    )

    assert output.exists()
    assert output.suffix == ".png"
