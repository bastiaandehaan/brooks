import pandas as pd
from dataclasses import dataclass

from strategies.h2l2 import Side
from backtest.runner import _simulate_trade_outcome


@dataclass(frozen=True)
class DummyTrade:
    execute_ts: pd.Timestamp
    side: Side
    stop: float
    tp: float


def test_both_hit_same_bar_is_worst_case_loss_long():
    idx = pd.to_datetime(["2026-01-08 15:55:00"], utc=True)
    m5 = pd.DataFrame(
        {
            "high": [105.0],  # TP geraakt
            "low": [95.0],    # SL geraakt
            "open": [100.0],
            "close": [100.0],
        },
        index=idx,
    )

    t = DummyTrade(execute_ts=idx[0], side=Side.LONG, stop=99.0, tp=101.0)
    out = _simulate_trade_outcome(m5, t)
    assert out == -1.0


def test_both_hit_same_bar_is_worst_case_loss_short():
    idx = pd.to_datetime(["2026-01-08 15:55:00"], utc=True)
    m5 = pd.DataFrame(
        {
            "high": [105.0],  # SL geraakt (short)
            "low": [95.0],    # TP geraakt
            "open": [100.0],
            "close": [100.0],
        },
        index=idx,
    )

    t = DummyTrade(execute_ts=idx[0], side=Side.SHORT, stop=101.0, tp=99.0)
    out = _simulate_trade_outcome(m5, t)
    assert out == -1.0
