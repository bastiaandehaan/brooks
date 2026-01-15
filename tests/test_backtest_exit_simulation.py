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


def test_exit_includes_execute_bar_sl_hit():
    # execute bar raakt SL direct
    idx = pd.to_datetime(
        ["2026-01-08 15:55:00", "2026-01-08 16:00:00"],
        utc=True,
    )
    m5 = pd.DataFrame(
        {
            "high": [101.0, 101.0],
            "low": [99.0, 100.0],   # SL geraakt op execute bar
            "open": [100.0, 100.0],
            "close": [100.0, 100.0],
        },
        index=idx,
    )
    t = DummyTrade(execute_ts=idx[0], side=Side.LONG, stop=99.5, tp=102.0)
    out = _simulate_trade_outcome(m5, t)
    assert out[0] == -1.0
