# tests/test_h2l2_next_open.py
import pandas as pd

from strategies.h2l2 import H2L2Params, Side, plan_next_open_trade
from utils.symbol_spec import SymbolSpec


def _spec() -> SymbolSpec:
    return SymbolSpec(
        name="US500.cash",
        digits=2,
        point=0.01,
        tick_size=0.01,
        tick_value=0.01,
        contract_size=1.0,
        volume_min=0.01,
        volume_step=0.01,
        volume_max=100.0,
    )


def test_next_open_with_current_bar_executes_on_last_index():
    # 6 bars, waarbij de laatste de "current forming bar" kan zijn.
    idx = pd.to_datetime(
        [
            "2026-01-08 15:30:00",
            "2026-01-08 15:35:00",
            "2026-01-08 15:40:00",
            "2026-01-08 15:45:00",
            "2026-01-08 15:50:00",  # signal bar (closed)
            "2026-01-08 15:55:00",  # execute bar open
        ],
        utc=True,
    )

    m5 = pd.DataFrame(
        {
            "open": [100, 100.1, 100.2, 100.3, 100.4, 100.5],
            "high": [100.2, 100.3, 100.4, 100.5, 100.9, 101.0],
            "low": [99.9, 100.0, 100.1, 100.2, 100.3, 100.2],
            "close": [100.1, 100.2, 100.3, 100.4, 100.8, 100.7],
        },
        index=idx,
    )

    t = plan_next_open_trade(
        m5,
        trend=Side.LONG,
        spec=_spec(),
        p=H2L2Params(pullback_bars=2, min_risk_price_units=0.1, stop_buffer=0.0),
        timeframe_minutes=5,
    )

    assert t is not None
    assert t.execute_ts == m5.index[-1]
    assert t.signal_ts == m5.index[-2]
    assert t.side == Side.LONG


def test_next_open_closed_bars_only_executes_on_synthetic_next_bar():
    """
    All bars closed. Need 6 bars so bar 4 can be evaluated
    (bar 5 exists for next_bar in plan_h2l2_trades loop).
    """
    idx = pd.to_datetime(
        [
            "2026-01-08 15:30:00",
            "2026-01-08 15:35:00",
            "2026-01-08 15:40:00",
            "2026-01-08 15:45:00",
            "2026-01-08 15:50:00",  # signal bar (closed)
            "2026-01-08 15:55:00",  # next bar (needed for loop evaluation!)
        ],
        utc=True,
    )

    m5 = pd.DataFrame(
        {
            "open": [100, 100.1, 100.2, 100.3, 100.4, 100.45],
            "high": [100.2, 100.3, 100.4, 100.5, 100.9, 100.95],
            "low": [99.9, 100.0, 100.1, 100.2, 100.3, 100.35],
            "close": [100.1, 100.2, 100.3, 100.4, 100.8, 100.85],
        },
        index=idx,
    )

    t = plan_next_open_trade(
        m5,
        trend=Side.LONG,
        spec=_spec(),
        p=H2L2Params(pullback_bars=2, min_risk_price_units=0.1, stop_buffer=0.0),
        timeframe_minutes=5,
    )

    assert t is not None
    assert t.signal_ts == m5.index[4]  # Bar 4 = signal
    assert t.execute_ts == m5.index[5]  # Bar 5 = execute
    assert t.side == Side.LONG