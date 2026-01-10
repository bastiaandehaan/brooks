import pandas as pd

from strategies.h2l2 import Side, H2L2Params, plan_next_open_trade
from utils.symbol_spec import SymbolSpec


def _df(rows):
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts")


def _spec():
    return SymbolSpec(
        name="US500.cash",
        digits=2,
        point=0.01,
        tick_size=0.01,
        tick_value=0.01,
        contract_size=1.0,
        volume_min=0.01,
        volume_step=0.01,
        volume_max=1000.0,
    )


def test_next_open_with_current_bar_executes_on_last_index():
    # last row represents the "current" bar (execute on its open)
    m5 = _df([
        {"ts": "2026-01-01T15:30:00Z", "open": 100, "high": 101, "low": 99, "close": 100.5},
        {"ts": "2026-01-01T15:35:00Z", "open": 100.5, "high": 100.6, "low": 98.8, "close": 99.2},
        {"ts": "2026-01-01T15:40:00Z", "open": 99.2, "high": 100.7, "low": 99.0, "close": 100.6},
        {"ts": "2026-01-01T15:45:00Z", "open": 100.6, "high": 100.65, "low": 98.7, "close": 99.0},
        {"ts": "2026-01-01T15:50:00Z", "open": 99.0, "high": 100.8, "low": 98.9, "close": 100.75},
        {"ts": "2026-01-01T15:55:00Z", "open": 100.75, "high": 101.0, "low": 100.2, "close": 100.8},  # current bar
    ])

    t = plan_next_open_trade(m5, Side.LONG, _spec(), H2L2Params(min_risk_price_units=0.5), timeframe_minutes=5)
    assert t is not None
    assert t.execute_ts == m5.index[-1]
    assert t.side == Side.LONG


def test_next_open_closed_bars_only_executes_on_synthetic_next_bar():
    # no current bar; last row is last CLOSED bar
    m5 = _df([
        {"ts": "2026-01-01T15:30:00Z", "open": 100, "high": 101, "low": 99, "close": 100.5},
        {"ts": "2026-01-01T15:35:00Z", "open": 100.5, "high": 100.6, "low": 98.8, "close": 99.2},
        {"ts": "2026-01-01T15:40:00Z", "open": 99.2, "high": 100.7, "low": 99.0, "close": 100.6},
        {"ts": "2026-01-01T15:45:00Z", "open": 100.6, "high": 100.65, "low": 98.7, "close": 99.0},
        {"ts": "2026-01-01T15:50:00Z", "open": 99.0, "high": 100.8, "low": 98.9, "close": 100.75},  # signal bar (closed)
    ])

    last_ts = m5.index[-1]
    expected_exec = last_ts + pd.Timedelta(minutes=5)

    t = plan_next_open_trade(m5, Side.LONG, _spec(), H2L2Params(min_risk_price_units=0.5), timeframe_minutes=5)
    assert t is not None
    assert t.signal_ts == last_ts
    assert t.execute_ts == expected_exec
    assert t.side == Side.LONG
