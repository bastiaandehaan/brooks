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
        p=H2L2Params(pullback_bars=2, min_risk_points=0.1),
        timeframe_minutes=5,
    )

    assert t is not None
    assert t.execute_ts == m5.index[-1]
    assert t.signal_ts == m5.index[-2]
    assert t.side == Side.LONG


def test_next_open_closed_bars_only_executes_on_synthetic_next_bar():
    # Alleen gesloten bars (laatste is de signal bar). We verwachten execute op +5 min.
    idx = pd.to_datetime(
        [
            "2026-01-08 15:30:00",
            "2026-01-08 15:35:00",
            "2026-01-08 15:40:00",
            "2026-01-08 15:45:00",
            "2026-01-08 15:50:00",  # signal bar (closed)
        ],
        utc=True,
    )

    m5 = pd.DataFrame(
        {
            "open": [100, 100.1, 100.2, 100.3, 100.4],
            "high": [100.2, 100.3, 100.4, 100.5, 100.9],
            "low": [99.9, 100.0, 100.1, 100.2, 100.3],
            "close": [100.1, 100.2, 100.3, 100.4, 100.8],
        },
        index=idx,
    )

    t = plan_next_open_trade(
        m5,
        trend=Side.LONG,
        spec=_spec(),
        p=H2L2Params(pullback_bars=2, min_risk_points=0.1),
        timeframe_minutes=5,
    )

    assert t is not None
    assert t.signal_ts == m5.index[-1]
    assert t.execute_ts == (m5.index[-1] + pd.Timedelta(minutes=5))
    assert t.side == Side.LONG


def test_next_open_fallback_disabled_returns_none_on_toy_data():
    # Zelfde toy-data als hierboven, maar fallback uit (pullback_bars=0).
    # Dan verwachten we None (want bar-counting setup zit niet in deze monotone sample).
    idx = pd.to_datetime(
        [
            "2026-01-08 15:30:00",
            "2026-01-08 15:35:00",
            "2026-01-08 15:40:00",
            "2026-01-08 15:45:00",
            "2026-01-08 15:50:00",
        ],
        utc=True,
    )

    m5 = pd.DataFrame(
        {
            "open": [100, 100.1, 100.2, 100.3, 100.4],
            "high": [100.2, 100.3, 100.4, 100.5, 100.9],
            "low": [99.9, 100.0, 100.1, 100.2, 100.3],
            "close": [100.1, 100.2, 100.3, 100.4, 100.8],
        },
        index=idx,
    )

    t = plan_next_open_trade(
        m5,
        trend=Side.LONG,
        spec=_spec(),
        p=H2L2Params(pullback_bars=0, min_risk_price_units=0.1),
        timeframe_minutes=5,
    )
    assert t is None
