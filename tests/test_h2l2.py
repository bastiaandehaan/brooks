import pandas as pd

from strategies.h2l2 import H2L2Params, Side, plan_h2l2_trades
from utils.symbol_spec import SymbolSpec


def _df(rows):
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts")


def test_plans_h2_long_next_open():
    m5 = _df(
        [
            {"ts": "2026-01-01T15:30:00Z", "open": 100, "high": 101, "low": 99, "close": 100.5},
            {
                "ts": "2026-01-01T15:35:00Z",
                "open": 100.5,
                "high": 100.6,
                "low": 98.8,
                "close": 99.2,
            },
            {
                "ts": "2026-01-01T15:40:00Z",
                "open": 99.2,
                "high": 100.7,
                "low": 99.0,
                "close": 100.6,
            },
            {
                "ts": "2026-01-01T15:45:00Z",
                "open": 100.6,
                "high": 100.65,
                "low": 98.7,
                "close": 99.0,
            },
            {
                "ts": "2026-01-01T15:50:00Z",
                "open": 99.0,
                "high": 100.8,
                "low": 98.9,
                "close": 100.75,
            },
            {
                "ts": "2026-01-01T15:55:00Z",
                "open": 100.75,
                "high": 101.0,
                "low": 100.2,
                "close": 100.8,
            },
        ]
    )

    spec = SymbolSpec(
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

    trades = plan_h2l2_trades(m5, Side.LONG, spec, H2L2Params(min_risk_price_units=0.5))
    assert len(trades) == 1
    assert trades[0].signal_ts == m5.index[4]
    assert trades[0].execute_ts == m5.index[5]
