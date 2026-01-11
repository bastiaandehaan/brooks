import pandas as pd
import pytest

from strategies.h2l2 import PlannedTrade, Side
from backtest.runner import select_top_trades_per_day


def _t(ts_exec: str, risk: float, idx: int) -> PlannedTrade:
    exec_ts = pd.Timestamp(ts_exec, tz="UTC")
    signal_ts = exec_ts - pd.Timedelta(minutes=5)
    entry = 100.0
    stop = entry - risk if idx % 2 == 0 else entry + risk
    tp = entry + 2.0 * abs(entry - stop) if idx % 2 == 0 else entry - 2.0 * abs(entry - stop)

    return PlannedTrade(
        signal_ts=signal_ts,
        execute_ts=exec_ts,
        side=Side.LONG,
        entry=entry,
        stop=stop,
        tp=tp,
        reason="test",
    )


def test_select_top2_per_day_is_deterministic_and_prefers_tighter_risk():
    trades = [
        _t("2026-01-08 15:30:00", 2.0, 1),
        _t("2026-01-08 15:35:00", 0.5, 2),
        _t("2026-01-08 15:40:00", 1.5, 3),
        _t("2026-01-08 15:45:00", 0.7, 4),
        _t("2026-01-08 15:50:00", 1.0, 5),
    ]

    trades_shuffled = [trades[i] for i in [3, 0, 4, 2, 1]]

    picked = select_top_trades_per_day(trades_shuffled, max_per_day=2, day_tz="America/New_York")
    assert len(picked) == 2

    risks = sorted([abs(t.entry - t.stop) for t in picked])
    assert risks == pytest.approx([0.5, 0.7], rel=0, abs=1e-9)

    assert picked[0].execute_ts < picked[1].execute_ts
