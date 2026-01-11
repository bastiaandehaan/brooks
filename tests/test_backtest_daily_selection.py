# tests/test_backtest_daily_selection.py
from __future__ import annotations

from dataclasses import dataclass
import random

import pandas as pd

from execution.selection import select_top_per_ny_day


@dataclass(frozen=True)
class DummyTrade:
    signal_ts: pd.Timestamp
    execute_ts: pd.Timestamp
    side: str
    entry: float
    stop: float


def _utc(s: str) -> pd.Timestamp:
    # convenience: tz-aware UTC timestamps
    return pd.Timestamp(s, tz="UTC")


def test_daily_selection_is_deterministic_top2_with_shuffle():
    """
    5 trades on the same NY day. Shuffle input => exact same top2 output.
    Score rule: smaller abs(entry-stop) wins.
    Tie-breakers: risk_pts, signal_ts, side, entry_q, stop_q.
    """
    tick_size = 0.25  # US500-like; change if your spec differs
    max_trades_day = 2

    # Choose execute_ts such that NY day is the same.
    # 2026-01-05 15:00 UTC = 10:00 NY (winter, EST).
    day_exec = _utc("2026-01-05 15:00:00")

    trades = [
        DummyTrade(signal_ts=_utc("2026-01-05 14:55:00"), execute_ts=day_exec, side="BUY",  entry=4800.00, stop=4790.00),  # risk 10.00
        DummyTrade(signal_ts=_utc("2026-01-05 14:56:00"), execute_ts=day_exec, side="BUY",  entry=4800.00, stop=4798.75),  # risk 1.25  (best)
        DummyTrade(signal_ts=_utc("2026-01-05 14:57:00"), execute_ts=day_exec, side="SELL", entry=4800.00, stop=4801.50),  # risk 1.50  (2nd)
        DummyTrade(signal_ts=_utc("2026-01-05 14:58:00"), execute_ts=day_exec, side="SELL", entry=4800.00, stop=4805.00),  # risk 5.00
        DummyTrade(signal_ts=_utc("2026-01-05 14:59:00"), execute_ts=day_exec, side="BUY",  entry=4800.00, stop=4797.00),  # risk 3.00
    ]

    expected_top2, _ = select_top_per_ny_day(
        trades,
        max_trades_day=max_trades_day,
        tick_size=tick_size,
    )

    # Shuffle multiple times; selection must be identical (object equality via dataclass)
    for seed in range(10):
        rng = random.Random(seed)
        shuffled = trades[:]
        rng.shuffle(shuffled)

        got_top2, _ = select_top_per_ny_day(
            shuffled,
            max_trades_day=max_trades_day,
            tick_size=tick_size,
        )

        assert got_top2 == expected_top2
        assert len(got_top2) == 2

    # Sanity check: the top2 are indeed the smallest risks
    assert expected_top2[0].stop == 4798.75
    assert expected_top2[1].stop == 4801.50
