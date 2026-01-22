# tests/test_backtest_daily_selection.py
from __future__ import annotations

import random
from dataclasses import dataclass

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
    return pd.Timestamp(s, tz="UTC")


def test_daily_selection_is_deterministic_top2_with_shuffle():
    """
    5 trades on the same NY day. Shuffle input => exact same top2 output.
    CHRONOLOGICAL mode: earliest signal_ts wins (default behavior)
    """
    tick_size = 0.25
    max_trades_day = 2

    day_exec = _utc("2026-01-05 15:00:00")

    trades = [
        DummyTrade(
            signal_ts=_utc("2026-01-05 14:55:00"),
            execute_ts=day_exec,
            side="BUY",
            entry=4800.00,
            stop=4790.00,
        ),  # FIRST
        DummyTrade(
            signal_ts=_utc("2026-01-05 14:56:00"),
            execute_ts=day_exec,
            side="BUY",
            entry=4800.00,
            stop=4798.75,
        ),  # SECOND
        DummyTrade(
            signal_ts=_utc("2026-01-05 14:57:00"),
            execute_ts=day_exec,
            side="SELL",
            entry=4800.00,
            stop=4801.50,
        ),
        DummyTrade(
            signal_ts=_utc("2026-01-05 14:58:00"),
            execute_ts=day_exec,
            side="SELL",
            entry=4800.00,
            stop=4805.00,
        ),
        DummyTrade(
            signal_ts=_utc("2026-01-05 14:59:00"),
            execute_ts=day_exec,
            side="BUY",
            entry=4800.00,
            stop=4797.00,
        ),
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

    # FIX: Chronological mode picks first 2 by signal_ts
    assert expected_top2[0].signal_ts == _utc("2026-01-05 14:55:00")  # Earliest
    assert expected_top2[1].signal_ts == _utc("2026-01-05 14:56:00")  # Second
