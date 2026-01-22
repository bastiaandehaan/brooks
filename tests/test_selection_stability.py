# tests/test_selection_stability.py
"""
STABILITY TEST: Verify selection is deterministic across shuffles.
This is the CRITICAL test that was failing.
"""

import random
from dataclasses import dataclass

import pandas as pd
import pytest

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


def test_selection_deterministic_with_same_signal_time():
    """
    CRITICAL TEST: 5 trades with IDENTICAL signal_ts.
    Selection must be deterministic regardless of input order.
    """
    day_exec = _utc("2026-01-05 15:00:00")
    same_signal = _utc("2026-01-05 14:55:00")  # ALL trades have same signal time

    trades = [
        DummyTrade(
            signal_ts=same_signal,
            execute_ts=day_exec,
            side="BUY",
            entry=4800.00,
            stop=4790.00,
        ),  # Trade A
        DummyTrade(
            signal_ts=same_signal,
            execute_ts=day_exec,
            side="BUY",
            entry=4800.00,
            stop=4798.75,
        ),  # Trade B
        DummyTrade(
            signal_ts=same_signal,
            execute_ts=day_exec,
            side="SELL",
            entry=4800.00,
            stop=4801.50,
        ),  # Trade C
        DummyTrade(
            signal_ts=same_signal,
            execute_ts=day_exec,
            side="SELL",
            entry=4800.00,
            stop=4805.00,
        ),  # Trade D
        DummyTrade(
            signal_ts=same_signal,
            execute_ts=day_exec,
            side="BUY",
            entry=4800.00,
            stop=4797.00,
        ),  # Trade E
    ]

    # Get baseline selection
    expected, _ = select_top_per_ny_day(
        trades,
        max_trades_day=2,
        tick_size=0.25,
    )

    # Verify shuffle stability (100 iterations)
    for seed in range(100):
        rng = random.Random(seed)
        shuffled = trades[:]
        rng.shuffle(shuffled)

        got, _ = select_top_per_ny_day(
            shuffled,
            max_trades_day=2,
            tick_size=0.25,
        )

        # MUST select same trades (by object identity)
        assert got == expected, f"Selection changed with seed {seed}!"

    # Verify we select FIRST 2 trades in original list order
    assert len(expected) == 2
    assert expected[0] is trades[0]  # Trade A
    assert expected[1] is trades[1]  # Trade B


def test_selection_chronological_when_different_times():
    """Verify chronological selection when signal times differ"""
    day_exec = _utc("2026-01-05 15:00:00")

    trades = [
        DummyTrade(
            signal_ts=_utc("2026-01-05 14:57:00"),  # Third
            execute_ts=day_exec,
            side="BUY",
            entry=4800.00,
            stop=4790.00,
        ),
        DummyTrade(
            signal_ts=_utc("2026-01-05 14:55:00"),  # First
            execute_ts=day_exec,
            side="BUY",
            entry=4800.00,
            stop=4798.75,
        ),
        DummyTrade(
            signal_ts=_utc("2026-01-05 14:56:00"),  # Second
            execute_ts=day_exec,
            side="SELL",
            entry=4800.00,
            stop=4801.50,
        ),
    ]

    selected, _ = select_top_per_ny_day(
        trades,
        max_trades_day=2,
        tick_size=0.25,
    )

    # Should select EARLIEST 2 by signal_ts
    assert len(selected) == 2
    assert selected[0].signal_ts == _utc("2026-01-05 14:55:00")
    assert selected[1].signal_ts == _utc("2026-01-05 14:56:00")


def test_selection_filters_bad_data():
    """Verify NaN/Inf filtering"""
    day = _utc("2026-01-05 15:00:00")

    trades = [
        DummyTrade(day, day, "BUY", 100.0, 99.0),  # Valid
        DummyTrade(day, day, "BUY", 100.0, float("nan")),  # Invalid (NaN)
        DummyTrade(day, day, "BUY", float("inf"), 99.0),  # Invalid (Inf)
        DummyTrade(day, day, "BUY", 100.0, 100.0),  # Invalid (zero risk)
    ]

    selected, stats = select_top_per_ny_day(
        trades,
        max_trades_day=5,
        tick_size=0.25,
        tz_ny="America/New_York",
    )

    # Only 1 valid trade
    assert len(selected) == 1
    assert selected[0].stop == 99.0

    # Stats should reflect filtering
    assert len(stats) == 1
    assert stats[0].candidates == 1  # Only 1 valid
    assert stats[0].selected == 1


def test_selection_respects_daily_limit():
    """Verify max_trades_day limit"""
    day = _utc("2026-01-05 15:00:00")

    # 10 valid trades same day
    trades = [
        DummyTrade(
            signal_ts=_utc(f"2026-01-05 14:{50 + i}:00"),
            execute_ts=day,
            side="BUY",
            entry=4800.0,
            stop=4799.0 - i * 0.1,
        )
        for i in range(10)
    ]

    selected, stats = select_top_per_ny_day(
        trades,
        max_trades_day=3,
        tick_size=0.25,
    )

    assert len(selected) == 3
    assert stats[0].candidates == 10
    assert stats[0].selected == 3
    assert stats[0].rejected == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
