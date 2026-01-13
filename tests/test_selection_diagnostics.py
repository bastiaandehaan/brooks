# tests/test_selection_diagnostics.py
import pandas as pd
from dataclasses import dataclass

from execution.selection import select_top_per_ny_day


@dataclass
class T:
    signal_ts: pd.Timestamp
    execute_ts: pd.Timestamp
    side: str
    entry: float
    stop: float


def test_selection_deterministic_and_filters_bad_rows():
    tz = "UTC"
    day = pd.Timestamp("2025-01-02 15:00", tz=tz)

    # 3 candidates same day; one has NaN stop -> must be skipped
    trades = [
        T(day, day, "LONG", 100.0, 99.0),  # risk 1
        T(day, day, "LONG", 100.0, float("nan")),  # bad
        T(day, day, "SHORT", 100.0, 98.0),  # risk 2
    ]

    sel, stats = select_top_per_ny_day(trades, max_trades_day=1, tick_size=0.25, tz_ny="America/New_York",
                                       log_daily=False)
    assert len(sel) == 1
    assert sel[0].stop == 99.0  # min risk picked deterministically (chronological if same time)

    # FIX: Stats are now returned (not empty)
    assert len(stats) == 1  # One day processed
    assert stats[0].candidates == 2  # 2 valid trades (1 filtered out due to NaN)
    assert stats[0].selected == 1
    assert stats[0].rejected == 1