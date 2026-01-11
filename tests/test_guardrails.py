from zoneinfo import ZoneInfo
import pandas as pd
import pytest

from execution.guardrails import Guardrails, apply_guardrails
from strategies.h2l2 import PlannedTrade, Side


def _utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_session_filter_new_york():
    g = Guardrails(session_tz="America/New_York", day_tz="America/New_York", session_start="09:30", session_end="15:00")

    # FIX: extra args toegevoegd
    plans = [
        PlannedTrade(Side.LONG, _utc("2026-01-09T12:00:00Z"), _utc("2026-01-09T12:05:00Z"), 100.0, 99.0, 101.0,
                     "reason1")
    ]

    acc, rej = apply_guardrails(plans, g)
    assert len(acc) == 0
    assert len(rej) == 1
    assert rej[0][1] == "outside session"

    plans2 = [
        PlannedTrade(Side.LONG, _utc("2026-01-09T15:00:00Z"), _utc("2026-01-09T15:05:00Z"), 100.0, 99.0, 101.0,
                     "reason2")
    ]
    acc, rej = apply_guardrails(plans2, g)
    assert len(acc) == 1
    assert len(rej) == 0


def test_max_trades_per_day_new_york():
    g = Guardrails(max_trades_per_day=2)

    # FIX: extra args toegevoegd
    plans = [
        PlannedTrade(Side.LONG, _utc("2026-01-09T15:00:00Z"), _utc("2026-01-09T15:05:00Z"), 100.0, 99.0, 101.0, "a"),
        PlannedTrade(Side.LONG, _utc("2026-01-09T16:00:00Z"), _utc("2026-01-09T16:05:00Z"), 100.0, 99.0, 101.0, "b"),
        PlannedTrade(Side.LONG, _utc("2026-01-09T17:00:00Z"), _utc("2026-01-09T17:05:00Z"), 100.0, 99.0, 101.0, "c"),
    ]

    acc, rej = apply_guardrails(plans, g)
    assert len(acc) == 2
    assert len(rej) == 1
    assert rej[0][1] == "max trades per day"