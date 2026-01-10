import pandas as pd

from execution.guardrails import Guardrails, apply_guardrails
from strategies.h2l2 import PlannedTrade, Side


def _utc(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert("UTC") if "Z" not in ts else pd.Timestamp(ts)


def test_session_filter_new_york():
    g = Guardrails(session_tz="America/New_York", day_tz="America/New_York", session_start="09:30", session_end="15:00")

    # 12:05Z = 07:05 NY => outside
    plans = [PlannedTrade(Side.LONG, _utc("2026-01-09T12:00:00Z"), _utc("2026-01-09T12:05:00Z"), 1.0, "x")]
    acc, rej = apply_guardrails(plans, g)
    assert len(acc) == 0
    assert len(rej) == 1

    # 15:05Z = 10:05 NY => inside
    plans = [PlannedTrade(Side.LONG, _utc("2026-01-09T15:00:00Z"), _utc("2026-01-09T15:05:00Z"), 1.0, "y")]
    acc, rej = apply_guardrails(plans, g)
    assert len(acc) == 1
    assert len(rej) == 0


def test_max_trades_per_day_new_york():
    g = Guardrails(max_trades_per_day=2)

    plans = [
        PlannedTrade(Side.LONG, _utc("2026-01-09T15:00:00Z"), _utc("2026-01-09T15:05:00Z"), 1.0, "a"),
        PlannedTrade(Side.LONG, _utc("2026-01-09T16:00:00Z"), _utc("2026-01-09T16:05:00Z"), 1.0, "b"),
        PlannedTrade(Side.LONG, _utc("2026-01-09T17:00:00Z"), _utc("2026-01-09T17:05:00Z"), 1.0, "c"),
    ]
    acc, rej = apply_guardrails(plans, g)
    assert len(acc) == 2
    assert len(rej) == 1
