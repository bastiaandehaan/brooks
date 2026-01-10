import pandas as pd

from execution.guardrails import Guardrails, apply_guardrails
from strategies.h2l2 import PlannedTrade, Side


def _t(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert("UTC") if "Z" not in ts else pd.Timestamp(ts)


def test_max_trades_per_day_enforced():
    # 3 trades same local day (Brussels) => only 2 accepted
    plans = [
        PlannedTrade(Side.LONG, _t("2026-01-09T18:00:00Z"), _t("2026-01-09T18:05:00Z"), 1.0, "a"),
        PlannedTrade(Side.LONG, _t("2026-01-09T18:10:00Z"), _t("2026-01-09T18:15:00Z"), 1.0, "b"),
        PlannedTrade(Side.LONG, _t("2026-01-09T18:20:00Z"), _t("2026-01-09T18:25:00Z"), 1.0, "c"),
    ]
    g = Guardrails(max_trades_per_day=2)
    acc, rej = apply_guardrails(plans, g)
    assert len(acc) == 2
    assert len(rej) == 1


def test_session_filter_rejects_outside_window():
    # 12:00 UTC = 13:00 Brussels (outside 15:30-21:00)
    plans = [
        PlannedTrade(Side.LONG, _t("2026-01-09T12:00:00Z"), _t("2026-01-09T12:05:00Z"), 1.0, "x"),
    ]
    g = Guardrails()
    acc, rej = apply_guardrails(plans, g)
    assert len(acc) == 0
    assert len(rej) == 1
