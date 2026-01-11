# execution/selection.py
from __future__ import annotations

import logging
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from strategies.h2l2 import PlannedTrade

logger = logging.getLogger(__name__)

NY_TZ = "America/New_York"


def trade_risk_price_units(t: PlannedTrade) -> float:
    """Absolute risk in price units (entry-stop)."""
    return float(abs(t.entry - t.stop))


def score_trade(t: PlannedTrade) -> float:
    """
    MVP score (KISS):
    - prefer smaller risk (tighter stop) => higher score
    """
    r = trade_risk_price_units(t)
    if not np.isfinite(r) or r <= 0:
        return -1e18
    return -r  # smaller risk => higher score


def select_top_trades_per_day(
    trades: list[PlannedTrade],
    *,
    max_per_day: int,
    day_tz: str = NY_TZ,
) -> list[PlannedTrade]:
    """
    Deterministic selection:
    - group by execute_ts NY day
    - sort by score desc
    - stable tie-breakers:
        1) score desc (i.e. risk asc)
        2) signal_ts asc
        3) side stable
        4) entry asc
        5) stop asc
    """
    if max_per_day <= 0 or not trades:
        return []

    tz = ZoneInfo(day_tz)

    buckets: dict[pd.Timestamp, list[PlannedTrade]] = {}
    for t in trades:
        exec_ts = t.execute_ts
        if exec_ts.tzinfo is None:
            # guardrails should reject this earlier; keep deterministic behavior anyway
            continue
        day_key = exec_ts.tz_convert(tz).normalize()
        buckets.setdefault(day_key, []).append(t)

    selected: list[PlannedTrade] = []
    for day_key in sorted(buckets.keys()):
        day_trades = buckets[day_key]

        def _sort_key(x: PlannedTrade):
            return (
                -score_trade(x),
                pd.Timestamp(x.signal_ts),
                str(x.side),
                float(x.entry),
                float(x.stop),
            )

        day_trades_sorted = sorted(day_trades, key=_sort_key)
        chosen = day_trades_sorted[:max_per_day]
        selected.extend(chosen)

        logger.info(
            "selection: day=%s candidates=%d selected=%d top_risks=%s",
            str(day_key.date()),
            len(day_trades),
            len(chosen),
            [round(trade_risk_price_units(t), 6) for t in chosen[: min(3, len(chosen))]],
        )

    return selected
