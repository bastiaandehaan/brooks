# execution/guardrails.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from strategies.h2l2 import PlannedTrade

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Guardrails:
    # Interpret session times in this timezone (NY session)
    session_tz: str = "America/New_York"
    # Count "trades per day" in this timezone (usually same as session_tz)
    day_tz: str = "America/New_York"

    # NY regular session (you can stop earlier)
    session_start: str = "09:30"
    session_end: str = "15:00"

    max_trades_per_day: int = 2
    one_trade_per_execute_ts: bool = True


def _parse_hhmm(s: str) -> tuple[int, int]:
    hh, mm = s.split(":")
    return int(hh), int(mm)


def _in_session(exec_ts_utc: pd.Timestamp, g: Guardrails) -> bool:
    tz = ZoneInfo(g.session_tz)
    ts_local = exec_ts_utc.tz_convert(tz)

    sh, sm = _parse_hhmm(g.session_start)
    eh, em = _parse_hhmm(g.session_end)

    start = ts_local.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = ts_local.replace(hour=eh, minute=em, second=0, microsecond=0)
    return start <= ts_local <= end


def apply_guardrails(
    plans: Iterable[PlannedTrade],
    g: Guardrails,
) -> Tuple[List[PlannedTrade], List[tuple[PlannedTrade, str]]]:
    """
    Enforces:
    - session window (based on execute_ts converted to session_tz)
    - max trades per day (based on day_tz)
    - optional: one trade per execute timestamp
    """
    day_tz = ZoneInfo(g.day_tz)

    accepted: List[PlannedTrade] = []
    rejected: List[tuple[PlannedTrade, str]] = []

    trades_per_day: dict[pd.Timestamp, int] = {}
    used_exec_ts: set[pd.Timestamp] = set()

    for t in plans:
        exec_ts = t.execute_ts
        if exec_ts.tzinfo is None:
            rejected.append((t, "execute_ts is naive (timezone missing)"))
            continue

        if not _in_session(exec_ts, g):
            rejected.append((t, "outside session"))
            continue

        if g.one_trade_per_execute_ts and exec_ts in used_exec_ts:
            rejected.append((t, "one trade per execute_ts"))
            continue

        day_key = exec_ts.tz_convert(day_tz).normalize()
        n = trades_per_day.get(day_key, 0)
        if n >= g.max_trades_per_day:
            rejected.append((t, "max trades per day"))
            continue

        accepted.append(t)
        used_exec_ts.add(exec_ts)
        trades_per_day[day_key] = n + 1

    logger.info(
        "Guardrails: accepted=%d rejected=%d (session=%s-%s %s max/day=%d day_tz=%s)",
        len(accepted), len(rejected),
        g.session_start, g.session_end, g.session_tz,
        g.max_trades_per_day, g.day_tz
    )
    return accepted, rejected
