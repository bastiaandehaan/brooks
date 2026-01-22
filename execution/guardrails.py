from __future__ import annotations

import logging
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

from strategies.h2l2 import PlannedTrade

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Guardrails:
    session_tz: str = "America/New_York"
    day_tz: str = "America/New_York"
    session_start: str = "09:30"
    session_end: str = "15:00"
    max_trades_per_day: int = 2
    one_trade_per_execute_ts: bool = True


def _parse_hhmm(s: str) -> tuple[int, int]:
    hh, mm = s.split(":")
    return int(hh), int(mm)


def _in_session(exec_ts_utc: pd.Timestamp, g: Guardrails) -> bool:
    try:
        tz = ZoneInfo(g.session_tz)
        ts_local = exec_ts_utc.tz_convert(tz)
    except Exception as e:
        logger.error("GUARDRAIL tz conversion failed: %s", e)
        return False

    sh, sm = _parse_hhmm(g.session_start)
    eh, em = _parse_hhmm(g.session_end)

    t = ts_local.time()
    start_time = t.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end_time = t.replace(hour=eh, minute=em, second=0, microsecond=0)

    return start_time <= t <= end_time


def apply_guardrails(
    plans: list[PlannedTrade],
    g: Guardrails,
) -> tuple[list[PlannedTrade], list[tuple[PlannedTrade, str]]]:
    """
    Apply guardrails with minimal logging (no spam per rejected trade).
    """
    day_tz = ZoneInfo(g.day_tz)
    accepted: list[PlannedTrade] = []
    rejected: list[tuple[PlannedTrade, str]] = []

    trades_per_day: dict[pd.Timestamp, int] = {}
    used_exec_ts: set[pd.Timestamp] = set()

    # Count rejection reasons
    reject_counts = {"outside_session": 0, "naive_ts": 0, "duplicate_ts": 0, "max_per_day": 0}

    for t in plans:
        exec_ts = t.execute_ts

        if exec_ts.tzinfo is None:
            rejected.append((t, "naive_ts"))
            reject_counts["naive_ts"] += 1
            continue

        if not _in_session(exec_ts, g):
            rejected.append((t, "outside_session"))
            reject_counts["outside_session"] += 1
            continue

        if g.one_trade_per_execute_ts and exec_ts in used_exec_ts:
            rejected.append((t, "duplicate_ts"))
            reject_counts["duplicate_ts"] += 1
            continue

        day_key = exec_ts.tz_convert(day_tz).normalize()
        n = trades_per_day.get(day_key, 0)

        if n >= g.max_trades_per_day:
            rejected.append((t, "max_per_day"))
            reject_counts["max_per_day"] += 1
            continue

        trades_per_day[day_key] = n + 1
        used_exec_ts.add(exec_ts)
        accepted.append(t)

    # Summary logging only (no per-trade spam)
    logger.info(
        "Guardrails: accepted=%d rejected=%d (session=%d, max_day=%d, dupe=%d)",
        len(accepted),
        len(rejected),
        reject_counts["outside_session"],
        reject_counts["max_per_day"],
        reject_counts["duplicate_ts"],
    )

    return accepted, rejected
