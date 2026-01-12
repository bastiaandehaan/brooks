# execution/selection.py
"""
Brooks Daily Selection - FIXED
Oude bug: kleinste stop = hoogste score → stop-out bias
Nieuwe regel: FIRST COME FIRST SERVED binnen session (chronologisch)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Optional

import pandas as pd

logger = logging.getLogger(__name__)

NY_TZ = "America/New_York"


@dataclass(frozen=True)
class SelectionStats:
    ny_day: str
    candidates: int
    selected: int
    rejected: int


def _as_utc_ts(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        logger.debug("select: naive timestamp, assuming UTC: %s", t)
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def _ny_day(execute_ts_utc: pd.Timestamp, tz_ny: str) -> pd.Timestamp:
    return execute_ts_utc.tz_convert(tz_ny).normalize()


def _is_finite(*vals: float) -> bool:
    for v in vals:
        if not pd.notna(v):
            return False
        if v != v:  # NaN
            return False
        if v in (float("inf"), float("-inf")):
            return False
    return True


def select_top_per_ny_day(
        candidates: Iterable[Any],
        *,
        max_trades_day: int,
        tick_size: float,  # kept for backwards compat, maar niet gebruikt
        tz_ny: str = NY_TZ,
        log_daily: bool = True,
        score_mode: str = "chronological",  # backwards compat, ignore
        warn_on_bad_rows: bool = True,
) -> Tuple[List[Any], List[SelectionStats]]:
    """
    Brooks Selection Rule - SIMPEL:

    Neem eerste N trades per dag (chronologisch op signal_ts).
    Waarom? Brooks: "Trade the first good setup you see. Don't wait for perfect."

    Oude bug: selecteerde kleinste stops → stop-out bias.
    Fix: chronologische volgorde = natuurlijke priority (eerste setups zijn vaak best).
    """
    cand_list = list(candidates)

    if max_trades_day <= 0 or not cand_list:
        return [], []

    # Group by NY day
    buckets: dict[pd.Timestamp, list[tuple[pd.Timestamp, Any]]] = {}
    bad_rows = 0

    for c in cand_list:
        try:
            exec_utc = _as_utc_ts(getattr(c, "execute_ts"))
            sig_utc = _as_utc_ts(getattr(c, "signal_ts"))

            entry = float(getattr(c, "entry"))
            stop = float(getattr(c, "stop"))

            if not _is_finite(entry, stop):
                bad_rows += 1
                if warn_on_bad_rows:
                    logger.warning("select: skipping non-finite entry/stop: %r", c)
                continue

            if abs(entry - stop) <= 0:
                bad_rows += 1
                if warn_on_bad_rows:
                    logger.warning("select: skipping zero risk: %r", c)
                continue

            day_key = _ny_day(exec_utc, tz_ny)
            buckets.setdefault(day_key, []).append((sig_utc, c))

        except Exception as e:
            bad_rows += 1
            if warn_on_bad_rows:
                logger.warning("select: skipping invalid candidate: %s", e)
            continue

    if bad_rows and log_daily:
        logger.info("selection: skipped_bad_rows=%d", bad_rows)

    selected: list[Any] = []
    stats: list[SelectionStats] = []

    for day_key in sorted(buckets.keys()):
        items = buckets[day_key]
        if not items:
            continue

        # Sort chronologically (signal_ts asc)
        items.sort(key=lambda x: x[0])

        # Take first N
        chosen = items[:max_trades_day]
        chosen_trades = [c for _sig, c in chosen]
        selected.extend(chosen_trades)

        if log_daily:
            st = SelectionStats(
                ny_day=str(day_key.date()),
                candidates=len(items),
                selected=len(chosen_trades),
                rejected=len(items) - len(chosen_trades),
            )
            stats.append(st)

            logger.info(
                "selection: ny_day=%s candidates=%d selected=%d rejected=%d (chronological order)",
                st.ny_day, st.candidates, st.selected, st.rejected
            )

    return selected, stats


# Backwards compatible wrapper
def select_top_trades_per_day(
        trades: list[Any],
        *,
        max_per_day: int,
        day_tz: str = NY_TZ,
        tick_size: float = 0.25,
) -> list[Any]:
    selected, _ = select_top_per_ny_day(
        trades,
        max_trades_day=max_per_day,
        tick_size=tick_size,
        tz_ny=day_tz,
        log_daily=True,
    )
    return selected