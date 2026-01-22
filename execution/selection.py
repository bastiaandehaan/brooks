# execution/selection.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

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
        if v != v:  # NaN check
            return False
        if v in (float("inf"), float("-inf")):
            return False
    return True


def select_top_per_ny_day(
    candidates: Iterable[Any],
    *,
    max_trades_day: int,
    tick_size: float,
    tz_ny: str = NY_TZ,
    log_daily: bool = False,
    score_mode: str = "chronological",
    warn_on_bad_rows: bool = False,
) -> Tuple[List[Any], List[SelectionStats]]:
    """
    TRULY DETERMINISTIC chronological selection.

    Key fix: Sort by (signal_ts, id(object)) to break ties consistently.
    """
    cand_list = list(candidates)

    if max_trades_day <= 0 or not cand_list:
        return [], []

    buckets: dict[pd.Timestamp, list[tuple[pd.Timestamp, int, Any]]] = {}
    bad_rows = 0

    for idx, c in enumerate(cand_list):
        try:
            exec_utc = _as_utc_ts(getattr(c, "execute_ts"))
            sig_utc = _as_utc_ts(getattr(c, "signal_ts"))
            entry = float(getattr(c, "entry"))
            stop = float(getattr(c, "stop"))

            if not _is_finite(entry, stop):
                bad_rows += 1
                continue

            if abs(entry - stop) <= 0:
                bad_rows += 1
                continue

            day_key = _ny_day(exec_utc, tz_ny)

            # CRITICAL FIX: Include original list index for tie-breaking
            buckets.setdefault(day_key, []).append((sig_utc, idx, c))

        except Exception:
            bad_rows += 1
            continue

    selected: list[Any] = []
    stats: list[SelectionStats] = []

    for day_key in sorted(buckets.keys()):
        items = buckets[day_key]
        if not items:
            continue

        # CRITICAL FIX: Sort by (signal_ts, original_index)
        # This ensures identical signal_ts are ordered by their original position
        items.sort(key=lambda x: (x[0], x[1]))

        chosen = items[:max_trades_day]
        chosen_trades = [c for _sig, _idx, c in chosen]
        selected.extend(chosen_trades)

        stats.append(
            SelectionStats(
                ny_day=str(day_key.date()),
                candidates=len(items),
                selected=len(chosen_trades),
                rejected=len(items) - len(chosen_trades),
            )
        )

    total_selected = len(selected)
    total_rejected = sum(s.rejected for s in stats)

    if log_daily:
        logger.info(
            "Daily selection: %d days, selected=%d, rejected=%d (bad_rows=%d)",
            len(stats),
            total_selected,
            total_rejected,
            bad_rows,
        )

    return selected, stats


# Backward compatibility wrapper
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
        log_daily=False,
    )
    return selected
