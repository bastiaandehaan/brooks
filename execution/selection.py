# execution/selection.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Tuple

import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectionStats:
    ny_day: str
    candidates: int
    selected: int
    top_risks_pts: List[float]


def _as_utc_ts(ts: Any) -> pd.Timestamp:
    """
    Accepts pd.Timestamp / datetime / str.
    Returns tz-aware UTC pd.Timestamp.
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        # We assume UTC if naive; this is intentional and explicit.
        # If this is wrong in your pipeline, fix upstream to always provide tz-aware.
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def _quantize(value: float, tick_size: float) -> float:
    if tick_size <= 0:
        raise ValueError("tick_size must be > 0")
    return round(value / tick_size) * tick_size


def _risk_pts(entry: float, stop: float, tick_size: float) -> float:
    # risk in price units, quantized to tick_size for float-stable ordering
    r = abs(float(entry) - float(stop))
    return _quantize(r, tick_size)


def _ny_day_from_execute_ts(execute_ts_utc: pd.Timestamp, tz_ny: str) -> pd.Timestamp:
    return execute_ts_utc.tz_convert(tz_ny).normalize()


def select_top_per_ny_day(
    candidates: Iterable[Any],
    *,
    max_trades_day: int,
    tick_size: float,
    tz_ny: str = "America/New_York",
    log_daily: bool = True,
) -> Tuple[List[Any], List[SelectionStats]]:
    """
    Deterministic selection per NY day.
    MVP score: smaller risk (=abs(entry-stop)) is better.

    - group by NY day (based on execute_ts)
    - stable deterministic sort per day using:
        (risk_pts asc, signal_ts asc, side asc, entry_q asc, stop_q asc)
    - pick top max_trades_day

    Requirements on candidate object (no datamodel changes needed):
      - candidate.execute_ts
      - candidate.signal_ts
      - candidate.side  (string-ish)
      - candidate.entry (float-ish)
      - candidate.stop  (float-ish)
    """
    cand_list = list(candidates)
    if max_trades_day <= 0 or not cand_list:
        return [], []

    if tick_size <= 0:
        raise ValueError("tick_size must be > 0")

    # group candidates by NY day
    by_day: Dict[pd.Timestamp, List[Tuple[float, pd.Timestamp, str, float, float, Any]]] = {}

    for c in cand_list:
        exec_utc = _as_utc_ts(getattr(c, "execute_ts"))
        sig_utc = _as_utc_ts(getattr(c, "signal_ts"))
        side = str(getattr(c, "side"))
        entry = float(getattr(c, "entry"))
        stop = float(getattr(c, "stop"))

        day = _ny_day_from_execute_ts(exec_utc, tz_ny)

        entry_q = _quantize(entry, tick_size)
        stop_q = _quantize(stop, tick_size)
        risk_q = _risk_pts(entry_q, stop_q, tick_size)

        by_day.setdefault(day, []).append((risk_q, sig_utc, side, entry_q, stop_q, c))

    selected: List[Any] = []
    stats: List[SelectionStats] = []

    for day in sorted(by_day.keys()):
        items = by_day[day]

        # Deterministic ordering, fully stable given quantization + tie-breakers
        items.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

        chosen = items[:max_trades_day]
        selected.extend([c for *_rest, c in chosen])

        if log_daily:
            top_risks = [float(r) for r, *_ in chosen[: min(3, len(chosen))]]
            st = SelectionStats(
                ny_day=str(day.date()),
                candidates=len(items),
                selected=len(chosen),
                top_risks_pts=top_risks,
            )
            stats.append(st)
            log.info(
                "selection: ny_day=%s candidates=%d selected=%d top_risk_pts=%s",
                st.ny_day,
                st.candidates,
                st.selected,
                [round(x, 6) for x in st.top_risks_pts],
            )

    return selected, stats
