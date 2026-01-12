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
    top_risk_pts: List[float]


def _as_utc_ts(ts: Any) -> pd.Timestamp:
    """
    Convert input to tz-aware UTC pd.Timestamp.
    If naive: we EXPLICITLY assume UTC (upstream should ideally provide tz-aware).
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def _quantize(value: float, tick_size: float) -> float:
    if tick_size <= 0:
        raise ValueError("tick_size must be > 0")
    return round(value / tick_size) * tick_size


def _risk_pts(entry: float, stop: float, tick_size: float) -> float:
    """
    Risk in price units (abs(entry-stop)), quantized to tick_size
    for float-stable deterministic ordering.
    """
    r = abs(float(entry) - float(stop))
    return _quantize(r, tick_size)


def _ny_day(execute_ts_utc: pd.Timestamp, tz_ny: str) -> pd.Timestamp:
    return execute_ts_utc.tz_convert(tz_ny).normalize()


def select_top_per_ny_day(
    candidates: Iterable[Any],
    *,
    max_trades_day: int,
    tick_size: float,
    tz_ny: str = NY_TZ,
    log_daily: bool = True,
) -> Tuple[List[Any], List[SelectionStats]]:
    """
    Deterministic daily selection (MVP/KISS).

    Score: smaller risk (abs(entry-stop)) is better.
    Deterministic tie-breakers (stable):
      1) risk_pts asc
      2) signal_ts asc
      3) side (string) asc
      4) entry_q asc
      5) stop_q asc

    Requirements on candidate objects (duck typing):
      - candidate.execute_ts
      - candidate.signal_ts
      - candidate.side
      - candidate.entry
      - candidate.stop
    """
    cand_list = list(candidates)

    if max_trades_day <= 0 or not cand_list:
        return [], []

    if tick_size <= 0:
        raise ValueError("tick_size must be > 0")

    # group: NY day -> sortable tuples
    buckets: dict[pd.Timestamp, list[tuple[float, pd.Timestamp, str, float, float, Any]]] = {}

    for c in cand_list:
        exec_utc = _as_utc_ts(getattr(c, "execute_ts"))
        sig_utc = _as_utc_ts(getattr(c, "signal_ts"))
        side = str(getattr(c, "side"))
        entry = float(getattr(c, "entry"))
        stop = float(getattr(c, "stop"))

        day_key = _ny_day(exec_utc, tz_ny)

        entry_q = _quantize(entry, tick_size)
        stop_q = _quantize(stop, tick_size)
        risk_q = _risk_pts(entry_q, stop_q, tick_size)

        buckets.setdefault(day_key, []).append((risk_q, sig_utc, side, entry_q, stop_q, c))

    selected: list[Any] = []
    stats: list[SelectionStats] = []

    for day_key in sorted(buckets.keys()):
        items = buckets[day_key]

        # deterministic ordering
        items.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

        chosen = items[:max_trades_day]
        chosen_trades = [c for *_rest, c in chosen]
        selected.extend(chosen_trades)

        if log_daily:
            top_risks = [float(r) for r, *_ in chosen[: min(3, len(chosen))]]
            st = SelectionStats(
                ny_day=str(day_key.date()),
                candidates=len(items),
                selected=len(chosen_trades),
                top_risk_pts=top_risks,
            )
            stats.append(st)
            logger.info(
                "selection: ny_day=%s candidates=%d selected=%d top_risk_pts=%s",
                st.ny_day,
                st.candidates,
                st.selected,
                [round(x, 6) for x in st.top_risk_pts],
            )

    return selected, stats


# Backwards-compatible wrapper (in case older code imports this name)
def select_top_trades_per_day(
    trades: list[Any],
    *,
    max_per_day: int,
    day_tz: str = NY_TZ,
    tick_size: float = 0.25,
) -> list[Any]:
    """
    Compatibility wrapper for older call sites.
    Uses select_top_per_ny_day under the hood.
    """
    selected, _ = select_top_per_ny_day(
        trades,
        max_trades_day=max_per_day,
        tick_size=tick_size,
        tz_ny=day_tz,
        log_daily=True,
    )
    return selected
