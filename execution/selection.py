# execution/selection.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)

NY_TZ = "America/New_York"


@dataclass(frozen=True)
class SelectionStats:
    ny_day: str
    candidates: int
    selected: int
    top_risk_pts: List[float]
    # diagnostics (new)
    min_risk_pts: Optional[float] = None
    median_risk_pts: Optional[float] = None
    max_risk_pts: Optional[float] = None
    side_counts: Optional[dict[str, int]] = None


def _as_utc_ts(ts: Any) -> pd.Timestamp:
    """
    Convert input to tz-aware UTC pd.Timestamp.
    If naive: we EXPLICITLY assume UTC (upstream should ideally provide tz-aware).
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        logger.debug("select: naive timestamp encountered, assuming UTC: %s", t)
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


def _is_finite(*vals: float) -> bool:
    for v in vals:
        if not pd.notna(v):
            return False
        if v != v:  # NaN
            return False
        if v in (float("inf"), float("-inf")):
            return False
    return True


ScoreMode = Literal["min_risk", "max_risk"]


def select_top_per_ny_day(
    candidates: Iterable[Any],
    *,
    max_trades_day: int,
    tick_size: float,
    tz_ny: str = NY_TZ,
    log_daily: bool = True,
    score_mode: ScoreMode = "min_risk",
    warn_on_bad_rows: bool = True,
) -> Tuple[List[Any], List[SelectionStats]]:
    """
    Deterministic daily selection (MVP/KISS).

    Default score: smaller risk (abs(entry-stop)) is better (min_risk).
    Optional: max_risk (useful as a diagnostic knob to see if selection is the culprit).

    Deterministic tie-breakers (stable):
      1) score asc/desc depending on score_mode
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

    if score_mode not in ("min_risk", "max_risk"):
        raise ValueError("score_mode must be 'min_risk' or 'max_risk'")

    # group: NY day -> sortable tuples
    # tuple layout: (score, sig_utc, side, entry_q, stop_q, risk_q, c)
    buckets: dict[pd.Timestamp, list[tuple[float, pd.Timestamp, str, float, float, float, Any]]] = {}

    bad_rows = 0

    for c in cand_list:
        exec_utc = _as_utc_ts(getattr(c, "execute_ts"))
        sig_utc = _as_utc_ts(getattr(c, "signal_ts"))
        side = str(getattr(c, "side"))

        entry = float(getattr(c, "entry"))
        stop = float(getattr(c, "stop"))

        if not _is_finite(entry, stop):
            bad_rows += 1
            if warn_on_bad_rows:
                logger.warning("select: skipping non-finite entry/stop: entry=%s stop=%s obj=%r", entry, stop, c)
            continue

        day_key = _ny_day(exec_utc, tz_ny)

        entry_q = _quantize(entry, tick_size)
        stop_q = _quantize(stop, tick_size)
        risk_q = _risk_pts(entry_q, stop_q, tick_size)

        # guard: risk=0 is almost always garbage / rounding artifact
        if risk_q <= 0:
            bad_rows += 1
            if warn_on_bad_rows:
                logger.warning(
                    "select: skipping zero/negative risk after quantize: entry_q=%.5f stop_q=%.5f risk_q=%.5f obj=%r",
                    entry_q, stop_q, risk_q, c
                )
            continue

        score = risk_q if score_mode == "min_risk" else -risk_q
        buckets.setdefault(day_key, []).append((score, sig_utc, side, entry_q, stop_q, risk_q, c))

    if bad_rows and log_daily:
        logger.info("selection: skipped_bad_rows=%d (non-finite or risk<=0)", bad_rows)

    selected: list[Any] = []
    stats: list[SelectionStats] = []

    for day_key in sorted(buckets.keys()):
        items = buckets[day_key]
        if not items:
            continue

        # deterministic ordering
        items.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

        chosen = items[:max_trades_day]
        chosen_trades = [c for *_rest, c in chosen]
        selected.extend(chosen_trades)

        if log_daily:
            # diagnostics on risk distribution
            all_risks = [float(risk_q) for *_, risk_q, _c in [(x[0], x[1], x[2], x[3], x[4], x[5], x[6]) for x in items]]
            # ^ keep simple / explicit; items already contains risk_q at index 5
            all_risks = [float(x[5]) for x in items]
            all_risks_sorted = sorted(all_risks)
            n = len(all_risks_sorted)
            min_r = all_risks_sorted[0]
            max_r = all_risks_sorted[-1]
            med_r = all_risks_sorted[n // 2] if n else None

            chosen_risks = [float(x[5]) for x in chosen]
            top_risks = [float(r) for r in chosen_risks[: min(3, len(chosen_risks))]]

            # side counts (chosen)
            sc: dict[str, int] = {}
            for x in chosen:
                side = str(x[2])
                sc[side] = sc.get(side, 0) + 1

            st = SelectionStats(
                ny_day=str(day_key.date()),
                candidates=len(items),
                selected=len(chosen_trades),
                top_risk_pts=top_risks,
                min_risk_pts=float(min_r),
                median_risk_pts=float(med_r) if med_r is not None else None,
                max_risk_pts=float(max_r),
                side_counts=sc,
            )
            stats.append(st)

            logger.info(
                "selection: ny_day=%s mode=%s candidates=%d selected=%d "
                "risk[min/med/max]=[%s/%s/%s] chosen_top_risk=%s sides=%s",
                st.ny_day,
                score_mode,
                st.candidates,
                st.selected,
                round(st.min_risk_pts, 6) if st.min_risk_pts is not None else None,
                round(st.median_risk_pts, 6) if st.median_risk_pts is not None else None,
                round(st.max_risk_pts, 6) if st.max_risk_pts is not None else None,
                [round(x, 6) for x in st.top_risk_pts],
                st.side_counts,
            )

            # extra DEBUG: show first few chosen timestamps and risks
            if logger.isEnabledFor(logging.DEBUG):
                preview = []
                for x in chosen[:5]:
                    sig_utc = x[1]
                    risk_q = x[5]
                    preview.append((str(sig_utc), float(risk_q), str(x[2])))
                logger.debug("selection: ny_day=%s chosen_preview=%s", st.ny_day, preview)

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
