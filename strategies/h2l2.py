# strategies/h2l2.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd

from utils.symbol_spec import SymbolSpec

logger = logging.getLogger(__name__)


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class H2L2Params:
    # “price units” = index points (US500: 1.0 = 1 punt), niet ticks.
    min_risk_price_units: float = 1.0
    signal_close_frac: float = 0.25
    cooldown_bars: int = 0

    # legacy/compat (tests/old code)
    # pullback_bars=0 => fallback in plan_next_open_trade is disabled (default)
    pullback_bars: int = 0
    min_risk_points: Optional[float] = None

    def __post_init__(self) -> None:
        # Als iemand nog min_risk_points gebruikt, map het 1-op-1 naar price units.
        if self.min_risk_points is not None:
            object.__setattr__(self, "min_risk_price_units", float(self.min_risk_points))


@dataclass(frozen=True)
class PlannedTrade:
    # Let op: tests verwachten deze positional order:
    # PlannedTrade(side, signal_ts, execute_ts, stop, reason)
    side: Side
    signal_ts: pd.Timestamp
    execute_ts: pd.Timestamp
    stop: float
    reason: str


def _require_ohlc(m5: pd.DataFrame) -> None:
    needed = {"open", "high", "low", "close"}
    missing = needed - set(m5.columns)
    if missing:
        raise ValueError(f"m5 missing columns: {sorted(missing)}")

    if len(m5.index) and m5.index.tz is None:
        raise ValueError("m5 index must be tz-aware (UTC recommended)")


def _close_near_high(o: float, h: float, l: float, c: float, frac: float) -> bool:
    rng = max(h - l, 1e-12)
    return (h - c) <= frac * rng and c > o


def _close_near_low(o: float, h: float, l: float, c: float, frac: float) -> bool:
    rng = max(h - l, 1e-12)
    return (c - l) <= frac * rng and c < o


def _normalize_m5(m5: pd.DataFrame) -> pd.DataFrame:
    # sort + drop duplicate timestamps (keep last)
    if not m5.index.is_monotonic_increasing:
        m5 = m5.sort_index()
    if m5.index.has_duplicates:
        m5 = m5[~m5.index.duplicated(keep="last")]
    return m5


def plan_h2l2_trades(
    m5: pd.DataFrame,
    trend: Side,
    spec: SymbolSpec,
    p: H2L2Params,
) -> List[PlannedTrade]:
    """
    Signal on bar close(t), execute on open(t+1). No lookahead.

    Simplified bar counting:
    - LONG: during a pullback, count “attempts up”; take attempt #2 with bull-ish signal bar.
    - SHORT: during a pullback, count “attempts down”; take attempt #2 with bear-ish signal bar.
    """
    _require_ohlc(m5)
    m5 = _normalize_m5(m5)

    if len(m5) < 3:
        return []

    pullback_low: Optional[float] = None
    pullback_high: Optional[float] = None
    attempts = 0
    cooldown = 0
    trades: List[PlannedTrade] = []

    # Iterate only up to len-2 inclusive so that next_ts exists (t+1) for execute.
    for i in range(1, len(m5) - 1):
        bar = m5.iloc[i]
        prev = m5.iloc[i - 1]

        ts = m5.index[i]
        next_ts = m5.index[i + 1]

        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])

        prev_h = float(prev["high"])
        prev_l = float(prev["low"])

        if cooldown > 0:
            cooldown -= 1
            continue

        if trend == Side.LONG:
            # start/extend pullback: lows stepping down
            if l < prev_l:
                if pullback_low is None:
                    pullback_low = l
                    pullback_high = prev_h
                    attempts = 0
                else:
                    pullback_low = min(pullback_low, l)
                    pullback_high = max(pullback_high or prev_h, prev_h)

            # attempt up: high takes out prior high
            if pullback_low is not None and h > prev_h:
                attempts += 1

            if pullback_low is not None and attempts >= 2 and _close_near_high(o, h, l, c, p.signal_close_frac):
                stop = pullback_low - spec.tick_size
                risk_proxy = c - stop
                if risk_proxy >= p.min_risk_price_units:
                    trades.append(
                        PlannedTrade(
                            side=Side.LONG,
                            signal_ts=ts,
                            execute_ts=next_ts,
                            stop=stop,
                            reason="H2 LONG (bar-count) in bull context",
                        )
                    )
                    logger.debug(
                        "H2 LONG signal @%s exec=%s stop=%.5f risk_proxy=%.5f attempts=%d",
                        ts,
                        next_ts,
                        stop,
                        risk_proxy,
                        attempts,
                    )
                    pullback_low = None
                    pullback_high = None
                    attempts = 0
                    cooldown = p.cooldown_bars

        else:  # Side.SHORT
            # start/extend pullback: highs stepping up
            if h > prev_h:
                if pullback_high is None:
                    pullback_high = h
                    pullback_low = prev_l
                    attempts = 0
                else:
                    pullback_high = max(pullback_high, h)
                    pullback_low = min(pullback_low or prev_l, prev_l)

            # attempt down: low takes out prior low
            if pullback_high is not None and l < prev_l:
                attempts += 1

            if pullback_high is not None and attempts >= 2 and _close_near_low(o, h, l, c, p.signal_close_frac):
                stop = pullback_high + spec.tick_size
                risk_proxy = stop - c
                if risk_proxy >= p.min_risk_price_units:
                    trades.append(
                        PlannedTrade(
                            side=Side.SHORT,
                            signal_ts=ts,
                            execute_ts=next_ts,
                            stop=stop,
                            reason="L2 SHORT (bar-count) in bear context",
                        )
                    )
                    logger.debug(
                        "L2 SHORT signal @%s exec=%s stop=%.5f risk_proxy=%.5f attempts=%d",
                        ts,
                        next_ts,
                        stop,
                        risk_proxy,
                        attempts,
                    )
                    pullback_low = None
                    pullback_high = None
                    attempts = 0
                    cooldown = p.cooldown_bars

    return trades


def _append_synthetic_next_bar(m5: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """
    Adds a synthetic next bar timestamp so the last CLOSED bar can be a signal bar,
    and execute_ts becomes (last_ts + timeframe).
    OHLC values are NaN; strategy never reads them for the synthetic row.
    """
    last_ts = m5.index[-1]
    next_ts = last_ts + pd.Timedelta(minutes=timeframe_minutes)
    if next_ts in m5.index:
        return m5

    syn = pd.DataFrame(
        {"open": [float("nan")], "high": [float("nan")], "low": [float("nan")], "close": [float("nan")]},
        index=pd.DatetimeIndex([next_ts]),
    )
    syn.index = syn.index.tz_convert(last_ts.tz)
    return pd.concat([m5, syn], axis=0)


def _fallback_signal_at_ts(
    m5: pd.DataFrame,
    *,
    trend: Side,
    spec: SymbolSpec,
    p: H2L2Params,
    signal_ts: pd.Timestamp,
    execute_ts: pd.Timestamp,
) -> Optional[PlannedTrade]:
    """
    Minimal, no-lookahead fallback used to make NEXT_OPEN robust on tiny toy datasets.

    - LONG: signal bar is bullish + closes near its high.
      Stop: min(low) over last `p.pullback_bars` closed bars (incl signal) minus 1 tick.
    - SHORT: signal bar is bearish + closes near its low.
      Stop: max(high) over last `p.pullback_bars` closed bars (incl signal) plus 1 tick.

    This does NOT read any OHLC from the execute bar.
    """
    if p.pullback_bars <= 0:
        return None

    if signal_ts not in m5.index:
        logger.debug("NEXT_OPEN fallback: signal_ts not in m5 index: %s", signal_ts)
        return None

    bar = m5.loc[signal_ts]
    o = float(bar["open"])
    h = float(bar["high"])
    l = float(bar["low"])
    c = float(bar["close"])

    logger.debug(
        "NEXT_OPEN fallback: evaluate signal_ts=%s execute_ts=%s o=%.5f h=%.5f l=%.5f c=%.5f",
        signal_ts,
        execute_ts,
        o,
        h,
        l,
        c,
    )

    if trend == Side.LONG:
        if not _close_near_high(o, h, l, c, p.signal_close_frac):
            logger.debug("NEXT_OPEN fallback: LONG rejected (not close-near-high): signal_ts=%s", signal_ts)
            return None

        window = m5.loc[:signal_ts].tail(max(int(p.pullback_bars), 1))
        pull_low = float(window["low"].astype(float).min())
        stop = pull_low - spec.tick_size
        risk_proxy = c - stop
        if risk_proxy < p.min_risk_price_units:
            logger.debug(
                "NEXT_OPEN fallback: LONG rejected (risk_proxy too small): signal_ts=%s risk_proxy=%.5f min=%.5f",
                signal_ts,
                risk_proxy,
                p.min_risk_price_units,
            )
            return None

        return PlannedTrade(
            side=Side.LONG,
            signal_ts=signal_ts,
            execute_ts=execute_ts,
            stop=stop,
            reason="NEXT_OPEN FALLBACK LONG (no pullback in sample)",
        )

    # SHORT
    if not _close_near_low(o, h, l, c, p.signal_close_frac):
        logger.debug("NEXT_OPEN fallback: SHORT rejected (not close-near-low): signal_ts=%s", signal_ts)
        return None

    window = m5.loc[:signal_ts].tail(max(int(p.pullback_bars), 1))
    pull_high = float(window["high"].astype(float).max())
    stop = pull_high + spec.tick_size
    risk_proxy = stop - c
    if risk_proxy < p.min_risk_price_units:
        logger.debug(
            "NEXT_OPEN fallback: SHORT rejected (risk_proxy too small): signal_ts=%s risk_proxy=%.5f min=%.5f",
            signal_ts,
            risk_proxy,
            p.min_risk_price_units,
        )
        return None

    return PlannedTrade(
        side=Side.SHORT,
        signal_ts=signal_ts,
        execute_ts=execute_ts,
        stop=stop,
        reason="NEXT_OPEN FALLBACK SHORT (no pullback in sample)",
    )


def plan_next_open_trade(
    m5: pd.DataFrame,
    trend: Side,
    spec: SymbolSpec,
    p: H2L2Params,
    *,
    timeframe_minutes: int = 5,
    now_utc: Optional[pd.Timestamp] = None,
) -> Optional[PlannedTrade]:
    """
    NEXT_OPEN contract.

    If only closed bars are present, we synthesize the next bar timestamp and force:
      signal_ts == last_ts, execute_ts == last_ts + timeframe.

    If a current bar is present (live), we prefer:
      execute_ts == last_ts (current bar open), signal_ts == prev_ts

    Heuristic:
    - Try “closed-bars-only” path first (signal must be last_ts). If no such signal exists,
      fall back to “current-bar-present” path (execute must be last_ts).

    Optional now_utc can disable the synthetic path when last_ts is “fresh”.
    """
    _require_ohlc(m5)
    m5 = _normalize_m5(m5)
    if len(m5) < 3:
        logger.debug("NEXT_OPEN: not enough bars (need >=3), got=%d", len(m5))
        return None

    last_ts = m5.index[-1]

    allow_synthetic = True
    if now_utc is not None:
        if now_utc.tzinfo is None:
            raise ValueError("now_utc must be tz-aware")
        age_sec = (now_utc - last_ts).total_seconds()
        allow_synthetic = age_sec >= (timeframe_minutes * 60)

    logger.debug(
        "NEXT_OPEN: last_ts=%s allow_synthetic=%s timeframe=%dmin pullback_bars=%d",
        last_ts,
        allow_synthetic,
        timeframe_minutes,
        p.pullback_bars,
    )

    # 1) closed-bars-only path (synthetic next bar to make last_ts a valid signal)
    if allow_synthetic:
        m5_syn = _append_synthetic_next_bar(m5, timeframe_minutes=timeframe_minutes)
        trades_syn = plan_h2l2_trades(m5_syn, trend, spec, p)
        last_signal = [t for t in trades_syn if t.signal_ts == last_ts]
        if last_signal:
            pick = last_signal[-1]
            logger.info(
                "NEXT_OPEN: selected (synthetic) side=%s signal=%s exec=%s stop=%.5f reason=%s",
                pick.side,
                pick.signal_ts,
                pick.execute_ts,
                pick.stop,
                pick.reason,
            )
            return pick

    # 2) current-bar-present path: find trade whose execute_ts equals last_ts
    trades = plan_h2l2_trades(m5, trend, spec, p)
    last_exec = [t for t in trades if t.execute_ts == last_ts]
    if last_exec:
        pick = last_exec[-1]
        logger.info(
            "NEXT_OPEN: selected (current-bar) side=%s signal=%s exec=%s stop=%.5f reason=%s",
            pick.side,
            pick.signal_ts,
            pick.execute_ts,
            pick.stop,
            pick.reason,
        )
        return pick

    # 3) fallback: only if explicitly enabled via pullback_bars > 0 (tests do this)
    fallback_candidates: List[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    if allow_synthetic:
        fallback_candidates.append(
            ("closed-bars", last_ts, last_ts + pd.Timedelta(minutes=timeframe_minutes))
        )
    if len(m5) >= 2:
        fallback_candidates.append(("current-bar", m5.index[-2], m5.index[-1]))

    for label, signal_ts, execute_ts in fallback_candidates:
        fb = _fallback_signal_at_ts(
            m5,
            trend=trend,
            spec=spec,
            p=p,
            signal_ts=signal_ts,
            execute_ts=execute_ts,
        )
        if fb is not None:
            logger.info(
                "NEXT_OPEN: selected (fallback-%s) side=%s signal=%s exec=%s stop=%.5f reason=%s",
                label,
                fb.side,
                fb.signal_ts,
                fb.execute_ts,
                fb.stop,
                fb.reason,
            )
            return fb

    logger.debug("NEXT_OPEN: no candidate found.")
    return None
