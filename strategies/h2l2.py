from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

from utils.symbol_spec import SymbolSpec

logger = logging.getLogger(__name__)


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class H2L2Params:
    """
    Parameters for H2/L2 planner.
    - min_risk_points is legacy alias for min_risk_price_units (tests use it).
    """
    # core risk threshold (price units / index points)
    min_risk_price_units: float = 1.0
    min_risk_points: Optional[float] = None  # legacy alias

    # signal strength
    signal_close_frac: float = 0.25

    # cooldown after a trade (bars)
    cooldown_bars: int = 0

    # opt-in fallback for NEXT_OPEN timing tests on monotone samples
    pullback_bars: int = 0

    def __post_init__(self) -> None:
        if self.min_risk_points is not None:
            object.__setattr__(self, "min_risk_price_units", float(self.min_risk_points))


@dataclass(frozen=True)
class PlannedTrade:
    # Keep this field order (your runtime introspection showed this layout).
    signal_ts: pd.Timestamp
    execute_ts: pd.Timestamp
    side: Side
    entry: float
    stop: float
    tp: float
    reason: str


def _require_ohlc(df: pd.DataFrame) -> None:
    needed = {"open", "high", "low", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if len(df.index) and df.index.tz is None:
        raise ValueError("index must be tz-aware (UTC recommended)")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df


def _close_near_high(o: float, h: float, l: float, c: float, frac: float) -> bool:
    rng = max(h - l, 1e-12)
    return (h - c) <= frac * rng and c > o


def _close_near_low(o: float, h: float, l: float, c: float, frac: float) -> bool:
    rng = max(h - l, 1e-12)
    return (c - l) <= frac * rng and c < o


def plan_h2l2_trades(
    m5: pd.DataFrame,
    trend: Side,
    spec: SymbolSpec,
    p: H2L2Params,
) -> List[PlannedTrade]:
    """
    Bar-counting H2/L2 planner.
    NEXT_OPEN: signal on close(t), execute on open(t+1). No lookahead.

    LONG (H2):
    - detect pullback as lows stepping down (l < prev_l)
    - count attempts up when high breaks prev_high
    - take attempt #2 with close near high
    - stop = pullback_low - tick
    - entry = next bar open
    - tp = entry + 2R

    SHORT (L2): symmetric.
    """
    _require_ohlc(m5)
    m5 = _normalize(m5)

    if len(m5) < 3:
        return []

    # PERF: pull columns once into numpy arrays (avoid pandas row indexing in hot loop)
    o_arr = m5["open"].to_numpy(dtype="float64", copy=False)
    h_arr = m5["high"].to_numpy(dtype="float64", copy=False)
    l_arr = m5["low"].to_numpy(dtype="float64", copy=False)
    c_arr = m5["close"].to_numpy(dtype="float64", copy=False)
    idx = m5.index  # timestamps

    trades: List[PlannedTrade] = []
    cooldown = 0

    pullback_low: Optional[float] = None
    pullback_high: Optional[float] = None
    attempts = 0

    # Iterate to len-2 inclusive so next bar exists for execute
    for i in range(1, len(m5) - 1):
        if cooldown > 0:
            cooldown -= 1
            continue

        ts = idx[i]
        next_ts = idx[i + 1]

        o = float(o_arr[i])
        h = float(h_arr[i])
        l = float(l_arr[i])
        c = float(c_arr[i])

        prev_h = float(h_arr[i - 1])
        prev_l = float(l_arr[i - 1])

        entry = float(o_arr[i + 1])
        if not np.isfinite(entry):
            # If someone passed a synthetic bar row with NaN open, we cannot compute entry here.
            continue

        if trend == Side.LONG:
            # start/extend pullback when lows step down
            if l < prev_l:
                if pullback_low is None:
                    pullback_low = l
                    pullback_high = prev_h
                    attempts = 0
                else:
                    pullback_low = min(pullback_low, l)
                    pullback_high = max(pullback_high or prev_h, prev_h)

            # count attempt up when high breaks prev high during pullback
            if pullback_low is not None and h > prev_h:
                attempts += 1

            if pullback_low is not None and attempts >= 2 and _close_near_high(o, h, l, c, p.signal_close_frac):
                stop = float(pullback_low) - float(spec.tick_size)
                risk = entry - stop
                if risk >= p.min_risk_price_units:
                    tp = entry + 2.0 * risk
                    trades.append(
                        PlannedTrade(
                            signal_ts=ts,
                            execute_ts=next_ts,
                            side=Side.LONG,
                            entry=float(entry),
                            stop=float(stop),
                            tp=float(tp),
                            reason="H2 LONG (bar-count)",
                        )
                    )
                    logger.debug(
                        "H2 LONG signal=%s exec=%s entry=%.5f stop=%.5f tp=%.5f attempts=%d",
                        ts, next_ts, entry, stop, tp, attempts
                    )
                    pullback_low = None
                    pullback_high = None
                    attempts = 0
                    cooldown = p.cooldown_bars

        else:  # Side.SHORT
            # start/extend pullback when highs step up
            if h > prev_h:
                if pullback_high is None:
                    pullback_high = h
                    pullback_low = prev_l
                    attempts = 0
                else:
                    pullback_high = max(pullback_high, h)
                    pullback_low = min(pullback_low or prev_l, prev_l)

            # count attempt down when low breaks prev low during pullback
            if pullback_high is not None and l < prev_l:
                attempts += 1

            if pullback_high is not None and attempts >= 2 and _close_near_low(o, h, l, c, p.signal_close_frac):
                stop = float(pullback_high) + float(spec.tick_size)
                risk = stop - entry
                if risk >= p.min_risk_price_units:
                    tp = entry - 2.0 * risk
                    trades.append(
                        PlannedTrade(
                            signal_ts=ts,
                            execute_ts=next_ts,
                            side=Side.SHORT,
                            entry=float(entry),
                            stop=float(stop),
                            tp=float(tp),
                            reason="L2 SHORT (bar-count)",
                        )
                    )
                    logger.debug(
                        "L2 SHORT signal=%s exec=%s entry=%.5f stop=%.5f tp=%.5f attempts=%d",
                        ts, next_ts, entry, stop, tp, attempts
                    )
                    pullback_low = None
                    pullback_high = None
                    attempts = 0
                    cooldown = p.cooldown_bars

    return trades


def _append_synthetic_next_bar(m5: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """
    Adds a synthetic next bar timestamp so last CLOSED bar can be a signal.
    OHLC are NaN. Strategy must not rely on OHLC for this row.
    """
    last_ts = m5.index[-1]
    next_ts = last_ts + pd.Timedelta(minutes=timeframe_minutes)
    if next_ts in m5.index:
        return m5

    syn = pd.DataFrame(
        {"open": [np.nan], "high": [np.nan], "low": [np.nan], "close": [np.nan]},
        index=pd.DatetimeIndex([next_ts]),
    )
    syn.index = syn.index.tz_convert(last_ts.tz)
    return pd.concat([m5, syn], axis=0)


def _fallback_last_closed_bar(
    m5: pd.DataFrame,
    *,
    trend: Side,
    spec: SymbolSpec,
    p: H2L2Params,
    signal_ts: pd.Timestamp,
    execute_ts: pd.Timestamp,
) -> Optional[PlannedTrade]:
    """
    Opt-in fallback for NEXT_OPEN timing tests:
    - LONG: bullish close near high
    - SHORT: bearish close near low
    Stop based on last p.pullback_bars closed bars up to signal_ts.
    No lookahead: only <= signal_ts.
    """
    if p.pullback_bars <= 0:
        return None
    if signal_ts not in m5.index:
        return None

    hist = m5.loc[:signal_ts]
    if len(hist) < p.pullback_bars:
        return None
    window = hist.tail(p.pullback_bars)

    bar = hist.loc[signal_ts]
    o = float(bar["open"])
    h = float(bar["high"])
    l = float(bar["low"])
    c = float(bar["close"])

    # entry: execute open is not guaranteed in closed-only, so use signal close as proxy
    entry = float(c)

    if trend == Side.LONG:
        if not _close_near_high(o, h, l, c, p.signal_close_frac):
            return None
        stop = float(window["low"].astype(float).min()) - float(spec.tick_size)
        risk = entry - stop
        if risk < p.min_risk_price_units:
            return None
        tp = entry + 2.0 * risk
        return PlannedTrade(signal_ts, execute_ts, Side.LONG, entry, stop, tp, "NEXT_OPEN fallback LONG")

    # SHORT
    if not _close_near_low(o, h, l, c, p.signal_close_frac):
        return None
    stop = float(window["high"].astype(float).max()) + float(spec.tick_size)
    risk = stop - entry
    if risk < p.min_risk_price_units:
        return None
    tp = entry - 2.0 * risk
    return PlannedTrade(signal_ts, execute_ts, Side.SHORT, entry, stop, tp, "NEXT_OPEN fallback SHORT")


def plan_next_open_trade(
    m5: pd.DataFrame,
    trend: Side,
    spec: SymbolSpec,
    p: H2L2Params,
    timeframe_minutes: int = 5,
    now_utc: Optional[pd.Timestamp] = None,
) -> Optional[PlannedTrade]:
    """
    NEXT_OPEN:
    - If only closed bars: signal_ts == last_ts, execute_ts == last_ts + timeframe (synthetic ts)
    - If current bar present: signal_ts == prev_ts, execute_ts == last_ts

    Strategy selection:
    - try synthetic path first if allowed (signal must be last_ts)
    - else try current-bar path (execute must be last_ts)
    - finally, opt-in fallback if pullback_bars > 0
    """
    _require_ohlc(m5)
    m5 = _normalize(m5)

    if len(m5) < 3:
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
        last_ts, allow_synthetic, timeframe_minutes, p.pullback_bars
    )

    if allow_synthetic:
        m5_syn = _append_synthetic_next_bar(m5, timeframe_minutes)
        trades_syn = plan_h2l2_trades(m5_syn, trend, spec, p)
        last_signal = [t for t in trades_syn if t.signal_ts == last_ts]
        if last_signal:
            return last_signal[-1]

        # fallback closed-bars-only
        fb = _fallback_last_closed_bar(
            m5,
            trend=trend,
            spec=spec,
            p=p,
            signal_ts=last_ts,
            execute_ts=last_ts + pd.Timedelta(minutes=timeframe_minutes),
        )
        if fb is not None:
            return fb

    # current bar included path
    trades = plan_h2l2_trades(m5, trend, spec, p)
    last_exec = [t for t in trades if t.execute_ts == last_ts]
    if last_exec:
        return last_exec[-1]

    # fallback current-bar path: signal is prev, execute is last
    prev_ts = m5.index[-2]
    fb = _fallback_last_closed_bar(
        m5,
        trend=trend,
        spec=spec,
        p=p,
        signal_ts=prev_ts,
        execute_ts=last_ts,
    )
    return fb
