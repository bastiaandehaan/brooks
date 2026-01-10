# strategies/h2l2.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Union

import pandas as pd

from utils.symbol_spec import SymbolSpec

logger = logging.getLogger(__name__)


class Side(Enum):
    LONG = auto()
    SHORT = auto()


@dataclass(frozen=True)
class H2L2Params:
    """
    Minimal Brooks H2/L2 idee:
    - LONG: higher-low pullback + bull reversal
    - SHORT: lower-high pullback + bear reversal

    Dit is (bewust) simpel gehouden; we doen vooral correct dataflow + no-lookahead.
    """
    pullback_bars: int = 2
    min_risk_points: float = 2.0  # proxy; later vervangen door echte sizing constraints


@dataclass(frozen=True)
class PlannedTrade:
    symbol: str
    side: Side
    signal_ts: pd.Timestamp   # laatst gesloten bar
    execute_ts: pd.Timestamp  # open van eerstvolgende bar (NEXT_OPEN)
    stop_price: float
    reason: str


def _require_ohlc(m5: pd.DataFrame) -> None:
    needed = {"open", "high", "low", "close"}
    missing = needed - set(m5.columns)
    if missing:
        raise ValueError(f"m5 missing columns: {sorted(missing)}")


def plan_h2l2_trades(
    m5: pd.DataFrame,
    trend: Side,
    spec: SymbolSpec,
    p: H2L2Params,
) -> List[PlannedTrade]:
    """
    Produceert trades op basis van gesloten bars, met execute op de volgende bar open.
    Contract: gebruikt NOOIT de laatste row als signaalbar (die is 'current forming' in live).
    """
    _require_ohlc(m5)
    if len(m5) < (p.pullback_bars + 3):
        return []

    # Zorg dat de index oplopend is
    if not m5.index.is_monotonic_increasing:
        m5 = m5.sort_index()

    trades: List[PlannedTrade] = []

    # i loopt tot len-2 zodat m5.index[i+1] bestaat (execute_ts)
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

        if trend == Side.LONG:
            # bull reversal proxy: close > open
            # higher-low proxy: low > prev_low
            if (c > o) and (l > prev_l):
                stop = l - spec.tick_size  # onder de signal bar low
                risk_proxy = c - stop
                if risk_proxy >= p.min_risk_points:
                    trades.append(
                        PlannedTrade(
                            symbol=spec.name,
                            side=Side.LONG,
                            signal_ts=ts,
                            execute_ts=next_ts,
                            stop_price=stop,
                            reason="H2 proxy: HL + bull close",
                        )
                    )

        if trend == Side.SHORT:
            # bear reversal proxy: close < open
            # lower-high proxy: high < prev_high
            if (c < o) and (h < prev_h):
                stop = h + spec.tick_size  # boven de signal bar high
                risk_proxy = stop - c
                if risk_proxy >= p.min_risk_points:
                    trades.append(
                        PlannedTrade(
                            symbol=spec.name,
                            side=Side.SHORT,
                            signal_ts=ts,
                            execute_ts=next_ts,
                            stop_price=stop,
                            reason="L2 proxy: LH + bear close",
                        )
                    )

    return trades


def _append_synthetic_next_bar(m5: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """
    Voeg een extra (synthetic) bar toe met timestamp = last_ts + timeframe.
    OHLC values zijn NaN; plan_h2l2_trades gebruikt de laatste row alleen als 'next_ts'.
    """
    if timeframe_minutes <= 0:
        raise ValueError("timeframe_minutes must be > 0")

    last_ts = m5.index[-1]
    next_ts = last_ts + pd.Timedelta(minutes=int(timeframe_minutes))

    # Maak lege row met juiste kolommen
    row = {col: float("nan") for col in ["open", "high", "low", "close"]}
    extra = pd.DataFrame([row], index=pd.DatetimeIndex([next_ts], name=m5.index.name))

    out = pd.concat([m5, extra], axis=0)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out


def plan_next_open_trade(
    m5: pd.DataFrame,
    trend: Side,
    spec: SymbolSpec,
    p: H2L2Params,
    *,
    timeframe_minutes: int = 5,
    now_utc: Optional[Union[pd.Timestamp, "datetime.datetime"]] = None,
) -> Optional[PlannedTrade]:
    """
    NEXT_OPEN planner:
    - probeert eerst een trade te vinden met execute_ts == laatste index (MT5 current bar aanwezig)
    - als die niet bestaat, en de laatste bar *waarschijnlijk gesloten is*,
      dan voegen we 1 synthetic next bar toe (voor datasets met alleen gesloten bars)
      en zoeken we execute_ts == synthetic last index.

    Belangrijk: om look-ahead te vermijden in live:
    - als now_utc meegegeven is, doen we de synthetic fallback alleen wanneer
      now_utc - last_ts >= timeframe_minutes (dus: last_ts is niet de current forming bar).
    """
    _require_ohlc(m5)
    if len(m5) < 3:
        return None

    if not m5.index.is_monotonic_increasing:
        m5 = m5.sort_index()

    last_ts = m5.index[-1]

    # 1) normale poging: current bar is aanwezig → execute op last_ts
    trades = plan_h2l2_trades(m5, trend, spec, p)
    candidates = [t for t in trades if t.execute_ts == last_ts]
    if candidates:
        chosen = candidates[-1]
        logger.debug(
            "NEXT_OPEN candidate found on current bar: signal=%s execute=%s side=%s",
            chosen.signal_ts, chosen.execute_ts, chosen.side.name
        )
        return chosen

    # 2) fallback: alleen als last bar waarschijnlijk gesloten is
    allow_fallback = True
    if now_utc is not None:
        now_ts = pd.Timestamp(now_utc).tz_convert("UTC") if pd.Timestamp(now_utc).tzinfo else pd.Timestamp(now_utc, tz="UTC")
        last_ts_utc = last_ts.tz_convert("UTC") if last_ts.tzinfo else last_ts.tz_localize("UTC")
        age_sec = (now_ts - last_ts_utc).total_seconds()
        allow_fallback = age_sec >= (timeframe_minutes * 60)

    if not allow_fallback:
        logger.debug(
            "No NEXT_OPEN candidate; fallback blocked (last bar likely forming). last_ts=%s",
            last_ts,
        )
        return None

    m5_plus = _append_synthetic_next_bar(m5, timeframe_minutes=timeframe_minutes)
    new_last_ts = m5_plus.index[-1]
    trades2 = plan_h2l2_trades(m5_plus, trend, spec, p)
    candidates2 = [t for t in trades2 if t.execute_ts == new_last_ts]
    if not candidates2:
        logger.debug(
            "No NEXT_OPEN candidate after synthetic bar. last_ts=%s synthetic_ts=%s",
            last_ts, new_last_ts,
        )
        return None

    chosen = candidates2[-1]
    logger.debug(
        "NEXT_OPEN candidate found via synthetic bar: signal=%s execute=%s side=%s",
        chosen.signal_ts, chosen.execute_ts, chosen.side.name
    )
    return chosen
