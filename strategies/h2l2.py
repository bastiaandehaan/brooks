# strategies/h2l2.py
"""
Brooks H2/L2 - SIMPEL zoals Brooks het bedoelde
Geen bar counting state machines. Alleen: pullback + rejection bar.
"""
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
    Brooks H2/L2 Parameters - KISS versie
    """
    # Swing detection: kijk N bars terug voor swing low/high
    pullback_bars: int = 3

    # Signal strength: close moet near high/low zijn
    signal_close_frac: float = 0.30  # binnen 30% van range

    # Risk management
    min_risk_price_units: float = 2.0  # minimaal 2pt risico (US500)
    stop_buffer: float = 1.0  # extra ruimte onder/boven swing

    # Legacy alias (voor oude tests)
    min_risk_points: Optional[float] = None

    # Cooldown (optioneel, meestal 0)
    cooldown_bars: int = 0

    def __post_init__(self) -> None:
        if self.min_risk_points is not None:
            object.__setattr__(self, "min_risk_price_units", float(self.min_risk_points))


@dataclass(frozen=True)
class PlannedTrade:
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


def _is_rejection_bar(o: float, h: float, l: float, c: float, side: Side, frac: float) -> bool:
    """
    Brooks rejection bar:
    - LONG: close near high + bullish body (c > o)
    - SHORT: close near low + bearish body (c < o)
    """
    bar_range = max(h - l, 1e-12)

    if side == Side.LONG:
        close_near_high = (h - c) <= frac * bar_range
        bullish = c > o
        return close_near_high and bullish
    else:  # SHORT
        close_near_low = (c - l) <= frac * bar_range
        bearish = c < o
        return close_near_low and bearish


def plan_h2l2_trades(
        m5: pd.DataFrame,
        trend: Side,
        spec: SymbolSpec,
        p: H2L2Params,
) -> List[PlannedTrade]:
    """
    Brooks H2/L2 - PURE implementatie:

    1. Detecteer swing low/high in laatste N bars (pullback_bars)
    2. Check of huidige bar rejection toont (close near extreme)
    3. Stop = swing +/- buffer
    4. Entry = volgende bar open (NEXT_OPEN)
    5. TP = 2R

    GEEN bar counting. GEEN attempt tracking. GEEN complex state.
    """
    _require_ohlc(m5)
    m5 = _normalize(m5)

    if len(m5) < p.pullback_bars + 2:
        return []

    trades: List[PlannedTrade] = []
    cooldown = 0

    for i in range(p.pullback_bars, len(m5) - 1):
        if cooldown > 0:
            cooldown -= 1
            continue

        bar = m5.iloc[i]
        next_bar = m5.iloc[i + 1]

        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])

        entry = float(next_bar["open"])
        if not np.isfinite(entry):
            continue

        # Detecteer swing in lookback window (inclusief current bar)
        window = m5.iloc[i - p.pullback_bars + 1: i + 1]
        swing_low = float(window["low"].min())
        swing_high = float(window["high"].max())

        # Skip doji's (geen range = geen rejection)
        if h - l < 0.01:
            continue

        # Check rejection bar
        if not _is_rejection_bar(o, h, l, c, trend, p.signal_close_frac):
            continue

        # Risk calculation
        if trend == Side.LONG:
            stop = swing_low - p.stop_buffer
            risk = entry - stop

            if risk < p.min_risk_price_units:
                continue

            tp = entry + 2.0 * risk

            trades.append(PlannedTrade(
                signal_ts=bar.name,
                execute_ts=next_bar.name,
                side=Side.LONG,
                entry=entry,
                stop=stop,
                tp=tp,
                reason=f"H2 LONG: rejection after {p.pullback_bars}bar swing",
            ))

            logger.debug(
                "H2 LONG signal=%s exec=%s entry=%.2f stop=%.2f tp=%.2f risk=%.2f",
                bar.name, next_bar.name, entry, stop, tp, risk
            )

            cooldown = p.cooldown_bars

        else:  # SHORT
            stop = swing_high + p.stop_buffer
            risk = stop - entry

            if risk < p.min_risk_price_units:
                continue

            tp = entry - 2.0 * risk

            trades.append(PlannedTrade(
                signal_ts=bar.name,
                execute_ts=next_bar.name,
                side=Side.SHORT,
                entry=entry,
                stop=stop,
                tp=tp,
                reason=f"L2 SHORT: rejection after {p.pullback_bars}bar swing",
            ))

            logger.debug(
                "L2 SHORT signal=%s exec=%s entry=%.2f stop=%.2f tp=%.2f risk=%.2f",
                bar.name, next_bar.name, entry, stop, tp, risk
            )

            cooldown = p.cooldown_bars

    return trades


def plan_next_open_trade(
        m5: pd.DataFrame,
        trend: Side,
        spec: SymbolSpec,
        p: H2L2Params,
        timeframe_minutes: int = 5,
        now_utc: Optional[pd.Timestamp] = None,
) -> Optional[PlannedTrade]:
    """
    NEXT_OPEN wrapper: vind laatste trade die execute op laatste/next bar.

    Backwards compatible met je tests, maar gebruikt nieuwe simpele logica.
    """
    _require_ohlc(m5)
    m5 = _normalize(m5)

    if len(m5) < p.pullback_bars + 2:
        return None

    # Plan alle trades
    all_trades = plan_h2l2_trades(m5, trend, spec, p)

    if not all_trades:
        return None

    last_ts = m5.index[-1]

    # Strategie 1: Execute op laatste bar (current bar scenario)
    last_exec = [t for t in all_trades if t.execute_ts == last_ts]
    if last_exec:
        return last_exec[-1]

    # Strategie 2: Signal op laatste bar, execute op synthetic next bar
    if now_utc is not None and now_utc.tzinfo is not None:
        age_sec = (now_utc - last_ts).total_seconds()
        if age_sec >= (timeframe_minutes * 60):
            # Closed bars scenario: check of laatste bar signal is
            last_signal = [t for t in all_trades if t.signal_ts == last_ts]
            if last_signal:
                # Update execute_ts naar synthetic next bar
                t = last_signal[-1]
                return PlannedTrade(
                    signal_ts=t.signal_ts,
                    execute_ts=last_ts + pd.Timedelta(minutes=timeframe_minutes),
                    side=t.side,
                    entry=t.entry,
                    stop=t.stop,
                    tp=t.tp,
                    reason=t.reason,
                )

    return None