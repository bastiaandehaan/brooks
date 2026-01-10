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
    min_risk_price_units: float = 1.0       # e.g. 1.0 = 1 index point
    signal_close_frac: float = 0.25         # close in top/bottom 25% of bar range
    cooldown_bars: int = 0                  # optional cool-down after a signal


@dataclass(frozen=True)
class PlannedTrade:
    side: Side
    signal_ts: pd.Timestamp
    execute_ts: pd.Timestamp
    stop: float
    reason: str


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
    Signal on bar close(t), execute on open(t+1). No lookahead.

    This is a simplified, objective "bar counting" approach:
    - In bull context: count attempts up during a pullback; take attempt #2 with a bull-ish signal bar.
    - In bear context: count attempts down during a pullback; take attempt #2 with a bear-ish signal bar.
    """
    if len(m5) < 3:
        return []

    # Defensive: require OHLC
    for col in ("open", "high", "low", "close"):
        if col not in m5.columns:
            raise ValueError(f"Missing column: {col}")

    trades: List[PlannedTrade] = []
    attempts = 0
    pullback_low: Optional[float] = None
    pullback_high: Optional[float] = None
    cooldown = 0

    # iterate i in [1, len-2], because execute_ts uses i+1
    for i in range(1, len(m5) - 1):
        ts = m5.index[i]
        next_ts = m5.index[i + 1]

        o = float(m5["open"].iloc[i])
        h = float(m5["high"].iloc[i])
        l = float(m5["low"].iloc[i])
        c = float(m5["close"].iloc[i])

        prev_h = float(m5["high"].iloc[i - 1])
        prev_l = float(m5["low"].iloc[i - 1])

        if cooldown > 0:
            cooldown -= 1
            continue

        if trend == Side.LONG:
            # Start pullback when we print a lower low
            if pullback_low is None and l < prev_l:
                pullback_low = l
                pullback_high = h
                attempts = 0

            if pullback_low is not None:
                pullback_low = min(pullback_low, l)
                pullback_high = max(pullback_high or h, h)

                # attempt increments when high > prior high (a push up)
                if h > prev_h:
                    attempts += 1

                    # H2 condition
                    if attempts >= 2 and _close_near_high(o, h, l, c, p.signal_close_frac):
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
                            # reset after signal
                            pullback_low = None
                            pullback_high = None
                            attempts = 0
                            cooldown = p.cooldown_bars

        else:  # Side.SHORT
            # Start pullback when we print a higher high
            if pullback_high is None and h > prev_h:
                pullback_high = h
                pullback_low = l
                attempts = 0

            if pullback_high is not None:
                pullback_high = max(pullback_high, h)
                pullback_low = min(pullback_low or l, l)

                # attempt increments when low < prior low (a push down)
                if l < prev_l:
                    attempts += 1

                    # L2 condition
                    if attempts >= 2 and _close_near_low(o, h, l, c, p.signal_close_frac):
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
                            # reset after signal
                            pullback_low = None
                            pullback_high = None
                            attempts = 0
                            cooldown = p.cooldown_bars

    return trades


def plan_next_open_trade(
    m5: pd.DataFrame,
    trend: Side,
    spec: SymbolSpec,
    p: H2L2Params,
) -> Optional[PlannedTrade]:
    """
    Live-safe: return at most 1 trade whose execute_ts equals the last bar timestamp.
    This aligns with NEXT_OPEN: signal bar is the last closed bar, execute on current bar open.

    Requires that m5 includes the current (still forming) bar at the end.
    If you feed only closed bars, this will not return a trade.
    """
    if len(m5) < 3:
        return None

    trades = plan_h2l2_trades(m5, trend, spec, p)
    if not trades:
        return None

    last_exec_ts = m5.index[-1]
    candidates = [t for t in trades if t.execute_ts == last_exec_ts]
    if not candidates:
        return None

    return candidates[-1]
