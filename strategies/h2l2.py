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
    min_risk_price_units: float = 1.0       # 1.0 = 1 indexpunt
    signal_close_frac: float = 0.25         # close in top/bottom 25%
    cooldown_bars: int = 0                  # optional


@dataclass(frozen=True)
class PlannedTrade:
    side: Side
    signal_ts: pd.Timestamp
    execute_ts: pd.Timestamp
    stop: float
    reason: str


def _close_near_high(o, h, l, c, frac: float) -> bool:
    rng = max(h - l, 1e-12)
    return (h - c) <= frac * rng and c > o


def _close_near_low(o, h, l, c, frac: float) -> bool:
    rng = max(h - l, 1e-12)
    return (c - l) <= frac * rng and c < o


def plan_h2l2_trades(m5: pd.DataFrame, trend: Side, spec: SymbolSpec, p: H2L2Params) -> List[PlannedTrade]:
    """
    Signal on bar close(t), execute on open(t+1). No lookahead.
    """
    if len(m5) < 3:
        return []

    trades: List[PlannedTrade] = []
    attempts = 0
    pullback_low: Optional[float] = None
    pullback_high: Optional[float] = None
    cooldown = 0

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

                # attempt increments when high > prior high
                if h > prev_h:
                    attempts += 1
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
                            pullback_low = None
                            pullback_high = None
                            attempts = 0
                            cooldown = p.cooldown_bars

        else:  # Side.SHORT
            if pullback_high is None and h > prev_h:
                pullback_high = h
                pullback_low = l
                attempts = 0

            if pullback_high is not None:
                pullback_high = max(pullback_high, h)
                pullback_low = min(pullback_low or l, l)

                if l < prev_l:
                    attempts += 1
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
                            pullback_low = None
                            pullback_high = None
                            attempts = 0
                            cooldown = p.cooldown_bars

    return trades
