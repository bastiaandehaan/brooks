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
    min_risk_price_units: float = 1.0
    signal_close_frac: float = 0.25
    cooldown_bars: int = 0
    pullback_bars: int = 2

    @property
    def risk_dist(self) -> float:
        return self.min_risk_price_units


@dataclass(frozen=True)
class PlannedTrade:
    signal_ts: pd.Timestamp
    execute_ts: pd.Timestamp
    side: Side
    entry: float
    stop: float
    tp: float
    reason: str


def plan_next_open_trade(
        m5: pd.DataFrame,
        trend: Side,
        spec: SymbolSpec,
        p: H2L2Params,
        timeframe_minutes: int,
        now_utc: pd.Timestamp | None = None
) -> Optional[PlannedTrade]:
    """
    Price Action Logica:
    Zoekt naar een sterke reversal bar na een pullback in de trend.
    """
    if len(m5) < 3:
        return None

    # Context:
    # [-1] = Huidige Open Bar (executie)
    # [-2] = Signal Bar (net gesloten)
    # [-3] = Bar voor Signal

    signal_bar = m5.iloc[-2]
    prev_bar = m5.iloc[-3]
    curr_idx = m5.index[-1]

    risk = p.risk_dist

    # --- LONG SETUP ---
    if trend == Side.LONG:
        # Pullback: High lager dan vorige high, of rode candle
        is_pullback = (signal_bar["high"] < prev_bar["high"]) or (signal_bar["close"] < signal_bar["open"])

        # Signal Bar Strength
        bar_range = signal_bar["high"] - signal_bar["low"]
        if bar_range == 0: return None

        # Close in upper % (1.0 = High)
        close_loc = (signal_bar["close"] - signal_bar["low"]) / bar_range
        is_strong_close = close_loc > (1.0 - p.signal_close_frac)

        if is_pullback and is_strong_close:
            entry = float(m5.iloc[-1]["open"])
            stop = entry - risk
            tp = entry + (risk * 2.0)

            return PlannedTrade(
                signal_ts=m5.index[-2],
                execute_ts=curr_idx,
                side=Side.LONG,
                entry=entry,
                stop=stop,
                tp=tp,
                reason=f"Bull Reversal (Str={close_loc:.2f})"
            )

    # --- SHORT SETUP ---
    elif trend == Side.SHORT:
        # Pullback: Low hoger dan vorige low, of groene candle
        is_pullback = (signal_bar["low"] > prev_bar["low"]) or (signal_bar["close"] > signal_bar["open"])

        bar_range = signal_bar["high"] - signal_bar["low"]
        if bar_range == 0: return None

        # Close in lower % (0.0 = Low)
        close_loc = (signal_bar["high"] - signal_bar["close"]) / bar_range
        is_strong_close = close_loc > (1.0 - p.signal_close_frac)

        if is_pullback and is_strong_close:
            entry = float(m5.iloc[-1]["open"])
            stop = entry + risk
            tp = entry - (risk * 2.0)

            return PlannedTrade(
                signal_ts=m5.index[-2],
                execute_ts=curr_idx,
                side=Side.SHORT,
                entry=entry,
                stop=stop,
                tp=tp,
                reason=f"Bear Reversal (Str={close_loc:.2f})"
            )

    return None


def plan_h2l2_trades(m5: pd.DataFrame, trend: Side, spec: SymbolSpec, p: H2L2Params) -> List[PlannedTrade]:
    trades = []
    # Start bij 50 voor buffer
    for i in range(50, len(m5)):
        current_slice = m5.iloc[:i + 1]
        trade = plan_next_open_trade(current_slice, trend, spec, p, timeframe_minutes=5)
        if trade:
            trades.append(trade)
    return trades