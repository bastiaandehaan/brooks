from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd
import numpy as np
from utils.symbol_spec import SymbolSpec

logger = logging.getLogger(__name__)


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class H2L2Params:
    # Risk settings
    use_atr_risk: bool = True  # NIEUW: Gebruik dynamische ATR stop?
    atr_period: int = 14
    atr_multiplier_stop: float = 1.5
    atr_multiplier_tp: float = 3.0  # 1:2 ratio (1.5 * 2 = 3.0)

    # Fallback voor vaste risk als ATR uit staat
    min_risk_price_units: float = 2.0

    signal_close_frac: float = 0.25
    pullback_bars: int = 2


@dataclass(frozen=True)
class PlannedTrade:
    signal_ts: pd.Timestamp
    execute_ts: pd.Timestamp
    side: Side
    entry: float
    stop: float
    tp: float
    reason: str


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Berekent de Average True Range (ATR)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def plan_next_open_trade(
        m5: pd.DataFrame,
        trend: Side,
        spec: SymbolSpec,
        p: H2L2Params,
        timeframe_minutes: int,
        now_utc: pd.Timestamp | None = None
) -> Optional[PlannedTrade]:
    if len(m5) < p.atr_period + 5:
        return None

    signal_bar = m5.iloc[-2]
    prev_bar = m5.iloc[-3]
    curr_idx = m5.index[-1]

    # --- Bepaal Risico (ATR of Vast) ---
    if p.use_atr_risk:
        # We berekenen ATR over de data t/m de signal bar
        # (We kunnen de huidige open bar niet gebruiken voor ATR)
        closed_data = m5.iloc[:-1]
        atr_series = calculate_atr(closed_data, p.atr_period)
        current_atr = atr_series.iloc[-1]

        if pd.isna(current_atr) or current_atr <= 0:
            return None

        risk_dist = current_atr * p.atr_multiplier_stop
        reward_dist = current_atr * p.atr_multiplier_tp
    else:
        risk_dist = p.min_risk_price_units
        reward_dist = risk_dist * 2.0

    # --- LONG ---
    if trend == Side.LONG:
        # Pullback: Lower High of Rode bar
        is_pullback = (signal_bar["high"] < prev_bar["high"]) or (signal_bar["close"] < signal_bar["open"])

        # Signal Bar Strength
        bar_range = signal_bar["high"] - signal_bar["low"]
        if bar_range == 0: return None

        close_loc = (signal_bar["close"] - signal_bar["low"]) / bar_range
        is_strong_close = close_loc > (1.0 - p.signal_close_frac)

        if is_pullback and is_strong_close:
            entry = float(m5.iloc[-1]["open"])
            stop = entry - risk_dist
            tp = entry + reward_dist

            # Extra check: stop mag niet boven entry liggen (door rare ATR)
            if stop >= entry: return None

            return PlannedTrade(
                signal_ts=m5.index[-2],
                execute_ts=curr_idx,
                side=Side.LONG,
                entry=entry,
                stop=stop,
                tp=tp,
                reason=f"Bull Rev (ATR={risk_dist:.2f})"
            )

    # --- SHORT ---
    elif trend == Side.SHORT:
        # Pullback: Higher Low of Groene bar
        is_pullback = (signal_bar["low"] > prev_bar["low"]) or (signal_bar["close"] > signal_bar["open"])

        bar_range = signal_bar["high"] - signal_bar["low"]
        if bar_range == 0: return None

        close_loc = (signal_bar["high"] - signal_bar["close"]) / bar_range
        is_strong_close = close_loc > (1.0 - p.signal_close_frac)

        if is_pullback and is_strong_close:
            entry = float(m5.iloc[-1]["open"])
            stop = entry + risk_dist
            tp = entry - reward_dist

            if stop <= entry: return None

            return PlannedTrade(
                signal_ts=m5.index[-2],
                execute_ts=curr_idx,
                side=Side.SHORT,
                entry=entry,
                stop=stop,
                tp=tp,
                reason=f"Bear Rev (ATR={risk_dist:.2f})"
            )

    return None

# Noot: plan_h2l2_trades functie is verwijderd/verplaatst naar runner
# omdat de loop nu in de runner zit (voor rolling trend).