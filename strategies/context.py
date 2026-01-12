# strategies/context.py
"""
Brooks Trend Filter - SIMPEL en EFFECTIEF
Regel: Trade met de trend. Trend = EMA richting + prijs positie.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Trend(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"


@dataclass(frozen=True)
class TrendParams:
    ema_period: int = 20
    ema_slope_bars: int = 5  # kijk 5 bars terug voor richting
    min_slope: float = 0.10  # minimaal 0.10 punten per 5 bars (filters range)
    # Brooks: "Trade met de trend. Maar alleen als trend DUIDELIJK is."


@dataclass(frozen=True)
class TrendMetrics:
    last_close: float
    last_ema: float
    ema_slope: float
    close_above_ema: bool


def infer_trend_m15(m15: pd.DataFrame, p: TrendParams) -> tuple[Optional[Trend], TrendMetrics]:
    """
    Brooks Trend Rule (simpel):
    - BULL: close > EMA EN EMA stijgend
    - BEAR: close < EMA EN EMA dalend
    - Geen extra filters. Period.
    """
    need = p.ema_period + p.ema_slope_bars
    if len(m15) < need:
        close = float(m15["close"].iloc[-1]) if len(m15) else 0.0
        metrics = TrendMetrics(close, close, 0.0, False)
        return None, metrics

    close = m15["close"].astype(float)
    ema = close.ewm(span=p.ema_period, adjust=False).mean()

    last_close = float(close.iloc[-1])
    last_ema = float(ema.iloc[-1])
    prev_ema = float(ema.iloc[-1 - p.ema_slope_bars])

    ema_slope = last_ema - prev_ema
    close_above_ema = last_close > last_ema

    metrics = TrendMetrics(
        last_close=last_close,
        last_ema=last_ema,
        ema_slope=ema_slope,
        close_above_ema=close_above_ema,
    )

    # Brooks regel: simpel maar met DUIDELIJKE trend requirement
    if close_above_ema and ema_slope >= p.min_slope:
        return Trend.BULL, metrics

    if not close_above_ema and ema_slope <= -p.min_slope:
        return Trend.BEAR, metrics

    return None, metrics


def infer_trend_m15_series(m15: pd.DataFrame, p: TrendParams) -> pd.Series:
    """
    Vectorized versie voor backtest runner (O(n) performance).
    Zelfde logica als infer_trend_m15, maar voor alle bars tegelijk.
    """
    if m15.empty:
        return pd.Series([], dtype="object", index=m15.index)

    close = m15["close"].astype(float)
    ema = close.ewm(span=p.ema_period, adjust=False).mean()

    ema_slope = ema - ema.shift(p.ema_slope_bars)
    close_above_ema = close > ema

    # Bull: close > EMA EN slope >= min_slope
    bull = close_above_ema & (ema_slope >= p.min_slope)

    # Bear: close < EMA EN slope <= -min_slope
    bear = (~close_above_ema) & (ema_slope <= -p.min_slope)

    out = pd.Series([None] * len(m15), index=m15.index, dtype="object")
    out[bull] = Trend.BULL
    out[bear] = Trend.BEAR

    # Warmup: eerste bars zijn unreliable
    need = p.ema_period + p.ema_slope_bars
    if len(out) >= need:
        out.iloc[: need - 1] = None

    return out