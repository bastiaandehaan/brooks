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
    slope_lookback: int = 10
    min_slope: float = 0.0  # keep simple; can tune later


def infer_trend_m15(m15: pd.DataFrame, p: TrendParams) -> Optional[Trend]:
    """
    Objective MVP trend:
    - EMA(20)
    - close above EMA + EMA slope up => BULL
    - close below EMA + EMA slope down => BEAR
    Else None (skip trading)
    """
    if len(m15) < max(p.ema_period, p.slope_lookback) + 2:
        return None

    close = m15["close"].astype(float)
    ema = close.ewm(span=p.ema_period, adjust=False).mean()

    slope = float(ema.iloc[-1] - ema.iloc[-1 - p.slope_lookback])
    last_close = float(close.iloc[-1])
    last_ema = float(ema.iloc[-1])

    if last_close > last_ema and slope > p.min_slope:
        return Trend.BULL
    if last_close < last_ema and slope < -p.min_slope:
        return Trend.BEAR
    return None
