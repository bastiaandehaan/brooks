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
    confirm_bars: int = 20
    min_above_frac: float = 0.60
    min_close_ema_dist: float = 0.50     # avoid "touching EMA" noise
    min_slope: float = 0.20              # price units over slope_lookback bars
    pullback_allowance: float = 2.00     # allow last close slightly on wrong side (US500 points)


@dataclass(frozen=True)
class TrendMetrics:
    last_close: float
    last_ema: float
    ema_slope: float
    above_frac: float
    below_frac: float
    close_ema_dist: float
    last_close_minus_ema: float


def infer_trend_m15(m15: pd.DataFrame, p: TrendParams) -> tuple[Optional[Trend], TrendMetrics]:
    need = max(p.ema_period, p.slope_lookback, p.confirm_bars) + 2
    if len(m15) < need:
        close = float(m15["close"].iloc[-1]) if len(m15) else 0.0
        metrics = TrendMetrics(close, close, 0.0, 0.0, 0.0, 0.0, 0.0)
        return None, metrics

    close = m15["close"].astype(float)
    ema = close.ewm(span=p.ema_period, adjust=False).mean()

    last_close = float(close.iloc[-1])
    last_ema = float(ema.iloc[-1])
    last_close_minus_ema = last_close - last_ema

    ema_slope = float(ema.iloc[-1] - ema.iloc[-1 - p.slope_lookback])

    w = p.confirm_bars
    above_frac = float((close.iloc[-w:] > ema.iloc[-w:]).mean())
    below_frac = float((close.iloc[-w:] < ema.iloc[-w:]).mean())

    dist = abs(last_close_minus_ema)

    metrics = TrendMetrics(
        last_close=last_close,
        last_ema=last_ema,
        ema_slope=ema_slope,
        above_frac=above_frac,
        below_frac=below_frac,
        close_ema_dist=float(dist),
        last_close_minus_ema=float(last_close_minus_ema),
    )

    # Bull: slope up + majority above EMA + not "touching EMA" + allow shallow pullback below EMA
    bull_ok = (
        ema_slope >= p.min_slope
        and above_frac >= p.min_above_frac
        and dist >= p.min_close_ema_dist
        and (last_close_minus_ema >= 0 or abs(last_close_minus_ema) <= p.pullback_allowance)
    )

    # Bear: slope down + majority below EMA + not "touching EMA" + allow shallow pullback above EMA
    bear_ok = (
        ema_slope <= -p.min_slope
        and below_frac >= p.min_above_frac
        and dist >= p.min_close_ema_dist
        and (last_close_minus_ema <= 0 or abs(last_close_minus_ema) <= p.pullback_allowance)
    )

    if bull_ok and not bear_ok:
        return Trend.BULL, metrics
    if bear_ok and not bull_ok:
        return Trend.BEAR, metrics

    return None, metrics
