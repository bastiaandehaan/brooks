# strategies/regime.py
"""
Market Regime Detection - FIXED VERSION
Correctly normalize price range by AVERAGE ATR over the same period
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class MarketRegime(str, Enum):
    """Market state classification"""

    TRENDING = "TRENDING"  # Clear directional move - TRADE
    CHOPPY = "CHOPPY"  # Range-bound, low volatility - SKIP
    UNKNOWN = "UNKNOWN"  # Not enough data


@dataclass(frozen=True)
class RegimeParams:
    """Configuration for regime detection"""

    atr_period: int = 14  # ATR calculation period
    range_period: int = 20  # Price range lookback
    chop_threshold: float = 2.5  # Range must be > (threshold × AVG_ATR) to be trending
    min_bars: int = 50  # Minimum bars needed for valid detection


@dataclass(frozen=True)
class RegimeMetrics:
    """Diagnostics from regime detection"""

    regime: MarketRegime
    price_range: float  # High-Low over range_period
    avg_atr: float  # Average ATR over range_period (KEY FIX!)
    chop_ratio: float  # price_range / (chop_threshold × avg_atr)
    bars_analyzed: int


def detect_regime_series(df: pd.DataFrame, params: RegimeParams) -> pd.Series:
    """
    VECTORIZED regime detection - CORRECTED VERSION

    KEY FIX: Compare price_range to AVERAGE ATR over the same period,
    not to single-bar ATR!

    Returns:
        Series with MarketRegime values (TRENDING/CHOPPY/UNKNOWN)
    """
    if df.empty:
        return pd.Series([], dtype="object", index=df.index)

    # Convert to float
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    # Calculate ATR (bar range)
    bar_range = high - low
    atr = bar_range.rolling(params.atr_period, min_periods=params.atr_period).mean()

    # 🔧 FIX: Average ATR over range_period (same window as price range!)
    avg_atr = atr.rolling(params.range_period, min_periods=params.range_period).mean()

    # Price range over range_period
    range_high = close.rolling(params.range_period, min_periods=params.range_period).max()
    range_low = close.rolling(params.range_period, min_periods=params.range_period).min()
    price_range = range_high - range_low

    # 🔧 FIX: Chop ratio = price_range / (threshold × avg_atr)
    # This normalizes: how many "average volatility units" did price move?
    threshold_range = params.chop_threshold * avg_atr
    chop_ratio = price_range / threshold_range

    # Classify regime
    regime = pd.Series([MarketRegime.UNKNOWN] * len(df), index=df.index, dtype="object")

    # Valid data mask (no NaN in calculations)
    valid_mask = ~(avg_atr.isna() | price_range.isna() | chop_ratio.isna())

    # Where valid AND chop_ratio < 1.0 → CHOPPY (moved less than threshold)
    choppy_mask = valid_mask & (chop_ratio < 1.0)
    regime[choppy_mask] = MarketRegime.CHOPPY

    # Where valid AND chop_ratio >= 1.0 → TRENDING (moved more than threshold)
    trending_mask = valid_mask & (chop_ratio >= 1.0)
    regime[trending_mask] = MarketRegime.TRENDING

    return regime


def detect_regime(df: pd.DataFrame, params: RegimeParams) -> tuple[MarketRegime, RegimeMetrics]:
    """
    Detect regime for LAST bar in dataframe (for main.py live trading)

    CORRECTED: Uses average ATR over range_period for proper normalization
    """
    min_required = params.atr_period + params.range_period  # 34 bars

    if len(df) < min_required:
        return MarketRegime.UNKNOWN, RegimeMetrics(
            regime=MarketRegime.UNKNOWN,
            price_range=0.0,
            avg_atr=0.0,
            chop_ratio=0.0,
            bars_analyzed=len(df),
        )

    # Convert to float
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    # Calculate ATR
    bar_range = high - low
    atr = bar_range.rolling(params.atr_period).mean()

    # 🔧 FIX: Average ATR over range_period
    avg_atr = atr.rolling(params.range_period).mean().iloc[-1]

    # Price range
    range_high = close.rolling(params.range_period).max().iloc[-1]
    range_low = close.rolling(params.range_period).min().iloc[-1]
    price_range = range_high - range_low

    # Check for NaN
    if pd.isna(avg_atr) or pd.isna(price_range) or avg_atr <= 0:
        return MarketRegime.UNKNOWN, RegimeMetrics(
            regime=MarketRegime.UNKNOWN,
            price_range=0.0,
            avg_atr=0.0,
            chop_ratio=0.0,
            bars_analyzed=len(df),
        )

    # 🔧 FIX: Chop ratio with normalized threshold
    threshold_range = params.chop_threshold * avg_atr
    chop_ratio = price_range / threshold_range

    # Classify
    if chop_ratio < 1.0:
        regime = MarketRegime.CHOPPY
    else:
        regime = MarketRegime.TRENDING

    metrics = RegimeMetrics(
        regime=regime,
        price_range=price_range,
        avg_atr=avg_atr,
        chop_ratio=chop_ratio,
        bars_analyzed=len(df),
    )

    return regime, metrics


def should_trade_today(df: pd.DataFrame, params: RegimeParams) -> tuple[bool, str]:
    """
    Simple decision: should we trade today?

    Returns:
        (should_trade, reason) tuple
    """
    regime, metrics = detect_regime(df, params)

    if regime == MarketRegime.UNKNOWN:
        return (
            False,
            f"Not enough data ({metrics.bars_analyzed} bars, need {params.atr_period + params.range_period})",
        )

    if regime == MarketRegime.CHOPPY:
        return False, (
            f"Market is CHOPPY (range={metrics.price_range:.1f}, "
            f"ATR={metrics.avg_atr:.1f}, ratio={metrics.chop_ratio:.2f}). "
            "Waiting for trending conditions."
        )

    return True, f"Market is TRENDING (chop_ratio={metrics.chop_ratio:.2f} > 1.0). Safe to trade."


def filter_trading_days(
    daily_bars: list[pd.DataFrame], params: RegimeParams
) -> list[tuple[pd.DataFrame, bool, str]]:
    """
    Filter which days to trade based on regime

    Args:
        daily_bars: List of dataframes, one per day
        params: Regime detection config

    Returns:
        List of (df, should_trade, reason) tuples
    """
    results = []
    for df in daily_bars:
        trade, reason = should_trade_today(df, params)
        results.append((df, trade, reason))
    return results
