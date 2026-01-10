import pandas as pd
from strategies.context import infer_trend_m15, Trend, TrendParams


def _mk(closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    df = pd.DataFrame({"close": closes, "open": closes, "high": closes, "low": closes}, index=idx)
    return df


def test_infer_trend_bull():
    df = _mk(list(range(1, 80)))
    t = infer_trend_m15(df, TrendParams(min_slope=0.0))
    assert t == Trend.BULL


def test_infer_trend_bear():
    df = _mk(list(range(80, 0, -1)))
    t = infer_trend_m15(df, TrendParams(min_slope=0.0))
    assert t == Trend.BEAR
