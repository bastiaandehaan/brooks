import pandas as pd
from strategies.context import infer_trend_m15, Trend, TrendParams


def _mk(closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    df = pd.DataFrame({"close": closes, "open": closes, "high": closes, "low": closes}, index=idx)
    return df


def test_infer_trend_bull():
    # clear uptrend with enough separation
    closes = [100 + i * 0.5 for i in range(120)]
    df = _mk(closes)
    t, _ = infer_trend_m15(df, TrendParams(min_slope=0.1, min_close_ema_dist=0.1))
    assert t == Trend.BULL


def test_infer_trend_bear():
    closes = [200 - i * 0.5 for i in range(120)]
    df = _mk(closes)
    t, _ = infer_trend_m15(df, TrendParams(min_slope=0.1, min_close_ema_dist=0.1))
    assert t == Trend.BEAR
