# strategies/regime_debug.py
"""
Debug tool to find optimal regime threshold
"""
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from utils.mt5_client import Mt5Client
from utils.mt5_data import fetch_rates, RatesRequest
from strategies.regime import RegimeParams


def calculate_threshold_for_target_choppy_pct(
        m15_data: pd.DataFrame,
        target_choppy_pct: float,
        params: RegimeParams
) -> float:
    """
    Find threshold that gives desired % of choppy bars

    Args:
        m15_data: M15 OHLC data
        target_choppy_pct: Desired % choppy bars (e.g., 7.8)
        params: RegimeParams (for ATR/range periods)

    Returns:
        Threshold value that gives target_choppy_pct
    """
    # Calculate ATR
    high = m15_data["high"].values
    low = m15_data["low"].values
    close = m15_data["close"].values

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.concatenate([[close[0]], close[:-1]])),
            np.abs(low - np.concatenate([[close[0]], close[:-1]]))
        )
    )
    atr = pd.Series(tr, index=m15_data.index).rolling(params.atr_period).mean()

    # Calculate range
    rolling_high = m15_data["high"].rolling(params.range_period).max()
    rolling_low = m15_data["low"].rolling(params.range_period).min()
    price_range = rolling_high - rolling_low

    # Chop ratio
    chop_ratio = (price_range / atr.replace(0, np.nan)).dropna()

    # Find threshold for target percentile
    # If we want 7.8% choppy, we want 92.2 percentile
    target_percentile = 100 - target_choppy_pct
    threshold = np.percentile(chop_ratio, target_percentile)

    # Stats
    stats = {
        "mean": chop_ratio.mean(),
        "median": chop_ratio.median(),
        "std": chop_ratio.std(),
        "p10": np.percentile(chop_ratio, 10),
        "p25": np.percentile(chop_ratio, 25),
        "p75": np.percentile(chop_ratio, 75),
        "p90": np.percentile(chop_ratio, 90),
        "p92": np.percentile(chop_ratio, 92),
        "p95": np.percentile(chop_ratio, 95),
        "optimal_threshold": threshold,
        "bars_total": len(chop_ratio),
    }

    return threshold, stats, chop_ratio


def print_threshold_analysis(stats: dict, chop_ratio: pd.Series, threshold: float):
    """Pretty print threshold analysis"""
    print("\n" + "=" * 80)
    print("  📊 REGIME THRESHOLD ANALYSIS")
    print("=" * 80)
    print(f"\nChop Ratio Statistics:")
    print(f"  Mean      : {stats['mean']:.2f}")
    print(f"  Median    : {stats['median']:.2f}")
    print(f"  Std Dev   : {stats['std']:.2f}")
    print(f"\nPercentiles:")
    print(f"  10th : {stats['p10']:.2f}")
    print(f"  25th : {stats['p25']:.2f}")
    print(f"  75th : {stats['p75']:.2f}")
    print(f"  90th : {stats['p90']:.2f}")
    print(f"  92nd : {stats['p92']:.2f}")
    print(f"  95th : {stats['p95']:.2f}")

    print(f"\n🎯 OPTIMAL THRESHOLD: {threshold:.2f}")

    # Test with different thresholds
    print("\n" + "=" * 80)
    print("  📈 THRESHOLD IMPACT TABLE")
    print("=" * 80)
    print(f"{'Threshold':<12} {'Choppy %':<12} {'Tradable %':<12} {'Bars':<12}")
    print("-" * 80)

    for test_threshold in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0, 9.0, 10.0]:
        choppy_bars = (chop_ratio <= test_threshold).sum()
        choppy_pct = 100 * choppy_bars / len(chop_ratio)
        tradable_pct = 100 - choppy_pct

        marker = " ⭐" if abs(test_threshold - threshold) < 0.5 else ""

        print(f"{test_threshold:<12.1f} {choppy_pct:<12.1f} {tradable_pct:<12.1f} {choppy_bars:<12}{marker}")

    print("=" * 80 + "\n")


def main():
    """Run threshold analysis on live MT5 data"""
    print("\n🔍 REGIME THRESHOLD CALCULATOR")
    print("=" * 80)

    # Connect to MT5
    client = Mt5Client(mt5_module=mt5)
    if not client.initialize():
        print("❌ Failed to connect to MT5")
        return 1

    # Fetch data (340 days worth)
    days = 340
    m15_bars = days * 96 * 2

    print(f"\n→ Fetching {m15_bars} M15 bars ({days} days)...")
    req = RatesRequest("US500.cash", mt5.TIMEFRAME_M15, m15_bars)
    m15_data = fetch_rates(mt5, req)

    print(f"  ✅ Got {len(m15_data)} M15 bars")

    # Calculate optimal threshold for 7.8% choppy
    target_choppy = 7.8
    params = RegimeParams(atr_period=14, range_period=20)

    print(f"\n→ Calculating optimal threshold for {target_choppy}% choppy bars...")
    threshold, stats, chop_ratio = calculate_threshold_for_target_choppy_pct(
        m15_data, target_choppy, params
    )

    # Print analysis
    print_threshold_analysis(stats, chop_ratio, threshold)

    # Recommendations
    print("💡 RECOMMENDATIONS:")
    print(f"   1. Use threshold = {threshold:.1f} for ~{target_choppy}% choppy")
    print(f"   2. Current threshold (2.0) gives ~{100 * (chop_ratio <= 2.0).sum() / len(chop_ratio):.1f}% choppy")
    print(f"   3. For conservative filter (5% choppy), use threshold = {stats['p95']:.1f}")
    print(f"   4. For aggressive filter (10% choppy), use threshold = {stats['p90']:.1f}")

    print("\n🚀 RUN BACKTEST WITH OPTIMAL THRESHOLD:")
    print(f"   python -m backtest.runner --days 340 --regime-filter --chop-threshold {threshold:.1f}")
    print()

    client.shutdown()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())