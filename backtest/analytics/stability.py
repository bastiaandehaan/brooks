# backtest/analytics/stability.py
"""
Temporal Stability Analysis

Analyzes performance consistency over time to detect:
- Performance decay/improvement
- Regime dependency
- Seasonal patterns
- Strategy degradation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PeriodPerformance:
    """Performance metrics for a time period"""

    period: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    trades: int
    net_r: float
    avg_r: float
    winrate: float
    profit_factor: float
    max_dd_r: float

    # Quality metrics
    sharpe: float
    avg_win_r: float
    avg_loss_r: float


@dataclass
class StabilityMetrics:
    """Temporal stability analysis results"""

    # Period breakdown
    periods: List[PeriodPerformance]

    # Stability metrics
    avg_r_std: float  # Std of period avg_r (lower = more stable)
    avg_r_cv: float  # Coefficient of variation

    # Trend analysis
    performance_trend: str  # "IMPROVING", "STABLE", "DEGRADING"
    trend_slope: float  # Linear regression slope of avg_r over time

    # Consistency
    positive_periods: int
    negative_periods: int
    pct_positive_periods: float


def analyze_stability(
    trades_df: pd.DataFrame,
    *,
    period: str = "Q",  # Q=quarter, M=month, Y=year
    min_trades_per_period: int = 5,
) -> StabilityMetrics:
    """
    Analyze performance stability across time periods.

    Args:
        trades_df: DataFrame with trades (must have 'entry_time', 'net_r')
        period: Pandas frequency string ('Q', 'M', 'Y')
        min_trades_per_period: Skip periods with fewer trades

    Returns:
        StabilityMetrics object
    """
    if trades_df.empty:
        raise ValueError("Cannot analyze empty trades DataFrame")

    if "entry_time" not in trades_df.columns or "net_r" not in trades_df.columns:
        raise ValueError("trades_df must contain 'entry_time' and 'net_r'")

    # Ensure datetime
    df = trades_df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df = df.sort_values("entry_time")

    # Group by period
    df["period"] = df["entry_time"].dt.to_period(period)

    periods: List[PeriodPerformance] = []

    for period_val, group in df.groupby("period"):
        if len(group) < min_trades_per_period:
            logger.debug(f"Skipping {period_val}: only {len(group)} trades")
            continue

        results = group["net_r"].values

        n_trades = len(results)
        net_r = float(results.sum())
        avg_r = float(results.mean())

        wins = results[results > 0]
        losses = results[results < 0]

        winrate = len(wins) / n_trades if n_trades > 0 else 0.0

        total_wins = float(wins.sum()) if len(wins) > 0 else 0.0
        total_losses = float(abs(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float("inf")

        # Calculate max DD for this period
        equity = np.cumsum(results)
        running_max = np.maximum.accumulate(equity)
        drawdown = equity - running_max
        max_dd_r = float(drawdown.min())

        # Sharpe (trade-level)
        std = float(results.std(ddof=1)) if len(results) > 1 else 0.0
        sharpe = (avg_r / std) if std > 0 else 0.0

        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

        periods.append(
            PeriodPerformance(
                period=str(period_val),
                start_date=group["entry_time"].min(),
                end_date=group["entry_time"].max(),
                trades=n_trades,
                net_r=net_r,
                avg_r=avg_r,
                winrate=winrate,
                profit_factor=profit_factor,
                max_dd_r=max_dd_r,
                sharpe=sharpe,
                avg_win_r=avg_win,
                avg_loss_r=avg_loss,
            )
        )

    if not periods:
        raise ValueError("No valid periods found (increase data or reduce min_trades_per_period)")

    # Calculate stability metrics
    avg_r_values = [p.avg_r for p in periods]

    avg_r_std = float(np.std(avg_r_values, ddof=1))
    avg_r_mean = float(np.mean(avg_r_values))
    avg_r_cv = abs(avg_r_std / avg_r_mean) if avg_r_mean != 0 else float("inf")

    # Trend analysis (simple linear regression)
    x = np.arange(len(periods))
    y = np.array(avg_r_values)

    if len(x) > 1:
        trend_slope = float(np.polyfit(x, y, 1)[0])
    else:
        trend_slope = 0.0

    # Classify trend
    if trend_slope > 0.001:  # Improving
        performance_trend = "IMPROVING"
    elif trend_slope < -0.001:  # Degrading
        performance_trend = "DEGRADING"
    else:
        performance_trend = "STABLE"

    # Consistency
    positive_periods = sum(1 for p in periods if p.net_r > 0)
    negative_periods = len(periods) - positive_periods
    pct_positive = (positive_periods / len(periods) * 100.0) if periods else 0.0

    return StabilityMetrics(
        periods=periods,
        avg_r_std=avg_r_std,
        avg_r_cv=avg_r_cv,
        performance_trend=performance_trend,
        trend_slope=trend_slope,
        positive_periods=positive_periods,
        negative_periods=negative_periods,
        pct_positive_periods=pct_positive,
    )


def print_stability_report(metrics: StabilityMetrics) -> None:
    """Print formatted stability analysis report."""

    print("\n" + "=" * 80)
    print("  📅 TEMPORAL STABILITY ANALYSIS")
    print("=" * 80)

    print(f"\n📊 PERIOD BREAKDOWN ({len(metrics.periods)} periods):")
    print(
        f"  {'Period':<12} {'Trades':>7} {'Net R':>9} {'Avg R':>9} {'WR%':>7} {'PF':>7} {'MaxDD':>9}"
    )
    print("  " + "-" * 78)

    for p in metrics.periods:
        print(
            f"  {p.period:<12} {p.trades:7d} {p.net_r:+9.2f} {p.avg_r:+9.4f} {p.winrate * 100:7.1f} {p.profit_factor:7.2f} {p.max_dd_r:9.2f}"
        )

    print(f"\n📈 STABILITY METRICS:")
    print(f"  Avg R Std Dev     : {metrics.avg_r_std:.4f}R")
    print(f"  Coeff of Variation: {metrics.avg_r_cv:.3f}  ", end="")
    if metrics.avg_r_cv < 0.5:
        print("(LOW - very consistent)")
    elif metrics.avg_r_cv < 1.0:
        print("(MODERATE - acceptable variation)")
    else:
        print("(HIGH - inconsistent)")

    print(f"\n📉 TREND ANALYSIS:")
    print(f"  Performance trend : {metrics.performance_trend}")
    print(f"  Trend slope       : {metrics.trend_slope:+.6f}R per period")

    if metrics.performance_trend == "IMPROVING":
        print("  ✅ Strategy appears to be improving over time")
        print("     → Recent periods outperform earlier ones")
    elif metrics.performance_trend == "DEGRADING":
        print("  ⚠️  Strategy shows performance decay")
        print("     → Recent periods underperform earlier ones")
        print("     → May indicate market adaptation or overfitting")
    else:
        print("  ✅ Strategy performance is stable over time")
        print("     → No significant trend detected")

    print(f"\n🎯 CONSISTENCY:")
    print(f"  Positive periods  : {metrics.positive_periods} ({metrics.pct_positive_periods:.1f}%)")
    print(f"  Negative periods  : {metrics.negative_periods}")

    if metrics.pct_positive_periods >= 70:
        print("  ✅ HIGH CONSISTENCY: >70% positive periods")
    elif metrics.pct_positive_periods >= 50:
        print("  ⚠️  MODERATE: 50-70% positive periods")
    else:
        print("  🔴 LOW CONSISTENCY: <50% positive periods")
        print("     → Strategy has many losing periods")

    # Best/worst periods
    best_period = max(metrics.periods, key=lambda p: p.net_r)
    worst_period = min(metrics.periods, key=lambda p: p.net_r)

    print(f"\n🏆 BEST PERIOD:")
    print(f"  {best_period.period}: {best_period.net_r:+.2f}R from {best_period.trades} trades")
    print(f"  WR: {best_period.winrate * 100:.1f}%, PF: {best_period.profit_factor:.2f}")

    print(f"\n💀 WORST PERIOD:")
    print(f"  {worst_period.period}: {worst_period.net_r:+.2f}R from {worst_period.trades} trades")
    print(f"  WR: {worst_period.winrate * 100:.1f}%, PF: {worst_period.profit_factor:.2f}")

    print("\n" + "=" * 80)


def compare_early_vs_late(
    trades_df: pd.DataFrame,
    *,
    split_pct: float = 0.5,
) -> Dict[str, float]:
    """
    Compare early vs late performance (simple in-sample vs out-of-sample).

    Args:
        trades_df: DataFrame with trades
        split_pct: Where to split (0.5 = 50/50)

    Returns:
        Dict with comparison metrics
    """
    df = trades_df.copy()
    df = df.sort_values("entry_time")

    split_idx = int(len(df) * split_pct)

    early = df.iloc[:split_idx]
    late = df.iloc[split_idx:]

    early_net_r = float(early["net_r"].sum())
    late_net_r = float(late["net_r"].sum())

    early_avg_r = float(early["net_r"].mean())
    late_avg_r = float(late["net_r"].mean())

    early_wr = float((early["net_r"] > 0).mean())
    late_wr = float((late["net_r"] > 0).mean())

    return {
        "early_trades": len(early),
        "late_trades": len(late),
        "early_net_r": early_net_r,
        "late_net_r": late_net_r,
        "early_avg_r": early_avg_r,
        "late_avg_r": late_avg_r,
        "early_wr": early_wr,
        "late_wr": late_wr,
        "avg_r_delta": late_avg_r - early_avg_r,
        "wr_delta": late_wr - early_wr,
    }


# Example usage
if __name__ == "__main__":
    # Simulate trade data
    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=328, freq="D")

    # Simulate slight performance improvement over time
    time_factor = np.linspace(0, 0.1, 328)
    results = np.random.normal(0.15, 1.0, 328) + time_factor

    trades = pd.DataFrame(
        {
            "entry_time": dates,
            "net_r": results,
        }
    )

    # Analyze
    metrics = analyze_stability(trades, period="Q", min_trades_per_period=10)
    print_stability_report(metrics)

    # Compare early vs late
    print("\n" + "=" * 80)
    print("  🔄 EARLY vs LATE COMPARISON")
    print("=" * 80)

    comparison = compare_early_vs_late(trades)
    print(f"\n  Early (first 50%):")
    print(f"    Trades  : {comparison['early_trades']}")
    print(f"    Net R   : {comparison['early_net_r']:+.2f}R")
    print(f"    Avg R   : {comparison['early_avg_r']:+.4f}R")
    print(f"    Winrate : {comparison['early_wr'] * 100:.1f}%")

    print(f"\n  Late (last 50%):")
    print(f"    Trades  : {comparison['late_trades']}")
    print(f"    Net R   : {comparison['late_net_r']:+.2f}R")
    print(f"    Avg R   : {comparison['late_avg_r']:+.4f}R")
    print(f"    Winrate : {comparison['late_wr'] * 100:.1f}%")

    print(f"\n  Delta:")
    print(f"    Avg R   : {comparison['avg_r_delta']:+.4f}R")
    print(f"    Winrate : {comparison['wr_delta'] * 100:+.1f}%")

    if comparison["avg_r_delta"] > 0:
        print("\n  ✅ Late performance better than early (good sign)")
    else:
        print("\n  ⚠️  Late performance worse than early (possible decay)")

    print("=" * 80)
