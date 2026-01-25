# backtest/analytics/distribution.py
"""
Trade Result Distribution Analysis

Analyzes the distribution of trade outcomes to detect:
- Tail dependency (is profit concentrated in few winners?)
- Skewness (asymmetric risk/reward)
- Outlier impact
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DistributionMetrics:
    """Statistical distribution metrics for trade results"""

    # Basic stats
    mean: float
    median: float
    std: float
    skewness: float
    kurtosis: float

    # Percentiles
    p01: float
    p05: float
    p10: float
    p25: float
    p75: float
    p90: float
    p95: float
    p99: float

    # Tail analysis
    top_10pct_contribution: float  # % of total profit from top 10% trades
    top_5pct_contribution: float
    worst_10pct_impact: float  # % of total loss from worst 10% trades

    # Outlier detection
    best_trade: float
    worst_trade: float
    best_5_sum: float
    worst_5_sum: float

    # Tail dependency ratio
    tail_dependency_score: float  # >0.5 = profit concentrated in tails


def analyze_distribution(results_r: pd.Series) -> DistributionMetrics:
    """
    Analyze trade result distribution.

    Args:
        results_r: Series of R-values per trade

    Returns:
        DistributionMetrics object
    """
    results = results_r.values
    n_trades = len(results)

    if n_trades == 0:
        raise ValueError("Cannot analyze empty result set")

    # Basic statistics
    mean = float(np.mean(results))
    median = float(np.median(results))
    std = float(np.std(results, ddof=1))

    # Higher moments
    skewness = float(pd.Series(results).skew())
    kurtosis = float(pd.Series(results).kurtosis())

    # Percentiles
    p01 = float(np.percentile(results, 1))
    p05 = float(np.percentile(results, 5))
    p10 = float(np.percentile(results, 10))
    p25 = float(np.percentile(results, 25))
    p75 = float(np.percentile(results, 75))
    p90 = float(np.percentile(results, 90))
    p95 = float(np.percentile(results, 95))
    p99 = float(np.percentile(results, 99))

    # Sort for tail analysis
    sorted_results = np.sort(results)

    # Best/worst trades
    best_trade = float(sorted_results[-1])
    worst_trade = float(sorted_results[0])
    best_5_sum = float(sorted_results[-5:].sum())
    worst_5_sum = float(sorted_results[:5].sum())

    # Tail contribution analysis
    total_profit = float(results[results > 0].sum())
    total_loss = float(abs(results[results < 0].sum()))

    # Top 10% trades contribution
    top_10pct_count = max(1, int(n_trades * 0.1))
    top_10pct_sum = float(sorted_results[-top_10pct_count:].sum())
    top_10pct_contribution = (top_10pct_sum / total_profit * 100.0) if total_profit > 0 else 0.0

    # Top 5% trades contribution
    top_5pct_count = max(1, int(n_trades * 0.05))
    top_5pct_sum = float(sorted_results[-top_5pct_count:].sum())
    top_5pct_contribution = (top_5pct_sum / total_profit * 100.0) if total_profit > 0 else 0.0

    # Worst 10% trades impact
    worst_10pct_count = max(1, int(n_trades * 0.1))
    worst_10pct_sum = float(abs(sorted_results[:worst_10pct_count].sum()))
    worst_10pct_impact = (worst_10pct_sum / total_loss * 100.0) if total_loss > 0 else 0.0

    # Tail dependency score
    # If top 10% contributes >50% of profit, tail dependency is high
    # If worst 10% contributes >50% of losses, risk is concentrated
    tail_dependency_score = (top_10pct_contribution + worst_10pct_impact) / 200.0

    return DistributionMetrics(
        mean=mean,
        median=median,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis,
        p01=p01,
        p05=p05,
        p10=p10,
        p25=p25,
        p75=p75,
        p90=p90,
        p95=p95,
        p99=p99,
        top_10pct_contribution=top_10pct_contribution,
        top_5pct_contribution=top_5pct_contribution,
        worst_10pct_impact=worst_10pct_impact,
        best_trade=best_trade,
        worst_trade=worst_trade,
        best_5_sum=best_5_sum,
        worst_5_sum=worst_5_sum,
        tail_dependency_score=tail_dependency_score,
    )


def print_distribution_report(metrics: DistributionMetrics) -> None:
    """Print formatted distribution analysis report."""

    print("\n" + "=" * 80)
    print("  📊 TRADE RESULT DISTRIBUTION ANALYSIS")
    print("=" * 80)

    print("\n📈 CENTRAL TENDENCY:")
    print(f"  Mean          : {metrics.mean:+.4f}R")
    print(f"  Median        : {metrics.median:+.4f}R")
    print(f"  Std Dev       : {metrics.std:.4f}R")

    print("\n📐 SHAPE:")
    print(f"  Skewness      : {metrics.skewness:+.3f}  ", end="")
    if metrics.skewness > 0.5:
        print("(RIGHT-SKEWED: big winners exist)")
    elif metrics.skewness < -0.5:
        print("(LEFT-SKEWED: big losers exist)")
    else:
        print("(SYMMETRIC)")

    print(f"  Kurtosis      : {metrics.kurtosis:+.3f}  ", end="")
    if metrics.kurtosis > 3:
        print("(FAT TAILS: extreme outcomes likely)")
    elif metrics.kurtosis < -1:
        print("(THIN TAILS: consistent outcomes)")
    else:
        print("(NORMAL-LIKE)")

    print("\n📊 PERCENTILES:")
    print(f"  P01 (worst 1%): {metrics.p01:+.4f}R")
    print(f"  P05 (VaR 95%)  : {metrics.p05:+.4f}R")
    print(f"  P10            : {metrics.p10:+.4f}R")
    print(f"  P25            : {metrics.p25:+.4f}R")
    print(f"  P50 (median)   : {metrics.median:+.4f}R")
    print(f"  P75            : {metrics.p75:+.4f}R")
    print(f"  P90            : {metrics.p90:+.4f}R")
    print(f"  P95            : {metrics.p95:+.4f}R")
    print(f"  P99 (best 1%)  : {metrics.p99:+.4f}R")

    print("\n🎯 TAIL CONCENTRATION:")
    print(f"  Top 10% trades contribute: {metrics.top_10pct_contribution:.1f}% of total profit")
    print(f"  Top 5% trades contribute : {metrics.top_5pct_contribution:.1f}% of total profit")
    print(f"  Worst 10% trades are     : {metrics.worst_10pct_impact:.1f}% of total losses")

    print("\n⚠️  TAIL DEPENDENCY SCORE:")
    print(f"  Score: {metrics.tail_dependency_score:.3f}  ", end="")
    if metrics.tail_dependency_score > 0.6:
        print("(HIGH - profit concentrated in few trades)")
        print("  ⚠️  WARNING: Strategy relies heavily on catching big moves")
        print("  ⚠️  Risk: Missing one tail trade significantly impacts results")
    elif metrics.tail_dependency_score > 0.4:
        print("(MODERATE - some concentration)")
        print("  ℹ️  Strategy benefits from outliers but not critically dependent")
    else:
        print("(LOW - profit well-distributed)")
        print("  ✅ Strategy has consistent edge across many trades")

    print("\n🏆 OUTLIERS:")
    print(f"  Best trade    : {metrics.best_trade:+.4f}R")
    print(f"  Worst trade   : {metrics.worst_trade:+.4f}R")
    print(f"  Best 5 sum    : {metrics.best_5_sum:+.4f}R")
    print(f"  Worst 5 sum   : {metrics.worst_5_sum:+.4f}R")

    print("\n" + "=" * 80)


def create_distribution_histogram_data(
    results_r: pd.Series, bins: int = 50
) -> Dict[str, np.ndarray]:
    """
    Create histogram data for plotting (separate from visualization).

    Returns:
        Dict with 'counts', 'edges', 'centers' arrays
    """
    counts, edges = np.histogram(results_r.values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    return {
        "counts": counts,
        "edges": edges,
        "centers": centers,
    }


# Example usage
if __name__ == "__main__":
    # Simulate some trade results
    np.random.seed(42)

    # Create realistic distribution (mix of small losses, small wins, rare big wins)
    base_trades = np.random.normal(0.1, 0.8, 900)  # Mostly small positive
    outliers = np.random.choice([3.0, 4.0, 5.0], 50)  # Big winners
    bad_losses = np.random.choice([-1.5, -2.0], 50)  # Some big losses

    results = pd.Series(np.concatenate([base_trades, outliers, bad_losses]))

    # Analyze
    metrics = analyze_distribution(results)
    print_distribution_report(metrics)
