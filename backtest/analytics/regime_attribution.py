# backtest/analytics/regime_attribution.py
"""
Regime Attribution Analysis

Breaks down performance by market regime to answer:
- Does the edge come from the strategy or the regime filter?
- Which regimes are profitable/unprofitable?
- How much does regime filtering improve results?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimePerformance:
    """Performance metrics for a single regime"""

    regime: str
    trades: int
    net_r: float
    avg_r: float
    winrate: float
    profit_factor: float
    avg_win_r: float
    avg_loss_r: float
    best_trade_r: float
    worst_trade_r: float

    # Contribution to total
    pct_of_total_trades: float
    pct_of_total_profit: float


@dataclass
class RegimeAttribution:
    """Complete regime attribution analysis"""

    # Overall (if no regime filter was used)
    total_trades: int
    total_net_r: float

    # Per-regime breakdown
    regimes: Dict[str, RegimePerformance]

    # Regime filter effectiveness
    filter_enabled: bool
    trades_skipped: int
    skipped_would_have_made: float  # Hypothetical R if traded anyway


def analyze_regime_attribution(
    trades_df: pd.DataFrame,
    *,
    regime_column: str = "regime_at_entry",
    filter_enabled: bool = True,
) -> RegimeAttribution:
    """
    Analyze performance attribution by market regime.

    Args:
        trades_df: DataFrame with trades (must have 'net_r' and regime column)
        regime_column: Column name containing regime labels
        filter_enabled: Whether regime filter was active

    Returns:
        RegimeAttribution object
    """
    if trades_df.empty:
        raise ValueError("Cannot analyze empty trades DataFrame")

    if "net_r" not in trades_df.columns:
        raise ValueError("trades_df must contain 'net_r' column")

    total_trades = len(trades_df)
    total_net_r = float(trades_df["net_r"].sum())

    # Check if regime data exists
    if regime_column not in trades_df.columns:
        logger.warning(f"Column '{regime_column}' not found - regime analysis skipped")
        return RegimeAttribution(
            total_trades=total_trades,
            total_net_r=total_net_r,
            regimes={},
            filter_enabled=False,
            trades_skipped=0,
            skipped_would_have_made=0.0,
        )

    # Analyze each regime
    regimes: Dict[str, RegimePerformance] = {}

    for regime_val in trades_df[regime_column].dropna().unique():
        regime_trades = trades_df[trades_df[regime_column] == regime_val]

        if len(regime_trades) == 0:
            continue

        results = regime_trades["net_r"].values

        n_trades = len(results)
        net_r = float(results.sum())
        avg_r = float(results.mean())

        wins = results[results > 0]
        losses = results[results < 0]

        winrate = len(wins) / n_trades if n_trades > 0 else 0.0

        total_wins = float(wins.sum()) if len(wins) > 0 else 0.0
        total_losses = float(abs(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float("inf")

        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

        best_trade = float(results.max())
        worst_trade = float(results.min())

        pct_of_total_trades = (n_trades / total_trades) * 100.0
        pct_of_total_profit = (net_r / total_net_r * 100.0) if total_net_r != 0 else 0.0

        regimes[str(regime_val)] = RegimePerformance(
            regime=str(regime_val),
            trades=n_trades,
            net_r=net_r,
            avg_r=avg_r,
            winrate=winrate,
            profit_factor=profit_factor,
            avg_win_r=avg_win,
            avg_loss_r=avg_loss,
            best_trade_r=best_trade,
            worst_trade_r=worst_trade,
            pct_of_total_trades=pct_of_total_trades,
            pct_of_total_profit=pct_of_total_profit,
        )

    # Estimate trades skipped (would need full candidate data)
    # For now, use placeholder
    trades_skipped = 0
    skipped_would_have_made = 0.0

    return RegimeAttribution(
        total_trades=total_trades,
        total_net_r=total_net_r,
        regimes=regimes,
        filter_enabled=filter_enabled,
        trades_skipped=trades_skipped,
        skipped_would_have_made=skipped_would_have_made,
    )


def print_regime_attribution_report(attr: RegimeAttribution) -> None:
    """Print formatted regime attribution report."""

    print("\n" + "=" * 80)
    print("  🌍 REGIME ATTRIBUTION ANALYSIS")
    print("=" * 80)

    if not attr.regimes:
        print("\n  ℹ️  No regime data available")
        print(f"  Total trades: {attr.total_trades}")
        print(f"  Total Net R : {attr.total_net_r:+.2f}R")
        print("=" * 80)
        return

    print(f"\n📊 OVERALL:")
    print(f"  Total trades  : {attr.total_trades}")
    print(f"  Total Net R   : {attr.total_net_r:+.2f}R")
    print(f"  Filter active : {'YES' if attr.filter_enabled else 'NO'}")

    # Sort regimes by Net R contribution
    sorted_regimes = sorted(attr.regimes.values(), key=lambda r: r.net_r, reverse=True)

    print(f"\n📈 PERFORMANCE BY REGIME:")
    print(
        f"  {'Regime':<15} {'Trades':>7} {'Net R':>9} {'Avg R':>9} {'WR%':>7} {'PF':>7} {'% Trades':>9} {'% Profit':>9}"
    )
    print("  " + "-" * 78)

    for regime in sorted_regimes:
        print(
            f"  {regime.regime:<15} {regime.trades:7d} {regime.net_r:+9.2f} {regime.avg_r:+9.4f} {regime.winrate * 100:7.1f} {regime.profit_factor:7.2f} {regime.pct_of_total_trades:8.1f}% {regime.pct_of_total_profit:+8.1f}%"
        )

    # Detailed breakdown
    print(f"\n🔍 DETAILED REGIME BREAKDOWN:")

    for regime in sorted_regimes:
        print(f"\n  📍 {regime.regime.upper()}")
        print(f"     Trades        : {regime.trades} ({regime.pct_of_total_trades:.1f}% of total)")
        print(
            f"     Net R         : {regime.net_r:+.2f}R ({regime.pct_of_total_profit:+.1f}% of total profit)"
        )
        print(f"     Avg R/trade   : {regime.avg_r:+.4f}R")
        print(f"     Winrate       : {regime.winrate * 100:.1f}%")
        print(f"     Profit Factor : {regime.profit_factor:.2f}")
        print(f"     Avg Win       : {regime.avg_win_r:+.4f}R")
        print(f"     Avg Loss      : {regime.avg_loss_r:+.4f}R")
        print(f"     Best trade    : {regime.best_trade_r:+.4f}R")
        print(f"     Worst trade   : {regime.worst_trade_r:+.4f}R")

    # Key insights
    print(f"\n💡 KEY INSIGHTS:")

    # Find best/worst performing regimes
    best_regime = max(sorted_regimes, key=lambda r: r.avg_r)
    worst_regime = min(sorted_regimes, key=lambda r: r.avg_r)

    print(f"  Best regime   : {best_regime.regime} ({best_regime.avg_r:+.4f}R avg)")
    print(f"  Worst regime  : {worst_regime.regime} ({worst_regime.avg_r:+.4f}R avg)")

    # Check if any regime is significantly negative
    negative_regimes = [r for r in sorted_regimes if r.net_r < 0]
    if negative_regimes:
        print(f"\n  ⚠️  NEGATIVE REGIMES:")
        for regime in negative_regimes:
            print(f"     {regime.regime}: {regime.net_r:.2f}R from {regime.trades} trades")
            print(f"     → Consider strengthening filter or avoiding this regime")

    # Check regime filter effectiveness
    if attr.filter_enabled and len(sorted_regimes) > 1:
        print(f"\n  ✅ REGIME FILTER EFFECTIVENESS:")

        # If we have CHOPPY vs TRENDING data
        if "CHOPPY" in attr.regimes and "TRENDING" in attr.regimes:
            choppy = attr.regimes["CHOPPY"]
            trending = attr.regimes["TRENDING"]

            print(f"     TRENDING : {trending.net_r:+.2f}R from {trending.trades} trades")
            print(f"     CHOPPY   : {choppy.net_r:+.2f}R from {choppy.trades} trades")

            if choppy.net_r < 0:
                print(f"     → Filter WORKING: Choppy trades are unprofitable")
                print(f"     → Skipping choppy periods saved {abs(choppy.net_r):.2f}R")
            elif trending.avg_r > choppy.avg_r:
                print(
                    f"     → Filter EFFECTIVE: Trending ({trending.avg_r:+.4f}R) > Choppy ({choppy.avg_r:+.4f}R)"
                )
            else:
                print(f"     ⚠️  Filter QUESTIONABLE: Choppy performs similarly")

    print("\n" + "=" * 80)


def calculate_regime_filter_value(
    trades_with_regime: pd.DataFrame,
    trades_without_regime: pd.DataFrame,
) -> Dict[str, float]:
    """
    Calculate value added by regime filter (if you have both datasets).

    Args:
        trades_with_regime: Trades executed WITH regime filter
        trades_without_regime: Hypothetical trades WITHOUT filter

    Returns:
        Dict with filter impact metrics
    """
    net_r_with = float(trades_with_regime["net_r"].sum())
    net_r_without = float(trades_without_regime["net_r"].sum())

    trades_with = len(trades_with_regime)
    trades_without = len(trades_without_regime)
    trades_filtered = trades_without - trades_with

    r_saved = net_r_with - net_r_without

    return {
        "net_r_with_filter": net_r_with,
        "net_r_without_filter": net_r_without,
        "r_saved_by_filter": r_saved,
        "trades_with_filter": trades_with,
        "trades_without_filter": trades_without,
        "trades_filtered_out": trades_filtered,
        "filter_effectiveness": (
            (r_saved / abs(net_r_without) * 100.0) if net_r_without != 0 else 0.0
        ),
    }


# Example usage
if __name__ == "__main__":
    # Simulate trade data
    np.random.seed(42)

    trades = pd.DataFrame(
        {
            "net_r": np.random.normal(0.2, 1.0, 328),
            "regime_at_entry": np.random.choice(["TRENDING", "CHOPPY"], 328, p=[0.7, 0.3]),
        }
    )

    # Make choppy slightly worse
    choppy_mask = trades["regime_at_entry"] == "CHOPPY"
    trades.loc[choppy_mask, "net_r"] -= 0.15

    # Analyze
    attr = analyze_regime_attribution(trades, filter_enabled=True)
    print_regime_attribution_report(attr)
