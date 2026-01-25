# backtest/ftmo_risk_report.py
"""
FTMO Risk Assessment Report - Integrated Analytics

Combines all analytics modules to answer:
"Can we safely scale to 1.25-1.5% risk or 3 trades/day?"

This is the final decision-making report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from backtest.analytics.distribution import (
    analyze_distribution,
    print_distribution_report,
)
from backtest.analytics.drawdown_duration import (
    analyze_drawdown_durations,
    print_drawdown_duration_report,
)
from backtest.analytics.monte_carlo import (
    monte_carlo_trade_resample,
    print_monte_carlo_report,
    simulate_scaled_risk,
)
from backtest.analytics.regime_attribution import (
    analyze_regime_attribution,
    print_regime_attribution_report,
)
from backtest.analytics.stability import (
    analyze_stability,
    compare_early_vs_late,
    print_stability_report,
)

logger = logging.getLogger(__name__)


@dataclass
class ScalingRecommendation:
    """Final scaling recommendation"""

    # Risk scaling
    can_scale_risk: bool
    max_safe_risk_pct: float
    risk_rationale: str

    # Trade frequency scaling
    can_increase_frequency: bool
    max_safe_trades_per_day: int
    frequency_rationale: str

    # Overall assessment
    confidence_level: str  # "HIGH", "MODERATE", "LOW"
    key_risks: list[str]
    strengths: list[str]


def generate_ftmo_risk_report(
    trades_df: pd.DataFrame,
    equity_curve_r: pd.Series,
    *,
    current_risk_pct: float = 1.0,
    current_max_trades_day: int = 2,
    n_monte_carlo_sims: int = 10000,
) -> ScalingRecommendation:
    """
    Generate comprehensive FTMO risk assessment report.

    Args:
        trades_df: Backtest trades (with 'entry_time', 'net_r', etc.)
        equity_curve_r: Cumulative R curve (datetime indexed)
        current_risk_pct: Current risk per trade
        current_max_trades_day: Current max trades per day
        n_monte_carlo_sims: Number of Monte Carlo simulations

    Returns:
        ScalingRecommendation object
    """
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  FTMO RISK ASSESSMENT REPORT - SCALING ANALYSIS".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    print(f"\nCurrent Configuration:")
    print(f"  Risk per trade     : {current_risk_pct:.2f}%")
    print(f"  Max trades per day : {current_max_trades_day}")
    print(f"  Total trades       : {len(trades_df)}")
    print(f"  Net R              : {trades_df['net_r'].sum():+.2f}R")

    # Extract results series
    results_r = trades_df["net_r"]

    # =================================================================
    # 1. DISTRIBUTION ANALYSIS
    # =================================================================
    print("\n" + "█" * 80)
    print("█  SECTION 1: TRADE RESULT DISTRIBUTION")
    print("█" * 80)

    dist_metrics = analyze_distribution(results_r)
    print_distribution_report(dist_metrics)

    # =================================================================
    # 2. MONTE CARLO ANALYSIS
    # =================================================================
    print("\n" + "█" * 80)
    print("█  SECTION 2: MONTE CARLO BOOTSTRAP ANALYSIS")
    print("█" * 80)

    mc_metrics = monte_carlo_trade_resample(
        results_r,
        n_simulations=n_monte_carlo_sims,
        ftmo_daily_limit_r=5.0,
        ftmo_total_limit_r=10.0,
        seed=42,
    )
    print_monte_carlo_report(mc_metrics, current_risk_pct=current_risk_pct)

    # Test scaled risk scenarios
    print("\n" + "=" * 80)
    print("  RISK SCALING SCENARIOS")
    print("=" * 80)

    scaling_df = simulate_scaled_risk(
        results_r,
        risk_multipliers=[1.0, 1.25, 1.5, 2.0],
        n_simulations=5000,
    )

    # =================================================================
    # 3. DRAWDOWN DURATION
    # =================================================================
    print("\n" + "█" * 80)
    print("█  SECTION 3: DRAWDOWN DURATION ANALYSIS")
    print("█" * 80)

    dd_metrics = analyze_drawdown_durations(equity_curve_r, min_depth_r=0.5)
    print_drawdown_duration_report(dd_metrics)

    # =================================================================
    # 4. REGIME ATTRIBUTION
    # =================================================================
    print("\n" + "█" * 80)
    print("█  SECTION 4: REGIME ATTRIBUTION")
    print("█" * 80)

    regime_attr = analyze_regime_attribution(trades_df, filter_enabled=True)
    print_regime_attribution_report(regime_attr)

    # =================================================================
    # 5. TEMPORAL STABILITY
    # =================================================================
    print("\n" + "█" * 80)
    print("█  SECTION 5: TEMPORAL STABILITY")
    print("█" * 80)

    stability_metrics = analyze_stability(trades_df, period="Q", min_trades_per_period=10)
    print_stability_report(stability_metrics)

    # Early vs Late comparison
    print("\n" + "=" * 80)
    print("  EARLY vs LATE PERFORMANCE")
    print("=" * 80)

    comparison = compare_early_vs_late(trades_df, split_pct=0.5)
    print(f"\n  First 50%:")
    print(f"    Trades  : {comparison['early_trades']}")
    print(f"    Avg R   : {comparison['early_avg_r']:+.4f}R")
    print(f"    Winrate : {comparison['early_wr'] * 100:.1f}%")

    print(f"\n  Last 50%:")
    print(f"    Trades  : {comparison['late_trades']}")
    print(f"    Avg R   : {comparison['late_avg_r']:+.4f}R")
    print(f"    Winrate : {comparison['late_wr'] * 100:.1f}%")

    print(f"\n  Performance Delta:")
    print(f"    Avg R   : {comparison['avg_r_delta']:+.4f}R")
    print(f"    Winrate : {comparison['wr_delta'] * 100:+.1f}%")

    # =================================================================
    # FINAL RECOMMENDATION
    # =================================================================
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  SCALING RECOMMENDATION".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    # Decision logic
    key_risks = []
    strengths = []

    # === RISK SCALING ASSESSMENT ===

    # Check 1: Tail dependency
    if dist_metrics.tail_dependency_score > 0.6:
        key_risks.append("HIGH TAIL DEPENDENCY: Profit concentrated in few trades")
        can_scale_risk = False
        max_safe_risk = current_risk_pct
    else:
        strengths.append("LOW TAIL DEPENDENCY: Profit well-distributed")
        can_scale_risk = True
        max_safe_risk = current_risk_pct * 1.5

    # Check 2: FTMO breach probability
    breach_threshold_125 = scaling_df[scaling_df["risk_multiplier"] == 1.25][
        "ftmo_breach_pct"
    ].values[0]
    breach_threshold_150 = scaling_df[scaling_df["risk_multiplier"] == 1.5][
        "ftmo_breach_pct"
    ].values[0]

    if breach_threshold_125 > 15.0:
        key_risks.append(f"FTMO BREACH RISK: {breach_threshold_125:.1f}% at 1.25x risk")
        can_scale_risk = False
        max_safe_risk = current_risk_pct
    elif breach_threshold_150 > 15.0:
        key_risks.append(f"MODERATE BREACH RISK: {breach_threshold_150:.1f}% at 1.5x risk")
        max_safe_risk = current_risk_pct * 1.25
    else:
        strengths.append(f"LOW BREACH RISK: {breach_threshold_150:.1f}% even at 1.5x risk")
        max_safe_risk = current_risk_pct * 1.5

    # Check 3: Drawdown duration
    if dd_metrics.max_duration > 90:
        key_risks.append(f"LONG DRAWDOWNS: Up to {dd_metrics.max_duration} days underwater")
        can_scale_risk = can_scale_risk and False
    elif dd_metrics.max_duration > 45:
        key_risks.append(f"MODERATE DD DURATION: {dd_metrics.max_duration} days max")
    else:
        strengths.append(f"QUICK RECOVERY: Max {dd_metrics.max_duration} days underwater")

    # Check 4: Performance stability
    if stability_metrics.performance_trend == "DEGRADING":
        key_risks.append("PERFORMANCE DECAY: Recent periods worse than early")
        can_scale_risk = False
        max_safe_risk = current_risk_pct
    elif stability_metrics.performance_trend == "IMPROVING":
        strengths.append("IMPROVING EDGE: Recent performance better than early")
    else:
        strengths.append("STABLE PERFORMANCE: No degradation over time")

    # Final risk decision
    if can_scale_risk:
        risk_rationale = (
            f"Monte Carlo + Distribution analysis support scaling to {max_safe_risk:.2f}%"
        )
    else:
        risk_rationale = "Current risk level is maximum safe threshold"

    # === FREQUENCY SCALING ASSESSMENT ===

    can_increase_frequency = True
    max_safe_trades = current_max_trades_day

    # Check 1: Current trade count
    avg_trades_per_day = len(trades_df) / (equity_curve_r.index[-1] - equity_curve_r.index[0]).days

    if avg_trades_per_day >= current_max_trades_day * 0.8:
        # Already hitting limit frequently
        can_increase_frequency = True
        max_safe_trades = 3
        strengths.append(
            f"FILTER EFFECTIVE: Using {avg_trades_per_day:.1f}/{current_max_trades_day} trades/day avg"
        )
    else:
        # Not using full capacity
        key_risks.append(
            f"LOW UTILIZATION: Only {avg_trades_per_day:.1f}/{current_max_trades_day} trades/day"
        )
        max_safe_trades = current_max_trades_day
        can_increase_frequency = False

    # Check 2: Regime filter effectiveness
    if regime_attr.regimes and "TRENDING" in regime_attr.regimes:
        trending_pct = regime_attr.regimes["TRENDING"].pct_of_total_trades
        if trending_pct > 60:
            strengths.append(f"STRONG REGIME FILTER: {trending_pct:.1f}% trades in trending")
            can_increase_frequency = True
            max_safe_trades = 3
        else:
            key_risks.append(f"WEAK REGIME FILTER: Only {trending_pct:.1f}% trending trades")

    if can_increase_frequency:
        frequency_rationale = f"Regime filter + utilization support {max_safe_trades} trades/day"
    else:
        frequency_rationale = "Current frequency is optimal given setup rate"

    # === CONFIDENCE LEVEL ===

    risk_score = len(key_risks)
    strength_score = len(strengths)

    if strength_score >= 4 and risk_score <= 1:
        confidence_level = "HIGH"
    elif strength_score >= 2 and risk_score <= 2:
        confidence_level = "MODERATE"
    else:
        confidence_level = "LOW"

    # === PRINT RECOMMENDATION ===

    print(f"\n🎯 RISK SCALING:")
    print(f"   Can scale?       : {'✅ YES' if can_scale_risk else '❌ NO'}")
    print(f"   Max safe risk    : {max_safe_risk:.2f}%")
    print(f"   Rationale        : {risk_rationale}")

    print(f"\n📊 FREQUENCY SCALING:")
    print(f"   Can increase?    : {'✅ YES' if can_increase_frequency else '❌ NO'}")
    print(f"   Max safe trades  : {max_safe_trades} per day")
    print(f"   Rationale        : {frequency_rationale}")

    print(f"\n🔒 CONFIDENCE LEVEL: {confidence_level}")

    print(f"\n✅ STRENGTHS:")
    for s in strengths:
        print(f"   • {s}")

    print(f"\n⚠️  KEY RISKS:")
    for r in key_risks:
        print(f"   • {r}")

    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  END OF REPORT".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")

    return ScalingRecommendation(
        can_scale_risk=can_scale_risk,
        max_safe_risk_pct=max_safe_risk,
        risk_rationale=risk_rationale,
        can_increase_frequency=can_increase_frequency,
        max_safe_trades_per_day=max_safe_trades,
        frequency_rationale=frequency_rationale,
        confidence_level=confidence_level,
        key_risks=key_risks,
        strengths=strengths,
    )


# Integration with existing backtest runner
def add_risk_analysis_to_backtest(backtest_result: Dict) -> None:
    """
    Add risk analysis to existing backtest runner.

    Call this at the end of run_backtest_from_config() in runner.py:

    ```python
    # At end of runner.py main():
    result = run_backtest_from_config(config, ...)

    if "error" not in result:
        from backtest.ftmo_risk_report import add_risk_analysis_to_backtest
        add_risk_analysis_to_backtest(result)
    ```
    """
    # This function would extract trades_df and equity_curve from backtest_result
    # and call generate_ftmo_risk_report()
    pass


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Simulate backtest data
    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=328, freq="D")
    results = np.random.normal(0.21, 1.0, 328)

    trades_df = pd.DataFrame(
        {
            "entry_time": dates,
            "net_r": results,
            "regime_at_entry": np.random.choice(["TRENDING", "CHOPPY"], 328, p=[0.75, 0.25]),
        }
    )

    equity_curve = pd.Series(np.cumsum(results), index=dates)

    # Generate report
    recommendation = generate_ftmo_risk_report(
        trades_df=trades_df,
        equity_curve_r=equity_curve,
        current_risk_pct=1.0,
        current_max_trades_day=2,
        n_monte_carlo_sims=10000,
    )

    print("\n📋 FINAL RECOMMENDATION OBJECT:")
    print(f"  Can scale risk       : {recommendation.can_scale_risk}")
    print(f"  Max safe risk        : {recommendation.max_safe_risk_pct:.2f}%")
    print(f"  Can increase freq    : {recommendation.can_increase_frequency}")
    print(f"  Max safe trades/day  : {recommendation.max_safe_trades_per_day}")
    print(f"  Confidence           : {recommendation.confidence_level}")
