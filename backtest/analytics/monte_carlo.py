# backtest/analytics/monte_carlo.py
"""
Monte Carlo Bootstrap Analysis

Resamples trade results to estimate:
- Confidence intervals for Net R and Max Drawdown
- FTMO breach probability
- Strategy robustness under different sequences
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation"""

    n_simulations: int

    # Net R distribution
    net_r_mean: float
    net_r_median: float
    net_r_p05: float
    net_r_p95: float
    net_r_std: float

    # Max DD distribution
    max_dd_mean: float
    max_dd_median: float
    max_dd_p05: float  # 5th percentile (worst DD)
    max_dd_p95: float  # 95th percentile (best DD)
    max_dd_std: float

    # FTMO breach analysis
    ftmo_daily_breach_pct: float  # % of sims breaching daily limit
    ftmo_total_breach_pct: float  # % of sims breaching total limit
    ftmo_any_breach_pct: float  # % of sims with any breach

    # Raw distributions (for plotting)
    net_r_distribution: np.ndarray
    max_dd_distribution: np.ndarray


def monte_carlo_trade_resample(
    results_r: pd.Series,
    *,
    n_simulations: int = 10000,
    ftmo_daily_limit_r: float = 5.0,  # 5% daily loss = 5R at 1% risk/trade
    ftmo_total_limit_r: float = 10.0,  # 10% total loss = 10R
    seed: Optional[int] = None,
) -> MonteCarloResults:
    """
    Monte Carlo bootstrap resampling of trades.

    Strategy:
    - Resample WITH REPLACEMENT from actual trade results
    - For each simulation, calculate Net R and Max DD
    - Track FTMO breaches

    Args:
        results_r: Actual trade results (R-values)
        n_simulations: Number of bootstrap samples
        ftmo_daily_limit_r: Daily loss limit in R-units
        ftmo_total_limit_r: Total loss limit in R-units
        seed: Random seed for reproducibility

    Returns:
        MonteCarloResults object
    """
    if seed is not None:
        np.random.seed(seed)

    results = results_r.values
    n_trades = len(results)

    if n_trades == 0:
        raise ValueError("Cannot run Monte Carlo on empty results")

    logger.info(f"Running Monte Carlo with {n_simulations:,d} simulations...")

    net_r_sims = np.zeros(n_simulations)
    max_dd_sims = np.zeros(n_simulations)
    daily_breach_count = 0
    total_breach_count = 0
    any_breach_count = 0

    for i in range(n_simulations):
        # Resample trades
        sim_results = np.random.choice(results, size=n_trades, replace=True)

        # Calculate equity curve
        equity = np.cumsum(sim_results)
        net_r_sims[i] = equity[-1]

        # Calculate max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = equity - running_max
        max_dd_sims[i] = drawdown.min()

        # FTMO breach detection (simplified daily check)
        # Assume trades spread across days, check max single-day loss
        # This is conservative: actual daily grouping would be more complex
        max_single_trade_loss = sim_results.min()
        daily_breach = max_single_trade_loss < -ftmo_daily_limit_r

        total_breach = max_dd_sims[i] < -ftmo_total_limit_r

        if daily_breach:
            daily_breach_count += 1
        if total_breach:
            total_breach_count += 1
        if daily_breach or total_breach:
            any_breach_count += 1

    # Calculate statistics
    net_r_mean = float(np.mean(net_r_sims))
    net_r_median = float(np.median(net_r_sims))
    net_r_p05 = float(np.percentile(net_r_sims, 5))
    net_r_p95 = float(np.percentile(net_r_sims, 95))
    net_r_std = float(np.std(net_r_sims, ddof=1))

    max_dd_mean = float(np.mean(max_dd_sims))
    max_dd_median = float(np.median(max_dd_sims))
    max_dd_p05 = float(np.percentile(max_dd_sims, 5))  # Worst DD
    max_dd_p95 = float(np.percentile(max_dd_sims, 95))  # Best DD
    max_dd_std = float(np.std(max_dd_sims, ddof=1))

    ftmo_daily_breach_pct = (daily_breach_count / n_simulations) * 100.0
    ftmo_total_breach_pct = (total_breach_count / n_simulations) * 100.0
    ftmo_any_breach_pct = (any_breach_count / n_simulations) * 100.0

    logger.info(f"Monte Carlo complete: {n_simulations:,d} simulations")
    logger.info(f"  Net R: {net_r_mean:.2f}R ± {net_r_std:.2f}R (mean ± std)")
    logger.info(f"  Max DD: {max_dd_mean:.2f}R ± {max_dd_std:.2f}R")
    logger.info(f"  FTMO breach rate: {ftmo_any_breach_pct:.2f}%")

    return MonteCarloResults(
        n_simulations=n_simulations,
        net_r_mean=net_r_mean,
        net_r_median=net_r_median,
        net_r_p05=net_r_p05,
        net_r_p95=net_r_p95,
        net_r_std=net_r_std,
        max_dd_mean=max_dd_mean,
        max_dd_median=max_dd_median,
        max_dd_p05=max_dd_p05,
        max_dd_p95=max_dd_p95,
        max_dd_std=max_dd_std,
        ftmo_daily_breach_pct=ftmo_daily_breach_pct,
        ftmo_total_breach_pct=ftmo_total_breach_pct,
        ftmo_any_breach_pct=ftmo_any_breach_pct,
        net_r_distribution=net_r_sims,
        max_dd_distribution=max_dd_sims,
    )


def print_monte_carlo_report(mc: MonteCarloResults, *, current_risk_pct: float = 1.0) -> None:
    """Print formatted Monte Carlo analysis report."""

    print("\n" + "=" * 80)
    print(f"  🎲 MONTE CARLO BOOTSTRAP ANALYSIS ({mc.n_simulations:,d} simulations)")
    print("=" * 80)

    print("\n💰 NET R DISTRIBUTION:")
    print(f"  Mean          : {mc.net_r_mean:+.2f}R")
    print(f"  Median        : {mc.net_r_median:+.2f}R")
    print(f"  Std Dev       : {mc.net_r_std:.2f}R")
    print(f"  90% CI        : [{mc.net_r_p05:+.2f}R, {mc.net_r_p95:+.2f}R]")
    print(f"  Range (90%)   : {mc.net_r_p95 - mc.net_r_p05:.2f}R")

    print("\n📉 MAX DRAWDOWN DISTRIBUTION:")
    print(f"  Mean          : {mc.max_dd_mean:.2f}R")
    print(f"  Median        : {mc.max_dd_median:.2f}R")
    print(f"  Std Dev       : {mc.max_dd_std:.2f}R")
    print(f"  90% CI        : [{mc.max_dd_p05:.2f}R, {mc.max_dd_p95:.2f}R]")
    print(f"  Worst 5%      : < {mc.max_dd_p05:.2f}R")

    print("\n🚨 FTMO BREACH PROBABILITY:")
    print(f"  Current risk  : {current_risk_pct:.2f}% per trade")
    print(f"  Daily breach  : {mc.ftmo_daily_breach_pct:.2f}% of simulations")
    print(f"  Total breach  : {mc.ftmo_total_breach_pct:.2f}% of simulations")
    print(f"  Any breach    : {mc.ftmo_any_breach_pct:.2f}% of simulations")

    # Risk assessment
    print("\n📊 RISK ASSESSMENT:")
    if mc.ftmo_any_breach_pct < 5.0:
        print("  ✅ LOW RISK: <5% breach probability")
        print("  → Strategy appears safe at current risk level")
    elif mc.ftmo_any_breach_pct < 15.0:
        print("  ⚠️  MODERATE RISK: 5-15% breach probability")
        print("  → Acceptable for demo, monitor closely")
    else:
        print("  🔴 HIGH RISK: >15% breach probability")
        print("  → Consider reducing risk per trade")

    # Upside potential
    prob_positive = (mc.net_r_distribution > 0).sum() / len(mc.net_r_distribution) * 100
    prob_target = (
        (mc.net_r_distribution >= 10.0).sum() / len(mc.net_r_distribution) * 100
    )  # 10R = 10% target

    print(f"\n📈 UPSIDE PROBABILITY:")
    print(f"  P(Net R > 0)  : {prob_positive:.1f}%")
    print(f"  P(Net R ≥ 10R): {prob_target:.1f}%  (FTMO target)")

    print("\n" + "=" * 80)


def simulate_scaled_risk(
    results_r: pd.Series,
    *,
    risk_multipliers: List[float] = [1.0, 1.25, 1.5, 2.0],
    n_simulations: int = 5000,
) -> pd.DataFrame:
    """
    Simulate strategy performance at different risk levels.

    Args:
        results_r: Base trade results at 1R
        risk_multipliers: Scaling factors to test
        n_simulations: Number of sims per multiplier

    Returns:
        DataFrame with results per risk level
    """
    logger.info("Simulating scaled risk scenarios...")

    results = []

    for mult in risk_multipliers:
        # Scale results by multiplier
        scaled_results = results_r * mult

        # Run Monte Carlo
        mc = monte_carlo_trade_resample(
            scaled_results,
            n_simulations=n_simulations,
            ftmo_daily_limit_r=5.0,  # Fixed FTMO limits
            ftmo_total_limit_r=10.0,
            seed=42,
        )

        results.append(
            {
                "risk_multiplier": mult,
                "risk_pct": mult * 1.0,  # Assuming base is 1%
                "net_r_mean": mc.net_r_mean,
                "net_r_p05": mc.net_r_p05,
                "net_r_p95": mc.net_r_p95,
                "max_dd_mean": mc.max_dd_mean,
                "max_dd_p05": mc.max_dd_p05,
                "ftmo_breach_pct": mc.ftmo_any_breach_pct,
            }
        )

    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("  ⚖️  RISK SCALING ANALYSIS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    return df


# Example usage
if __name__ == "__main__":
    # Simulate trade results
    np.random.seed(42)
    results = pd.Series(np.random.normal(0.2, 1.0, 328))  # 328 trades like backtest

    # Run Monte Carlo
    mc = monte_carlo_trade_resample(results, n_simulations=10000, seed=42)
    print_monte_carlo_report(mc, current_risk_pct=1.0)

    # Test scaled risk
    simulate_scaled_risk(results, risk_multipliers=[1.0, 1.25, 1.5, 2.0])
