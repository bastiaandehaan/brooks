#!/usr/bin/env python3
"""
Brooks Strategy Grid Search
Tests ALL strategy parameters to find optimal configuration.

Focus areas:
1. Context (trend filter)
2. H2/L2 (setup detection)
3. Risk Management
4. Execution timing
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from itertools import product
from datetime import datetime
import json

from backtest.runner import run_backtest

# Suppress verbose logging
import logging

logging.getLogger("execution.guardrails").setLevel(logging.WARNING)
logging.getLogger("Backtest").setLevel(logging.WARNING)


def grid_search_phase_1_context(days=60):
    """
    Phase 1: Optimize CONTEXT (trend filter)
    This has the biggest impact on Sharpe.
    """
    print("\n" + "=" * 80)
    print("  PHASE 1: CONTEXT OPTIMIZATION (Trend Filter)")
    print("=" * 80 + "\n")

    # Grid
    min_slopes = [0.10, 0.15, 0.20, 0.25]
    ema_periods = [15, 20, 25]

    results = []
    total = len(min_slopes) * len(ema_periods)
    counter = 0

    for min_slope, ema_period in product(min_slopes, ema_periods):
        counter += 1
        print(f"[{counter}/{total}] Testing: min_slope={min_slope:.2f}, ema_period={ema_period}")

        metrics = run_backtest(
            symbol="US500.cash",
            days=days,
            max_trades_day=2,
            min_slope=min_slope,
            ema_period=ema_period,
            # Defaults for other params
            pullback_bars=3,
            signal_close_frac=0.30,
            stop_buffer=2.0,
            min_risk_price_units=2.0,
            cooldown_bars=10,
        )

        if "error" not in metrics:
            results.append({
                "min_slope": min_slope,
                "ema_period": ema_period,
                **metrics
            })

    df = pd.DataFrame(results)

    # Sort by Sharpe
    df_sorted = df.sort_values("sharpe", ascending=False)

    print("\n" + "=" * 80)
    print("  TOP 5 CONFIGURATIONS (by Sharpe)")
    print("=" * 80 + "\n")
    print(
        df_sorted[["min_slope", "ema_period", "trades", "sharpe", "net_r", "winrate"]].head(10).to_string(index=False))

    # Best config
    best = df_sorted.iloc[0]
    print(f"\n🎯 BEST CONTEXT CONFIG:")
    print(f"   min_slope={best['min_slope']:.2f}, ema_period={int(best['ema_period'])}")
    print(f"   → Sharpe={best['sharpe']:.3f}, Net R={best['net_r']:.2f}, Trades={int(best['trades'])}")

    return best


def grid_search_phase_2_h2l2(days=60, best_context=None):
    """
    Phase 2: Optimize H2/L2 SETUP (using best context from phase 1)
    """
    print("\n" + "=" * 80)
    print("  PHASE 2: H2/L2 OPTIMIZATION (Setup Detection)")
    print("=" * 80 + "\n")

    if best_context is None:
        best_context = {"min_slope": 0.15, "ema_period": 20}

    # Grid
    pullback_bars_list = [3, 4, 5]
    signal_close_fracs = [0.20, 0.25, 0.30]

    results = []
    total = len(pullback_bars_list) * len(signal_close_fracs)
    counter = 0

    for pullback, close_frac in product(pullback_bars_list, signal_close_fracs):
        counter += 1
        print(f"[{counter}/{total}] Testing: pullback={pullback}, close_frac={close_frac:.2f}")

        metrics = run_backtest(
            symbol="US500.cash",
            days=days,
            max_trades_day=2,
            min_slope=best_context["min_slope"],
            ema_period=int(best_context["ema_period"]),
            pullback_bars=pullback,
            signal_close_frac=close_frac,
            # Defaults
            stop_buffer=2.0,
            min_risk_price_units=2.0,
            cooldown_bars=10,
        )

        if "error" not in metrics:
            results.append({
                "pullback_bars": pullback,
                "signal_close_frac": close_frac,
                **metrics
            })

    df = pd.DataFrame(results)
    df_sorted = df.sort_values("sharpe", ascending=False)

    print("\n" + "=" * 80)
    print("  TOP 5 CONFIGURATIONS (by Sharpe)")
    print("=" * 80 + "\n")
    print(df_sorted[["pullback_bars", "signal_close_frac", "trades", "sharpe", "net_r"]].head(5).to_string(index=False))

    best = df_sorted.iloc[0]
    print(f"\n🎯 BEST H2/L2 CONFIG:")
    print(f"   pullback_bars={int(best['pullback_bars'])}, signal_close_frac={best['signal_close_frac']:.2f}")
    print(f"   → Sharpe={best['sharpe']:.3f}")

    return best


def grid_search_phase_3_risk(days=60, best_context=None, best_h2l2=None):
    """
    Phase 3: Optimize RISK MANAGEMENT (using best context + h2l2)
    """
    print("\n" + "=" * 80)
    print("  PHASE 3: RISK MANAGEMENT OPTIMIZATION")
    print("=" * 80 + "\n")

    if best_context is None:
        best_context = {"min_slope": 0.15, "ema_period": 20}
    if best_h2l2 is None:
        best_h2l2 = {"pullback_bars": 3, "signal_close_frac": 0.30}

    # Grid
    stop_buffers = [1.0, 1.5, 2.0, 2.5]
    min_risks = [1.5, 2.0, 2.5]

    results = []
    total = len(stop_buffers) * len(min_risks)
    counter = 0

    for stop_buf, min_risk in product(stop_buffers, min_risks):
        counter += 1
        print(f"[{counter}/{total}] Testing: stop_buffer={stop_buf:.1f}, min_risk={min_risk:.1f}")

        metrics = run_backtest(
            symbol="US500.cash",
            days=days,
            max_trades_day=2,
            min_slope=best_context["min_slope"],
            ema_period=int(best_context["ema_period"]),
            pullback_bars=int(best_h2l2["pullback_bars"]),
            signal_close_frac=best_h2l2["signal_close_frac"],
            stop_buffer=stop_buf,
            min_risk_price_units=min_risk,
            cooldown_bars=10,
        )

        if "error" not in metrics:
            results.append({
                "stop_buffer": stop_buf,
                "min_risk": min_risk,
                **metrics
            })

    df = pd.DataFrame(results)
    df_sorted = df.sort_values("sharpe", ascending=False)

    print("\n" + "=" * 80)
    print("  TOP 5 CONFIGURATIONS (by Sharpe)")
    print("=" * 80 + "\n")
    print(df_sorted[["stop_buffer", "min_risk", "trades", "sharpe", "max_dd"]].head(5).to_string(index=False))

    best = df_sorted.iloc[0]
    print(f"\n🎯 BEST RISK CONFIG:")
    print(f"   stop_buffer={best['stop_buffer']:.1f}, min_risk={best['min_risk']:.1f}")
    print(f"   → Sharpe={best['sharpe']:.3f}, Max DD={best['max_dd']:.2f}R")

    return best


def grid_search_phase_4_execution(days=60, best_context=None, best_h2l2=None, best_risk=None):
    """
    Phase 4: Optimize EXECUTION (cooldown, max_trades_day)
    """
    print("\n" + "=" * 80)
    print("  PHASE 4: EXECUTION OPTIMIZATION")
    print("=" * 80 + "\n")

    if best_context is None:
        best_context = {"min_slope": 0.15, "ema_period": 20}
    if best_h2l2 is None:
        best_h2l2 = {"pullback_bars": 3, "signal_close_frac": 0.30}
    if best_risk is None:
        best_risk = {"stop_buffer": 2.0, "min_risk": 2.0}

    # Grid
    cooldowns = [0, 5, 10, 15]
    max_trades_days = [1, 2, 3]

    results = []
    total = len(cooldowns) * len(max_trades_days)
    counter = 0

    for cooldown, max_day in product(cooldowns, max_trades_days):
        counter += 1
        print(f"[{counter}/{total}] Testing: cooldown={cooldown}, max_trades_day={max_day}")

        metrics = run_backtest(
            symbol="US500.cash",
            days=days,
            max_trades_day=max_day,
            min_slope=best_context["min_slope"],
            ema_period=int(best_context["ema_period"]),
            pullback_bars=int(best_h2l2["pullback_bars"]),
            signal_close_frac=best_h2l2["signal_close_frac"],
            stop_buffer=best_risk["stop_buffer"],
            min_risk_price_units=best_risk["min_risk"],
            cooldown_bars=cooldown,
        )

        if "error" not in metrics:
            results.append({
                "cooldown": cooldown,
                "max_trades_day": max_day,
                **metrics
            })

    df = pd.DataFrame(results)
    df_sorted = df.sort_values("sharpe", ascending=False)

    print("\n" + "=" * 80)
    print("  TOP 5 CONFIGURATIONS (by Sharpe)")
    print("=" * 80 + "\n")
    print(df_sorted[["cooldown", "max_trades_day", "trades", "sharpe", "net_r"]].head(5).to_string(index=False))

    best = df_sorted.iloc[0]
    print(f"\n🎯 BEST EXECUTION CONFIG:")
    print(f"   cooldown={int(best['cooldown'])}, max_trades_day={int(best['max_trades_day'])}")
    print(f"   → Sharpe={best['sharpe']:.3f}")

    return best


def main():
    print("\n" + "🎯" * 40)
    print("  BROOKS COMPREHENSIVE STRATEGY GRID SEARCH")
    print("🎯" * 40 + "\n")

    start_time = datetime.now()

    # Phase 1: Context (most important!)
    best_context = grid_search_phase_1_context(days=60)

    # Phase 2: H2/L2 (using best context)
    best_h2l2 = grid_search_phase_2_h2l2(days=60, best_context=best_context)

    # Phase 3: Risk (using best context + h2l2)
    best_risk = grid_search_phase_3_risk(days=60, best_context=best_context, best_h2l2=best_h2l2)

    # Phase 4: Execution (using all best configs)
    best_execution = grid_search_phase_4_execution(
        days=60,
        best_context=best_context,
        best_h2l2=best_h2l2,
        best_risk=best_risk
    )

    # Final summary
    print("\n" + "=" * 80)
    print("  🏆 FINAL OPTIMAL CONFIGURATION")
    print("=" * 80 + "\n")

    optimal_config = {
        "context": {
            "min_slope": float(best_context["min_slope"]),
            "ema_period": int(best_context["ema_period"]),
        },
        "h2l2": {
            "pullback_bars": int(best_h2l2["pullback_bars"]),
            "signal_close_frac": float(best_h2l2["signal_close_frac"]),
        },
        "risk": {
            "stop_buffer": float(best_risk["stop_buffer"]),
            "min_risk_price_units": float(best_risk["min_risk"]),
        },
        "execution": {
            "cooldown_bars": int(best_execution["cooldown"]),
            "max_trades_day": int(best_execution["max_trades_day"]),
        },
        "performance": {
            "sharpe": float(best_execution["sharpe"]),
            "net_r": float(best_execution["net_r"]),
            "winrate": float(best_execution["winrate"]),
            "trades": int(best_execution["trades"]),
        }
    }

    print(json.dumps(optimal_config, indent=2))

    elapsed = datetime.now() - start_time
    print(f"\n⏱️  Total time: {elapsed}")

    # Save to file
    with open("optimal_config.json", "w") as f:
        json.dump(optimal_config, f, indent=2)
    print("\n💾 Saved to: optimal_config.json")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()