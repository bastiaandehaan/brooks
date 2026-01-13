#!/usr/bin/env python3
"""
Brooks Strategy Grid Search - UPDATED FOR PRODUCTION
Tests ALL strategy parameters to find optimal configuration.

CRITICAL: Now includes regime filter and costs (realistic!)

Focus areas:
1. Regime Filter (NEW - biggest impact!)
2. Context (trend filter)
3. H2/L2 (setup detection)
4. Risk Management
5. Execution timing
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


def grid_search_phase_0_regime(days=180):
    """
    Phase 0: Optimize REGIME FILTER (NEW - Most Critical!)
    This determines which days we trade at all.
    """
    print("\n" + "=" * 80)
    print("  PHASE 0: REGIME FILTER OPTIMIZATION (Skip Choppy Markets)")
    print("=" * 80 + "\n")

    # Grid
    chop_thresholds = [1.5, 2.0, 2.5, 3.0]
    regime_filter_options = [True, False]

    results = []
    total = len(chop_thresholds) * len(regime_filter_options)
    counter = 0

    for regime_filter, chop_threshold in product(regime_filter_options, chop_thresholds):
        if not regime_filter and chop_threshold != 2.5:
            continue  # Skip redundant tests when filter is off

        counter += 1
        filter_str = "ON" if regime_filter else "OFF"
        print(f"[{counter}/{total}] Testing: regime_filter={filter_str}, chop_threshold={chop_threshold:.1f}")

        metrics = run_backtest(
            symbol="US500.cash",
            days=days,
            max_trades_day=2,
            # Defaults for now
            min_slope=0.15,
            ema_period=20,
            pullback_bars=3,
            signal_close_frac=0.30,
            stop_buffer=1.0,  # Updated from 2.0
            min_risk_price_units=2.0,
            cooldown_bars=0,  # Updated from 10
            regime_filter=regime_filter,
            chop_threshold=chop_threshold,
            costs_per_trade_r=0.04,  # NEW: Realistic costs!
        )

        if "error" not in metrics:
            results.append({
                "regime_filter": regime_filter,
                "chop_threshold": chop_threshold,
                "choppy_segments_skipped": metrics.get("choppy_segments_skipped", 0),
                **metrics
            })

    df = pd.DataFrame(results)

    # Sort by Daily Sharpe (NEW metric!)
    sort_by = "daily_sharpe" if "daily_sharpe" in df.columns else "sharpe"
    df_sorted = df.sort_values(sort_by, ascending=False)

    print("\n" + "=" * 80)
    print("  TOP 5 CONFIGURATIONS (by Daily Sharpe)")
    print("=" * 80 + "\n")

    cols = ["regime_filter", "chop_threshold", "trades", "daily_sharpe" if "daily_sharpe" in df.columns else "sharpe",
            "net_r", "winrate", "max_dd"]
    print(df_sorted[cols].head(5).to_string(index=False))

    # Best config
    best = df_sorted.iloc[0]
    print(f"\n🎯 BEST REGIME CONFIG:")
    print(f"   regime_filter={best['regime_filter']}, chop_threshold={best['chop_threshold']:.1f}")
    sharpe_metric = best.get('daily_sharpe', best.get('sharpe', 0))
    print(f"   → Daily Sharpe={sharpe_metric:.3f}, Net R={best['net_r']:.2f}, Trades={int(best['trades'])}")

    if best['regime_filter']:
        print(f"   → Choppy segments skipped: {best.get('choppy_segments_skipped', 0)}")

    return best


def grid_search_phase_1_context(days=180, best_regime=None):
    """
    Phase 1: Optimize CONTEXT (trend filter)
    This has the biggest impact after regime.
    """
    print("\n" + "=" * 80)
    print("  PHASE 1: CONTEXT OPTIMIZATION (Trend Filter)")
    print("=" * 80 + "\n")

    if best_regime is None:
        best_regime = {"regime_filter": True, "chop_threshold": 2.0}

    # Grid - SIMPLIFIED (fewer combinations)
    min_slopes = [0.10, 0.15, 0.20]
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
            # Current best from production
            pullback_bars=3,
            signal_close_frac=0.30,
            stop_buffer=1.0,
            min_risk_price_units=2.0,
            cooldown_bars=0,
            regime_filter=best_regime["regime_filter"],
            chop_threshold=best_regime["chop_threshold"],
            costs_per_trade_r=0.04,
        )

        if "error" not in metrics:
            results.append({
                "min_slope": min_slope,
                "ema_period": ema_period,
                **metrics
            })

    df = pd.DataFrame(results)

    # Sort by Daily Sharpe
    sort_by = "daily_sharpe" if "daily_sharpe" in df.columns else "sharpe"
    df_sorted = df.sort_values(sort_by, ascending=False)

    print("\n" + "=" * 80)
    print("  TOP 5 CONFIGURATIONS (by Daily Sharpe)")
    print("=" * 80 + "\n")

    cols = ["min_slope", "ema_period", "trades", sort_by, "net_r", "winrate"]
    print(df_sorted[cols].head(5).to_string(index=False))

    # Best config
    best = df_sorted.iloc[0]
    sharpe_val = best.get('daily_sharpe', best.get('sharpe', 0))
    print(f"\n🎯 BEST CONTEXT CONFIG:")
    print(f"   min_slope={best['min_slope']:.2f}, ema_period={int(best['ema_period'])}")
    print(f"   → Daily Sharpe={sharpe_val:.3f}, Net R={best['net_r']:.2f}, Trades={int(best['trades'])}")

    return best


def grid_search_phase_2_h2l2(days=180, best_regime=None, best_context=None):
    """
    Phase 2: Optimize H2/L2 SETUP (using best regime + context from phase 0+1)
    """
    print("\n" + "=" * 80)
    print("  PHASE 2: H2/L2 OPTIMIZATION (Setup Detection)")
    print("=" * 80 + "\n")

    if best_regime is None:
        best_regime = {"regime_filter": True, "chop_threshold": 2.0}
    if best_context is None:
        best_context = {"min_slope": 0.15, "ema_period": 20}

    # Grid - SIMPLIFIED
    pullback_bars_list = [3, 4, 5]
    signal_close_fracs = [0.25, 0.30, 0.35]

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
            # Current production defaults
            stop_buffer=1.0,
            min_risk_price_units=2.0,
            cooldown_bars=0,
            regime_filter=best_regime["regime_filter"],
            chop_threshold=best_regime["chop_threshold"],
            costs_per_trade_r=0.04,
        )

        if "error" not in metrics:
            results.append({
                "pullback_bars": pullback,
                "signal_close_frac": close_frac,
                **metrics
            })

    df = pd.DataFrame(results)
    sort_by = "daily_sharpe" if "daily_sharpe" in df.columns else "sharpe"
    df_sorted = df.sort_values(sort_by, ascending=False)

    print("\n" + "=" * 80)
    print("  TOP 5 CONFIGURATIONS (by Daily Sharpe)")
    print("=" * 80 + "\n")

    cols = ["pullback_bars", "signal_close_frac", "trades", sort_by, "net_r"]
    print(df_sorted[cols].head(5).to_string(index=False))

    best = df_sorted.iloc[0]
    sharpe_val = best.get('daily_sharpe', best.get('sharpe', 0))
    print(f"\n🎯 BEST H2/L2 CONFIG:")
    print(f"   pullback_bars={int(best['pullback_bars'])}, signal_close_frac={best['signal_close_frac']:.2f}")
    print(f"   → Daily Sharpe={sharpe_val:.3f}")

    return best


def grid_search_phase_3_risk(days=180, best_regime=None, best_context=None, best_h2l2=None):
    """
    Phase 3: Optimize RISK MANAGEMENT (using best regime + context + h2l2)
    """
    print("\n" + "=" * 80)
    print("  PHASE 3: RISK MANAGEMENT OPTIMIZATION")
    print("=" * 80 + "\n")

    if best_regime is None:
        best_regime = {"regime_filter": True, "chop_threshold": 2.0}
    if best_context is None:
        best_context = {"min_slope": 0.15, "ema_period": 20}
    if best_h2l2 is None:
        best_h2l2 = {"pullback_bars": 3, "signal_close_frac": 0.30}

    # Grid - CRITICAL for FTMO compliance
    stop_buffers = [0.5, 1.0, 1.5, 2.0]
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
            cooldown_bars=0,
            regime_filter=best_regime["regime_filter"],
            chop_threshold=best_regime["chop_threshold"],
            costs_per_trade_r=0.04,
        )

        if "error" not in metrics:
            results.append({
                "stop_buffer": stop_buf,
                "min_risk": min_risk,
                **metrics
            })

    df = pd.DataFrame(results)
    sort_by = "daily_sharpe" if "daily_sharpe" in df.columns else "sharpe"
    df_sorted = df.sort_values(sort_by, ascending=False)

    print("\n" + "=" * 80)
    print("  TOP 5 CONFIGURATIONS (by Daily Sharpe)")
    print("=" * 80 + "\n")

    cols = ["stop_buffer", "min_risk", "trades", sort_by, "max_dd", "recovery_factor"]
    print(df_sorted[cols].head(5).to_string(index=False))

    best = df_sorted.iloc[0]
    sharpe_val = best.get('daily_sharpe', best.get('sharpe', 0))
    print(f"\n🎯 BEST RISK CONFIG:")
    print(f"   stop_buffer={best['stop_buffer']:.1f}, min_risk={best['min_risk']:.1f}")
    print(f"   → Daily Sharpe={sharpe_val:.3f}, Max DD={best['max_dd']:.2f}R")
    print(f"   → Recovery Factor={best.get('recovery_factor', 0):.2f}, MAR Ratio={best.get('mar_ratio', 0):.2f}")

    return best


def grid_search_phase_4_execution(days=180, best_regime=None, best_context=None,
                                  best_h2l2=None, best_risk=None):
    """
    Phase 4: Optimize EXECUTION (cooldown, max_trades_day)
    """
    print("\n" + "=" * 80)
    print("  PHASE 4: EXECUTION OPTIMIZATION")
    print("=" * 80 + "\n")

    if best_regime is None:
        best_regime = {"regime_filter": True, "chop_threshold": 2.0}
    if best_context is None:
        best_context = {"min_slope": 0.15, "ema_period": 20}
    if best_h2l2 is None:
        best_h2l2 = {"pullback_bars": 3, "signal_close_frac": 0.30}
    if best_risk is None:
        best_risk = {"stop_buffer": 1.0, "min_risk": 2.0}

    # Grid
    cooldowns = [0, 5, 10, 15, 20]
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
            regime_filter=best_regime["regime_filter"],
            chop_threshold=best_regime["chop_threshold"],
            costs_per_trade_r=0.04,
        )

        if "error" not in metrics:
            results.append({
                "cooldown": cooldown,
                "max_trades_day": max_day,
                **metrics
            })

    df = pd.DataFrame(results)
    sort_by = "daily_sharpe" if "daily_sharpe" in df.columns else "sharpe"
    df_sorted = df.sort_values(sort_by, ascending=False)

    print("\n" + "=" * 80)
    print("  TOP 5 CONFIGURATIONS (by Daily Sharpe)")
    print("=" * 80 + "\n")

    cols = ["cooldown", "max_trades_day", "trades", sort_by, "net_r", "expectancy"]
    print(df_sorted[cols].head(5).to_string(index=False))

    best = df_sorted.iloc[0]
    sharpe_val = best.get('daily_sharpe', best.get('sharpe', 0))
    print(f"\n🎯 BEST EXECUTION CONFIG:")
    print(f"   cooldown={int(best['cooldown'])}, max_trades_day={int(best['max_trades_day'])}")
    print(f"   → Daily Sharpe={sharpe_val:.3f}, Expectancy={best.get('expectancy', 0):+.4f}R")

    return best


def validate_final_config(config, days=340):
    """
    Validate final config on longer period (340 days like production)
    """
    print("\n" + "=" * 80)
    print("  🔬 FINAL VALIDATION (340 Days)")
    print("=" * 80 + "\n")

    metrics = run_backtest(
        symbol="US500.cash",
        days=days,
        max_trades_day=config["execution"]["max_trades_day"],
        min_slope=config["context"]["min_slope"],
        ema_period=config["context"]["ema_period"],
        pullback_bars=config["h2l2"]["pullback_bars"],
        signal_close_frac=config["h2l2"]["signal_close_frac"],
        stop_buffer=config["risk"]["stop_buffer"],
        min_risk_price_units=config["risk"]["min_risk_price_units"],
        cooldown_bars=config["execution"]["cooldown_bars"],
        regime_filter=config["regime"]["regime_filter"],
        chop_threshold=config["regime"]["chop_threshold"],
        costs_per_trade_r=0.04,
    )

    print("\n📊 VALIDATION RESULTS:")
    print(f"   Trades        : {metrics.get('trades', 0)}")
    print(f"   Net R         : {metrics.get('net_r', 0):+.2f}R")
    print(f"   Daily Sharpe  : {metrics.get('daily_sharpe', metrics.get('sharpe', 0)):.3f}")
    print(f"   Winrate       : {metrics.get('winrate', 0) * 100:.1f}%")
    print(f"   Max DD        : {metrics.get('max_dd', 0):.2f}R")
    print(f"   Recovery Factor: {metrics.get('recovery_factor', 0):.2f}")
    print(f"   MAR Ratio     : {metrics.get('mar_ratio', 0):.2f}")

    # Check if meets minimum standards
    sharpe = metrics.get('daily_sharpe', metrics.get('sharpe', 0))
    if sharpe < 1.5:
        print("\n⚠️  WARNING: Daily Sharpe < 1.5 (target for FTMO)")
    else:
        print(f"\n✅ EXCELLENT: Daily Sharpe {sharpe:.3f} > 1.5 (FTMO ready!)")

    return metrics


def main():
    print("\n" + "🎯" * 40)
    print("  BROOKS COMPREHENSIVE STRATEGY GRID SEARCH")
    print("  Updated for: Regime Filter + Costs + Daily Sharpe")
    print("🎯" * 40 + "\n")

    start_time = datetime.now()

    # Phase 0: Regime Filter (NEW - Most Important!)
    best_regime = grid_search_phase_0_regime(days=180)

    # Phase 1: Context (trend filter)
    best_context = grid_search_phase_1_context(days=180, best_regime=best_regime)

    # Phase 2: H2/L2 (using best regime + context)
    best_h2l2 = grid_search_phase_2_h2l2(
        days=180,
        best_regime=best_regime,
        best_context=best_context
    )

    # Phase 3: Risk (using best regime + context + h2l2)
    best_risk = grid_search_phase_3_risk(
        days=180,
        best_regime=best_regime,
        best_context=best_context,
        best_h2l2=best_h2l2
    )

    # Phase 4: Execution (using all best configs)
    best_execution = grid_search_phase_4_execution(
        days=180,
        best_regime=best_regime,
        best_context=best_context,
        best_h2l2=best_h2l2,
        best_risk=best_risk
    )

    # Build final config
    optimal_config = {
        "regime": {
            "regime_filter": bool(best_regime["regime_filter"]),
            "chop_threshold": float(best_regime["chop_threshold"]),
        },
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
        "costs": {
            "costs_per_trade_r": 0.04,
        },
        "performance_180d": {
            "daily_sharpe": float(best_execution.get('daily_sharpe', best_execution.get('sharpe', 0))),
            "net_r": float(best_execution["net_r"]),
            "winrate": float(best_execution["winrate"]),
            "trades": int(best_execution["trades"]),
            "max_dd": float(best_execution.get("max_dd", 0)),
        }
    }

    # Validate on 340 days (production length)
    validation_metrics = validate_final_config(optimal_config, days=340)

    optimal_config["performance_340d"] = {
        "daily_sharpe": float(validation_metrics.get('daily_sharpe', validation_metrics.get('sharpe', 0))),
        "net_r": float(validation_metrics.get("net_r", 0)),
        "winrate": float(validation_metrics.get("winrate", 0)),
        "trades": int(validation_metrics.get("trades", 0)),
        "max_dd": float(validation_metrics.get("max_dd", 0)),
        "recovery_factor": float(validation_metrics.get("recovery_factor", 0)),
        "mar_ratio": float(validation_metrics.get("mar_ratio", 0)),
    }

    # Final summary
    print("\n" + "=" * 80)
    print("  🏆 FINAL OPTIMAL CONFIGURATION")
    print("=" * 80 + "\n")

    print(json.dumps(optimal_config, indent=2))

    elapsed = datetime.now() - start_time
    print(f"\n⏱️  Total time: {elapsed}")

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimal_config_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(optimal_config, f, indent=2)
    print(f"\n💾 Saved to: {filename}")

    # Also save as latest
    with open("optimal_config.json", "w") as f:
        json.dump(optimal_config, f, indent=2)
    print("💾 Saved to: optimal_config.json (latest)")

    # Compare to current production config
    print("\n" + "=" * 80)
    print("  📊 COMPARISON TO CURRENT PRODUCTION")
    print("=" * 80 + "\n")

    production_config = {
        "regime_filter": True,
        "chop_threshold": 2.0,
        "min_slope": 0.15,
        "ema_period": 20,
        "pullback_bars": 3,
        "signal_close_frac": 0.30,
        "stop_buffer": 1.0,
        "min_risk": 2.0,
        "cooldown": 0,
        "max_trades_day": 2,
    }

    print("CURRENT PRODUCTION:")
    print(json.dumps(production_config, indent=2))

    print("\nNEW OPTIMAL:")
    new_config = {
        "regime_filter": optimal_config["regime"]["regime_filter"],
        "chop_threshold": optimal_config["regime"]["chop_threshold"],
        "min_slope": optimal_config["context"]["min_slope"],
        "ema_period": optimal_config["context"]["ema_period"],
        "pullback_bars": optimal_config["h2l2"]["pullback_bars"],
        "signal_close_frac": optimal_config["h2l2"]["signal_close_frac"],
        "stop_buffer": optimal_config["risk"]["stop_buffer"],
        "min_risk": optimal_config["risk"]["min_risk_price_units"],
        "cooldown": optimal_config["execution"]["cooldown_bars"],
        "max_trades_day": optimal_config["execution"]["max_trades_day"],
    }
    print(json.dumps(new_config, indent=2))

    # Highlight differences
    differences = []
    for key in production_config:
        if production_config[key] != new_config[key]:
            differences.append(f"  {key}: {production_config[key]} → {new_config[key]}")

    if differences:
        print("\n⚠️  DIFFERENCES FOUND:")
        for diff in differences:
            print(diff)
        print("\n⚠️  RECOMMENDATION: Test new config on DEMO before using in FTMO!")
    else:
        print("\n✅ OPTIMAL = PRODUCTION (Already using best config!)")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()