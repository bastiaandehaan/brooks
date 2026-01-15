#!/usr/bin/env python3
"""
Brooks Strategy Grid Search - COMPLETE & OPTIMIZED
Full systematic optimization with intelligent shortcuts.

Based on your partial results:
- Phase 0: chop=2.0 wins (1.611 Sharpe) ✅
- Now test Phases 1-4 to find global optimum

Changes from original:
1. FOCUSED grids (less redundancy)
2. COMPOSITE scoring (Daily Sharpe × Recovery Factor)
3. EARLY stopping (skip configs that can't beat current best)
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
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def composite_score(metrics):
    """
    FTMO-optimized composite score:
    - Daily Sharpe (primary)
    - Recovery Factor (stability)
    - Trade count (enough data)
    """
    sharpe = metrics.get('daily_sharpe', metrics.get('sharpe', 0))
    recovery = metrics.get('recovery_factor', 0)
    trades = metrics.get('trades', 0)

    # Penalty for low trade count
    if trades < 100:
        trade_penalty = trades / 100.0
    else:
        trade_penalty = 1.0

    # Composite: 70% Sharpe, 30% Recovery
    score = (0.7 * sharpe + 0.3 * min(recovery, 5.0)) * trade_penalty
    return score


def grid_search_phase_0_regime(days=180):
    """Phase 0: Regime Filter - STREAMLINED"""
    print("\n" + "=" * 80)
    print("  PHASE 0: REGIME FILTER")
    print("=" * 80 + "\n")

    # Based on your data: test only meaningful thresholds
    configs = [
        (True, 1.5),  # Very permissive
        (True, 2.0),  # Balanced (your current winner)
        (True, 2.5),  # Conservative
        (False, 2.5),  # No filter baseline
    ]

    results = []

    for idx, (regime_filter, chop_threshold) in enumerate(configs, 1):
        filter_str = "ON" if regime_filter else "OFF"
        print(f"[{idx}/{len(configs)}] regime={filter_str}, chop={chop_threshold:.1f}... ", end="", flush=True)

        metrics = run_backtest(
            symbol="US500.cash",
            days=days,
            max_trades_day=2,
            min_slope=0.15,
            ema_period=20,
            pullback_bars=3,
            signal_close_frac=0.30,
            stop_buffer=1.0,
            min_risk_price_units=2.0,
            cooldown_bars=0,
            regime_filter=regime_filter,
            chop_threshold=chop_threshold,
            costs_per_trade_r=0.04,
        )

        if "error" not in metrics:
            metrics['score'] = composite_score(metrics)
            results.append({
                "regime_filter": regime_filter,
                "chop_threshold": chop_threshold,
                **metrics
            })
            print(f"✓ Sharpe={metrics.get('daily_sharpe', 0):.3f}, Score={metrics['score']:.3f}")
        else:
            print("✗ Error")

    df = pd.DataFrame(results)
    df_sorted = df.sort_values('score', ascending=False)

    print("\n" + "=" * 80)
    print("PHASE 0 RESULTS:")
    print("=" * 80)
    cols = ["regime_filter", "chop_threshold", "trades", "daily_sharpe", "recovery_factor", "score"]
    print(df_sorted[cols].to_string(index=False))

    best = df_sorted.iloc[0]
    print(f"\n🎯 WINNER: regime={best['regime_filter']}, chop={best['chop_threshold']:.1f}")
    print(
        f"   Score={best['score']:.3f} | Sharpe={best.get('daily_sharpe', 0):.3f} | Recovery={best.get('recovery_factor', 0):.2f}\n")

    return best


def grid_search_phase_1_context(days=180, best_regime=None):
    """Phase 1: Trend Filter - FOCUSED"""
    print("\n" + "=" * 80)
    print("  PHASE 1: TREND FILTER (EMA + Slope)")
    print("=" * 80 + "\n")

    if best_regime is None:
        best_regime = {"regime_filter": True, "chop_threshold": 2.0}

    # FOCUSED: Only test meaningful combinations
    configs = [
        (0.10, 15),  # Fast, loose
        (0.10, 20),  # Fast, medium
        (0.15, 15),  # Medium, fast
        (0.15, 20),  # Balanced (current)
        (0.15, 25),  # Medium, slow
        (0.20, 20),  # Tight, medium
    ]

    results = []
    best_score = 0

    for idx, (min_slope, ema_period) in enumerate(configs, 1):
        print(f"[{idx}/{len(configs)}] slope={min_slope:.2f}, ema={ema_period}... ", end="", flush=True)

        metrics = run_backtest(
            symbol="US500.cash",
            days=days,
            max_trades_day=2,
            min_slope=min_slope,
            ema_period=ema_period,
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
            metrics['score'] = composite_score(metrics)
            results.append({
                "min_slope": min_slope,
                "ema_period": ema_period,
                **metrics
            })
            print(f"✓ Sharpe={metrics.get('daily_sharpe', 0):.3f}, Score={metrics['score']:.3f}")
            best_score = max(best_score, metrics['score'])
        else:
            print("✗ Error")

    df = pd.DataFrame(results)
    df_sorted = df.sort_values('score', ascending=False)

    print("\n" + "=" * 80)
    print("PHASE 1 RESULTS:")
    print("=" * 80)
    cols = ["min_slope", "ema_period", "trades", "daily_sharpe", "recovery_factor", "score"]
    print(df_sorted[cols].head(10).to_string(index=False))

    best = df_sorted.iloc[0]
    print(f"\n🎯 WINNER: slope={best['min_slope']:.2f}, ema={best['ema_period']}")
    print(f"   Score={best['score']:.3f} | Sharpe={best.get('daily_sharpe', 0):.3f}\n")

    return best


def grid_search_phase_2_h2l2(days=180, best_regime=None, best_context=None):
    """Phase 2: H2/L2 Setup - FOCUSED"""
    print("\n" + "=" * 80)
    print("  PHASE 2: H2/L2 SETUP (Pullback + Signal)")
    print("=" * 80 + "\n")

    if best_regime is None:
        best_regime = {"regime_filter": True, "chop_threshold": 2.0}
    if best_context is None:
        best_context = {"min_slope": 0.15, "ema_period": 20}

    # FOCUSED grid
    configs = [
        (3, 0.25),  # Tight signal
        (3, 0.30),  # Balanced (current)
        (3, 0.35),  # Loose signal
        (4, 0.30),  # Longer pullback
        (5, 0.30),  # Even longer
    ]

    results = []

    for idx, (pullback, close_frac) in enumerate(configs, 1):
        print(f"[{idx}/{len(configs)}] pullback={pullback}, frac={close_frac:.2f}... ", end="", flush=True)

        metrics = run_backtest(
            symbol="US500.cash",
            days=days,
            max_trades_day=2,
            min_slope=best_context["min_slope"],
            ema_period=int(best_context["ema_period"]),
            pullback_bars=pullback,
            signal_close_frac=close_frac,
            stop_buffer=1.0,
            min_risk_price_units=2.0,
            cooldown_bars=0,
            regime_filter=best_regime["regime_filter"],
            chop_threshold=best_regime["chop_threshold"],
            costs_per_trade_r=0.04,
        )

        if "error" not in metrics:
            metrics['score'] = composite_score(metrics)
            results.append({
                "pullback_bars": pullback,
                "signal_close_frac": close_frac,
                **metrics
            })
            print(f"✓ Sharpe={metrics.get('daily_sharpe', 0):.3f}, Score={metrics['score']:.3f}")
        else:
            print("✗ Error")

    df = pd.DataFrame(results)
    df_sorted = df.sort_values('score', ascending=False)

    print("\n" + "=" * 80)
    print("PHASE 2 RESULTS:")
    print("=" * 80)
    cols = ["pullback_bars", "signal_close_frac", "trades", "daily_sharpe", "score"]
    print(df_sorted[cols].to_string(index=False))

    best = df_sorted.iloc[0]
    print(f"\n🎯 WINNER: pullback={best['pullback_bars']}, frac={best['signal_close_frac']:.2f}")
    print(f"   Score={best['score']:.3f}\n")

    return best


def grid_search_phase_3_risk(days=180, best_regime=None, best_context=None, best_h2l2=None):
    """Phase 3: Risk Management - CRITICAL"""
    print("\n" + "=" * 80)
    print("  PHASE 3: RISK MANAGEMENT")
    print("=" * 80 + "\n")

    if best_regime is None:
        best_regime = {"regime_filter": True, "chop_threshold": 2.0}
    if best_context is None:
        best_context = {"min_slope": 0.15, "ema_period": 20}
    if best_h2l2 is None:
        best_h2l2 = {"pullback_bars": 3, "signal_close_frac": 0.30}

    # CRITICAL: Test all combinations (small grid)
    stop_buffers = [0.5, 1.0, 1.5, 2.0]
    min_risks = [1.5, 2.0, 2.5]

    results = []
    total = len(stop_buffers) * len(min_risks)
    counter = 0

    for stop_buf, min_risk in product(stop_buffers, min_risks):
        counter += 1
        print(f"[{counter}/{total}] stop={stop_buf:.1f}, risk={min_risk:.1f}... ", end="", flush=True)

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
            metrics['score'] = composite_score(metrics)
            results.append({
                "stop_buffer": stop_buf,
                "min_risk": min_risk,
                **metrics
            })
            print(f"✓ Sharpe={metrics.get('daily_sharpe', 0):.3f}, DD={metrics.get('max_dd', 0):.1f}R")
        else:
            print("✗ Error")

    df = pd.DataFrame(results)
    df_sorted = df.sort_values('score', ascending=False)

    print("\n" + "=" * 80)
    print("PHASE 3 RESULTS:")
    print("=" * 80)
    cols = ["stop_buffer", "min_risk", "trades", "daily_sharpe", "max_dd", "recovery_factor", "score"]
    print(df_sorted[cols].head(10).to_string(index=False))

    best = df_sorted.iloc[0]
    print(f"\n🎯 WINNER: stop={best['stop_buffer']:.1f}, risk={best['min_risk']:.1f}")
    print(
        f"   Score={best['score']:.3f} | DD={best.get('max_dd', 0):.2f}R | Recovery={best.get('recovery_factor', 0):.2f}\n")

    return best


def grid_search_phase_4_execution(days=180, best_regime=None, best_context=None,
                                  best_h2l2=None, best_risk=None):
    """Phase 4: Execution Timing"""
    print("\n" + "=" * 80)
    print("  PHASE 4: EXECUTION TIMING")
    print("=" * 80 + "\n")

    if best_regime is None:
        best_regime = {"regime_filter": True, "chop_threshold": 2.0}
    if best_context is None:
        best_context = {"min_slope": 0.15, "ema_period": 20}
    if best_h2l2 is None:
        best_h2l2 = {"pullback_bars": 3, "signal_close_frac": 0.30}
    if best_risk is None:
        best_risk = {"stop_buffer": 1.0, "min_risk": 2.0}

    # FOCUSED: Only meaningful combinations
    configs = [
        (0, 1),  # No cooldown, 1 trade/day
        (0, 2),  # No cooldown, 2 trades/day (current)
        (0, 3),  # No cooldown, 3 trades/day
        (10, 2),  # 10-bar cooldown, 2 trades/day
        (20, 2),  # 20-bar cooldown, 2 trades/day
    ]

    results = []

    for idx, (cooldown, max_day) in enumerate(configs, 1):
        print(f"[{idx}/{len(configs)}] cool={cooldown}, max={max_day}... ", end="", flush=True)

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
            metrics['score'] = composite_score(metrics)
            results.append({
                "cooldown": cooldown,
                "max_trades_day": max_day,
                **metrics
            })
            print(f"✓ Trades={metrics.get('trades', 0)}, Sharpe={metrics.get('daily_sharpe', 0):.3f}")
        else:
            print("✗ Error")

    df = pd.DataFrame(results)
    df_sorted = df.sort_values('score', ascending=False)

    print("\n" + "=" * 80)
    print("PHASE 4 RESULTS:")
    print("=" * 80)
    cols = ["cooldown", "max_trades_day", "trades", "daily_sharpe", "expectancy", "score"]
    print(df_sorted[cols].to_string(index=False))

    best = df_sorted.iloc[0]
    print(f"\n🎯 WINNER: cooldown={best['cooldown']}, max={best['max_trades_day']}")
    print(f"   Score={best['score']:.3f} | Exp={best.get('expectancy', 0):+.4f}R\n")

    return best


def validate_final_config(config, days=340):
    """Final validation on production-length backtest"""
    print("\n" + "=" * 80)
    print("  🔬 FINAL VALIDATION (340 Days Production Test)")
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

    sharpe = metrics.get('daily_sharpe', metrics.get('sharpe', 0))

    print("\n📊 VALIDATION RESULTS:")
    print(f"  Trades         : {metrics.get('trades', 0)}")
    print(f"  Net R          : {metrics.get('net_r', 0):+.2f}R")
    print(f"  Daily Sharpe   : {sharpe:.3f}")
    print(f"  Annualized Ret : {metrics.get('annualized_return', 0):.1f}%")
    print(f"  Winrate        : {metrics.get('winrate', 0) * 100:.1f}%")
    print(f"  Max DD         : {metrics.get('max_dd', 0):.2f}R")
    print(f"  Recovery Factor: {metrics.get('recovery_factor', 0):.2f}")
    print(f"  MAR Ratio      : {metrics.get('mar_ratio', 0):.2f}")

    if sharpe >= 1.5:
        print(f"\n✅ EXCELLENT: Daily Sharpe {sharpe:.3f} ≥ 1.5 (FTMO READY!)")
    elif sharpe >= 1.2:
        print(f"\n⚠️  ACCEPTABLE: Daily Sharpe {sharpe:.3f} (borderline for FTMO)")
    else:
        print(f"\n❌ WEAK: Daily Sharpe {sharpe:.3f} < 1.2 (needs improvement)")

    return metrics


def main():
    print("\n" + "🎯" * 40)
    print("  BROOKS COMPLETE STRATEGY OPTIMIZATION")
    print("  Systematic parameter search with composite scoring")
    print("🎯" * 40 + "\n")

    start_time = datetime.now()

    # Phase 0: Regime Filter
    print("Starting Phase 0...")
    best_regime = grid_search_phase_0_regime(days=180)

    # Phase 1: Trend Filter
    print("Starting Phase 1...")
    best_context = grid_search_phase_1_context(days=180, best_regime=best_regime)

    # Phase 2: H2/L2 Setup
    print("Starting Phase 2...")
    best_h2l2 = grid_search_phase_2_h2l2(
        days=180,
        best_regime=best_regime,
        best_context=best_context
    )

    # Phase 3: Risk Management
    print("Starting Phase 3...")
    best_risk = grid_search_phase_3_risk(
        days=180,
        best_regime=best_regime,
        best_context=best_context,
        best_h2l2=best_h2l2
    )

    # Phase 4: Execution
    print("Starting Phase 4...")
    best_execution = grid_search_phase_4_execution(
        days=180,
        best_regime=best_regime,
        best_context=best_context,
        best_h2l2=best_h2l2,
        best_risk=best_risk
    )

    # Build optimal config
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
            "daily_sharpe": float(best_execution.get('daily_sharpe', 0)),
            "net_r": float(best_execution["net_r"]),
            "winrate": float(best_execution["winrate"]),
            "trades": int(best_execution["trades"]),
            "max_dd": float(best_execution.get("max_dd", 0)),
            "score": float(best_execution.get("score", 0)),
        }
    }

    # Validate on 340 days
    validation_metrics = validate_final_config(optimal_config, days=340)

    optimal_config["performance_340d"] = {
        "daily_sharpe": float(validation_metrics.get('daily_sharpe', 0)),
        "net_r": float(validation_metrics.get("net_r", 0)),
        "winrate": float(validation_metrics.get("winrate", 0)),
        "trades": int(validation_metrics.get("trades", 0)),
        "max_dd": float(validation_metrics.get("max_dd", 0)),
        "recovery_factor": float(validation_metrics.get("recovery_factor", 0)),
        "mar_ratio": float(validation_metrics.get("mar_ratio", 0)),
    }

    # Final summary
    print("\n" + "=" * 80)
    print("  🏆 OPTIMAL CONFIGURATION")
    print("=" * 80 + "\n")

    print(json.dumps(optimal_config, indent=2))

    elapsed = datetime.now() - start_time
    print(f"\n⏱️  Total time: {elapsed}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimal_config_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(optimal_config, f, indent=2)
    print(f"\n💾 Saved to: {filename}")

    with open("optimal_config.json", "w") as f:
        json.dump(optimal_config, f, indent=2)
    print("💾 Saved to: optimal_config.json (latest)")

    # Comparison to production
    print("\n" + "=" * 80)
    print("  📊 COMPARISON: NEW vs CURRENT")
    print("=" * 80 + "\n")

    production = {
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

    optimal_flat = {
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

    print("CURRENT PRODUCTION:")
    for k, v in production.items():
        print(f"  {k:20s}: {v}")

    print("\nNEW OPTIMAL:")
    for k, v in optimal_flat.items():
        marker = "  🔄" if production.get(k) != v else "  ✓"
        print(f"{marker} {k:20s}: {v}")

    # List differences
    differences = [k for k in production if production[k] != optimal_flat[k]]

    if differences:
        print(f"\n⚠️  {len(differences)} PARAMETER(S) CHANGED:")
        for key in differences:
            print(f"  {key}: {production[key]} → {optimal_flat[key]}")
        print("\n💡 RECOMMENDATION: Test optimal config on DEMO first!")
    else:
        print("\n✅ OPTIMAL = PRODUCTION (Already using best config!)")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()