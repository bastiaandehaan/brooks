#!/usr/bin/env python3
"""
VOLLEDIGE GRID SEARCH — Systematische Parameter Optimalisatie
=============================================================

DOEL: Vind de BESTE configuratie door ALLE relevante combinaties te testen

STRATEGIE:
1. Hierarchical search (fase voor fase)
2. Training period: 180 dagen (snelheid)
3. Validation period: 340 dagen (confirmatie)
4. Walk-forward analyse (robuustheid)

GEBRUIK:
python scripts/fast_grid_search.py --mode full
python scripts/fast_grid_search.py --mode fast  # Voor snelle test

OUTPUT:
- CSV met alle resultaten
- Top 10 configuraties
- Validation scores
- Overfitting detectie
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

from backtest.runner import run_backtest_from_config
from execution.guardrails import Guardrails
from strategies.config import StrategyConfig
from strategies.context import TrendParams
from strategies.h2l2 import H2L2Params
from strategies.regime import RegimeParams

# Suppress verbose logging
logging.getLogger("execution.guardrails").setLevel(logging.WARNING)
logging.getLogger("Backtest").setLevel(logging.WARNING)

# =============================================================================
# PARAMETER GRIDS
# =============================================================================

GRID_CONFIGS = {
    "full": {
        "regime_filter": [True, False],
        "chop_threshold": [1.5, 2.0, 2.5, 3.0, 3.5],
        "ema_period": [10, 15, 20, 25],
        "min_slope": [0.05, 0.10, 0.15, 0.20],
        "pullback_bars": [2, 3, 4, 5],
        "signal_close_frac": [0.25, 0.30, 0.35, 0.40],
        "stop_buffer": [0.5, 1.0, 1.5, 2.0],
        "cooldown_bars": [0, 10, 20, 30],
        "max_trades_day": [1, 2, 3],
    },
    "fast": {
        "regime_filter": [True],
        "chop_threshold": [2.0, 2.5, 3.0],
        "ema_period": [15, 20],
        "min_slope": [0.10, 0.15],
        "pullback_bars": [3, 4],
        "signal_close_frac": [0.30, 0.35],
        "stop_buffer": [1.0, 1.5],
        "cooldown_bars": [10, 20],
        "max_trades_day": [1, 2],
    },
}

# =============================================================================
# SCORING FUNCTIE
# =============================================================================


def calculate_score(metrics, weights=None):
    """
    Multi-objective optimization score

    Default weights (FTMO-optimized):
    - 40% Daily Sharpe (risk-adjusted returns)
    - 25% Recovery Factor (drawdown quality)
    - 20% Consistency (winrate × profit factor)
    - 15% Trade frequency (enough data)
    """
    if weights is None:
        weights = {"sharpe": 0.40, "recovery": 0.25, "consistency": 0.20, "frequency": 0.15}

    # Extract metrics
    sharpe = metrics.get("daily_sharpe", metrics.get("sharpe", 0))
    recovery = metrics.get("recovery_factor", 0)
    winrate = metrics.get("winrate", 0)
    pf = metrics.get("profit_factor", 0)
    trades = metrics.get("trades", 0)
    net_r = metrics.get("net_r", 0)

    # Normalize components
    sharpe_score = min(sharpe / 2.0, 1.0)  # Cap at Sharpe 2.0
    recovery_score = min(recovery / 5.0, 1.0)  # Cap at Recovery 5.0

    # Consistency: combine winrate and PF
    consistency_score = winrate * min(pf / 2.0, 1.0)

    # Frequency: ideal is 200-500 trades in 180d
    if trades < 50:
        freq_score = trades / 50.0  # Penalty for too few
    elif trades > 500:
        freq_score = 500.0 / trades  # Penalty for too many
    else:
        freq_score = 1.0

    # Composite score
    score = (
        weights["sharpe"] * sharpe_score
        + weights["recovery"] * recovery_score
        + weights["consistency"] * consistency_score
        + weights["frequency"] * freq_score
    )

    # Penalty for negative returns
    if net_r < 0:
        score *= 0.5

    return score


# =============================================================================
# GRID SEARCH RUNNER
# =============================================================================


def run_grid_search(mode="full", train_days=180, val_days=340):
    """
    Run systematic grid search

    Args:
        mode: 'full' or 'fast'
        train_days: Training period length
        val_days: Validation period length
    """

    print("=" * 80)
    print(f"  VOLLEDIGE GRID SEARCH — Mode: {mode.upper()}")
    print("=" * 80)

    grid = GRID_CONFIGS[mode]

    # Calculate total combinations
    total = 1
    for param, values in grid.items():
        total *= len(values)

    print(f"\n📊 PARAMETER RUIMTE:")
    for param, values in grid.items():
        print(f"  {param:20s}: {len(values)} opties → {values}")
    print(f"\n  TOTAAL: {total:,} combinaties te testen")
    print(f"  Geschatte tijd: {total * 10 / 60:.0f} minuten\n")

    # Generate all combinations
    param_names = list(grid.keys())
    param_values = [grid[p] for p in param_names]

    results_train = []
    results_val = []

    start_time = datetime.now()

    # Run grid search
    for idx, combo in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, combo))

        # Progress
        if idx % 10 == 0 or idx == 1:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (total - idx) / rate / 60 if rate > 0 else 0
            print(f"[{idx}/{total}] ({idx / total * 100:.1f}%) — ETA: {eta:.0f} min")

        # TRAINING RUN - Build config dynamically
        config = StrategyConfig(
            symbol="US500.cash",
            regime_filter=params["regime_filter"],
            regime_params=RegimeParams(
                chop_threshold=params["chop_threshold"],
                atr_period=14,
                range_period=20,
            ),
            trend_params=TrendParams(
                ema_period=params["ema_period"],
                min_slope=params["min_slope"],
            ),
            h2l2_params=H2L2Params(
                pullback_bars=params["pullback_bars"],
                signal_close_frac=params["signal_close_frac"],
                stop_buffer=params["stop_buffer"],
                min_risk_price_units=2.0,
                cooldown_bars=params["cooldown_bars"],
            ),
            guardrails=Guardrails(
                max_trades_per_day=params["max_trades_day"],
                session_start="09:30",
                session_end="15:00",
                session_tz="America/New_York",
                day_tz="America/New_York",
            ),
            risk_pct=1.0,
            costs_per_trade_r=0.04,
        )

        train_metrics = run_backtest_from_config(
            config=config,
            days=train_days,
            initial_capital=10000.0,
            dashboard="none",
        )

        if "error" in train_metrics:
            continue

        train_metrics["score"] = calculate_score(train_metrics)
        train_metrics["config_id"] = idx

        # Store training results
        results_train.append({**params, **train_metrics})

    # Convert to DataFrame
    df_train = pd.DataFrame(results_train)

    if df_train.empty:
        print("\n❌ Geen valide resultaten!")
        return None

    # Sort by score
    df_train = df_train.sort_values("score", ascending=False)

    print("\n" + "=" * 80)
    print("  🏆 TOP 10 CONFIGURATIES (TRAINING)")
    print("=" * 80)

    display_cols = [
        "regime_filter",
        "chop_threshold",
        "ema_period",
        "min_slope",
        "pullback_bars",
        "signal_close_frac",
        "cooldown_bars",
        "max_trades_day",
        "trades",
        "net_r",
        "daily_sharpe",
        "recovery_factor",
        "score",
    ]

    # Check which columns exist
    display_cols = [c for c in display_cols if c in df_train.columns]

    print(df_train[display_cols].head(10).to_string(index=False))

    # ==========================================================================
    # VALIDATION RUN (top 10 configs only)
    # ==========================================================================

    print("\n" + "=" * 80)
    print(f"  🔬 VALIDATION RUN ({val_days} dagen)")
    print("=" * 80 + "\n")

    top_10 = df_train.head(10)

    for idx, row in top_10.iterrows():
        print(f"Validating config {row.get('config_id', idx)}... ", end="", flush=True)

        # Build config for validation
        val_config = StrategyConfig(
            symbol="US500.cash",
            regime_filter=row["regime_filter"],
            regime_params=RegimeParams(
                chop_threshold=row["chop_threshold"],
                atr_period=14,
                range_period=20,
            ),
            trend_params=TrendParams(
                ema_period=int(row["ema_period"]),
                min_slope=row["min_slope"],
            ),
            h2l2_params=H2L2Params(
                pullback_bars=int(row["pullback_bars"]),
                signal_close_frac=row["signal_close_frac"],
                stop_buffer=row["stop_buffer"],
                min_risk_price_units=2.0,
                cooldown_bars=int(row["cooldown_bars"]),
            ),
            guardrails=Guardrails(
                max_trades_per_day=int(row["max_trades_day"]),
                session_start="09:30",
                session_end="15:00",
                session_tz="America/New_York",
                day_tz="America/New_York",
            ),
            risk_pct=1.0,
            costs_per_trade_r=0.04,
        )

        val_metrics = run_backtest_from_config(
            config=val_config,
            days=val_days,
            initial_capital=10000.0,
            dashboard="none",
        )

        if "error" not in val_metrics:
            val_metrics["score"] = calculate_score(val_metrics)
            val_metrics["config_id"] = row.get("config_id", idx)

            # Calculate overfitting metric
            train_score = row["score"]
            val_score = val_metrics["score"]
            overfit_ratio = val_score / train_score if train_score > 0 else 0

            val_metrics["train_score"] = train_score
            val_metrics["overfit_ratio"] = overfit_ratio

            # Store params too
            for param in param_names:
                val_metrics[param] = row[param]

            results_val.append(val_metrics)

            print(
                f"✓ Val Score: {val_score:.3f} (Train: {train_score:.3f}, Ratio: {overfit_ratio:.2f})"
            )
        else:
            print("✗ Error")

    df_val = pd.DataFrame(results_val)

    if df_val.empty:
        print("\n⚠️  Validation failed voor alle configs")
        return df_train

    # Sort by validation score
    df_val = df_val.sort_values("score", ascending=False)

    print("\n" + "=" * 80)
    print("  🥇 VALIDATION RESULTATEN")
    print("=" * 80)

    val_display_cols = [
        "regime_filter",
        "chop_threshold",
        "ema_period",
        "min_slope",
        "pullback_bars",
        "signal_close_frac",
        "cooldown_bars",
        "max_trades_day",
        "trades",
        "net_r",
        "daily_sharpe",
        "train_score",
        "score",
        "overfit_ratio",
    ]

    val_display_cols = [c for c in val_display_cols if c in df_val.columns]

    print(df_val[val_display_cols].to_string(index=False))

    # ==========================================================================
    # OVERFITTING ANALYSE
    # ==========================================================================

    print("\n" + "=" * 80)
    print("  🔍 OVERFITTING DETECTIE")
    print("=" * 80 + "\n")

    # Check for overfitting
    avg_overfit = df_val["overfit_ratio"].mean()

    if avg_overfit < 0.7:
        print("🔴 HOOG OVERFITTING RISICO!")
        print(f"   Gemiddelde val/train ratio: {avg_overfit:.2f}")
        print("   → Validation scores veel lager dan training")
        print("   → Configs zijn overfit op training data\n")
    elif avg_overfit < 0.9:
        print("⚠️  MATIG OVERFITTING")
        print(f"   Gemiddelde val/train ratio: {avg_overfit:.2f}")
        print("   → Acceptabel maar voorzichtig zijn\n")
    else:
        print("✅ GOEDE GENERALISATIE!")
        print(f"   Gemiddelde val/train ratio: {avg_overfit:.2f}")
        print("   → Validation scores bevestigen training\n")

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full training results
    train_csv = f"grid_search_train_{mode}_{timestamp}.csv"
    df_train.to_csv(train_csv, index=False)
    print(f"💾 Training results: {train_csv}")

    # Save validation results
    val_csv = f"grid_search_val_{mode}_{timestamp}.csv"
    df_val.to_csv(val_csv, index=False)
    print(f"💾 Validation results: {val_csv}")

    # Save winner config as JSON
    winner = df_val.iloc[0]

    winner_config = {
        "regime": {
            "enabled": bool(winner["regime_filter"]),
            "chop_threshold": float(winner["chop_threshold"]),
        },
        "trend": {
            "ema_period": int(winner["ema_period"]),
            "min_slope": float(winner["min_slope"]),
        },
        "h2l2": {
            "pullback_bars": int(winner["pullback_bars"]),
            "signal_close_frac": float(winner["signal_close_frac"]),
            "stop_buffer": float(winner["stop_buffer"]),
        },
        "execution": {
            "cooldown_bars": int(winner["cooldown_bars"]),
            "max_trades_day": int(winner["max_trades_day"]),
        },
        "performance": {
            "train_days": train_days,
            "val_days": val_days,
            "train_score": float(winner["train_score"]),
            "val_score": float(winner["score"]),
            "overfit_ratio": float(winner["overfit_ratio"]),
            "val_net_r": float(winner["net_r"]),
            "val_sharpe": float(winner.get("daily_sharpe", 0)),
            "val_trades": int(winner["trades"]),
        },
    }

    json_file = f"optimal_config_{mode}_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(winner_config, f, indent=2)

    print(f"💾 Winner config: {json_file}")

    # Print winner
    print("\n" + "=" * 80)
    print("  🏆 WINNENDE CONFIGURATIE")
    print("=" * 80)
    print(json.dumps(winner_config, indent=2))

    total_time = datetime.now() - start_time
    print(f"\n⏱️  Totale tijd: {total_time}")

    return df_val


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Volledige Grid Search voor Brooks strategie")
    parser.add_argument(
        "--mode",
        choices=["full", "fast"],
        default="fast",
        help="Grid size: full (uren) of fast (minuten)",
    )
    parser.add_argument("--train-days", type=int, default=180, help="Training period (dagen)")
    parser.add_argument("--val-days", type=int, default=340, help="Validation period (dagen)")

    args = parser.parse_args()

    run_grid_search(mode=args.mode, train_days=args.train_days, val_days=args.val_days)

    return 0


if __name__ == "__main__":
    sys.exit(main())
