#!/usr/bin/env python3
"""
Convert Optimizer JSON Output to YAML Strategy Config

Usage:
    python tools/convert_optimizer_json_to_yaml.py optimal_config_20260113_162721.json

Outputs:
    config/strategies/us500_optimal_TIMESTAMP.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml


def convert_optimizer_json_to_yaml(json_path: str, output_path: str | None = None) -> str:
    """
    Convert optimizer JSON format to StrategyConfig YAML format.

    Args:
        json_path: Path to optimizer JSON file
        output_path: Optional custom output path

    Returns:
        Path to generated YAML file
    """
    # Read JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📖 Loaded optimizer config: {json_path}")
    print(f"   Keys: {list(data.keys())}")

    # Build YAML structure (map optimizer keys to StrategyConfig structure)
    yaml_config = {
        "symbol": data.get("symbol", "US500.cash"),
        "regime": {
            "enabled": data.get("regime_filter", True),
            "chop_threshold": data.get("chop_threshold", 2.5),
            "atr_period": 14,  # Default (not typically in optimizer output)
            "range_period": 20,
        },
        "trend": {
            "ema_period": data.get("ema_period", 20),
            "min_slope": data.get("min_slope", 0.15),
        },
        "h2l2": {
            "pullback_bars": data.get("pullback_bars", 3),
            "signal_close_frac": data.get("signal_close_frac", 0.30),
            "min_risk_price_units": data.get("min_risk_price_units", 2.0),
            "stop_buffer": data.get("stop_buffer", 1.0),
            "cooldown_bars": data.get("cooldown_bars", 0),
        },
        "guardrails": {
            "session_tz": "America/New_York",
            "day_tz": "America/New_York",
            "session_start": "09:30",
            "session_end": "16:00",
            "max_trades_per_day": data.get("max_trades_day", 2),
        },
        "risk": {
            "risk_pct": data.get("risk_pct", 1.0),
        },
        "costs": {
            "per_trade_r": data.get("costs_per_trade_r", 0.04),
        },
    }

    # Generate output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"config/strategies/us500_optimal_{timestamp}.yaml"

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write YAML
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Converted to YAML: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("  📋 STRATEGY CONFIG SUMMARY")
    print("=" * 80)
    print(f"  Symbol: {yaml_config['symbol']}")
    print(f"  Regime Filter: {yaml_config['regime']['enabled']}")
    if yaml_config["regime"]["enabled"]:
        print(f"  Chop Threshold: {yaml_config['regime']['chop_threshold']}")
    print(f"  EMA Period: {yaml_config['trend']['ema_period']}")
    print(f"  Pullback Bars: {yaml_config['h2l2']['pullback_bars']}")
    print(f"  Signal Close Frac: {yaml_config['h2l2']['signal_close_frac']}")
    print(f"  Stop Buffer: {yaml_config['h2l2']['stop_buffer']}")
    print(f"  Cooldown: {yaml_config['h2l2']['cooldown_bars']}")
    print(f"  Max Trades/Day: {yaml_config['guardrails']['max_trades_per_day']}")
    print(f"  Risk %: {yaml_config['risk']['risk_pct']}")
    print(f"  Costs (R): {yaml_config['costs']['per_trade_r']}")
    print("=" * 80)

    # Add optimizer metrics to YAML as comments if available
    if "metrics" in data or any(k in data for k in ["net_r", "winrate", "daily_sharpe_r"]):
        print("\n💡 TIP: Add these optimizer results as comments to the YAML:")
        if "net_r" in data:
            print(f"   # Net R: {data['net_r']:.2f}")
        if "winrate" in data:
            print(f"   # Winrate: {data['winrate'] * 100:.1f}%")
        if "daily_sharpe_r" in data:
            print(f"   # Daily Sharpe: {data['daily_sharpe_r']:.3f}")

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert optimizer JSON to StrategyConfig YAML")
    parser.add_argument(
        "json_file", help="Path to optimizer JSON file (e.g., optimal_config_20260113_162721.json)"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Custom output path (default: config/strategies/us500_optimal_TIMESTAMP.yaml)",
    )

    args = parser.parse_args()

    if not Path(args.json_file).exists():
        print(f"❌ Error: File not found: {args.json_file}")
        return 1

    try:
        convert_optimizer_json_to_yaml(args.json_file, args.output)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
