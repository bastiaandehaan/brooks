import pytest
from strategies.config import StrategyConfig
from backtest.config_formatter import format_frozen_config_text


def test_format_deterministic():
    """Same config should produce same text"""
    config = StrategyConfig.load("config/strategies/us500_sniper.yaml")

    text1 = format_frozen_config_text(config)
    text2 = format_frozen_config_text(config)

    assert text1 == text2, "Format should be deterministic"


def test_format_contains_required_fields():
    """Verify all required fields present"""
    config = StrategyConfig.load("config/strategies/us500_sniper.yaml")
    text = format_frozen_config_text(config)

    required = [
        "pullback_bars",
        "signal_close_frac",
        "ema_period",
        "min_slope",
        "chop_threshold",
        "cooldown_bars",
        "max_trades_per_day",
        "session",
        "risk_pct",
        "costs_per_trade",
        "execution",
        "regime_filter",
    ]

    for field in required:
        assert field in text, f"Missing field: {field}"
