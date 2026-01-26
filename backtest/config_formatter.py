# backtest/config_formatter.py
"""
Config Formatter - Single Source of Truth for Config Display
Used by both logging and PNG generation to prevent drift.
"""

from strategies.config import StrategyConfig


def format_frozen_config_text(config: StrategyConfig) -> str:
    """
    Generate deterministic config text for drift checking.

    This EXACT format is used in:
    - runner.py logging (after config load)
    - visualiser_v2.py PNG config box

    Smoke test verifies: log text == PNG text

    Args:
        config: Strategy configuration

    Returns:
        Formatted config string (deterministic)
    """
    lines = [
        "STRATEGY CONFIG (FROZEN)",
        "=" * 50,
        f"pullback_bars: {config.h2l2_params.pullback_bars}",
        f"signal_close_frac: {config.h2l2_params.signal_close_frac:.2f}",
        f"ema_period: {config.trend_params.ema_period}",
        f"min_slope: {config.trend_params.min_slope:.2f}",
        f"chop_threshold: {config.regime_params.chop_threshold:.1f}",
        f"cooldown_bars: {config.h2l2_params.cooldown_bars}",
        f"max_trades_per_day: {config.guardrails.max_trades_per_day}",
        f"session: {config.guardrails.session_start}-{config.guardrails.session_end} {config.guardrails.session_tz}",
        f"risk_pct: {config.risk_pct:.1f}",
        f"costs_per_trade: {config.costs_per_trade_r:.4f}R",
        f"execution: NEXT_OPEN",
        f"regime_filter: {'ON' if config.regime_filter else 'OFF'}",
        "=" * 50,
    ]

    return "\n".join(lines)
