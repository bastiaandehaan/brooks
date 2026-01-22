# strategies/config.py
"""
SINGLE SOURCE OF TRUTH for Strategy Configuration
Zero-drift guarantee between backtest and live execution.

Features:
- YAML/JSON loading with automatic format detection
- Key aliasing (max_trades_day vs max_trades_per_day)
- Config hash logging for audit trail
- Validation with clear error messages
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from execution.guardrails import Guardrails
from strategies.context import TrendParams
from strategies.h2l2 import H2L2Params
from strategies.regime import RegimeParams

logger = logging.getLogger(__name__)

# Key aliases for backwards compatibility and common naming variations
KEY_ALIASES = {
    "max_trades_day": "max_trades_per_day",
    "max_trades_per_day": "max_trades_per_day",
    "ema": "ema_period",
    "min_risk": "min_risk_price_units",
    "cooldown": "cooldown_bars",
    "costs": "costs_per_trade_r",
    "per_trade_r": "costs_per_trade_r",
}


def _normalize_keys(data: Dict[str, Any], aliases: Dict[str, str]) -> Dict[str, Any]:
    """Apply key aliasing to dict (non-recursive for top-level keys)"""
    normalized = {}
    for key, value in data.items():
        canonical_key = aliases.get(key, key)
        normalized[canonical_key] = value
    return normalized


@dataclass(frozen=True)
class StrategyConfig:
    """
    Complete strategy configuration - Single Source of Truth
    Used by BOTH backtest and live to prevent parameter drift
    """

    # Instrument
    symbol: str = "US500.cash"

    # Regime Filter
    regime_filter: bool = True
    regime_params: RegimeParams = None

    # Trend Context
    trend_params: TrendParams = None

    # H2/L2 Strategy
    h2l2_params: H2L2Params = None

    # Execution Guardrails
    guardrails: Guardrails = None

    # Risk Management
    risk_pct: float = 1.0

    # Transaction Costs (R-units for consistency)
    costs_per_trade_r: float = 0.04

    def __post_init__(self):
        """Initialize default sub-configs if not provided"""
        if self.regime_params is None:
            object.__setattr__(self, "regime_params", RegimeParams())
        if self.trend_params is None:
            object.__setattr__(self, "trend_params", TrendParams())
        if self.h2l2_params is None:
            object.__setattr__(self, "h2l2_params", H2L2Params())
        if self.guardrails is None:
            object.__setattr__(self, "guardrails", Guardrails())

    @classmethod
    def load(cls, filepath: str) -> "StrategyConfig":
        """
        MAIN LOADER - Auto-detects YAML or JSON and loads config.
        Logs config hash for audit trail.

        Usage:
            config = StrategyConfig.load("config/strategies/us500_optimal.yaml")
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        # Read raw content for hash
        raw_content = path.read_text(encoding="utf-8")
        config_hash = hashlib.sha256(raw_content.encode()).hexdigest()[:12]

        # Auto-detect format
        suffix = path.suffix.lower()
        if suffix in [".yaml", ".yml"]:
            config = cls.from_yaml(filepath)
        elif suffix == ".json":
            config = cls.from_json(filepath)
        else:
            raise ValueError(f"Unsupported config format: {suffix}. Use .yaml or .json")

        logger.info("=" * 80)
        logger.info("📋 LOADED STRATEGY CONFIG")
        logger.info("=" * 80)
        logger.info(f"  File: {filepath}")
        logger.info(f"  Hash: {config_hash}")
        logger.info(f"  Symbol: {config.symbol}")
        logger.info(f"  Regime Filter: {config.regime_filter}")
        if config.regime_filter:
            logger.info(f"  Chop Threshold: {config.regime_params.chop_threshold}")
        logger.info(f"  Risk per Trade: {config.risk_pct}%")
        logger.info(f"  Max Trades/Day: {config.guardrails.max_trades_per_day}")
        logger.info("=" * 80)

        # Validate
        valid, error = config.validate()
        if not valid:
            raise ValueError(f"Config validation failed: {error}")

        return config

    @classmethod
    def from_yaml(cls, filepath: str) -> "StrategyConfig":
        """Load from YAML file with alias support"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load YAML from {filepath}: {e}")
            raise

        return cls._from_dict(data)

    @classmethod
    def from_json(cls, filepath: str) -> "StrategyConfig":
        """Load from JSON file with alias support"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON from {filepath}: {e}")
            raise

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig":
        """Internal: construct from normalized dict"""
        # Extract and normalize subsections
        regime_data = _normalize_keys(data.get("regime", {}), KEY_ALIASES)
        trend_data = _normalize_keys(data.get("trend", {}), KEY_ALIASES)
        h2l2_data = _normalize_keys(data.get("h2l2", {}), KEY_ALIASES)
        guard_data = _normalize_keys(data.get("guardrails", {}), KEY_ALIASES)
        risk_data = _normalize_keys(data.get("risk", {}), KEY_ALIASES)
        costs_data = _normalize_keys(data.get("costs", {}), KEY_ALIASES)

        return cls(
            symbol=data.get("symbol", "US500.cash"),
            regime_filter=regime_data.get("enabled", False),
            regime_params=RegimeParams(
                chop_threshold=regime_data.get("chop_threshold", 2.5),
                atr_period=regime_data.get("atr_period", 14),
                range_period=regime_data.get("range_period", 20),
            ),
            trend_params=TrendParams(
                ema_period=trend_data.get("ema_period", 20),
                min_slope=trend_data.get("min_slope", 0.15),
            ),
            h2l2_params=H2L2Params(
                pullback_bars=h2l2_data.get("pullback_bars", 3),
                signal_close_frac=h2l2_data.get("signal_close_frac", 0.30),
                min_risk_price_units=h2l2_data.get("min_risk_price_units", 2.0),
                stop_buffer=h2l2_data.get("stop_buffer", 1.0),
                cooldown_bars=h2l2_data.get("cooldown_bars", 0),
            ),
            guardrails=Guardrails(
                session_tz=guard_data.get("session_tz", "America/New_York"),
                day_tz=guard_data.get("day_tz", "America/New_York"),
                session_start=guard_data.get("session_start", "09:30"),
                session_end=guard_data.get("session_end", "16:00"),
                max_trades_per_day=guard_data.get("max_trades_per_day", 2),
            ),
            risk_pct=risk_data.get("risk_pct", 1.0),
            costs_per_trade_r=costs_data.get("costs_per_trade_r", 0.04),
        )

    def to_yaml(self, filepath: str) -> None:
        """Save config to YAML (for exporting live config)"""
        data = {
            "symbol": self.symbol,
            "regime": {
                "enabled": self.regime_filter,
                "chop_threshold": self.regime_params.chop_threshold,
                "atr_period": self.regime_params.atr_period,
                "range_period": self.regime_params.range_period,
            },
            "trend": {
                "ema_period": self.trend_params.ema_period,
                "min_slope": self.trend_params.min_slope,
            },
            "h2l2": {
                "pullback_bars": self.h2l2_params.pullback_bars,
                "signal_close_frac": self.h2l2_params.signal_close_frac,
                "min_risk_price_units": self.h2l2_params.min_risk_price_units,
                "stop_buffer": self.h2l2_params.stop_buffer,
                "cooldown_bars": self.h2l2_params.cooldown_bars,
            },
            "guardrails": {
                "session_tz": self.guardrails.session_tz,
                "day_tz": self.guardrails.day_tz,
                "session_start": self.guardrails.session_start,
                "session_end": self.guardrails.session_end,
                "max_trades_per_day": self.guardrails.max_trades_per_day,
            },
            "risk": {
                "risk_pct": self.risk_pct,
            },
            "costs": {
                "per_trade_r": self.costs_per_trade_r,
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Config saved to {filepath}")

    def validate(self) -> tuple[bool, Optional[str]]:
        """Pre-flight validation"""
        if self.regime_params.chop_threshold < 0:
            return False, "chop_threshold must be >= 0"

        if self.trend_params.ema_period < 2:
            return False, "ema_period must be >= 2"

        if self.h2l2_params.pullback_bars < 1:
            return False, "pullback_bars must be >= 1"

        if not 0 < self.h2l2_params.signal_close_frac < 1:
            return False, "signal_close_frac must be between 0 and 1"

        if self.h2l2_params.stop_buffer < 0:
            return False, "stop_buffer must be >= 0"

        if self.risk_pct <= 0 or self.risk_pct > 10:
            return False, "risk_pct must be between 0 and 10"

        if self.guardrails.max_trades_per_day < 1:
            return False, "max_trades_per_day must be >= 1"

        return True, None

    def get_hash(self) -> str:
        """Get reproducible hash of config for audit trail"""
        # Convert to dict and sort keys for deterministic hash
        data_dict = {
            "symbol": self.symbol,
            "regime_filter": self.regime_filter,
            "regime_chop": self.regime_params.chop_threshold,
            "trend_ema": self.trend_params.ema_period,
            "trend_slope": self.trend_params.min_slope,
            "h2l2_pullback": self.h2l2_params.pullback_bars,
            "h2l2_frac": self.h2l2_params.signal_close_frac,
            "h2l2_stop_buffer": self.h2l2_params.stop_buffer,
            "h2l2_cooldown": self.h2l2_params.cooldown_bars,
            "risk_pct": self.risk_pct,
            "max_trades_day": self.guardrails.max_trades_per_day,
        }
        json_str = json.dumps(data_dict, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:12]
