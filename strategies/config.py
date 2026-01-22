# strategies/config.py
"""
Shared strategy configuration
Ensures main.py and backtest/runner.py use EXACT same parameters
to prevent parameter drift between simulation and live execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import yaml

from execution.guardrails import Guardrails

# Importeren van parameter classes uit de bestaande strategie pakketten.
# Dit garandeert dat de config direct mapt op de logica in de strategies map.
from strategies.context import TrendParams
from strategies.h2l2 import H2L2Params
from strategies.regime import RegimeParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyConfig:
    """
    Complete strategy configuration.
    Used by BOTH live and backtest environments to prevent drift.
    Acts as the Single Source of Truth for the trading session.
    """

    # Instrument Symbol (bv. US500.cash)
    symbol: str = "US500.cash"

    # Regime Filter Configuratie
    # Bepaalt of we filteren op choppy markten
    regime_filter: bool = True
    regime_params: RegimeParams = None

    # Trend Context Configuratie
    # Instellingen voor EMA periode en slope detectie
    trend_params: TrendParams = None

    # H2/L2 Strategie Configuratie
    # Parameters voor pullback detectie en signal bars
    h2l2_params: H2L2Params = None

    # Execution Guardrails
    # Tijdfilters en dagelijkse limieten
    guardrails: Guardrails = None

    # Risk Management
    # Percentage van account balans per trade
    risk_pct: float = 1.0

    # Transactiekosten (voornamelijk voor backtesting simulatie)
    # Uitgedrukt in R-units om consistentie te bewaren
    costs_per_trade_r: float = 0.04

    def __post_init__(self):
        """
        Initialize default sub-configs if they were not provided during instantiation.
        This ensures the object is always fully populated and ready for use.
        Because the dataclass is frozen, we must use object.__setattr__.
        """
        if self.regime_params is None:
            object.__setattr__(self, "regime_params", RegimeParams())
        if self.trend_params is None:
            object.__setattr__(self, "trend_params", TrendParams())
        if self.h2l2_params is None:
            object.__setattr__(self, "h2l2_params", H2L2Params())
        if self.guardrails is None:
            object.__setattr__(self, "guardrails", Guardrails())

    @classmethod
    def from_args(cls, args) -> StrategyConfig:
        """
        Create config from CLI arguments.
        This provides backward compatibility for main.py and backtest/runner.py
        which currently rely on argparse. It maps the flat argument structure
        to the hierarchical config structure.
        """
        return cls(
            symbol=getattr(args, "symbol", "US500.cash"),
            regime_filter=getattr(args, "regime_filter", False),
            regime_params=RegimeParams(
                chop_threshold=getattr(args, "chop_threshold", 2.5),
            ),
            trend_params=TrendParams(
                ema_period=getattr(args, "ema", 20),
                min_slope=getattr(args, "min_slope", 0.15),
            ),
            h2l2_params=H2L2Params(
                pullback_bars=getattr(args, "pullback_bars", 3),
                signal_close_frac=getattr(args, "signal_close_frac", 0.30),
                min_risk_price_units=getattr(args, "min_risk", 2.0),
                stop_buffer=getattr(args, "stop_buffer", 1.0),
                cooldown_bars=getattr(args, "cooldown", 0),
            ),
            guardrails=Guardrails(
                session_tz=getattr(args, "session_tz", "America/New_York"),
                day_tz=getattr(args, "day_tz", "America/New_York"),
                session_start=getattr(args, "session_start", "09:30"),
                session_end=getattr(args, "session_end", "16:00"),
                max_trades_per_day=getattr(args, "max_trades_day", 2),
            ),
            risk_pct=getattr(args, "risk_pct", 1.0),
            costs_per_trade_r=getattr(args, "costs", 0.04),
        )

    @classmethod
    def from_yaml(cls, filepath: str) -> StrategyConfig:
        """
        Load configuration from a YAML file.
        This is the preferred method for production deployment, ensuring
        reproducibility and separation of concerns.
        """
        try:
            with open(filepath) as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load YAML config from {filepath}: {e}")
            raise

        # Extract subsections with defaults if missing
        # This robustness prevents crashes on partial config files
        regime_data = data.get("regime", {})
        trend_data = data.get("trend", {})
        h2l2_data = data.get("h2l2", {})
        guard_data = data.get("guardrails", {})
        risk_data = data.get("risk", {})
        costs_data = data.get("costs", {})

        return cls(
            symbol=data.get("symbol", "US500.cash"),
            regime_filter=regime_data.get("enabled", False),
            regime_params=RegimeParams(
                chop_threshold=regime_data.get("chop_threshold", 2.5),
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
                max_trades_per_day=guard_data.get("max_trades_day", 2),
            ),
            risk_pct=risk_data.get("risk_pct", 1.0),
            costs_per_trade_r=costs_data.get("per_trade_r", 0.04),
        )

    def to_yaml(self, filepath: str) -> None:
        """
        Save the current configuration state to a YAML file.
        Useful for exporting the exact config used during a live session
        for audit purposes.
        """
        data = {
            "symbol": self.symbol,
            "regime": {
                "enabled": self.regime_filter,
                "chop_threshold": self.regime_params.chop_threshold,
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
                "max_trades_day": self.guardrails.max_trades_per_day,
            },
            "risk": {
                "risk_pct": self.risk_pct,
            },
            "costs": {
                "per_trade_r": self.costs_per_trade_r,
            },
        }

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def validate(self) -> tuple[bool, str | None]:
        """
        Validate configuration parameters against logical constraints.
        This serves as a 'Pre-Flight Check' before trading starts.
        Returns: (is_valid, error_message)
        """
        # Validatie van Regime Parameters
        if self.regime_params.chop_threshold < 0:
            return False, "chop_threshold must be >= 0"

        # Validatie van Trend Parameters
        if self.trend_params.ema_period < 2:
            return False, "ema_period must be >= 2"

        # Validatie van H2L2 Parameters
        if self.h2l2_params.pullback_bars < 1:
            return False, "pullback_bars must be >= 1"

        if not 0 < self.h2l2_params.signal_close_frac < 1:
            return False, "signal_close_frac must be between 0 and 1"

        if self.h2l2_params.stop_buffer < 0:
            return False, "stop_buffer must be >= 0"

        # Validatie van Risico Management
        if self.risk_pct <= 0 or self.risk_pct > 10:
            return False, "risk_pct must be between 0 and 10 (reasonable range for FTMO)"

        # Validatie van Guardrails
        if self.guardrails.max_trades_per_day < 1:
            return False, "max_trades_per_day must be >= 1"

        return True, None
