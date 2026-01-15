# strategies/config.py
"""
Shared strategy configuration
Ensures main.py and backtest/runner.py use EXACT same parameters
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import yaml

from strategies.context import TrendParams
from strategies.h2l2 import H2L2Params
from strategies.regime import RegimeParams
from execution.guardrails import Guardrails


@dataclass(frozen=True)
class StrategyConfig:
    """
    Complete strategy configuration
    Used by BOTH live and backtest to prevent drift
    """
    # Symbol
    symbol: str = "US500.cash"

    # Regime filter
    regime_filter: bool = True
    regime_params: RegimeParams = None

    # Trend context
    trend_params: TrendParams = None

    # H2/L2 strategy
    h2l2_params: H2L2Params = None

    # Guardrails
    guardrails: Guardrails = None

    # Risk
    risk_pct: float = 1.0

    # Costs (for backtest)
    costs_per_trade_r: float = 0.04

    def __post_init__(self):
        """Initialize default sub-configs if not provided"""
        if self.regime_params is None:
            object.__setattr__(self, 'regime_params', RegimeParams())
        if self.trend_params is None:
            object.__setattr__(self, 'trend_params', TrendParams())
        if self.h2l2_params is None:
            object.__setattr__(self, 'h2l2_params', H2L2Params())
        if self.guardrails is None:
            object.__setattr__(self, 'guardrails', Guardrails())

    @classmethod
    def from_args(cls, args) -> StrategyConfig:
        """Create config from CLI arguments (for main.py and backtest/runner.py)"""
        return cls(
            symbol=getattr(args, 'symbol', 'US500.cash'),
            regime_filter=getattr(args, 'regime_filter', False),
            regime_params=RegimeParams(
                chop_threshold=getattr(args, 'chop_threshold', 2.5),
            ),
            trend_params=TrendParams(
                ema_period=getattr(args, 'ema', 20),
                min_slope=getattr(args, 'min_slope', 0.15),
            ),
            h2l2_params=H2L2Params(
                pullback_bars=getattr(args, 'pullback_bars', 3),
                signal_close_frac=getattr(args, 'signal_close_frac', 0.30),
                min_risk_price_units=getattr(args, 'min_risk', 2.0),
                stop_buffer=getattr(args, 'stop_buffer', 1.0),
                cooldown_bars=getattr(args, 'cooldown', 0),
            ),
            guardrails=Guardrails(
                session_tz=getattr(args, 'session_tz', 'America/New_York'),
                day_tz=getattr(args, 'day_tz', 'America/New_York'),
                session_start=getattr(args, 'session_start', '09:30'),
                session_end=getattr(args, 'session_end', '16:00'),
                max_trades_per_day=getattr(args, 'max_trades_day', 2),
            ),
            risk_pct=getattr(args, 'risk_pct', 1.0),
            costs_per_trade_r=getattr(args, 'costs', 0.04),
        )

    @classmethod
    def from_yaml(cls, filepath: str) -> StrategyConfig:
        """Load config from YAML file"""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        return cls(
            symbol=data.get('symbol', 'US500.cash'),
            regime_filter=data.get('regime', {}).get('enabled', False),
            regime_params=RegimeParams(
                chop_threshold=data.get('regime', {}).get('chop_threshold', 2.5),
            ),
            trend_params=TrendParams(
                ema_period=data.get('trend', {}).get('ema_period', 20),
                min_slope=data.get('trend', {}).get('min_slope', 0.15),
            ),
            h2l2_params=H2L2Params(
                pullback_bars=data.get('h2l2', {}).get('pullback_bars', 3),
                signal_close_frac=data.get('h2l2', {}).get('signal_close_frac', 0.30),
                min_risk_price_units=data.get('h2l2', {}).get('min_risk_price_units', 2.0),
                stop_buffer=data.get('h2l2', {}).get('stop_buffer', 1.0),
                cooldown_bars=data.get('h2l2', {}).get('cooldown_bars', 0),
            ),
            guardrails=Guardrails(
                session_tz=data.get('guardrails', {}).get('session_tz', 'America/New_York'),
                day_tz=data.get('guardrails', {}).get('day_tz', 'America/New_York'),
                session_start=data.get('guardrails', {}).get('session_start', '09:30'),
                session_end=data.get('guardrails', {}).get('session_end', '16:00'),
                max_trades_per_day=data.get('guardrails', {}).get('max_trades_day', 2),
            ),
            risk_pct=data.get('risk', {}).get('risk_pct', 1.0),
            costs_per_trade_r=data.get('costs', {}).get('per_trade_r', 0.04),
        )

    def to_yaml(self, filepath: str) -> None:
        """Save config to YAML file"""
        data = {
            'symbol': self.symbol,
            'regime': {
                'enabled': self.regime_filter,
                'chop_threshold': self.regime_params.chop_threshold,
            },
            'trend': {
                'ema_period': self.trend_params.ema_period,
                'min_slope': self.trend_params.min_slope,
            },
            'h2l2': {
                'pullback_bars': self.h2l2_params.pullback_bars,
                'signal_close_frac': self.h2l2_params.signal_close_frac,
                'min_risk_price_units': self.h2l2_params.min_risk_price_units,
                'stop_buffer': self.h2l2_params.stop_buffer,
                'cooldown_bars': self.h2l2_params.cooldown_bars,
            },
            'guardrails': {
                'session_tz': self.guardrails.session_tz,
                'day_tz': self.guardrails.day_tz,
                'session_start': self.guardrails.session_start,
                'session_end': self.guardrails.session_end,
                'max_trades_day': self.guardrails.max_trades_per_day,
            },
            'risk': {
                'risk_pct': self.risk_pct,
            },
            'costs': {
                'per_trade_r': self.costs_per_trade_r,
            },
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate configuration parameters"""
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