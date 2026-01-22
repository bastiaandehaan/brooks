# execution/risk_manager.py
from __future__ import annotations

import logging
from dataclasses import dataclass

from utils.symbol_spec import SymbolSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskParams:
    min_risk_pts: float = 1.0
    fees_usd: float = 0.0


class RiskManager:
    """
    Fixed Risk Manager - NOW COMPATIBLE WITH ALL CALLERS

    Accepts both old-style (spec=SymbolSpec) and new-style (kwargs) calls.
    """

    def __init__(self, params: RiskParams = None):
        self.params = params or RiskParams()

    def size_position(
        self,
        *,
        balance: float,
        entry: float,
        stop: float,
        risk_pct: float,
        spec: SymbolSpec = None,
        tick_size: float = None,
        contract_size: float = None,
        fees_usd: float = None,
        **_ignored,
    ) -> tuple[float, float]:
        """
        Universal sizing method - works with ANY caller.

        Args:
            balance: Account balance
            entry: Entry price
            stop: Stop loss price
            risk_pct: Risk as PERCENTAGE (e.g., 0.5 for 0.5%)
            spec: SymbolSpec object (preferred)
            tick_size: Manual override (if spec not provided)
            contract_size: Manual override
            fees_usd: Trading costs

        Returns:
            (lots, risk_usd)
        """
        # Extract values from spec OR kwargs
        if spec is not None:
            _tick_size = float(spec.tick_size)
            _contract_size = float(spec.contract_size)
        else:
            if tick_size is None or contract_size is None:
                raise ValueError("Must provide either 'spec' or 'tick_size + contract_size'")
            _tick_size = float(tick_size)
            _contract_size = float(contract_size)

        _fees = self.params.fees_usd if fees_usd is None else float(fees_usd)

        # Calculate risk in price units
        risk_pts = abs(float(entry) - float(stop))

        if risk_pts <= 0:
            raise ValueError("Invalid stop: risk_pts must be > 0")

        if risk_pts < float(self.params.min_risk_pts):
            raise ValueError(
                f"Risk too small: {risk_pts:.4f} < min_risk_pts={self.params.min_risk_pts:.4f}"
            )

        # Calculate position size
        risk_usd_target = float(balance) * (float(risk_pct) / 100.0)
        usd_per_point_per_lot = _contract_size

        lots = (risk_usd_target - _fees) / (risk_pts * usd_per_point_per_lot)
        lots = max(0.0, lots)

        # Validation
        if spec is not None:
            lots = spec.round_volume_down(lots)

        logger.debug(
            "SIZING: balance=%.2f risk_pct=%.4f target=%.2f risk_pts=%.4f -> lots=%.4f",
            balance,
            risk_pct,
            risk_usd_target,
            risk_pts,
            lots,
        )

        return lots, max(0.0, risk_usd_target - _fees)

    # Legacy alias for backward compatibility
    def calculate_lot_size(
        self, balance: float, spec: SymbolSpec, entry: float, stop: float
    ) -> float:
        """Legacy method - redirects to size_position"""
        lots, _ = self.size_position(
            balance=balance,
            entry=entry,
            stop=stop,
            spec=spec,
            risk_pct=1.0,  # Default 1% risk
        )
        return lots
