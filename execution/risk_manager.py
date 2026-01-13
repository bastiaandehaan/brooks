# execution/risk_manager.py
from __future__ import annotations

from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class RiskParams:
    min_risk_pts: float = 1.0   # example; keep your existing defaults
    fees_usd: float = 0.0

class RiskManager:
    def __init__(self, params: RiskParams):
        self.params = params

    def size_position(
        self,
        *,
        balance: float,
        entry: float,
        stop: float,
        tick_size: float,
        contract_size: float,
        risk_pct: float | None = None,
        fees_usd: float | None = None,
        **_ignored: object,   # keeps older callers from breaking if they pass extra
    ) -> tuple[float, float]:
        """
        Returns (lots, risk_usd).
        risk_pct is percent of balance (e.g. 0.5 means 0.5%).
        """

        if risk_pct is None:
            raise TypeError("risk_pct is required for sizing (percent, e.g. 0.5)")

        fees = self.params.fees_usd if fees_usd is None else float(fees_usd)

        risk_pts = abs(float(entry) - float(stop))
        if risk_pts <= 0:
            raise ValueError("Invalid stop: risk_pts must be > 0")

        if risk_pts < float(self.params.min_risk_pts):
            raise ValueError(f"Risk too small: {risk_pts:.4f} < min_risk_pts={self.params.min_risk_pts:.4f}")

        risk_usd_target = float(balance) * (float(risk_pct) / 100.0)
        # USD per 1.0 price point for 1 lot:
        usd_per_point_per_lot = float(contract_size)

        lots = (risk_usd_target - fees) / (risk_pts * usd_per_point_per_lot)
        lots = max(0.0, lots)

        log.info(
            "SIZING: balance=%.2f risk_pct=%.4f risk_usd_target=%.2f risk_pts=%.4f contract=%.4f -> lots=%.4f",
            balance, risk_pct, risk_usd_target, risk_pts, contract_size, lots
        )
        return lots, max(0.0, risk_usd_target - fees)
