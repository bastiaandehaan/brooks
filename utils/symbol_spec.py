from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SymbolSpec:
    name: str
    digits: int
    point: float
    tick_size: float
    tick_value: float
    contract_size: float  # Let op: hier heet het contract_size
    volume_min: float
    volume_step: float
    volume_max: float

    @property
    def usd_per_price_unit_per_lot(self) -> float:
        """
        USD value for a 1.0 price move per 1.0 lot.
        """
        if self.tick_size <= 0:
            # Prevent division by zero
            if self.tick_value > 0:
                return 0.0
            return 0.0
        return float(self.tick_value) / float(self.tick_size)

    def round_volume_down(self, vol: float) -> float:
        if vol <= 0:
            return 0.0
        # Epsilon voor floating point onnauwkeurigheden
        steps = int((vol + 1e-9) / self.volume_step)
        v = steps * self.volume_step
        if v < self.volume_min:
            return 0.0
        return min(v, self.volume_max)

    @staticmethod
    def from_symbol_info(info: dict[str, Any]) -> SymbolSpec:
        """
        Factory method: Vertaalt ruwe MT5 data (dict) naar een schoon SymbolSpec object.
        Dit voorkomt fouten met veldnamen in de rest van de applicatie.
        """
        return SymbolSpec(
            name=str(info["name"]),
            digits=int(info["digits"]),
            point=float(info["point"]),
            # MT5 heet het 'trade_tick_size', wij noemen het 'tick_size'
            tick_size=float(info.get("trade_tick_size", 0.0)),
            # MT5 heet het 'trade_tick_value', wij noemen het 'tick_value'
            tick_value=float(info.get("trade_tick_value", 0.0)),
            # MT5 heet het 'trade_contract_size', wij noemen het 'contract_size'
            contract_size=float(info.get("trade_contract_size", 1.0)),
            volume_min=float(info["volume_min"]),
            volume_step=float(info["volume_step"]),
            volume_max=float(info["volume_max"]),
        )
