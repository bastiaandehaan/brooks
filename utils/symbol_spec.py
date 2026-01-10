# utils/symbol_spec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SymbolSpec:
    name: str
    digits: int
    point: float
    tick_size: float
    tick_value: float
    contract_size: float
    volume_min: float
    volume_step: float
    volume_max: float

    @property
    def usd_per_price_unit_per_lot(self) -> float:
        """
        USD value for a 1.0 price move per 1.0 lot.
        US500.cash example: tick_size=0.01, tick_value=0.01 => 1.0 USD per 1.0 move.
        """
        if self.tick_size <= 0:
            raise ValueError("tick_size must be > 0")
        return float(self.tick_value) / float(self.tick_size)

    def round_volume_down(self, vol: float) -> float:
        if vol <= 0:
            return 0.0
        steps = int(vol / self.volume_step)
        v = steps * self.volume_step
        if v < self.volume_min:
            return 0.0
        return min(v, self.volume_max)

    @staticmethod
    def from_symbol_info(info: Dict[str, Any]) -> "SymbolSpec":
        return SymbolSpec(
            name=str(info["name"]),
            digits=int(info["digits"]),
            point=float(info["point"]),
            tick_size=float(info["trade_tick_size"]),
            tick_value=float(info["trade_tick_value"]),
            contract_size=float(info["trade_contract_size"]),
            volume_min=float(info["volume_min"]),
            volume_step=float(info["volume_step"]),
            volume_max=float(info["volume_max"]),
        )
