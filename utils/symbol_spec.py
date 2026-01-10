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
    def value_per_point_per_lot(self) -> float:
        """
        USD value for a 1.00 price move (one index point) per 1.00 lot.
        Derived from tick_value / tick_size scaled to 1.0 point.
        """
        if self.tick_size <= 0:
            raise ValueError("tick_size must be > 0")
        return self.tick_value * (self.point / self.tick_size) if self.point != 1.0 else self.tick_value / self.tick_size

    def round_volume(self, vol: float) -> float:
        """
        Round volume DOWN to the nearest volume_step and clamp to [min, max].
        """
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
