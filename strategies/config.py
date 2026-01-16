import yaml
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# --- CORE IMPORTS ---
from strategies.context import TrendParams
from strategies.h2l2 import H2L2Params
from strategies.regime import RegimeParams
from execution.guardrails import Guardrails


@dataclass
class StrategyConfig:
    """
    De Single Source of Truth.
    Ondersteunt YAML (menselijk) en JSON (machine).
    """
    symbol: str = "US500.cash"

    # --- STRATEGIE COMPONENTEN ---
    trend: TrendParams = field(default_factory=TrendParams)
    h2l2: H2L2Params = field(default_factory=H2L2Params)
    regime: RegimeParams = field(default_factory=RegimeParams)

    # --- EXECUTIE & RISK ---
    guardrails: Guardrails = field(default_factory=Guardrails)
    risk_pct: float = 0.5

    # --- LEGACY COMPATIBILITEIT ---
    # Om te voorkomen dat oude YAML files crashen op onbekende keys
    ema_period: Optional[int] = None
    min_slope: Optional[float] = None

    @classmethod
    def from_yaml(cls, path: str):
        return cls._load(path, yaml.safe_load)

    @classmethod
    def from_json(cls, path: str):
        return cls._load(path, json.load)

    @classmethod
    def _load(cls, path: str, loader):
        p = Path(path)
        if not p.exists():
            # Veilige fallback als file mist
            print(f"⚠️ Config {path} niet gevonden, gebruik defaults.")
            return cls()

        with open(p, 'r') as f:
            data = loader(f)

        # Filter keys die niet in deze class staan (voorkomt crashes)
        valid_keys = cls.__annotations__.keys()
        clean_data = {k: v for k, v in data.items() if k in valid_keys}

        return cls(**clean_data)