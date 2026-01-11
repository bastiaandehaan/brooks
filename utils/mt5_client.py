from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils.symbol_spec import SymbolSpec

logger = logging.getLogger(__name__)


class Mt5Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Mt5ConnectionParams:
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    timeout_ms: int = 10_000


class Mt5Client:
    def __init__(self, mt5_module, params: Mt5ConnectionParams = Mt5ConnectionParams()):
        self._mt5 = mt5_module
        self._params = params
        self._initialized = False

    def initialize(self) -> bool:
        logger.info("Initializing MT5 connection...")
        kwargs: Dict[str, Any] = {"timeout": self._params.timeout_ms}
        if self._params.login is not None:
            kwargs["login"] = self._params.login
            kwargs["password"] = self._params.password
            kwargs["server"] = self._params.server

        if not self._mt5.initialize(**kwargs):
            code, msg = self._safe_last_error()
            logger.error(f"MT5 initialize failed: {code} {msg}")
            return False

        self._initialized = True

        term = self._mt5.terminal_info()
        acc = self._mt5.account_info()
        t_name = term.name if term else "Unknown"
        a_login = acc.login if acc else "Unknown"
        logger.info(f"MT5 connected. Terminal={t_name}, Account={a_login}")
        return True

    def shutdown(self) -> None:
        if self._initialized:
            self._mt5.shutdown()
            self._initialized = False
            logger.info("MT5 connection shutdown.")

    def symbols_search(self, group: str = "") -> List[str]:
        self._require_init()
        symbols = self._mt5.symbols_get(group)
        if symbols is None:
            return []
        return [s.name for s in symbols]

    def ensure_selected(self, symbol: str) -> bool:
        self._require_init()
        if not self._mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol {symbol}")
            return False
        return True

    def get_symbol_specification(self, symbol: str) -> Optional[SymbolSpec]:
        """
        Haalt specificaties op en retourneert een SymbolSpec object.
        """
        # 1. Haal de data op als dictionary via onze helper
        try:
            info_dict = self.symbol_info(symbol)
        except Mt5Error as e:
            logger.error(f"Error getting spec for {symbol}: {e}")
            return None

        # 2. Gebruik de factory methode in SymbolSpec om mapping fouten te voorkomen
        try:
            return SymbolSpec.from_symbol_info(info_dict)
        except Exception as e:
            logger.error(f"Failed to create SymbolSpec for {symbol}: {e}")
            return None

    def symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Wrapper rond mt5.symbol_info die altijd een dict teruggeeft."""
        self._require_init()

        # Zorg dat hij geselecteerd is
        if not self.ensure_selected(symbol):
            raise Mt5Error(f"Could not select {symbol}")

        info = self._mt5.symbol_info(symbol)
        if info is None:
            code, msg = self._safe_last_error()
            raise Mt5Error(f"symbol_info({symbol}) returned None: {code} {msg}")
        return symbol_info_to_dict(info)

    def _require_init(self) -> None:
        if not self._initialized:
            raise Mt5Error("MT5 not initialized. Call initialize() first.")

    def _safe_last_error(self) -> tuple:
        try:
            return self._mt5.last_error()
        except Exception:
            return -1, "unknown"


def symbol_info_to_dict(info_obj: Any) -> Dict[str, Any]:
    """Zet MT5 object om naar dict."""
    if hasattr(info_obj, "_asdict"):
        return dict(info_obj._asdict())

    if hasattr(info_obj, "__dict__"):
        return dict(info_obj.__dict__)

    # Fallback velden (dit dekt de meeste mocks en echte objecten)
    known_fields = [
        "name", "digits", "point",
        "trade_contract_size", "spread", "trade_stops_level",
        "volume_min", "volume_max", "volume_step",
        "trade_tick_size", "trade_tick_value"
    ]
    return {k: getattr(info_obj, k) for k in known_fields if hasattr(info_obj, k)}