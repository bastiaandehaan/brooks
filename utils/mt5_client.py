# utils/mt5_client.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class Mt5Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Mt5ConnectionParams:
    # Voor de meeste setups is alleen initialize() voldoende (MT5 terminal open + ingelogd).
    # Laat deze leeg tenzij je expliciet credentials/server via API wil doen.
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    timeout_ms: int = 10_000


class Mt5Client:
    """
    Dunne wrapper rond MetaTrader5 package.
    Doel: symbol discovery + info dumps, met nette logging en testbaarheid.
    """

    def __init__(self, mt5_module, params: Mt5ConnectionParams = Mt5ConnectionParams()):
        self._mt5 = mt5_module
        self._params = params
        self._initialized = False

    def initialize(self) -> None:
        logger.info("Initializing MT5 connection...")
        ok = self._mt5.initialize(
            login=self._params.login,
            password=self._params.password,
            server=self._params.server,
            timeout=self._params.timeout_ms,
        )
        if not ok:
            code, msg = self._safe_last_error()
            raise Mt5Error(f"MT5 initialize failed: {code} {msg}")

        self._initialized = True
        term = self._mt5.terminal_info()
        acc = self._mt5.account_info()
        logger.info("MT5 connected. Terminal=%s, Account=%s", getattr(term, "name", term), getattr(acc, "login", acc))

    def shutdown(self) -> None:
        if self._initialized:
            logger.info("Shutting down MT5 connection...")
            self._mt5.shutdown()
        self._initialized = False

    def symbols_list(self) -> Sequence[Any]:
        self._require_init()
        syms = self._mt5.symbols_get()
        if syms is None:
            code, msg = self._safe_last_error()
            raise Mt5Error(f"symbols_get() returned None: {code} {msg}")
        logger.info("Loaded %d symbols from MT5", len(syms))
        return syms

    def symbols_search(self, needle: str) -> List[Any]:
        """
        Case-insensitive substring match on symbol name.
        """
        self._require_init()
        needle_l = needle.lower().strip()
        if not needle_l:
            raise ValueError("needle must be non-empty")

        matches: List[Any] = []
        for s in self.symbols_list():
            name = getattr(s, "name", "")
            if needle_l in name.lower():
                matches.append(s)

        logger.info("Found %d symbols matching '%s'", len(matches), needle)
        return matches

    def ensure_selected(self, symbol: str, enable: bool = True) -> None:
        """
        Zorgt dat symbool in MarketWatch zichtbaar is (nodig voor sommige brokers/feeds).
        """
        self._require_init()
        info = self._mt5.symbol_info(symbol)
        if info is None:
            code, msg = self._safe_last_error()
            raise Mt5Error(f"symbol_info({symbol}) returned None: {code} {msg}")

        if getattr(info, "visible", False):
            logger.debug("Symbol already visible: %s", symbol)
            return

        ok = self._mt5.symbol_select(symbol, enable)
        if not ok:
            code, msg = self._safe_last_error()
            raise Mt5Error(f"symbol_select({symbol}, {enable}) failed: {code} {msg}")

        logger.info("Symbol selected in MarketWatch: %s", symbol)

    def symbol_info(self, symbol: str) -> Dict[str, Any]:
        self._require_init()
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
    """
    MetaTrader5 returns a namedtuple-like object (SymbolInfo).
    Prefer _asdict() when available; fallback to vars()/dir().
    """
    if hasattr(info_obj, "_asdict"):
        return dict(info_obj._asdict())

    if hasattr(info_obj, "__dict__"):
        return dict(info_obj.__dict__)

    # Last resort: collect public attributes
    out: Dict[str, Any] = {}
    for k in dir(info_obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(info_obj, k)
        except Exception:
            continue
        if callable(v):
            continue
        out[k] = v
    return out
