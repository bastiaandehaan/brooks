# tests/test_mt5_client.py
import types

import pytest
from utils.mt5_client import Mt5Client, Mt5ConnectionParams, symbol_info_to_dict


class DummySymbol:
    def __init__(self, name: str):
        self.name = name


class DummyInfo:
    def __init__(self, visible: bool):
        self.visible = visible

    def _asdict(self):
        return {"visible": self.visible, "digits": 2}


@pytest.fixture()
def mt5_mock():
    state = {"initialized": False, "selected": {}}

    def initialize(**kwargs):
        state["initialized"] = True
        return True

    def shutdown():
        state["initialized"] = False

    def symbols_get():
        return [DummySymbol("US500.cash"), DummySymbol("US30.cash"), DummySymbol("EURUSD")]

    def symbol_info(symbol):
        # visible false unless selected
        return DummyInfo(visible=state["selected"].get(symbol, False))

    def symbol_select(symbol, enable):
        state["selected"][symbol] = bool(enable)
        return True

    def terminal_info():
        return types.SimpleNamespace(name="DummyTerminal")

    def account_info():
        return types.SimpleNamespace(login=123)

    def last_error():
        return (0, "ok")

    return types.SimpleNamespace(
        initialize=initialize,
        shutdown=shutdown,
        symbols_get=symbols_get,
        symbol_info=symbol_info,
        symbol_select=symbol_select,
        terminal_info=terminal_info,
        account_info=account_info,
        last_error=last_error,
    )


def test_symbols_search_finds_us500(mt5_mock):
    c = Mt5Client(mt5_mock, Mt5ConnectionParams())
    c.initialize()
    matches = c.symbols_search("us500")
    assert len(matches) == 1
    assert matches[0].name == "US500.cash"
    c.shutdown()


def test_ensure_selected_calls_select(mt5_mock):
    c = Mt5Client(mt5_mock, Mt5ConnectionParams())
    c.initialize()
    c.ensure_selected("US500.cash")
    info = c.symbol_info("US500.cash")
    assert info["visible"] is True
    c.shutdown()


def test_symbol_info_to_dict_uses_asdict():
    d = symbol_info_to_dict(DummyInfo(visible=True))
    assert d["visible"] is True
    assert d["digits"] == 2
