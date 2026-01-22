from types import SimpleNamespace

import pytest

from utils.mt5_client import Mt5Client, Mt5ConnectionParams


@pytest.fixture
def mt5_mock():
    ns = SimpleNamespace()

    ns.initialize = lambda **kwargs: True
    ns.shutdown = lambda: None
    ns.last_error = lambda: (1, "Generic Error")

    # FIX: Accepteer argumenten (*args), want de code roept symbols_get("us500") aan
    def mock_symbols_get(*args, **kwargs):
        s1 = SimpleNamespace(name="US500.cash")
        return (s1,)

    ns.symbols_get = mock_symbols_get

    ns.symbol_select = lambda s, enable: True

    def mock_symbol_info(symbol):
        if symbol == "FAIL":
            return None
        # Zorg dat deze mock velden overeenkomen met wat SymbolSpec.from_symbol_info verwacht
        return SimpleNamespace(
            name=symbol,
            digits=2,
            point=0.01,
            trade_contract_size=1.0,  # MT5 naam
            spread=10,
            trade_stops_level=0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            trade_tick_size=0.01,
            trade_tick_value=0.01,
            _asdict=lambda: {
                "name": symbol,
                "digits": 2,
                "point": 0.01,
                "trade_contract_size": 1.0,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "trade_tick_size": 0.01,
                "trade_tick_value": 0.01,
            },
        )

    ns.symbol_info = mock_symbol_info

    ns.terminal_info = lambda: SimpleNamespace(name="MockTerminal")
    ns.account_info = lambda: SimpleNamespace(login=12345)

    return ns


def test_initialization_flow(mt5_mock):
    c = Mt5Client(mt5_mock)
    assert c.initialize() is True
    c.shutdown()


def test_symbol_info_fetch(mt5_mock):
    c = Mt5Client(mt5_mock)
    c.initialize()
    info = c.symbol_info("US500.cash")
    assert info["name"] == "US500.cash"


def test_symbols_search_finds_us500(mt5_mock):
    c = Mt5Client(mt5_mock, Mt5ConnectionParams())
    c.initialize()
    # Dit faalde eerst, nu niet meer door *args in mock_symbols_get
    matches = c.symbols_search("us500")
    assert "US500.cash" in matches


def test_ensure_selected_calls_select(mt5_mock):
    c = Mt5Client(mt5_mock, Mt5ConnectionParams())
    c.initialize()
    assert c.ensure_selected("US500.cash") is True
