# MT5 Symbol Discovery (FTMO) - Architecture

## Doel
Via de MetaTrader5 Python API:
1) alle beschikbare symbolen (namen) ophalen
2) details (SymbolInfo) dumpen
3) specifiek: US500 vinden en alle specs tonen

## Componenten
- /utils/mt5_client.py
  - Mt5Client: init/shutdown, symbols_list(), symbols_search(), symbol_info(), ensure_selected()
  - symbol_info_to_dict(): converteert MT5 namedtuple naar dict
- /examples/mt5_list_us500.py
  - zoekt naar symbolen met "US500" (case-insensitive)
  - selecteert het gekozen symbool in MarketWatch
  - print kernvelden + volledige JSON dump van SymbolInfo

## Logging
- INFO: connect status, aantal symbols, matches voor US500, gekozen symbool
- DEBUG: volledige SymbolInfo keys/values (optioneel)

## Teststrategie
- Unit test (mock MT5 module):
  - symbols_search() filtert correct op substring
  - symbol_info_to_dict() levert dict met verwachte keys
  - ensure_selected() roept symbol_select aan bij not-visible

## Definition of Done
- Script kan US500 symbolen tonen + details
- Test(s) groen
- Logging zichtbaar
