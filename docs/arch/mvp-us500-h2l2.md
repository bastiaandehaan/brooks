# MVP: US500.cash H2/L2 (Brooks) bot op MT5/FTMO

## Doel
Eén enkel instrument (US500.cash), één setup-familie (H2/L2 “second entries”), één sessie (NY),
met harde FTMO-guardrails:
- ENTRY = NEXT_OPEN (geen intra-bar fills)
- max 1 trade per bar/timestamp
- max trades per dag
- geen overlap/geen pyramiding (default)

Brooks-basis:
- Second entries hebben gemiddeld hogere kans dan first entries. 
- Bar counting: High 1/High 2 (bull context), Low 1/Low 2 (bear context).  :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

## Dataflow (live)
MT5 -> fetch bars -> clean -> context filter (HTF trend) -> setup detect (M5) -> order plan (signal) -> execute NEXT_OPEN -> manage exits.

## Modules
- utils/mt5_client.py: connect + symbol ops (bestaat)
- utils/mt5_data.py: rates ophalen + DataFrame clean (nieuw)
- utils/symbol_spec.py: tick/point/value + rounding (bestaat, correctheid fix)
- strategies/h2l2.py: bar-counting setup detectie (nieuw)
- execution/: order planning + risk sizing + guards (nieuw)
- backtest/: event-loop backtester met dezelfde guards (nieuw)

## Invariants
- Signal ontstaat op BAR_CLOSE(t)
- Execute pas op OPEN(t+1)
- No-lookahead: strategie ziet alleen bars <= t op moment van signal
- Eén trade per bar; max trades/dag; geen overlap

## Logging (minimaal)
INFO:
- context regime (bull/bear/none)
- setup gevonden (H2/L2) + reden + stopafstand
- trade geplaatst/geskipt (guardrail reason)
DEBUG:
- bar-count state (attempt_count, pullback_active)

## Tests
- Unit: bar counting + symbol spec math + data cleaning
- Property: incremental/no-lookahead + no-overlap + max-trades/day
- Integratie smoke: 200 bars ophalen van MT5 + 1 backtest-run zonder crash
