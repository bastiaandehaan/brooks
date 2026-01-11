# STATUS — Brooks US500.cash Trading Bot (planner-only)

## Project
- Repo: `brooks` (Python / PyCharm)
- Broker/Platform: MT5 (FTMO)
- Instrument: `US500.cash`
- Goal: state-of-the-art, MVP-first trading bot based on Al Brooks price action (H2/L2 pullback in trend)
- Current mode: **planner-only** (detect/plan/log/backtest). **No MT5 execution yet.**

---

## Core Contract (Non-negotiables)
### NEXT_OPEN
- Signal bar = **last CLOSED bar**
- Execute = **OPEN of the next bar**
- Must support both datasets:
  - MT5 live rates may include a **current forming bar**
  - Backtest/CSV may contain **closed bars only**
- Therefore: `plan_next_open_trade(...)` supports:
  - current bar included -> execute on last index
  - closed bars only -> execute on **synthetic next timestamp** (= last_ts + timeframe)

### No Look-Ahead
- Context inference uses only history up to `t` (`slice <= t`)
- Entry/exit simulation must not peek beyond current bar

### Risk/Exposure Guardrails
- Entries only during session: **09:30–15:00 America/New_York**
- Force exit (planned next step): **15:55 America/New_York**
- Max trades per NY day: **parameter** (default 2)
- One trade per timestamp
- No overlap planned (position state later)

---

## Repo Layout
- `utils/`
  - `mt5_client.py` — MT5 init/connect/shutdown + symbol discovery
  - `mt5_data.py` — fetch_rates + rates_to_df (sorted index, dupes removed, OHLC schema)
  - `symbol_spec.py` — SymbolSpec (tick_size/tick_value/usd per price unit/lot etc.)
- `strategies/`
  - `context.py` — M15 trend filter (EMA + slope + above/below fraction) => Trend.BULL/BEAR/NONE
  - `h2l2.py` — H2/L2 planner + `plan_next_open_trade` (NEXT_OPEN)
- `execution/`
  - `guardrails.py` — session filter + max trades/day (planner-only)
- `backtest/`
  - `runner.py` — rolling backtest runner (uses planner + guardrails)
  - `visualiser.py` — exports dashboard png
- `tests/`
  - H2/L2 semantics (NEXT_OPEN current bar + synthetic next bar)
  - Guardrails
  - Backtest exit simulation (including both-hit policy)
  - Daily selection determinism test

---

## Current Status
- Tests: ✅ `pytest -q` -> ALL GREEN
- Latest focus:
  - Backtest runner exists and runs on MT5 historical bars
  - Daily cap enforcement proven correct (NY day bucketing)
  - Deterministic selection step planned/added (top-N per NY day)

---

## Key Backtest Evidence (historical)
### Trades per day validation (NY day bucketing)
- Script: `scripts/check_trades_per_day.py`
- Example output:
  - raw signals: 856
  - accepted trades: 20
  - trades per NY day: exactly 2 per trading day

### Backtest results (before selection / older runner)
- 60d:
  - trades: 128
  - winrate: ~41%
  - net: +31R
  - PF: 1.41
  - Sharpe: 1.85
  - max DD: -12R
- 180d:
  - trades: 380
  - winrate: ~36%
  - net: +31R
  - PF: 1.13
  - Sharpe: 1.10
  - max DD: -21R

Interpretation: edge is modest over 180d; improvements should target **selection + context filter + realism** before parameter optimization.

---

## Recent Work / Fixes
- H2L2 API compatibility restored (tests expect `min_risk_points` alias / NEXT_OPEN semantics)
- Backtest runner aligned with `H2L2Params` API
- Added/maintained:
  - exit simulation policy: SL/TP hit in same bar => worst-case SL
  - include execute bar in future simulation
- Added tests:
  - `test_backtest_exit_simulation.py`
  - `test_backtest_both_hit_policy.py`
  - `test_backtest_daily_selection.py` (float safe: `pytest.approx`)

---

## Open TODO (Next Steps)
1) Backtest runner parameterization
   - Add CLI: `--max-trades-day` (default 2)
   - Use it in selection + guardrails

2) Selection improvement (no look-ahead)
   - MVP score currently may be risk-based (tight stops)
   - Improve to “quality score”:
     - trend quality metrics from `context.py`
     - signal strength (close-position/body size/pullback depth)

3) Backtest realism
   - Time-based exit at **15:55 NY**
   - Cost model: spread + slippage parametric

4) Only after above: grid search / walk-forward
   - minimal knobs: TP_R (0.75..2.0), maybe min_risk threshold
   - walk-forward train/test slices (avoid overfit)

---

## Commands
- Run tests:
  - `pytest -q`
- Plan candidates live (planner-only):
  - `python main.py --symbol US500.cash --m5-bars 500 --m15-bars 300 --max-trades-day 2`
- Backtest:
  - `python -m backtest.runner --days 10`
  - `python -m backtest.runner --days 60`
  - `python -m backtest.runner --days 180`
- Validate trades/day cap:
  - `python scripts/check_trades_per_day.py`

---

## Definition of Done (MVP Backtest v1)
- All tests green
- Runner supports:
  - session filtering
  - deterministic daily selection
  - SL/TP simulation incl. execute bar
  - both-hit policy
  - time-exit 15:55 NY
  - spread/slippage model
- Metrics stable over ≥ 180 days
- Logging at INFO/DEBUG for:
  - context
  - selection
  - guardrails rejections
  - exit outcomes
