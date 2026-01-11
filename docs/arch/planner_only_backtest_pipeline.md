# Planner-only Backtest Pipeline (MVP v1.1)

## Data
- MT5 rates M15 + M5 (UTC, tz-aware)
- Clean: sort index, drop dupes, OHLC schema

## Regime / Context (M15)
- infer_trend_m15(slice<=t) -> Trend.BULL/BEAR/NONE
- Trend "quality" metrics (slope, close-ema dist, above/below fraction)

## Signal / Planner (M5)
- plan_next_open_trade:
  - signal_bar = laatst gesloten bar
  - execute = open next bar
  - ondersteunt 'closed bars only' via synthetic next timestamp
- produceert PlannedTrade (entry/stop/tp/reason)

## Selection (nieuw)
- per NY-day: score candidates en kies deterministisch max N (N=2)
- één trade per timestamp
- (later) no-overlap / position state

## Guardrails
- session 09:30–15:00 NY (entries)
- max_trades_per_day=2 (executed)
- log accepted/rejected + reden

## Exit Simulation (MVP)
- SL/TP (execute bar inbegrepen)
- both-hit same bar => worst-case SL
- (nieuw) time-exit 15:55 NY
- (nieuw) kostenmodel: spread+slippage parametriseerbaar

## Metrics
- R-metrics: winrate, PF, Sharpe (op R), max DD depth+duration
- breakdown per day/week, hitrate per side, per trend bucket
