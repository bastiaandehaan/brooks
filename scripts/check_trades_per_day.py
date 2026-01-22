import MetaTrader5 as mt5
import pandas as pd

from execution.guardrails import Guardrails, apply_guardrails
from strategies.context import Trend, TrendParams, infer_trend_m15
from strategies.h2l2 import H2L2Params, Side, plan_next_open_trade
from utils.mt5_client import Mt5Client
from utils.mt5_data import RatesRequest, fetch_rates

SYMBOL = "US500.cash"
DAYS = 10
count_m5 = DAYS * 288
count_m15 = DAYS * 96 * 2

c = Mt5Client(mt5_module=mt5)
ok = c.initialize()
if not ok:
    raise SystemExit("MT5 init failed")

spec = c.get_symbol_specification(SYMBOL)
if spec is None:
    c.shutdown()
    raise SystemExit("No spec")

m15 = fetch_rates(mt5, RatesRequest(SYMBOL, mt5.TIMEFRAME_M15, count_m15))
m5 = fetch_rates(mt5, RatesRequest(SYMBOL, mt5.TIMEFRAME_M5, count_m5))

trends = []
tp = TrendParams()
for i in range(len(m15)):
    t, _ = infer_trend_m15(m15.iloc[: i + 1], tp)
    trends.append(t)

m15 = m15.copy()
m15["trend"] = trends
trend_data = m15[["trend"]]

m5t = pd.merge_asof(
    m5.sort_index(),
    trend_data.sort_index(),
    left_index=True,
    right_index=True,
    direction="backward",
)

p = H2L2Params(
    min_risk_price_units=1.0,
    signal_close_frac=0.30,
    pullback_bars=2,
    cooldown_bars=0,
)

raw = []
for i in range(200, len(m5t)):
    tr = m5t.iloc[i]["trend"]
    if tr not in (Trend.BULL, Trend.BEAR):
        continue
    side = Side.LONG if tr == Trend.BULL else Side.SHORT
    sl = m5t.iloc[i - 50 : i + 1]
    t = plan_next_open_trade(sl, side, spec, p, timeframe_minutes=5)
    if t:
        raw.append(t)

g = Guardrails(
    max_trades_per_day=2,
    session_start="09:30",
    session_end="15:00",
    day_tz="America/New_York",
    session_tz="America/New_York",
)

acc, rej = apply_guardrails(raw, g)

print("raw", len(raw), "accepted", len(acc), "rejected", len(rej))

ny = "America/New_York"
exec_ts = pd.Series([t.execute_ts for t in acc], dtype="datetime64[ns, UTC]")
days = exec_ts.dt.tz_convert(ny).dt.date

print("\nTRADES PER NY DAY:")
print(days.value_counts().sort_index())

c.shutdown()
