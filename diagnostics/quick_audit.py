# diagnostics/quick_audit.py
"""
Quick diagnostics voor H2/L2 strategie
Gebruik bestaande MT5 data + strategies
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import pandas as pd
from strategies.context import Trend, TrendParams, infer_trend_m15_series
from strategies.h2l2 import H2L2Params, Side, plan_h2l2_trades
from strategies.regime import RegimeParams, detect_regime_series

# Import uit jouw bestaande project
from utils.mt5_client import Mt5Client, Mt5ConnectionParams
from utils.mt5_data import RatesRequest, fetch_rates
from utils.symbol_spec import SymbolSpec


def fetch_data_from_mt5(symbol: str, start_date: datetime, end_date: datetime):
    """Fetch M5 and M15 data using your existing MT5 client"""

    # Initialize MT5
    client = Mt5Client(mt5, Mt5ConnectionParams())
    if not client.initialize():
        raise RuntimeError("Failed to initialize MT5")

    try:
        # Ensure symbol is selected
        if not client.ensure_selected(symbol):
            raise RuntimeError(f"Failed to select {symbol}")

        # Get symbol spec
        spec = client.get_symbol_specification(symbol)
        if spec is None:
            raise RuntimeError(f"Failed to get spec for {symbol}")

        print(f"📊 Fetching {symbol} data from MT5...")

        # Calculate approximate bar count (conservative estimate)
        days = (end_date - start_date).days
        m5_bars_per_day = 288  # 24h * 60min / 5min
        m15_bars_per_day = 96  # 24h * 60min / 15min

        # Fetch M5 data
        m5_req = RatesRequest(
            symbol=symbol,
            timeframe=mt5.TIMEFRAME_M5,
            count=min(days * m5_bars_per_day, 100000),  # Cap at 100k bars
            pos=0,
        )
        m5 = fetch_rates(mt5, m5_req)

        # Fetch M15 data
        m15_req = RatesRequest(
            symbol=symbol,
            timeframe=mt5.TIMEFRAME_M15,
            count=min(days * m15_bars_per_day, 100000),
            pos=0,
        )
        m15 = fetch_rates(mt5, m15_req)

        # Filter to date range (MT5 fetches from most recent backwards)
        m5 = m5[m5.index >= pd.Timestamp(start_date, tz="UTC")]
        m5 = m5[m5.index <= pd.Timestamp(end_date, tz="UTC")]

        m15 = m15[m15.index >= pd.Timestamp(start_date, tz="UTC")]
        m15 = m15[m15.index <= pd.Timestamp(end_date, tz="UTC")]

        print(f"✅ Loaded {len(m5)} M5 bars, {len(m15)} M15 bars")

        return m5, m15, spec

    finally:
        client.shutdown()


def get_session(ts):
    """Classify trading session"""
    ny_hour = ts.tz_convert("America/New_York").hour
    if 9 <= ny_hour < 11:
        return "NY_OPEN"
    elif 3 <= ny_hour < 9:
        return "LONDON"
    elif 11 <= ny_hour < 14:
        return "NY_MID"
    elif 14 <= ny_hour < 16:
        return "NY_CLOSE"
    else:
        return "OTHER"


def has_rejection_wick(bar, side):
    """Check wick-based rejection (ignoring body color)"""
    bar_range = bar["high"] - bar["low"]
    if bar_range < 0.01:
        return False

    if side == "LONG":
        lower_wick = min(bar["open"], bar["close"]) - bar["low"]
        return lower_wick > bar_range * 0.5
    else:
        upper_wick = bar["high"] - max(bar["open"], bar["close"])
        return upper_wick > bar_range * 0.5


def quick_audit():
    """Run alle diagnostics in 1 keer"""

    # Fetch data
    m5, m15, spec = fetch_data_from_mt5(
        symbol="US500.cash",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 12, 31),
    )

    if m5.empty or m15.empty:
        print("❌ No data fetched. Check MT5 connection and symbol.")
        return

    # Initialize strategy params
    h2l2_params = H2L2Params()
    trend_params = TrendParams()
    regime_params = RegimeParams()

    print("\n📈 Running strategy analysis...")

    # Detecteer regimes & trends
    m15["regime"] = detect_regime_series(m15, regime_params)
    m15["trend"] = infer_trend_m15_series(m15, trend_params)

    # === DIAGNOSTIEK 1: REGIME DISTRIBUTIE ===
    print("\n" + "=" * 60)
    print("DIAGNOSTIEK 1: Regime Distributie Per Kwartaal")
    print("=" * 60)

    m15["quarter"] = m15.index.to_period("Q")
    regime_dist = m15.groupby(["quarter", "regime"]).size().unstack(fill_value=0)
    regime_pct = regime_dist.div(regime_dist.sum(axis=1), axis=0) * 100

    print("\nPercentage bars per regime:")
    print(regime_pct.round(1))

    # Check if 2023 was more choppy than 2024
    if len(regime_pct) > 0:
        avg_2023 = regime_pct[
            regime_pct.index.astype(str).str.startswith("2023")
        ].mean()
        avg_2024 = regime_pct[
            regime_pct.index.astype(str).str.startswith("2024")
        ].mean()

        print("\n📊 Gemiddelde regime distributie:")
        print(f"2023: {avg_2023.to_dict()}")
        print(f"2024: {avg_2024.to_dict()}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    regime_pct.plot(kind="bar", stacked=True, ax=ax, colormap="RdYlGn")
    ax.set_title("Regime Distributie 2023-2024", fontsize=14, fontweight="bold")
    ax.set_ylabel("Percentage bars (%)")
    ax.set_xlabel("Kwartaal")
    ax.legend(title="Regime")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Create output directory if needed
    Path("diagnostics").mkdir(exist_ok=True)
    plt.savefig("diagnostics/regime_distribution.png", dpi=150)
    print("\n💾 Saved: diagnostics/regime_distribution.png")
    plt.close()

    # === DIAGNOSTIEK 2: SESSION ANALYSIS ===
    print("\n" + "=" * 60)
    print("DIAGNOSTIEK 2: Performance Per Sessie")
    print("=" * 60)

    # Generate signals for both BULL and BEAR trends
    all_signals = []

    # Get unique trends from M15
    bull_periods = m15[m15["trend"] == Trend.BULL]
    bear_periods = m15[m15["trend"] == Trend.BEAR]

    print(f"\nTrend periods found:")
    print(
        f"  BULL bars: {len(bull_periods)} ({len(bull_periods) / len(m15) * 100:.1f}%)"
    )
    print(
        f"  BEAR bars: {len(bear_periods)} ({len(bear_periods) / len(m15) * 100:.1f}%)"
    )

    # For simplicity, run H2/L2 on full M5 with both sides
    # (In real backtest, you'd align M15 trend with M5 bars)
    try:
        long_signals = plan_h2l2_trades(m5, Side.LONG, spec, h2l2_params)
        short_signals = plan_h2l2_trades(m5, Side.SHORT, spec, h2l2_params)
        all_signals = long_signals + short_signals
    except Exception as e:
        print(f"⚠️  Error generating signals: {e}")
        all_signals = []

    print(f"\n✅ Generated {len(all_signals)} total signals")

    if all_signals:
        # Group by session
        signal_df = pd.DataFrame(
            [
                {
                    "session": get_session(s.signal_ts),
                    "side": str(s.side),
                    "entry": s.entry,
                    "stop": s.stop,
                    "tp": s.tp,
                }
                for s in all_signals
            ]
        )

        session_counts = signal_df["session"].value_counts()
        print("\n📊 Signals per sessie:")
        for session, count in session_counts.items():
            pct = (count / len(signal_df)) * 100
            print(f"  {session:12s}: {count:4d} ({pct:5.1f}%)")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        session_counts.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title("Signal Distributie Per Sessie", fontsize=14, fontweight="bold")
        ax.set_ylabel("Aantal Signals")
        ax.set_xlabel("Trading Sessie")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("diagnostics/session_distribution.png", dpi=150)
        print("💾 Saved: diagnostics/session_distribution.png")
        plt.close()

    # === DIAGNOSTIEK 3: VOLATILITY REGIME ===
    print("\n" + "=" * 60)
    print("DIAGNOSTIEK 3: ATR Regimes")
    print("=" * 60)

    m15["atr_14"] = (m15["high"] - m15["low"]).rolling(14).mean()
    m15["atr_regime"] = pd.cut(
        m15["atr_14"], bins=[0, 8, 15, 100], labels=["LOW", "MED", "HIGH"]
    )

    atr_dist = m15.groupby(["quarter", "atr_regime"]).size().unstack(fill_value=0)
    atr_pct = atr_dist.div(atr_dist.sum(axis=1), axis=0) * 100

    print("\n📊 ATR regime distributie per kwartaal:")
    print(atr_pct.round(1))

    # Compare 2023 vs 2024
    if len(atr_pct) > 0:
        avg_atr_2023 = atr_pct[atr_pct.index.astype(str).str.startswith("2023")].mean()
        avg_atr_2024 = atr_pct[atr_pct.index.astype(str).str.startswith("2024")].mean()

        print("\n📊 Gemiddelde ATR regime:")
        print(f"2023: {avg_atr_2023.to_dict()}")
        print(f"2024: {avg_atr_2024.to_dict()}")

    # === DIAGNOSTIEK 4: REJECTION BAR AUDIT ===
    print("\n" + "=" * 60)
    print("DIAGNOSTIEK 4: Rejection Bar Analysis")
    print("=" * 60)

    # Sample bars to check for missed signals
    sample_size = min(2000, len(m5))
    sample = m5.sample(sample_size) if len(m5) > sample_size else m5

    rejection_with_wrong_body = 0
    for idx, bar in sample.iterrows():
        # Check LONG rejection with bearish body
        if has_rejection_wick(bar, "LONG") and bar["close"] < bar["open"]:
            rejection_with_wrong_body += 1
        # Check SHORT rejection with bullish body
        elif has_rejection_wick(bar, "SHORT") and bar["close"] > bar["open"]:
            rejection_with_wrong_body += 1

    missed_pct = (rejection_with_wrong_body / len(sample)) * 100
    print(
        f"\n⚠️  Bars met rejection wick maar 'verkeerde' body kleur: {rejection_with_wrong_body}/{len(sample)} ({missed_pct:.1f}%)"
    )
    print(f"    (Dit zijn potentieel gemiste signals door body color filter)")

    # === DIAGNOSTIEK 5: ROLLING WIN RATE SIMULATION ===
    print("\n" + "=" * 60)
    print("DIAGNOSTIEK 5: Edge Stability Check")
    print("=" * 60)

    if all_signals:
        # Simulate trade outcomes (simplified - just check if TP was hit first)
        signal_df["r_multiple"] = 0.0
        signal_df["outcome"] = "unknown"

        for idx, sig in enumerate(all_signals[:100]):  # Sample first 100 for speed
            # Find next bars after entry
            entry_time = sig.execute_ts
            future_bars = m5[m5.index > entry_time].head(50)  # Check next 50 bars

            if len(future_bars) == 0:
                continue

            # Check if TP or SL hit first
            if str(sig.side) == "Side.LONG":
                hit_tp = (future_bars["high"] >= sig.tp).any()
                hit_sl = (future_bars["low"] <= sig.stop).any()

                if hit_tp:
                    first_tp = future_bars[future_bars["high"] >= sig.tp].index[0]
                    first_sl = (
                        future_bars[future_bars["low"] <= sig.stop].index[0]
                        if hit_sl
                        else pd.Timestamp.max
                    )

                    if first_tp < first_sl:
                        signal_df.loc[idx, "r_multiple"] = 2.0
                        signal_df.loc[idx, "outcome"] = "win"
                    else:
                        signal_df.loc[idx, "r_multiple"] = -1.0
                        signal_df.loc[idx, "outcome"] = "loss"
                elif hit_sl:
                    signal_df.loc[idx, "r_multiple"] = -1.0
                    signal_df.loc[idx, "outcome"] = "loss"

            # Mirror for SHORT
            else:
                hit_tp = (future_bars["low"] <= sig.tp).any()
                hit_sl = (future_bars["high"] >= sig.stop).any()

                if hit_tp:
                    first_tp = future_bars[future_bars["low"] <= sig.tp].index[0]
                    first_sl = (
                        future_bars[future_bars["high"] >= sig.stop].index[0]
                        if hit_sl
                        else pd.Timestamp.max
                    )

                    if first_tp < first_sl:
                        signal_df.loc[idx, "r_multiple"] = 2.0
                        signal_df.loc[idx, "outcome"] = "win"
                    else:
                        signal_df.loc[idx, "r_multiple"] = -1.0
                        signal_df.loc[idx, "outcome"] = "loss"
                elif hit_sl:
                    signal_df.loc[idx, "r_multiple"] = -1.0
                    signal_df.loc[idx, "outcome"] = "loss"

        # Calculate win rate on simulated trades
        completed = signal_df[signal_df["outcome"] != "unknown"]
        if len(completed) > 0:
            win_rate = (completed["outcome"] == "win").sum() / len(completed) * 100
            avg_r = completed["r_multiple"].mean()

            print(f"\n📊 Simplified backtest (first 100 signals):")
            print(f"  Completed trades: {len(completed)}")
            print(f"  Win rate: {win_rate:.1f}%")
            print(f"  Avg R-multiple: {avg_r:.2f}R")
            print(f"  Expected R per trade: {avg_r:.2f}R (positive = profitable)")

    # === SUMMARY ===
    print("\n" + "=" * 60)
    print("SAMENVATTING")
    print("=" * 60)

    print(f"""
📊 Data periode: {m5.index[0].date()} tot {m5.index[-1].date()}
📈 Totaal signals: {len(all_signals)}

🎯 Key findings:
1. Regime: Check regime_distribution.png
2. Sessions: {"Meeste signals in " + session_counts.idxmax() + " sessie" if all_signals else "Geen signals"}
3. Volatility: Check console output hierboven
4. Rejection bars: ~{missed_pct:.0f}% mogelijk gemist door body filter

💡 Volgende stap:
   - Review de plots in diagnostics/ folder
   - Bepaal welke fix hoogste prioriteit heeft
   - Implementeer verbetering in main project
   - Re-run backtest om impact te valideren
""")


if __name__ == "__main__":
    try:
        quick_audit()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
