import itertools
import pandas as pd
from runner import run_backtest  # Importeer je verbeterde functie


def start_optimization():
    # --- CONFIGURATIE ---
    SYMBOL = "US500.cash"
    DAYS = 60
    COSTS = 0.04  # Reken altijd met kosten!

    # --- DE PARAMETER GRID ---
    # Pas de waarden hieronder aan om meer of minder combinaties te testen
    grid = {
        'ema_period': [10, 20, 50],
        'pullback_bars': [2, 3, 5],
        'signal_close_frac': [0.20, 0.30, 0.45],
        'chop_threshold': [1.5, 2.5, 3.5],
        'regime_filter': [True]
    }

    # Maak alle combinaties
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    all_results = []
    total = len(combinations)

    print(f"🚀 Start optimalisatie voor {SYMBOL}")
    print(f"📊 Totaal aantal combinaties: {total}")
    print("-" * 50)

    for i, params in enumerate(combinations, 1):
        print(f"[{i}/{total}] Testen: EMA={params['ema_period']}, PB={params['pullback_bars']}, "
              f"Frac={params['signal_close_frac']}, Chop={params['chop_threshold']}")

        # Voer de backtest uit met de parameters uit de grid
        res = run_backtest(
            symbol=SYMBOL,
            days=DAYS,
            ema_period=params['ema_period'],
            pullback_bars=params['pullback_bars'],
            signal_close_frac=params['signal_close_frac'],
            regime_filter=params['regime_filter'],
            chop_threshold=params['chop_threshold'],
            costs_per_trade_r=COSTS
        )

        # Als er trades zijn gedaan, sla het resultaat op
        if "error" not in res and res.get("trades", 0) > 0:
            # Voeg de gebruikte parameters toe aan de resultaten voor de analyse
            row = {**params, **res}
            all_results.append(row)
        else:
            print("   ⚠️ Geen trades gevonden voor deze combinatie.")

    # --- ANALYSE ---
    if not all_results:
        print("❌ Geen enkele combinatie leverde trades op.")
        return

    df = pd.DataFrame(all_results)

    # Sorteer op Sharpe Ratio (beste maatstaf voor stabiliteit)
    # Of gebruik 'net_r' voor maximale winst
    top_sharpe = df.sort_values("daily_sharpe_r", ascending=False).head(5)

    print("\n" + "=" * 80)
    print("🏆 TOP 5 PARAMETER COMBINATIES (Geselecteerd op Daily Sharpe)")
    print("=" * 80)

    cols_to_show = ['ema_period', 'pullback_bars', 'signal_close_frac', 'chop_threshold', 'net_r', 'daily_sharpe_r',
                    'trades', 'winrate']
    print(top_sharpe[cols_to_show].to_string(index=False))

    # Opslaan naar CSV voor diepere analyse in Excel
    df.to_csv("optimization_results.csv", index=False)
    print(f"\n✅ Alle resultaten zijn opgeslagen in 'optimization_results.csv'")


if __name__ == "__main__":
    start_optimization()