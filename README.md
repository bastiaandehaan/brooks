Dit is een overzichtelijke samenvatting van het **Brooks US500.cash Trading Bot** framework, gebaseerd op de door jou gedeelde documentatie.

---

## 🤖 Brooks US500.cash Trading Bot

Dit framework implementeert een **Al Brooks-stijl** price action systeem (H2/L2 second entries) voor **US500.cash** op MetaTrader 5 (FTMO). Het is ontworpen als een MVP (Minimum Viable Product) met een strikte focus op regels en backtesting.

### 🚀 Kernfunctionaliteiten

* **Live Planner:** Detecteert setups en genereert een `NEXT_OPEN` handelsplan (Entry, SL, TP en sizing). *Let op: uitvoering is handmatig, geen automatische orders.*
* **Backtest Runner:** Simuleert volledige handelsperiodes met exact dezelfde regels als de planner en genereert prestatie-dashboards.
* **Marktregime Filter:** Optionele module die 'choppy' markten vermijdt door ATR te vergelijken met de rolling price range.

---

### 📏 Strategie & Regels (Non-negotiables)

| Onderdeel | Specificatie |
| --- | --- |
| **Instrument** | `US500.cash` (M5/M15 tijdframe) |
| **Signaal** | Brooks H2/L2 setups op de M5 |
| **Context** | Trendfilter via M15 EMA + slope |
| **Executie** | `NEXT_OPEN`: Signaal op de laatste gesloten bar, executeer op de opening van de volgende bar |
| **Sessie** | Alleen New York sessie (**09:30–15:00** America/New_York) |
| **Risico** | Fixed-R (1R stop, 2R target). Maximaal 2 trades per dag |

---

### 📂 Structuur van het Framework

* **`strategies/`**: Bevat de logica voor trend-inferences (`context.py`), regime-detectie (`regime.py`) en de H2/L2 planning (`h2l2.py`).
* **`execution/`**: Beheert de guardrails (sessietijden, limieten) en risicomanagement.
* **`backtest/`**: De engine voor simulaties en visualisatie van equity curves en drawdowns.
* **`scripts/`**: Handige tools voor grid-searches (optimalisatie) en robuustheidstests.

---

### 💻 Gebruik (Quick Start)

**Live Planner draaien:**

```powershell
python main.py --symbol US500.cash --m5-bars 500 --m15-bars 300 --max-trades-day 2

```

**Backtest uitvoeren (bijv. 180 dagen):**

```powershell
python -m backtest.runner --symbol US500.cash --days 180 --max-trades-day 2

```

**Met regime filter:**
Voeg `--regime-filter --chop-threshold 2.5` toe aan je commando.

---

### 🛠️ Installatie Vereisten

1. **Python 3.10+**
2. **Windows** (vereist voor de MT5 terminal connectie)
3. **FTMO MT5 Account** lokaal geconfigureerd

> **Belangrijke realisme-check:** In de backtest wordt bij een 'both-hit' bar (zowel SL als TP geraakt in dezelfde bar) altijd uitgegaan van het worst-case scenario: de **Stop Loss**.


