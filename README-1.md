# Emevine Scalper Signal Bot

Rule-based M1/M5 scalping signal scanner for EUR/USD, GBP/USD, AUD/USD.

## Run it

```bash
pip install -r requirements.txt
streamlit run emevine_signal_bot.py
```

## What it does

- Pulls live OHLC data via `yfinance` for the selected pair/timeframe.
- Computes: 200 EMA, 48 SMA, 22 EMA on Weighted Close (HLCC/4), 10 EMA, 1 EMA,
  RSI(1), and a hand-rolled Parabolic SAR (step 0.02 / max 0.20).
- Flags a BUY/SELL only when **all five** confluence rules align (trend,
  cross, momentum, RSI extreme-rejection, PSAR flip).
- Computes SL from the recent swing high/low + buffer, and TP1/TP2 at your
  chosen fixed R:R (1:1.5 or 1:2).
- Plots everything on an interactive candlestick + RSI chart with entry/SL/TP
  markers, in the gold/pink/black/turquoise theme.
- Auto-reloads every 2 minutes.
- "Save Signals" writes the current signal log to a CSV in your chosen folder.
- "Clear Cache" wipes Streamlit's cached data fetch and reruns.

## Important limitations — read before using with real money

1. **Spread filter is manual, not live.** Yahoo Finance does not expose real
   broker bid/ask spreads for retail FX pairs. The sidebar "Live spread"
   field is something *you* update from your own broker's feed — the app
   blocks the signal banner above 1.2 pips but cannot see your broker's
   spread automatically.
2. **75% win rate is a design target, not a guarantee.** It reflects how
   tightly the five confluence filters are stacked, not a backtested,
   audited statistic. Forward-test on a demo account first.
3. **No order execution.** This is a signal/analysis dashboard only — it
   does not place trades. Wiring it to Deriv's or a broker's live trading
   API would be a separate, higher-stakes step.
4. **yfinance intraday limits:** 1-minute data is only available for the
   trailing ~7 days; 5-minute data for ~60 days. That's a Yahoo Finance
   constraint, not something this app can change.
