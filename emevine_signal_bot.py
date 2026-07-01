"""
==============================================================================
 EMEVINE SCALPER SIGNAL BOT
 Rule-based multi-indicator scalping signal scanner for EURUSD / GBPUSD / AUDUSD
 Timeframes: M1 / M5
------------------------------------------------------------------------------
 Indicator stack:
   - 200 EMA (Close)              -> major trend
   - 48 SMA  (Close)               -> medium-term dynamic S/R
   - 22 EMA  (Weighted Close HLCC/4) -> momentum trigger line
   - 10 EMA  (Close)               -> fast execution line
   - 1  EMA  (Close)               -> price-action tracker
   - RSI(1)  (Close)               -> extreme exhaustion filter (8 / 50 / 92)
   - Parabolic SAR (step 0.02, max 0.20) -> volatility / trend-flip filter

 IMPORTANT DISCLAIMER:
 This tool is for educational / informational purposes only. It is NOT
 financial advice and the 75% win-rate target is a DESIGN GOAL, not a
 guarantee. Forward-test on a demo account before risking real capital.
 yfinance does not provide true broker bid/ask spreads for FX -> the spread
 filter here uses a manual override that you should set from your own
 broker's live spread feed.
==============================================================================
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------------
# PAGE CONFIG + THEME (Gold / Pink / Black / Turquoise)
# ------------------------------------------------------------------------
st.set_page_config(
    page_title="Emevine Scalper Signal Bot",
    page_icon="\U0001F4C8",
    layout="wide",
    initial_sidebar_state="expanded",
)

GOLD = "#FFD700"
PINK = "#FF4FA3"
TURQ = "#1FE3CE"
BLACK = "#000000"
DARKBG = "#0B0B0F"

CUSTOM_CSS = f"""
<style>
.stApp {{
    background-color: {DARKBG};
}}
h1, h2, h3 {{
    color: {GOLD} !important;
    font-weight: 800 !important;
}}
p, label, span, div[data-testid="stMarkdownContainer"] {{
    color: {PINK};
}}
[data-testid="stSidebar"] {{
    background-color: {BLACK};
    border-right: 2px solid {GOLD};
}}
[data-testid="stMetric"] {{
    background-color: {GOLD};
    border-radius: 10px;
    padding: 10px;
    border: 2px solid {TURQ};
}}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
    color: {BLACK} !important;
    font-weight: 800 !important;
}}
.stButton>button {{
    background-color: {TURQ};
    color: {BLACK};
    font-weight: 800;
    border-radius: 8px;
    border: 2px solid {GOLD};
}}
.stButton>button:hover {{
    background-color: {GOLD};
    color: {BLACK};
    border: 2px solid {PINK};
}}
.signal-buy {{
    background-color: {TURQ};
    color: {BLACK};
    padding: 6px 10px;
    border-radius: 6px;
    font-weight: 800;
}}
.signal-sell {{
    background-color: {PINK};
    color: {BLACK};
    padding: 6px 10px;
    border-radius: 6px;
    font-weight: 800;
}}
hr {{ border-color: {GOLD}; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------------
# AUTO-REFRESH EVERY 2 MINUTES (120s) — pure HTML meta-refresh, no extra deps
# ------------------------------------------------------------------------
REFRESH_SECONDS = 120
st.markdown(f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">', unsafe_allow_html=True)

# ------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------------------
st.sidebar.markdown(f"<h2 style='color:{GOLD}'>\u2699\ufe0f Control Panel</h2>", unsafe_allow_html=True)

PAIR_MAP = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
}
pair_label = st.sidebar.selectbox("Currency Pair", list(PAIR_MAP.keys()), index=0)
ticker = PAIR_MAP[pair_label]

timeframe = st.sidebar.radio("Timeframe", ["M1", "M5"], index=1, horizontal=True)
interval = "1m" if timeframe == "M1" else "5m"
period = "5d" if timeframe == "M1" else "60d"

st.sidebar.markdown("---")
st.sidebar.markdown(f"<h3 style='color:{TURQ}'>Risk Management</h3>", unsafe_allow_html=True)
rr_choice = st.sidebar.selectbox("Risk:Reward (TP1)", ["1:1.5", "1:2.0"], index=0)
tp1_rr = 1.5 if rr_choice == "1:1.5" else 2.0
tp2_rr = 2.0 if tp1_rr == 1.5 else 2.5
sl_buffer_pips = st.sidebar.slider("SL buffer beyond swing (pips)", 1.0, 5.0, 2.5, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown(f"<h3 style='color:{TURQ}'>Spread Filter</h3>", unsafe_allow_html=True)
st.sidebar.caption("yfinance has no live broker bid/ask for FX — enter your broker's current spread manually.")
manual_spread = st.sidebar.number_input("Live spread (pips)", min_value=0.0, max_value=10.0, value=0.8, step=0.1)
max_spread = 1.2

st.sidebar.markdown("---")
st.sidebar.markdown(f"<h3 style='color:{TURQ}'>Parabolic SAR</h3>", unsafe_allow_html=True)
psar_step = st.sidebar.number_input("Step", value=0.020, step=0.001, format="%.3f")
psar_max = st.sidebar.number_input("Max", value=0.200, step=0.01, format="%.2f")

st.sidebar.markdown("---")
save_dir = st.sidebar.text_input("Save directory", value="signal_logs")
col_a, col_b = st.sidebar.columns(2)
save_clicked = col_a.button("\U0001F4BE Save Signals")
clear_clicked = col_b.button("\U0001F9F9 Clear Cache")

if clear_clicked:
    st.cache_data.clear()
    st.sidebar.success("Cache cleared.")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Auto-refresh: every {REFRESH_SECONDS//60} min")
st.sidebar.caption(f"Last load: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------------------------------------------------------------
# INDICATOR FUNCTIONS
# ------------------------------------------------------------------------
def weighted_close(df):
    """HLCC/4"""
    return (df["High"] + df["Low"] + 2 * df["Close"]) / 4


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def sma(series, period):
    return series.rolling(window=period).mean()


def rsi(series, period=1):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df, period=14):
    """Average True Range (Wilder smoothing)."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def volume_analysis(df, period=20):
    """Volume SMA + ratio of current volume to its average. Returns (vol_sma, vol_ratio)."""
    if "Volume" not in df.columns or df["Volume"].fillna(0).sum() == 0:
        empty = pd.Series(np.nan, index=df.index)
        return empty, empty
    vol_sma = df["Volume"].rolling(window=period).mean()
    vol_ratio = df["Volume"] / vol_sma.replace(0, np.nan)
    return vol_sma, vol_ratio


def calculate_psar(df, step=0.02, max_step=0.2):
    """Classic Wilder Parabolic SAR. Returns (psar series, bullish-bool series)."""
    high = df["High"].values
    low = df["Low"].values
    n = len(df)
    psar = np.zeros(n)
    bullish = np.zeros(n, dtype=bool)

    if n == 0:
        return pd.Series(dtype=float), pd.Series(dtype=bool)

    bull = True
    af = step
    ep = high[0]
    psar[0] = low[0]
    bullish[0] = True

    for i in range(1, n):
        prev_psar = psar[i - 1]
        if bull:
            val = prev_psar + af * (ep - prev_psar)
            val = min(val, low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < val:
                bull = False
                val = ep
                ep = low[i]
                af = step
        else:
            val = prev_psar + af * (ep - prev_psar)
            val = max(val, high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > val:
                bull = True
                val = ep
                ep = high[i]
                af = step

        if bull and high[i] > ep:
            ep = high[i]
            af = min(af + step, max_step)
        elif not bull and low[i] < ep:
            ep = low[i]
            af = min(af + step, max_step)

        psar[i] = val
        bullish[i] = bull

    return pd.Series(psar, index=df.index), pd.Series(bullish, index=df.index)


def pip_size(pair_label):
    # All three pairs here are standard 4-decimal pairs (no JPY)
    return 0.0001


# ------------------------------------------------------------------------
# DATA FETCH (cached)
# ------------------------------------------------------------------------
@st.cache_data(ttl=110, show_spinner="Fetching live price data...")
def fetch_data(ticker, interval, period):
    df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
    if df.empty:
        return df
    # yfinance sometimes returns MultiIndex columns even for single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna(how="all")
    return df


# ------------------------------------------------------------------------
# SIGNAL ENGINE
# ------------------------------------------------------------------------
def build_signals(df, psar_step, psar_max, pip, sl_buffer_pips, tp1_rr, tp2_rr, rsi_lookback=10):
    df = df.copy()
    df["wc"] = weighted_close(df)
    df["ema200"] = ema(df["Close"], 200)
    df["sma48"] = sma(df["Close"], 48)
    df["ema22"] = ema(df["wc"], 22)
    df["ema10"] = ema(df["Close"], 10)
    df["ema1"] = ema(df["Close"], 1)
    df["rsi1"] = rsi(df["Close"], 1)
    psar, bull = calculate_psar(df, step=psar_step, max_step=psar_max)
    df["psar"] = psar
    df["psar_bull"] = bull

    # --- Supplementary technical indicators (context/display only) ---
    # These do NOT alter the core 5-rule confluence logic below; they add
    # extra market context (trend strength, volatility, volume) on the chart.
    df["ema50"] = ema(df["Close"], 50)
    df["rsi14"] = rsi(df["Close"], 14)
    df["atr14"] = atr(df, 14)
    df["vol_sma20"], df["vol_ratio"] = volume_analysis(df, 20)

    df["signal"] = None
    df["sl"] = np.nan
    df["tp1"] = np.nan
    df["tp2"] = np.nan

    warmup = 200 + rsi_lookback
    buffer_price = sl_buffer_pips * pip

    for i in range(min(warmup, len(df) - 1), len(df)):
        price = df["Close"].iloc[i]
        ema200_v = df["ema200"].iloc[i]
        sma48_v = df["sma48"].iloc[i]
        ema10_now, ema10_prev = df["ema10"].iloc[i], df["ema10"].iloc[i - 1]
        ema22_now, ema22_prev = df["ema22"].iloc[i], df["ema22"].iloc[i - 1]
        ema1_now = df["ema1"].iloc[i]
        rsi_now = df["rsi1"].iloc[i]
        rsi_window = df["rsi1"].iloc[max(0, i - rsi_lookback): i + 1]
        psar_now = df["psar"].iloc[i]
        psar_bull_now = df["psar_bull"].iloc[i]

        if pd.isna(ema200_v) or pd.isna(sma48_v) or pd.isna(ema22_now):
            continue

        cross_up = (ema10_prev <= ema22_prev) and (ema10_now > ema22_now)
        cross_down = (ema10_prev >= ema22_prev) and (ema10_now < ema22_now)

        trend_up = (price > ema200_v) and (price > sma48_v)
        trend_down = (price < ema200_v) and (price < sma48_v)

        rsi_dip_reject = (rsi_window.min() <= 8) and (rsi_now > rsi_window.min())
        rsi_spike_reject = (rsi_window.max() >= 92) and (rsi_now < rsi_window.max())

        momentum_up = ema1_now >= ema10_now
        momentum_down = ema1_now <= ema10_now

        psar_below_price = psar_now < price
        psar_above_price = psar_now > price

        buy = (
            trend_up and cross_up and momentum_up
            and rsi_dip_reject and psar_below_price and psar_bull_now
        )
        sell = (
            trend_down and cross_down and momentum_down
            and rsi_spike_reject and psar_above_price and (not psar_bull_now)
        )

        idx = df.index[i]
        if buy:
            swing_low = df["Low"].iloc[max(0, i - 10): i + 1].min()
            sl_price = swing_low - buffer_price
            risk = price - sl_price
            if risk > 0:
                df.loc[idx, "signal"] = "BUY"
                df.loc[idx, "sl"] = sl_price
                df.loc[idx, "tp1"] = price + risk * tp1_rr
                df.loc[idx, "tp2"] = price + risk * tp2_rr
        elif sell:
            swing_high = df["High"].iloc[max(0, i - 10): i + 1].max()
            sl_price = swing_high + buffer_price
            risk = sl_price - price
            if risk > 0:
                df.loc[idx, "signal"] = "SELL"
                df.loc[idx, "sl"] = sl_price
                df.loc[idx, "tp1"] = price - risk * tp1_rr
                df.loc[idx, "tp2"] = price - risk * tp2_rr

    return df


# ------------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------------
st.markdown(
    f"<h1>\U0001F947 EMEVINE SCALPER SIGNAL BOT</h1>"
    f"<p style='color:{TURQ}; font-size:16px;'>SMC / ICT-style M1-M5 precision scalper — "
    f"200EMA / 48SMA / 22EMA(HLCC4) / 10EMA / 1EMA / RSI(1) / Parabolic SAR</p>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='color:{PINK}; font-size:13px;'>\u26A0\uFE0F Educational tool only — not financial advice. "
    f"75% target win rate is a design goal based on strict confluence rules, not a guarantee of future results. "
    f"START/STOP/CLOSE ALL/RESET control a <b>simulated paper-trading session</b> only — "
    f"this app has no broker connection and never places real orders.</p>",
    unsafe_allow_html=True,
)

raw = fetch_data(ticker, interval, period)

if raw.empty:
    st.error(
        f"No data returned for {pair_label} on {timeframe}. "
        "Yahoo Finance intraday FX data can be intermittently unavailable — try again shortly."
    )
    st.stop()

pip = pip_size(pair_label)
data = build_signals(raw, psar_step, psar_max, pip, sl_buffer_pips, tp1_rr, tp2_rr)

spread_blocked = manual_spread > max_spread
recent = data.tail(300)
last_row = data.iloc[-1]

# ------------------------------------------------------------------------
# BOT SESSION STATE (paper-trading simulation — NOT live broker execution)
# ------------------------------------------------------------------------
# NOTE: This app only fetches quotes from Yahoo Finance and has no broker
# connection. START/STOP/CLOSE ALL/RESET control a simulated session so you
# can track how the signal engine would have performed — no real orders are
# ever sent anywhere.
if "bot_running" not in st.session_state:
    st.session_state.bot_running = False
if "open_positions" not in st.session_state:
    st.session_state.open_positions = []
if "closed_positions" not in st.session_state:
    st.session_state.closed_positions = []
if "last_signal_time" not in st.session_state:
    st.session_state.last_signal_time = {}


def _exit_check(pos, price, rsi_now):
    """Return (exit_reason, exit_price) or (None, None)."""
    if pos["direction"] == "BUY":
        if price <= pos["sl"]:
            return "SL", pos["sl"]
        if price >= pos["tp2"]:
            return "TP2", pos["tp2"]
        if price >= pos["tp1"]:
            return "TP1", pos["tp1"]
        if rsi_now >= 92:
            return "RSI-extreme", price
    else:
        if price >= pos["sl"]:
            return "SL", pos["sl"]
        if price <= pos["tp2"]:
            return "TP2", pos["tp2"]
        if price <= pos["tp1"]:
            return "TP1", pos["tp1"]
        if rsi_now <= 8:
            return "RSI-extreme", price
    return None, None


def _close_position(pos, exit_price, reason, pip):
    direction_mult = 1 if pos["direction"] == "BUY" else -1
    pnl_pips = (exit_price - pos["entry"]) / pip * direction_mult
    pos.update({
        "status": "CLOSED",
        "exit": exit_price,
        "exit_reason": reason,
        "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pnl_pips": round(pnl_pips, 1),
    })
    return pos


# ---- Control buttons ----
st.markdown(f"<h3 style='color:{TURQ}'>Bot Controls</h3>", unsafe_allow_html=True)
b1, b2, b3, b4 = st.columns(4)
start_clicked = b1.button("\u25B6\ufe0f START BOT", use_container_width=True)
stop_clicked = b2.button("\u23F9\ufe0f STOP BOT", use_container_width=True)
close_all_clicked = b3.button("\u274C CLOSE ALL TRADES", use_container_width=True)
reset_clicked = b4.button("\U0001F504 RESET SESSION", use_container_width=True)

if start_clicked:
    st.session_state.bot_running = True
if stop_clicked:
    st.session_state.bot_running = False
if reset_clicked:
    st.session_state.bot_running = False
    st.session_state.open_positions = []
    st.session_state.closed_positions = []
    st.session_state.last_signal_time = {}
    st.rerun()
if close_all_clicked:
    for pos in st.session_state.open_positions:
        _close_position(pos, last_row["Close"], "Manual close-all", pip)
        st.session_state.closed_positions.append(pos)
    st.session_state.open_positions = []

status_badge = "\U0001F7E2 RUNNING" if st.session_state.bot_running else "\U0001F534 STOPPED"
st.markdown(
    f"<p style='color:{GOLD}; font-weight:800;'>Bot status: {status_badge} &nbsp;|&nbsp; "
    f"Open positions: {len(st.session_state.open_positions)} &nbsp;|&nbsp; "
    f"Closed this session: {len(st.session_state.closed_positions)}</p>",
    unsafe_allow_html=True,
)

# ---- Monitor open positions for SL/TP/RSI-extreme exits every rerun ----
still_open = []
for pos in st.session_state.open_positions:
    reason, exit_price = _exit_check(pos, last_row["Close"], last_row["rsi1"])
    if reason:
        _close_position(pos, exit_price, reason, pip)
        st.session_state.closed_positions.append(pos)
    else:
        still_open.append(pos)
st.session_state.open_positions = still_open

# ---- Auto-open a paper position on a fresh confluence signal, if bot running ----
if st.session_state.bot_running and not spread_blocked and pd.notna(last_row["signal"]):
    bar_time = str(data.index[-1])
    already_has_open_for_pair = any(p["pair"] == pair_label for p in st.session_state.open_positions)
    if (
        st.session_state.last_signal_time.get(pair_label) != bar_time
        and not already_has_open_for_pair
    ):
        st.session_state.open_positions.append({
            "id": f"{pair_label}-{bar_time}",
            "pair": pair_label,
            "direction": last_row["signal"],
            "entry": last_row["Close"],
            "sl": last_row["sl"],
            "tp1": last_row["tp1"],
            "tp2": last_row["tp2"],
            "open_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "OPEN",
        })
    st.session_state.last_signal_time[pair_label] = bar_time

# ---- Top metrics row ----
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(f"{pair_label} Price", f"{last_row['Close']:.5f}")
m2.metric("RSI(1)", f"{last_row['rsi1']:.1f}")
m3.metric("PSAR Trend", "Bullish" if last_row["psar_bull"] else "Bearish")
m4.metric("Spread (pips)", f"{manual_spread:.1f}", delta=("BLOCKED" if spread_blocked else "OK"))
m5.metric("Timeframe", timeframe)

# ---- Extra indicator metrics row ----
atr_pips = last_row["atr14"] / pip if pd.notna(last_row["atr14"]) else np.nan
vol_ratio_val = last_row["vol_ratio"] if pd.notna(last_row["vol_ratio"]) else None
n1, n2, n3, n4 = st.columns(4)
n1.metric("EMA 50", f"{last_row['ema50']:.5f}")
n2.metric("EMA 200", f"{last_row['ema200']:.5f}")
n3.metric("RSI(14)", f"{last_row['rsi14']:.1f}")
n4.metric("ATR(14)", f"{atr_pips:.1f} pips" if pd.notna(atr_pips) else "N/A")

if vol_ratio_val is not None:
    vol_state = "Elevated" if vol_ratio_val >= 1.5 else ("Below avg" if vol_ratio_val < 0.7 else "Normal")
    st.caption(f"Volume: {vol_ratio_val:.2f}\u00d7 20-bar average ({vol_state})")
else:
    st.caption("Volume: not available for this pair via Yahoo Finance (spot FX typically has no exchange volume).")

# ---- Open / closed position panels ----
if st.session_state.open_positions:
    st.markdown(f"<h4 style='color:{TURQ}'>Open Positions</h4>", unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(st.session_state.open_positions)[
            ["pair", "direction", "entry", "sl", "tp1", "tp2", "open_time"]
        ].style.format(precision=5, subset=["entry", "sl", "tp1", "tp2"]),
        use_container_width=True,
    )

if st.session_state.closed_positions:
    closed_df = pd.DataFrame(st.session_state.closed_positions)
    wins = (closed_df["pnl_pips"] > 0).sum()
    losses = (closed_df["pnl_pips"] <= 0).sum()
    total_pips = closed_df["pnl_pips"].sum()
    win_rate = 100 * wins / len(closed_df) if len(closed_df) else 0
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Session Trades", len(closed_df))
    s2.metric("Win Rate", f"{win_rate:.0f}%")
    s3.metric("Wins / Losses", f"{wins} / {losses}")
    s4.metric("Total Pips", f"{total_pips:+.1f}")
    with st.expander("Closed trades this session"):
        st.dataframe(
            closed_df[["pair", "direction", "entry", "exit", "exit_reason", "pnl_pips", "open_time", "exit_time"]]
            .iloc[::-1].style.format(precision=5, subset=["entry", "exit"]),
            use_container_width=True,
        )



if spread_blocked:
    st.markdown(
        f"<div class='signal-sell'>\U0001F6AB SPREAD FILTER ACTIVE — live spread "
        f"{manual_spread:.1f} pips exceeds the {max_spread} pip max. New trades are blocked.</div>",
        unsafe_allow_html=True,
    )

# ---- Latest signal callout ----
recent_signals = data[data["signal"].notna()].tail(5)
if not recent_signals.empty:
    last_sig = recent_signals.iloc[-1]
    css_class = "signal-buy" if last_sig["signal"] == "BUY" else "signal-sell"
    blocked_note = " (spread-blocked, no execution)" if spread_blocked else ""
    st.markdown(
        f"<div class='{css_class}'>Latest signal: {last_sig['signal']} @ {last_sig['Close']:.5f} "
        f"| SL {last_sig['sl']:.5f} | TP1 {last_sig['tp1']:.5f} | TP2 {last_sig['tp2']:.5f}"
        f"{blocked_note}</div>",
        unsafe_allow_html=True,
    )
else:
    st.info("No confluence signal on the most recent bars. Waiting for full setup alignment.")

st.markdown("---")

# ------------------------------------------------------------------------
# CHART
# ------------------------------------------------------------------------
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True, row_heights=[0.48, 0.20, 0.14, 0.18],
    vertical_spacing=0.025,
    subplot_titles=(f"{pair_label} — {timeframe}", "RSI(1) & RSI(14)", "ATR(14)", "Volume"),
)

fig.add_trace(go.Candlestick(
    x=recent.index, open=recent["Open"], high=recent["High"],
    low=recent["Low"], close=recent["Close"], name="Price",
    increasing_line_color=TURQ, decreasing_line_color=PINK,
), row=1, col=1)

fig.add_trace(go.Scatter(x=recent.index, y=recent["ema200"], name="200 EMA",
                          line=dict(color=GOLD, width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=recent.index, y=recent["ema50"], name="50 EMA",
                          line=dict(color="orange", width=1.3, dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=recent.index, y=recent["sma48"], name="48 SMA",
                          line=dict(color=TURQ, width=1.5, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=recent.index, y=recent["ema22"], name="22 EMA (HLCC/4)",
                          line=dict(color=PINK, width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=recent.index, y=recent["ema10"], name="10 EMA",
                          line=dict(color="white", width=1.2)), row=1, col=1)

fig.add_trace(go.Scatter(
    x=recent.index, y=recent["psar"], name="Parabolic SAR", mode="markers",
    marker=dict(size=4, color=np.where(recent["psar_bull"], GOLD, PINK)),
), row=1, col=1)

buys = recent[recent["signal"] == "BUY"]
sells = recent[recent["signal"] == "SELL"]

fig.add_trace(go.Scatter(
    x=buys.index, y=buys["Close"], mode="markers", name="BUY entry",
    marker=dict(symbol="triangle-up", size=14, color=TURQ, line=dict(color=GOLD, width=1.5)),
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=sells.index, y=sells["Close"], mode="markers", name="SELL entry",
    marker=dict(symbol="triangle-down", size=14, color=PINK, line=dict(color=GOLD, width=1.5)),
), row=1, col=1)

for label, col, color, dash in [
    ("TP1", "tp1", "lime", "dash"), ("TP2", "tp2", GOLD, "dot"), ("SL", "sl", "red", "dash"),
]:
    pts = recent[recent[col].notna()]
    if not pts.empty:
        fig.add_trace(go.Scatter(
            x=pts.index, y=pts[col], mode="markers", name=label,
            marker=dict(symbol="line-ew", size=10, color=color, line=dict(width=2, color=color)),
        ), row=1, col=1)

fig.add_trace(go.Scatter(x=recent.index, y=recent["rsi1"], name="RSI(1)",
                          line=dict(color=PINK, width=1.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=recent.index, y=recent["rsi14"], name="RSI(14)",
                          line=dict(color=GOLD, width=1.3, dash="dash")), row=2, col=1)
for level, color in [(92, PINK), (50, "white"), (8, TURQ)]:
    fig.add_hline(y=level, line=dict(color=color, width=1, dash="dot"), row=2, col=1)

fig.add_trace(go.Scatter(
    x=recent.index, y=recent["atr14"], name="ATR(14)", fill="tozeroy",
    line=dict(color=TURQ, width=1.3),
), row=3, col=1)

if recent["vol_sma20"].notna().any():
    vol_colors = np.where(recent["Close"] >= recent["Open"], TURQ, PINK)
    fig.add_trace(go.Bar(x=recent.index, y=recent["Volume"], name="Volume",
                          marker_color=vol_colors, opacity=0.7), row=4, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent["vol_sma20"], name="Vol SMA(20)",
                              line=dict(color=GOLD, width=1.3)), row=4, col=1)
else:
    fig.add_annotation(
        text="Volume data unavailable for this pair via Yahoo Finance",
        xref="x4 domain", yref="y4 domain", x=0.5, y=0.5, showarrow=False,
        font=dict(color=PINK, size=12), row=4, col=1,
    )

fig.update_layout(
    height=950,
    paper_bgcolor=BLACK, plot_bgcolor=DARKBG,
    font=dict(color=GOLD),
    legend=dict(bgcolor=BLACK, font=dict(color=PINK)),
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=40, b=10),
)
fig.update_xaxes(gridcolor="#222")
fig.update_yaxes(gridcolor="#222")

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------
# SIGNAL LOG TABLE
# ------------------------------------------------------------------------
st.markdown(f"<h3 style='color:{TURQ}'>Recent Signal Log</h3>", unsafe_allow_html=True)
log_cols = ["signal", "Close", "sl", "tp1", "tp2", "rsi1"]
signal_log = data[data["signal"].notna()][log_cols].tail(20).iloc[::-1]
signal_log = signal_log.rename(columns={"Close": "entry", "rsi1": "rsi"})
st.dataframe(signal_log.style.format(precision=5), use_container_width=True)

# ------------------------------------------------------------------------
# SAVE TO DIRECTORY
# ------------------------------------------------------------------------
if save_clicked:
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{pair_label.replace('/', '')}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fpath = os.path.join(save_dir, fname)
    export_df = data[data["signal"].notna()][log_cols].rename(columns={"Close": "entry", "rsi1": "rsi"})
    export_df.to_csv(fpath)
    st.success(f"Saved {len(export_df)} signals to {fpath}")

st.caption(
    "Core signal rules (unchanged): trend filter (200EMA/48SMA) -> 10/22 EMA cross with 1EMA momentum "
    "confluence -> RSI(1) extreme rejection (8/92) -> Parabolic SAR flip confirmation. "
    "SL = swing high/low + buffer. TP1/TP2 = fixed R:R, with early exit on opposing RSI extreme. "
    "EMA50, RSI(14), ATR(14) and Volume are supplementary context indicators and do not change entry logic. "
    "Bot controls run a simulated paper-trading session only — no live orders are placed."
)
