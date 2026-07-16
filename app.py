"""
ICT / Smart Money Concept / Price Action Signal Bot
----------------------------------------------------
A Streamlit app that:
  1. Auto-runs full analysis the moment a pair + timeframe is chosen
  2. Pulls REAL market data (yfinance)
  3. Computes trend (EMAs), market structure (swing HH/HL/LH/LL),
     support/resistance, liquidity pools (equal highs/lows),
     simplified ICT Order Blocks + Fair Value Gaps, and RSI
  4. Scores confluence into BUY / SELL / NO TRADE with Entry, SL,
     TP1-3, R:R and a Confidence Score
  5. Checks the HIGHER timeframe trend and warns on conflicts
  6. Pulls a REAL economic calendar feed and explains the likely
     ripple effect of upcoming high-impact news on the specific pair
     (this is scenario analysis based on historical tendencies -
     NOT a claim of knowing future news outcomes, which is impossible)

DISCLAIMER: Educational tool only. Not financial advice.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go

st.set_page_config(page_title="ICT / SMC Signal Bot", page_icon="📈", layout="wide",
                    initial_sidebar_state="expanded")

# ============================================================
# STYLING (dark, professional, colorful accents)
# ============================================================

CUSTOM_CSS = """
<style>
:root{
  --bg:#0b0f1a; --panel:#121826; --panel2:#161e2e; --border:#232c40;
  --green:#16c784; --green-soft:#0f2f26; --red:#ea3943; --red-soft:#341418;
  --amber:#f5a623; --blue:#3b82f6; --text:#e6e9f0; --muted:#8b93a7;
}
html, body, [class*="css"]  { color: var(--text); }
.stApp{
  background: radial-gradient(1200px 600px at 10% -10%, #131b2e 0%, var(--bg) 55%) fixed;
}
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0e1422 0%, #0b0f1a 100%);
  border-right: 1px solid var(--border);
}
h1, h2, h3 { letter-spacing: .2px; }
.hero{
  background: linear-gradient(120deg, #10192e 0%, #131b30 60%, #0f1626 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 22px 26px;
  margin-bottom: 18px;
  box-shadow: 0 8px 30px rgba(0,0,0,.35);
}
.hero h1{ margin:0 0 4px 0; font-size: 1.65rem; background: linear-gradient(90deg,#7cf5c4,#3b82f6 60%,#a78bfa);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p{ margin:0; color: var(--muted); font-size:.92rem; }

.badge{ display:inline-block; padding:5px 14px; border-radius:999px; font-weight:700; font-size:.85rem; letter-spacing:.4px; }
.badge-buy{ background:var(--green-soft); color:var(--green); border:1px solid var(--green); }
.badge-sell{ background:var(--red-soft); color:var(--red); border:1px solid var(--red); }
.badge-none{ background:#26314a; color:var(--muted); border:1px solid var(--border); }
.badge-bullish{ background:var(--green-soft); color:var(--green); border:1px solid var(--green); }
.badge-bearish{ background:var(--red-soft); color:var(--red); border:1px solid var(--red); }
.badge-ranging{ background:#2a2617; color:var(--amber); border:1px solid var(--amber); }

.card{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 18px;
  height: 100%;
}
.card .label{ color: var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.6px; margin-bottom:4px;}
.card .value{ font-size:1.35rem; font-weight:700; }
.value-buy{ color: var(--green); }
.value-sell{ color: var(--red); }
.value-neutral{ color: var(--text); }

.conf-wrap{ background:#1b2337; border-radius:999px; height:10px; overflow:hidden; border:1px solid var(--border);}
.conf-bar{ height:100%; border-radius:999px; background: linear-gradient(90deg, var(--red), var(--amber), var(--green)); }

.reason-item{
  background: var(--panel2);
  border-left: 3px solid var(--blue);
  border-radius: 8px;
  padding: 8px 12px;
  margin-bottom: 8px;
  font-size: .92rem;
}
.sr-pill{
  display:inline-block; padding:4px 10px; border-radius:8px; margin:3px 6px 3px 0; font-weight:600; font-size:.85rem;
}
.sr-support{ background: var(--green-soft); color: var(--green); border:1px solid var(--green); }
.sr-resistance{ background: var(--red-soft); color: var(--red); border:1px solid var(--red); }

.news-card{
  background: var(--panel);
  border: 1px solid var(--border);
  border-left: 4px solid var(--amber);
  border-radius: 10px;
  padding: 12px 16px;
  margin-bottom: 10px;
}
.news-high{ border-left-color: var(--red); }
.news-medium{ border-left-color: var(--amber); }
.news-low{ border-left-color: var(--green); }
.news-title{ font-weight:700; margin-bottom:2px; }
.news-meta{ color: var(--muted); font-size:.82rem; margin-bottom:6px;}
.news-ripple{ color:#c7cee0; font-size:.88rem; }

.warn-box{
  background: var(--red-soft); border:1px solid var(--red); color:#ffb4b8;
  border-radius:10px; padding:12px 16px; margin:10px 0; font-size:.92rem;
}
.disclaimer{ color: var(--muted); font-size:.8rem; }
</style>
"""

# ============================================================
# STATIC CONFIG
# ============================================================

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "CHF/JPY": "CHFJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "GBP/CHF": "GBPCHF=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/CAD": "EURCAD=X",
    "GBP/USD": "GBPUSD=X",
    "CAD/CHF": "CADCHF=X",
    "GBP/CAD": "GBPCAD=X",
    "AUD/NZD": "AUDNZD=X",
    "AUD/CHF": "AUDCHF=X",
    "EUR/AUD": "EURAUD=X",
    "XAU/USD (Gold)": "GC=F",
    "XAG/USD (Silver)": "SI=F",
    "BTC/USD": "BTC-USD",
    "Custom (type Yahoo ticker)": None,
}

TIMEFRAMES = {
    "5 Minutes":  dict(interval="5m",  period="5d",  resample=None),
    "15 Minutes": dict(interval="15m", period="5d",  resample=None),
    "30 Minutes": dict(interval="30m", period="1mo", resample=None),
    "1 Hour":     dict(interval="60m", period="1mo", resample=None),
    "4 Hours":    dict(interval="60m", period="3mo", resample="4h"),
    "Daily":      dict(interval="1d",  period="1y",  resample=None),
    "Weekly":     dict(interval="1wk", period="5y",  resample=None),
}

# Higher timeframe used to sanity-check the trade against the bigger picture
HTF_MAP = {
    "5 Minutes": "1 Hour",
    "15 Minutes": "1 Hour",
    "30 Minutes": "4 Hours",
    "1 Hour": "4 Hours",
    "4 Hours": "Daily",
    "Daily": "Weekly",
    "Weekly": "Weekly",
}

NEWS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Generic, historically-typical reaction map used for the ripple-effect
# explanation. This is general market tendency, not a guarantee.
EVENT_RIPPLE = {
    "interest rate": "Rate decisions are the single biggest driver of a currency. A hike or hawkish hold typically strengthens the currency; a cut or dovish tone typically weakens it.",
    "rate decision": "Rate decisions are the single biggest driver of a currency. A hike or hawkish hold typically strengthens the currency; a cut or dovish tone typically weakens it.",
    "fomc": "FOMC statements/press conferences drive USD volatility sharply. Hawkish language (inflation concern, fewer cuts) is USD-positive; dovish language is USD-negative.",
    "non-farm": "A strong NFP beat usually strengthens USD (more rate-hold/hike expectation); a big miss usually weakens USD (more rate-cut expectation).",
    "employment change": "Stronger-than-expected jobs data tends to strengthen the currency on rate-hold expectations; weak data tends to weaken it.",
    "unemployment": "A falling unemployment rate is currency-positive; a rising rate is currency-negative.",
    "cpi": "Hotter-than-expected inflation (CPI) usually strengthens the currency (rate-hike/hold bets); cooler inflation usually weakens it.",
    "inflation": "Hotter-than-expected inflation usually strengthens the currency (rate-hike/hold bets); cooler inflation usually weakens it.",
    "gdp": "Stronger GDP growth is currency-positive; a miss or contraction is currency-negative.",
    "pmi": "A PMI print above 50 and above forecast is currency-positive (expansion); below 50 or a miss is currency-negative (contraction).",
    "retail sales": "Stronger consumer spending data is currency-positive; weaker data is currency-negative.",
    "trade balance": "A wider-than-expected surplus is mildly currency-positive; a wider deficit is mildly currency-negative.",
    "consumer confidence": "Higher confidence readings are mildly currency-positive; lower readings are mildly currency-negative.",
    "housing": "Stronger housing data is mildly currency-positive; weaker data is mildly currency-negative.",
    "ism": "A beat above 50/forecast is currency-positive; a miss is currency-negative.",
}

# ============================================================
# DATA FETCHING
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_price_data(ticker: str, interval: str, period: str, resample: str | None):
    df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if resample:
        df = df.resample(resample).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna()
    return df


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news():
    try:
        resp = requests.get(NEWS_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# INDICATORS
# ============================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def find_swings(df: pd.DataFrame, left: int = 3, right: int = 3):
    """Simple fractal-based swing high/low detector."""
    highs, lows = [], []
    h, l = df["High"].values, df["Low"].values
    for i in range(left, len(df) - right):
        window_h = h[i - left:i + right + 1]
        if h[i] == window_h.max() and np.sum(window_h == h[i]) == 1:
            highs.append((df.index[i], h[i]))
        window_l = l[i - left:i + right + 1]
        if l[i] == window_l.min() and np.sum(window_l == l[i]) == 1:
            lows.append((df.index[i], l[i]))
    return highs, lows


def cluster_levels(levels, tolerance_pct=0.0012):
    """Merge nearby price levels into single clustered levels."""
    if not levels:
        return []
    vals = sorted([v for _, v in levels])
    clusters = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - clusters[-1][-1]) / clusters[-1][-1] <= tolerance_pct:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [round(sum(c) / len(c), 6) for c in clusters]


def detect_equal_levels(levels, tolerance_pct=0.0008):
    """Detect liquidity pools: 2+ swing points sitting at ~the same price."""
    vals = sorted([v for _, v in levels])
    pools = []
    i = 0
    while i < len(vals):
        j = i
        group = [vals[i]]
        while j + 1 < len(vals) and abs(vals[j + 1] - vals[i]) / vals[i] <= tolerance_pct:
            j += 1
            group.append(vals[j])
        if len(group) >= 2:
            pools.append(round(sum(group) / len(group), 6))
        i = j + 1
    return pools


def market_structure(swing_highs, swing_lows):
    """Classify trend from the last few swing points (HH/HL vs LH/LL)."""
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "Ranging", []
    last_highs = [v for _, v in swing_highs[-3:]]
    last_lows = [v for _, v in swing_lows[-3:]]
    notes = []
    bullish = last_highs[-1] > last_highs[0] and last_lows[-1] > last_lows[0]
    bearish = last_highs[-1] < last_highs[0] and last_lows[-1] < last_lows[0]
    if bullish:
        notes.append("Higher Highs and Higher Lows detected (bullish structure)")
        return "Bullish", notes
    if bearish:
        notes.append("Lower Highs and Lower Lows detected (bearish structure)")
        return "Bearish", notes
    notes.append("No clean HH/HL or LH/LL sequence - structure is choppy")
    return "Ranging", notes


def detect_order_blocks(df, atr_series):
    """Very simplified ICT order block detection:
    last opposite-colour candle before a strong impulse move."""
    bullish_ob, bearish_ob = None, None
    closes = df["Close"].values
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    a = atr_series.values
    n = len(df)
    for i in range(n - 2, 5, -1):
        if np.isnan(a[i]) or a[i] == 0:
            continue
        move = closes[i] - opens[i]
        if move > 1.4 * a[i] and opens[i - 1] > closes[i - 1] and bullish_ob is None:
            bullish_ob = (df.index[i - 1], lows[i - 1], highs[i - 1])
        if move < -1.4 * a[i] and closes[i - 1] > opens[i - 1] and bearish_ob is None:
            bearish_ob = (df.index[i - 1], lows[i - 1], highs[i - 1])
        if bullish_ob and bearish_ob:
            break
    return bullish_ob, bearish_ob


def detect_fvg(df):
    """3-candle Fair Value Gap detection (bullish + bearish), most recent first."""
    bullish_fvg, bearish_fvg = None, None
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    for i in range(n - 1, 1, -1):
        if lows[i] > highs[i - 2] and bullish_fvg is None:
            bullish_fvg = (highs[i - 2], lows[i])
        if highs[i] < lows[i - 2] and bearish_fvg is None:
            bearish_fvg = (lows[i - 2], highs[i])
        if bullish_fvg and bearish_fvg:
            break
    return bullish_fvg, bearish_fvg


def get_decimals(pair_label: str) -> int:
    if "JPY" in pair_label:
        return 3
    if "XAU" in pair_label or "Gold" in pair_label:
        return 2
    if "XAG" in pair_label or "Silver" in pair_label:
        return 3
    if "BTC" in pair_label:
        return 2
    return 5


# ============================================================
# CORE ANALYSIS ENGINE
# ============================================================

def analyze(df: pd.DataFrame, pair_label: str):
    df = df.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200) if len(df) > 210 else np.nan
    df["RSI"] = rsi(df["Close"], 14)
    df["ATR"] = atr(df, 14)

    price = float(df["Close"].iloc[-1])
    ema20, ema50 = float(df["EMA20"].iloc[-1]), float(df["EMA50"].iloc[-1])
    ema200 = df["EMA200"].iloc[-1]
    ema200 = float(ema200) if not pd.isna(ema200) else None
    cur_rsi = float(df["RSI"].iloc[-1])
    cur_atr = float(df["ATR"].iloc[-1]) if not pd.isna(df["ATR"].iloc[-1]) else price * 0.001

    swing_highs, swing_lows = find_swings(df)
    structure, structure_notes = market_structure(swing_highs, swing_lows)

    res_levels = sorted([lv for lv in cluster_levels(swing_highs) if lv > price])[:3]
    sup_levels = sorted([lv for lv in cluster_levels(swing_lows) if lv < price], reverse=True)[:3]

    liquidity_highs = detect_equal_levels(swing_highs)
    liquidity_lows = detect_equal_levels(swing_lows)

    bullish_ob, bearish_ob = detect_order_blocks(df, df["ATR"])
    bullish_fvg, bearish_fvg = detect_fvg(df)

    # ---------------- Confluence scoring ----------------
    score = 0
    reasons = []

    if ema200 is not None:
        if price > ema20 > ema50 > ema200:
            score += 2; reasons.append("Price is above EMA20/EMA50/EMA200 in bullish order — clear uptrend on this timeframe.")
        elif price < ema20 < ema50 < ema200:
            score -= 2; reasons.append("Price is below EMA20/EMA50/EMA200 in bearish order — clear downtrend on this timeframe.")
        else:
            reasons.append("Moving averages are mixed / overlapping — trend is not clean on this timeframe.")
    else:
        if price > ema20 > ema50:
            score += 1; reasons.append("Price is above EMA20 and EMA50 — short-term bullish bias.")
        elif price < ema20 < ema50:
            score -= 1; reasons.append("Price is below EMA20 and EMA50 — short-term bearish bias.")

    if structure == "Bullish":
        score += 2; reasons.append(structure_notes[0])
    elif structure == "Bearish":
        score -= 2; reasons.append(structure_notes[0])
    else:
        reasons.append(structure_notes[0])

    if cur_rsi >= 55 and cur_rsi < 70:
        score += 1; reasons.append(f"RSI at {cur_rsi:.1f} shows healthy bullish momentum (not yet overbought).")
    elif cur_rsi <= 45 and cur_rsi > 30:
        score -= 1; reasons.append(f"RSI at {cur_rsi:.1f} shows bearish momentum (not yet oversold).")
    elif cur_rsi >= 70:
        score -= 1; reasons.append(f"RSI at {cur_rsi:.1f} is overbought — pullback risk for new longs.")
    elif cur_rsi <= 30:
        score += 1; reasons.append(f"RSI at {cur_rsi:.1f} is oversold — bounce risk for new shorts.")

    # Price proximity to support/resistance (liquidity context)
    if sup_levels and abs(price - sup_levels[0]) / price <= 0.0025:
        score += 1; reasons.append(f"Price is sitting right on a key support level near {sup_levels[0]:.5f}.")
    if res_levels and abs(price - res_levels[0]) / price <= 0.0025:
        score -= 1; reasons.append(f"Price is sitting right on a key resistance level near {res_levels[0]:.5f}.")

    # Order block confluence
    if bullish_ob and bullish_ob[1] <= price <= bullish_ob[2] * 1.002:
        score += 1; reasons.append("Price is trading inside/near a bullish ICT Order Block.")
    if bearish_ob and bearish_ob[1] * 0.998 <= price <= bearish_ob[2]:
        score -= 1; reasons.append("Price is trading inside/near a bearish ICT Order Block.")

    # FVG confluence
    if bullish_fvg and bullish_fvg[0] <= price <= bullish_fvg[1]:
        score += 1; reasons.append("Price is filling a bullish Fair Value Gap — potential support reaction zone.")
    if bearish_fvg and bearish_fvg[1] <= price <= bearish_fvg[0]:
        score -= 1; reasons.append("Price is filling a bearish Fair Value Gap — potential resistance reaction zone.")

    # Liquidity sweep context
    if liquidity_lows and min(liquidity_lows, key=lambda x: abs(x - price)) < price and \
            abs(price - min(liquidity_lows, key=lambda x: abs(x - price))) / price <= 0.003:
        reasons.append("Equal lows detected just below price — resting buy-side liquidity that price may sweep before reversing up.")
    if liquidity_highs and abs(price - min(liquidity_highs, key=lambda x: abs(x - price))) / price <= 0.003:
        reasons.append("Equal highs detected just above price — resting sell-side liquidity that price may sweep before reversing down.")

    decimals = get_decimals(pair_label)
    signal = "NO TRADE"
    entry = sl = tp1 = tp2 = tp3 = rr = None

    BUY_THRESHOLD = 4
    SELL_THRESHOLD = -4

    if score >= BUY_THRESHOLD:
        signal = "BUY"
        entry = price
        base_low = min([lv for lv in sup_levels[:1]] + [price - cur_atr], default=price - cur_atr)
        sl = round(min(base_low, price - cur_atr * 1.2), decimals)
        risk = entry - sl
        tp1 = round(entry + risk * 1.0, decimals)
        tp2 = round(entry + risk * 2.0, decimals)
        tp3_struct = res_levels[-1] if res_levels else entry + risk * 3.0
        tp3 = round(max(tp3_struct, entry + risk * 2.5), decimals)
        rr = round((tp2 - entry) / risk, 2) if risk else None

    elif score <= SELL_THRESHOLD:
        signal = "SELL"
        entry = price
        base_high = max([lv for lv in res_levels[:1]] + [price + cur_atr], default=price + cur_atr)
        sl = round(max(base_high, price + cur_atr * 1.2), decimals)
        risk = sl - entry
        tp1 = round(entry - risk * 1.0, decimals)
        tp2 = round(entry - risk * 2.0, decimals)
        tp3_struct = sup_levels[-1] if sup_levels else entry - risk * 3.0
        tp3 = round(min(tp3_struct, entry - risk * 2.5), decimals)
        rr = round((entry - tp2) / risk, 2) if risk else None

    confidence = min(92, 50 + abs(score) * 6) if signal != "NO TRADE" else max(20, 50 - abs(score) * 3)

    return dict(
        price=round(price, decimals), entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3, rr=rr,
        signal=signal, confidence=confidence, structure=structure, reasons=reasons,
        support=[round(x, decimals) for x in sup_levels],
        resistance=[round(x, decimals) for x in res_levels],
        rsi=round(cur_rsi, 1), score=score, decimals=decimals,
        bullish_ob=bullish_ob, bearish_ob=bearish_ob,
        bullish_fvg=bullish_fvg, bearish_fvg=bearish_fvg,
        liquidity_highs=liquidity_highs, liquidity_lows=liquidity_lows,
        df=df,
    )


def htf_trend_only(df: pd.DataFrame):
    """Quick trend read for the higher timeframe (no full signal needed)."""
    d = df.copy()
    d["EMA20"] = ema(d["Close"], 20)
    d["EMA50"] = ema(d["Close"], 50)
    price = float(d["Close"].iloc[-1])
    e20, e50 = float(d["EMA20"].iloc[-1]), float(d["EMA50"].iloc[-1])
    if price > e20 > e50:
        return "Bullish"
    if price < e20 < e50:
        return "Bearish"
    return "Ranging"


# ============================================================
# NEWS / RIPPLE EFFECT
# ============================================================

def currencies_in_pair(pair_label: str):
    core = pair_label.split(" ")[0]
    parts = core.split("/")
    codes = []
    for p in parts:
        code = "".join([c for c in p if c.isalpha()])[:3].upper()
        if code:
            codes.append(code)
    if "BTC" in pair_label.upper():
        codes = ["USD"]
    if "XAU" in pair_label.upper():
        codes = ["USD"]
    if "XAG" in pair_label.upper():
        codes = ["USD"]
    return codes


def ripple_text(title: str, currency: str, base_ccy: str, quote_ccy: str):
    title_l = title.lower()
    match = None
    for key, text in EVENT_RIPPLE.items():
        if key in title_l:
            match = text
            break
    if not match:
        return None
    if currency == base_ccy:
        direction = f"a strong/beat result would tend to push this pair UP, a weak/miss result would tend to push it DOWN"
    elif currency == quote_ccy:
        direction = f"a strong/beat result would tend to push this pair DOWN, a weak/miss result would tend to push it UP"
    else:
        direction = "impact on this specific pair is indirect"
    return f"{match} Since this event is for {currency}, {direction}."


def render_news_section(pair_label: str):
    codes = currencies_in_pair(pair_label)
    data = fetch_news()
    st.subheader("📰 Upcoming News & Ripple Effect on This Pair")
    st.caption(
        "Real scheduled events from the economic calendar for the currencies in this pair. "
        "This is scenario analysis based on typical historical reactions — nobody can truly "
        "predict the actual outcome of unreleased data, so treat this as 'what usually happens if X', not a forecast."
    )
    if isinstance(data, dict) and "error" in data:
        st.warning(f"Could not load live economic calendar (network/feed issue: {data['error']}). "
                   f"Deploy this app with internet access to see live events.")
        return

    now = datetime.now(timezone.utc)
    relevant = []
    for ev in data:
        ccy = ev.get("country") or ev.get("currency")
        if ccy not in codes:
            continue
        try:
            ev_time = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
        except Exception:
            continue
        relevant.append((ev_time, ev, ccy))

    relevant.sort(key=lambda x: x[0])
    upcoming = [r for r in relevant if r[0] >= now - timedelta(hours=6)]

    if not upcoming:
        st.info("No high/medium-impact scheduled events found for this pair's currencies in the current feed window.")
        return

    base_ccy, quote_ccy = (codes[0], codes[1]) if len(codes) >= 2 else (codes[0], codes[0])

    impact_class = {"High": "news-high", "Medium": "news-medium", "Low": "news-low"}
    impact_emoji = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}

    for ev_time, ev, ccy in upcoming[:8]:
        impact = ev.get("impact", "Unknown")
        title = ev.get("title", "Event")
        forecast = ev.get("forecast", "n/a")
        previous = ev.get("previous", "n/a")
        badge = impact_emoji.get(impact, "⚪")
        css_cls = impact_class.get(impact, "")
        rt = ripple_text(title, ccy, base_ccy, quote_ccy)
        ripple_html = f'<div class="news-ripple">💡 {rt}</div>' if rt else ""
        st.markdown(
            f"""
            <div class="news-card {css_cls}">
              <div class="news-title">{badge} {ev_time.strftime('%a %d %b, %H:%M UTC')} — {ccy} — {title}</div>
              <div class="news-meta">Impact: {impact} &nbsp;|&nbsp; Forecast: {forecast} &nbsp;|&nbsp; Previous: {previous}</div>
              {ripple_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# CHART
# ============================================================

def render_chart(df: pd.DataFrame, analysis: dict, pair_label: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=pair_label,
        increasing_line_color="#16c784", increasing_fillcolor="#16c784",
        decreasing_line_color="#ea3943", decreasing_fillcolor="#ea3943",
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], line=dict(width=1.6, color="#f5a623"), name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], line=dict(width=1.6, color="#3b82f6"), name="EMA50"))
    if "EMA200" in df.columns and df["EMA200"].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], line=dict(width=1.6, color="#a78bfa"), name="EMA200"))

    for lv in analysis["support"]:
        fig.add_hline(y=lv, line_dash="dot", line_color="#16c784", opacity=0.6,
                      annotation_text=f"S {lv:.{analysis['decimals']}f}", annotation_font_color="#16c784",
                      annotation_position="right")
    for lv in analysis["resistance"]:
        fig.add_hline(y=lv, line_dash="dot", line_color="#ea3943", opacity=0.6,
                      annotation_text=f"R {lv:.{analysis['decimals']}f}", annotation_font_color="#ea3943",
                      annotation_position="right")

    if analysis["entry"]:
        fig.add_hline(y=analysis["entry"], line_color="#3b82f6", line_width=2,
                      annotation_text="ENTRY", annotation_font_color="#3b82f6")
    if analysis["sl"]:
        fig.add_hline(y=analysis["sl"], line_color="#ea3943", line_width=2, line_dash="dash",
                      annotation_text="SL", annotation_font_color="#ea3943")
    if analysis["tp1"]:
        fig.add_hline(y=analysis["tp1"], line_color="#16c784", line_width=1.4, line_dash="dash",
                      annotation_text="TP1", annotation_font_color="#16c784")
    if analysis["tp2"]:
        fig.add_hline(y=analysis["tp2"], line_color="#16c784", line_width=1.4, line_dash="dash",
                      annotation_text="TP2", annotation_font_color="#16c784")
    if analysis["tp3"]:
        fig.add_hline(y=analysis["tp3"], line_color="#16c784", line_width=1.4, line_dash="dash",
                      annotation_text="TP3", annotation_font_color="#16c784")

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#121826",
        plot_bgcolor="#121826",
        font=dict(color="#e6e9f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#232c40", showgrid=True),
        yaxis=dict(gridcolor="#232c40", showgrid=True),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <h1>📊 ICT · Smart Money Concept · Price Action Signal Bot</h1>
          <p>Automated multi-timeframe trend, market structure, liquidity, RSI and price-action confluence engine.
          Live data. Real economic calendar. Educational tool — not financial advice.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")
        pair_label = st.selectbox("Currency Pair", list(PAIRS.keys()), index=0)
        custom_ticker = None
        if PAIRS[pair_label] is None:
            custom_ticker = st.text_input("Enter Yahoo Finance ticker (e.g. USDSEK=X)", value="")
        tf_label = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=3)
        st.markdown("---")
        st.caption("Data: Yahoo Finance (yfinance). Economic calendar: public ForexFactory JSON feed.")
        run = st.button("🔄 Refresh Analysis", use_container_width=True)

    ticker = custom_ticker if PAIRS[pair_label] is None else PAIRS[pair_label]
    if not ticker:
        st.info("Enter a valid ticker in the sidebar to run analysis.")
        return

    tf_cfg = TIMEFRAMES[tf_label]

    with st.spinner("Fetching live price data..."):
        df = fetch_price_data(ticker, tf_cfg["interval"], tf_cfg["period"], tf_cfg["resample"])

    if df is None or len(df) < 30:
        st.error("Could not retrieve enough price data for this pair/timeframe. "
                 "Check the ticker symbol, your internet connection, or try a different timeframe.")
        return

    analysis = analyze(df, pair_label)

    # Higher timeframe check
    htf_label = HTF_MAP.get(tf_label, tf_label)
    htf_cfg = TIMEFRAMES[htf_label]
    htf_df = fetch_price_data(ticker, htf_cfg["interval"], htf_cfg["period"], htf_cfg["resample"])
    htf_trend = htf_trend_only(htf_df) if htf_df is not None and len(htf_df) > 30 else "Unknown"

    htf_warning = None
    if analysis["signal"] == "BUY" and htf_trend == "Bearish":
        htf_warning = f"⚠️ WARNING: This BUY signal is AGAINST the higher timeframe ({htf_label}) trend, which is Bearish. Trade with reduced size/caution."
    elif analysis["signal"] == "SELL" and htf_trend == "Bullish":
        htf_warning = f"⚠️ WARNING: This SELL signal is AGAINST the higher timeframe ({htf_label}) trend, which is Bullish. Trade with reduced size/caution."

    # -------- Colorful dashboard --------
    d = analysis["decimals"]
    sig = analysis["signal"]
    sig_badge_class = {"BUY": "badge-buy", "SELL": "badge-sell", "NO TRADE": "badge-none"}[sig]
    trend_badge_class = {"Bullish": "badge-bullish", "Bearish": "badge-bearish", "Ranging": "badge-ranging"}[analysis["structure"]]
    value_class = {"BUY": "value-buy", "SELL": "value-sell", "NO TRADE": "value-neutral"}[sig]

    top_l, top_r = st.columns([2, 1])
    with top_l:
        st.markdown(
            f"""
            <div class="card">
              <div class="label">Pair · Timeframe</div>
              <div class="value">{pair_label} &nbsp;·&nbsp; {tf_label}</div>
              <div style="margin-top:10px;">
                <span class="badge {sig_badge_class}">{sig}</span>
                &nbsp;
                <span class="badge {trend_badge_class}">TREND: {analysis['structure'].upper()}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_r:
        conf = analysis["confidence"]
        conf_color = "#16c784" if conf >= 70 else ("#f5a623" if conf >= 50 else "#ea3943")
        st.markdown(
            f"""
            <div class="card">
              <div class="label">Confidence Score</div>
              <div class="value" style="color:{conf_color};">{conf}%</div>
              <div class="conf-wrap" style="margin-top:8px;">
                <div class="conf-bar" style="width:{conf}%;"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    if sig != "NO TRADE":
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        cards = [
            (c1, "Entry", analysis["entry"], "value-neutral"),
            (c2, "Stop Loss", analysis["sl"], "value-sell"),
            (c3, "Take Profit 1", analysis["tp1"], "value-buy"),
            (c4, "Take Profit 2", analysis["tp2"], "value-buy"),
            (c5, "Take Profit 3", analysis["tp3"], "value-buy"),
            (c6, "Risk : Reward", f"1 : {analysis['rr']}" if analysis["rr"] else "n/a", "value-neutral"),
        ]
        for col, label, val, cls in cards:
            with col:
                display_val = f"{val:.{d}f}" if isinstance(val, (int, float)) else val
                st.markdown(
                    f"""<div class="card"><div class="label">{label}</div>
                    <div class="value {cls}">{display_val}</div></div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            """<div class="card"><div class="label">Result</div>
            <div class="value value-neutral">No high-probability confluence found — standing aside is the trade.</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    if htf_warning:
        st.markdown(f'<div class="warn-box">{htf_warning}</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Current Price", f"{analysis['price']:.{d}f}")
    m2.metric("RSI (14)", analysis["rsi"])
    m3.metric(f"Higher TF Trend ({htf_label})", htf_trend)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    sr_l, sr_r = st.columns(2)
    with sr_l:
        st.markdown("**🟢 Support Levels**")
        if analysis["support"]:
            pills = "".join([f'<span class="sr-pill sr-support">{lv:.{d}f}</span>' for lv in analysis["support"]])
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.caption("None clearly defined nearby.")
    with sr_r:
        st.markdown("**🔴 Resistance Levels**")
        if analysis["resistance"]:
            pills = "".join([f'<span class="sr-pill sr-resistance">{lv:.{d}f}</span>' for lv in analysis["resistance"]])
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.caption("None clearly defined nearby.")

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown("**🧠 Trade Reasoning**")
    if analysis["reasons"]:
        for r in analysis["reasons"]:
            st.markdown(f'<div class="reason-item">{r}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="reason-item">No sufficient confluence found for a high-probability setup.</div>', unsafe_allow_html=True)

    st.markdown(
        f"""<div class="warn-box">⚠️ Not financial advice. Always confirm with your own analysis and risk
        management. Never risk more than 1-2% of account equity per trade.
        {"No high-probability confluence was found — forcing a trade here would be low-quality." if sig == "NO TRADE" else ""}</div>""",
        unsafe_allow_html=True,
    )

    with st.expander("📋 Copy raw report (plain text)"):
        d_ = d
        lines = [
            f"PAIR: {pair_label}", f"TIMEFRAME: {tf_label}", "", f"SIGNAL: {sig}", "",
        ]
        if sig != "NO TRADE":
            lines += [
                f"ENTRY: {analysis['entry']:.{d_}f}", f"STOP LOSS: {analysis['sl']:.{d_}f}", "",
                f"TAKE PROFIT 1: {analysis['tp1']:.{d_}f}", f"TAKE PROFIT 2: {analysis['tp2']:.{d_}f}",
                f"TAKE PROFIT 3: {analysis['tp3']:.{d_}f}", "",
                f"RISK:REWARD = 1:{analysis['rr']}" if analysis["rr"] else "RISK:REWARD = n/a",
            ]
        else:
            lines += ["ENTRY: -", "STOP LOSS: -", "", "TAKE PROFIT 1: -", "TAKE PROFIT 2: -", "TAKE PROFIT 3: -", "", "RISK:REWARD = n/a"]
        lines += ["", f"CONFIDENCE: {analysis['confidence']}%", "", f"TREND: {analysis['structure']}", "", "SUPPORT LEVELS:"]
        lines += [f"* {lv:.{d_}f}" for lv in analysis["support"]] if analysis["support"] else ["* None clearly defined nearby"]
        lines += ["", "RESISTANCE LEVELS:"]
        lines += [f"* {lv:.{d_}f}" for lv in analysis["resistance"]] if analysis["resistance"] else ["* None clearly defined nearby"]
        lines += ["", "TRADE REASON:"]
        lines += [f"- {r}" for r in analysis["reasons"]] if analysis["reasons"] else ["- No sufficient confluence found for a high-probability setup."]
        lines += ["", "RISK WARNING:"]
        warn = ["This is not financial advice. Always confirm with your own analysis and risk management. "
                "Never risk more than 1-2% of account equity per trade."]
        if htf_warning:
            warn.append(htf_warning)
        if sig == "NO TRADE":
            warn.append("No high-probability confluence was found — forcing a trade here would be low-quality and is not recommended.")
        lines.append("\n".join(warn))
        st.code("\n".join(lines), language=None)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    render_chart(analysis["df"], analysis, pair_label)

    st.markdown("---")
    render_news_section(pair_label)

    st.markdown("---")
    st.markdown(
        """<p class="disclaimer">Methodology: Trend = EMA20/50/200 alignment. Structure = swing HH/HL vs LH/LL
        (fractal swing detection). Liquidity = equal highs/lows clustering. ICT concepts = simplified Order Block
        (last opposite candle before an impulse move ≥1.4×ATR) and 3-candle Fair Value Gaps. Momentum = RSI(14).
        Confidence score is a weighted confluence score, not a statistical win-rate guarantee.</p>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
