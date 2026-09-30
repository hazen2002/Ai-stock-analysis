import json
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Pro Stock Analysis & Watchlist Platform",
    page_icon="⚡",
    layout="wide",
)

# Keep the default list moderate. Add your long list through the sidebar.
DEFAULT_WATCHLIST = [
    "MSFT.US", "NVDA.US", "AAPL.US", "TSLA.US", "AMZN.US", "META.US",
    "GOOG.US", "AMD.US", "AVGO.US", "QQQ.US", "VOO.US", "TSM.US",
    "00700.HK", "00941.HK", "00005.HK", "03690.HK", "09988.HK",
]
WATCHLIST_PATH = Path(__file__).with_name("watchlist.json")

# Fundamentals change slowly. Prices/options use shorter TTLs.
PRICE_TTL = 60 * 30
FUNDAMENTAL_TTL = 60 * 60 * 12
SCANNER_TTL = 60 * 60
OPTIONS_TTL = 60 * 30


# ============================================================
# Symbol and watchlist helpers
# ============================================================
def to_yfinance_symbol(symbol):
    normalized = str(symbol).strip().upper()
    if normalized.endswith(".US"):
        normalized = normalized[:-3]
    if normalized.endswith(".HK"):
        hk_code = normalized[:-3]
        if hk_code.isdigit() and len(hk_code) == 5:
            normalized = f"{hk_code[1:]}.HK"
    special = {
        ".SKEW": "^SKEW",
        ".SPX": "^GSPC",
        ".RUT": "^RUT",
        "CLMAIN": "CL=F",
        "BRK.B": "BRK-B",
    }
    if normalized in special:
        return special[normalized]
    if normalized.endswith(".ESC"):
        normalized = normalized[:-4]
    return normalized


def load_watchlist():
    try:
        values = json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
        if isinstance(values, list):
            return list(dict.fromkeys(
                str(x).strip().upper() for x in values if str(x).strip()
            ))
    except (OSError, json.JSONDecodeError):
        pass
    return DEFAULT_WATCHLIST.copy()


def save_watchlist(values):
    WATCHLIST_PATH.write_text(
        json.dumps(values, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def is_rate_limit_error(error):
    text = str(error).lower()
    return any(token in text for token in (
        "too many requests", "rate limit", "ratelimit", "429"
    ))


def with_backoff(fn, attempts=4, base_delay=1.5):
    """Retry transient Yahoo failures with capped exponential backoff."""
    last_error = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as error:
            last_error = error
            if not is_rate_limit_error(error) or attempt == attempts - 1:
                raise
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 0.5), 12)
            time.sleep(delay)
    raise last_error


# ============================================================
# Data access layer
# Important: scanner uses ONE batch call, not one Ticker per symbol.
# ============================================================
@st.cache_data(ttl=SCANNER_TTL, show_spinner=False)
def download_scanner_batch(display_symbols_tuple):
    display_symbols = list(display_symbols_tuple)
    yf_symbols = [to_yfinance_symbol(x) for x in display_symbols]
    if not yf_symbols:
        return pd.DataFrame(), {}

    def request():
        return yf.download(
            tickers=yf_symbols,
            period="6mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            actions=False,
            threads=False,
            progress=False,
            timeout=30,
        )

    data = with_backoff(request)
    mapping = dict(zip(display_symbols, yf_symbols))
    return data, mapping


def extract_symbol_history(batch, yf_symbol, symbol_count):
    if batch is None or batch.empty:
        return pd.DataFrame()
    if symbol_count == 1:
        frame = batch.copy()
    elif isinstance(batch.columns, pd.MultiIndex):
        level0 = batch.columns.get_level_values(0)
        level1 = batch.columns.get_level_values(1)
        if yf_symbol in level0:
            frame = batch[yf_symbol].copy()
        elif yf_symbol in level1:
            frame = batch.xs(yf_symbol, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()
    return frame.dropna(how="all")


@st.cache_data(ttl=PRICE_TTL, show_spinner=False)
def fetch_price_history(symbol, period="2y"):
    yf_symbol = to_yfinance_symbol(symbol)

    def request():
        return yf.download(
            tickers=yf_symbol,
            period=period,
            interval="1d",
            auto_adjust=False,
            actions=False,
            threads=False,
            progress=False,
            timeout=30,
        )

    result = with_backoff(request)
    if isinstance(result.columns, pd.MultiIndex):
        # yfinance versions differ for one-symbol downloads.
        if yf_symbol in result.columns.get_level_values(1):
            result = result.xs(yf_symbol, axis=1, level=1)
        elif yf_symbol in result.columns.get_level_values(0):
            result = result[yf_symbol]
    return result.dropna(how="all")


@st.cache_data(ttl=FUNDAMENTAL_TTL, show_spinner=False)
def fetch_fundamentals(symbol):
    """One lazy fundamental request only when the user opens analysis."""
    ticker = yf.Ticker(to_yfinance_symbol(symbol))

    def request():
        # get_info is expensive, so it is isolated and cached for 12 hours.
        info = ticker.get_info() or {}
        return {
            "info": info,
            "balance_sheet": ticker.balance_sheet,
            "financials": ticker.financials,
            "cashflow": ticker.cashflow,
            "quarterly_financials": ticker.quarterly_financials,
            "quarterly_balance_sheet": ticker.quarterly_balance_sheet,
            "quarterly_cashflow": ticker.quarterly_cashflow,
        }

    return with_backoff(request, attempts=3, base_delay=2.0)


@st.cache_data(ttl=PRICE_TTL, show_spinner=False)
def get_vix_value():
    hist = fetch_price_history("^VIX", "5d")
    return float(hist["Close"].dropna().iloc[-1]) if not hist.empty else np.nan


@st.cache_data(ttl=PRICE_TTL, show_spinner=False)
def get_risk_free_rate():
    hist = fetch_price_history("^TNX", "5d")
    if hist.empty:
        return 0.042
    return float(hist["Close"].dropna().iloc[-1]) / 100.0


@st.cache_data(ttl=PRICE_TTL, show_spinner=False)
def get_cnn_fear_and_greed():
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Origin": "https://www.cnn.com",
            "Referer": "https://www.cnn.com/markets/fear-and-greed",
        })
        with urllib.request.urlopen(req, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        item = payload["fear_and_greed"]
        return round(float(item["score"])), str(item["rating"]).title()
    except Exception:
        return None, "Unavailable"


@st.cache_data(ttl=OPTIONS_TTL, show_spinner=False)
def fetch_options_analysis(symbol, current_price):
    try:
        ticker = yf.Ticker(to_yfinance_symbol(symbol))
        expiries = with_backoff(lambda: ticker.options or (), attempts=3)
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        chosen = []
        for expiry in expiries:
            expiry_date = pd.Timestamp(expiry)
            dte = (expiry_date - today).days
            if 0 <= dte <= 45:
                chosen.append((expiry, expiry_date))
        if not chosen:
            return {"available": False, "message": "No option expiry within 45 days."}

        frames = []
        # Cap the number of chains. This materially reduces requests.
        for expiry, expiry_date in chosen[:4]:
            chain = with_backoff(lambda e=expiry: ticker.option_chain(e), attempts=3)
            for option_type, frame in (("Call", chain.calls), ("Put", chain.puts)):
                if frame is not None and not frame.empty:
                    f = frame.copy()
                    f["Expiry"] = expiry_date.strftime("%Y-%m-%d")
                    f["Option Type"] = option_type
                    frames.append(f)
            time.sleep(0.35)

        if not frames:
            return {"available": False, "message": "Option chains are unavailable."}
        options = pd.concat(frames, ignore_index=True)
        for col in ["volume", "openInterest", "impliedVolatility", "lastPrice", "strike"]:
            options[col] = pd.to_numeric(options.get(col, 0), errors="coerce").fillna(0.0)
        options["Vol/OI Ratio"] = np.where(
            options["openInterest"] > 0,
            options["volume"] / options["openInterest"],
            np.nan,
        )
        options["Estimated Trade Value"] = options["volume"] * options["lastPrice"] * 100
        calls = options[options["Option Type"] == "Call"]
        puts = options[options["Option Type"] == "Put"]
        call_vol, put_vol = calls["volume"].sum(), puts["volume"].sum()
        call_oi, put_oi = calls["openInterest"].sum(), puts["openInterest"].sum()

        max_pain = np.nan
        strikes = sorted(options["strike"].unique())
        if strikes:
            pain = {}
            for settlement in strikes:
                pain[settlement] = (
                    ((settlement - calls["strike"]).clip(lower=0) * calls["openInterest"]).sum()
                    + ((puts["strike"] - settlement).clip(lower=0) * puts["openInterest"]).sum()
                )
            max_pain = min(pain, key=pain.get)

        unusual = options[
            (options["volume"] > 2.5 * options["openInterest"])
            & (options["volume"] > 1000)
        ].copy()
        unusual = unusual.sort_values("Estimated Trade Value", ascending=False)
        unusual = unusual.rename(columns={
            "strike": "Strike Price", "volume": "Volume",
            "openInterest": "Open Interest",
            "impliedVolatility": "Implied Volatility",
        })
        cols = ["Strike Price", "Expiry", "Option Type", "Volume", "Open Interest",
                "Vol/OI Ratio", "Implied Volatility", "Estimated Trade Value"]
        return {
            "available": True,
            "expirations": [x[0] for x in chosen[:4]],
            "pcr_volume": put_vol / call_vol if call_vol else np.nan,
            "pcr_oi": put_oi / call_oi if call_oi else np.nan,
            "max_pain": max_pain,
            "unusual": unusual[cols],
        }
    except Exception as error:
        return {"available": False, "message": str(error)}


# ============================================================
# Analytics
# ============================================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_technicals(df):
    d = df.copy().dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    d["MA10"] = d["Close"].rolling(10).mean()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA55"] = d["Close"].rolling(55).mean()
    d["MA250"] = d["Close"].rolling(250).mean()
    d["RSI14"] = calculate_rsi(d["Close"])
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["Hist"] = d["MACD"] - d["Signal"]
    middle = d["Close"].rolling(20).mean()
    std = d["Close"].rolling(20).std()
    d["BB_Middle"] = middle
    d["BB_Upper"] = middle + 2 * std
    d["BB_Lower"] = middle - 2 * std
    return d


def find_dual_divergences(history, lookback_days=90, window=3):
    if history.empty or len(history) < 40:
        return []
    data = add_technicals(history)
    cutoff = data.index.max() - pd.Timedelta(days=lookback_days)
    lows, highs = [], []
    low_values = data["Low"].to_numpy(float)
    high_values = data["High"].to_numpy(float)
    for i in range(window, len(data) - window):
        if data.index[i] < cutoff:
            continue
        if np.isnan(data["RSI14"].iloc[i]):
            continue
        if low_values[i] <= min(low_values[i-window:i]) and low_values[i] <= min(low_values[i+1:i+window+1]):
            lows.append((i, data.index[i], low_values[i], data["RSI14"].iloc[i], data["MACD"].iloc[i]))
        if high_values[i] >= max(high_values[i-window:i]) and high_values[i] >= max(high_values[i+1:i+window+1]):
            highs.append((i, data.index[i], high_values[i], data["RSI14"].iloc[i], data["MACD"].iloc[i]))

    signals = []
    for pivots, bullish in ((lows, True), (highs, False)):
        for first, second in zip(pivots, pivots[1:]):
            _, _, p1, r1, m1 = first
            i2, d2, p2, r2, m2 = second
            price_ok = p2 < p1 if bullish else p2 > p1
            indicator_ok = (r2 > r1 and m2 > m1) if bullish else (r2 < r1 and m2 < m1)
            if price_ok and indicator_ok:
                trigger = float(data["Close"].iloc[i2])
                latest = float(data["Close"].iloc[-1])
                signals.append({
                    "Signal Type": "Bullish" if bullish else "Bearish",
                    "Trigger Date": d2,
                    "Pivot Price": p2,
                    "Trigger Close": trigger,
                    "Latest Price": latest,
                    "Change (%)": (latest / trigger - 1) * 100 if trigger else np.nan,
                    "RSI": r2,
                    "MACD": m2,
                })
    return sorted(signals, key=lambda x: x["Trigger Date"])


def classify_trend(df):
    if len(df) < 55:
        return "Insufficient data"
    row = df.iloc[-1]
    if row["Close"] > row["MA10"] > row["MA20"] > row["MA55"]:
        return "Strong Bull"
    if row["Close"] > row["MA20"] > row["MA55"]:
        return "Weak Bull"
    if row["Close"] < row["MA10"] < row["MA20"] < row["MA55"]:
        return "Strong Bear"
    if row["Close"] < row["MA20"] < row["MA55"]:
        return "Weak Bear"
    return "Consolidation"


def support_resistance(df, window=5):
    if len(df) < 30:
        return float(df["Low"].min()), float(df["High"].max())
    close = float(df["Close"].iloc[-1])
    lows, highs = [], []
    for i in range(window, len(df) - window):
        if df["Low"].iloc[i] <= df["Low"].iloc[i-window:i+window+1].min():
            lows.append(float(df["Low"].iloc[i]))
        if df["High"].iloc[i] >= df["High"].iloc[i-window:i+window+1].max():
            highs.append(float(df["High"].iloc[i]))
    support = max([x for x in lows if x < close], default=float(df["Low"].min()))
    resistance = min([x for x in highs if x > close], default=float(df["High"].max()))
    return support, resistance


def basic_dcf(info, cashflow, balance_sheet, discount_rate):
    shares = float(info.get("sharesOutstanding") or 0)
    if shares <= 0 or cashflow.empty:
        return None
    fcf = None
    if "Free Cash Flow" in cashflow.index:
        vals = pd.to_numeric(cashflow.loc["Free Cash Flow"], errors="coerce").dropna()
        if not vals.empty:
            fcf = float(vals.iloc[0])
    if not fcf or fcf <= 0:
        return None
    growth = info.get("revenueGrowth")
    growth = float(growth) if isinstance(growth, (int, float)) and pd.notna(growth) else 0.08
    growth = min(max(growth, -0.05), 0.20)
    terminal_growth = min(0.03, discount_rate - 0.01)
    projected, pv = [], 0.0
    current = fcf
    for year in range(1, 6):
        faded_growth = growth * (1 - 0.10 * (year - 1))
        current *= 1 + faded_growth
        discounted = current / ((1 + discount_rate) ** year)
        projected.append((year, current, discounted))
        pv += discounted
    terminal = current * (1 + terminal_growth) / (discount_rate - terminal_growth)
    enterprise = pv + terminal / ((1 + discount_rate) ** 5)
    debt = cash = 0.0
    if not balance_sheet.empty:
        latest = balance_sheet.iloc[:, 0]
        debt = float(latest.get("Total Debt", 0) or 0)
        cash = float(latest.get("Cash Cash Equivalents And Short Term Investments", latest.get("Cash And Cash Equivalents", 0)) or 0)
    equity = max(enterprise + cash - debt, 0)
    return {
        "value_per_share": equity / shares,
        "enterprise_value": enterprise,
        "equity_value": equity,
        "growth": growth,
        "terminal_growth": terminal_growth,
        "projections": projected,
    }


# ============================================================
# Sidebar and navigation
# ============================================================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

st.sidebar.title("⭐ Watchlist Manager")
with st.sidebar.form("add_form"):
    new_symbol = st.text_input("Stock ticker").strip().upper()
    add = st.form_submit_button("Add ticker", use_container_width=True)
if add and new_symbol and new_symbol not in st.session_state.watchlist:
    st.session_state.watchlist.append(new_symbol)
    save_watchlist(st.session_state.watchlist)
    st.rerun()

with st.sidebar.expander(f"Saved tickers ({len(st.session_state.watchlist)})"):
    st.dataframe(pd.DataFrame({"Ticker": st.session_state.watchlist}), hide_index=True, height=250)
    if st.session_state.watchlist:
        remove_symbol = st.selectbox("Ticker to remove", st.session_state.watchlist)
        if st.button("Remove selected ticker", use_container_width=True):
            st.session_state.watchlist.remove(remove_symbol)
            save_watchlist(st.session_state.watchlist)
            st.rerun()

view = st.radio("Workspace", ["Scanner Dashboard", "Stock Analysis"], horizontal=True)


# ============================================================
# Scanner: only runs after explicit click and uses one batch call
# ============================================================
if view == "Scanner Dashboard":
    st.title("📡 Watchlist Dual Divergence Scanner")
    st.caption("The scanner is manual and cached. It does not re-query Yahoo on every Streamlit rerun.")

    c1, c2 = st.columns([1, 3])
    run_scan = c1.button("Run scanner", type="primary", use_container_width=True)
    c2.caption("RSI(14) and MACD must confirm at the same price pivot in the last 90 calendar days.")

    if run_scan:
        try:
            with st.spinner("Downloading watchlist in one batch..."):
                batch, mapping = download_scanner_batch(tuple(st.session_state.watchlist))
            rows, unavailable = [], []
            for display_symbol, yf_symbol in mapping.items():
                history = extract_symbol_history(batch, yf_symbol, len(mapping))
                if history.empty:
                    unavailable.append(display_symbol)
                    continue
                for signal in find_dual_divergences(history):
                    rows.append({"Ticker": display_symbol, **signal})
            st.session_state.scan_results = {
                "rows": rows,
                "unavailable": unavailable,
                "scanned": len(mapping),
            }
        except Exception as error:
            if is_rate_limit_error(error):
                st.error("Yahoo rate limit is active. Cached results remain available. Reduce the watchlist or retry after Yahoo lifts the block.")
            else:
                st.error(f"Scanner failed: {error}")

    result = st.session_state.scan_results
    if result is None:
        st.info("Select Run scanner to start. No Yahoo request is sent until you click it.")
    else:
        frame = pd.DataFrame(result["rows"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Bullish alerts", int((frame["Signal Type"] == "Bullish").sum()) if not frame.empty else 0)
        m2.metric("Bearish alerts", int((frame["Signal Type"] == "Bearish").sum()) if not frame.empty else 0)
        m3.metric("Tickers scanned", result["scanned"])
        if result["unavailable"]:
            st.warning("Unavailable: " + ", ".join(result["unavailable"]))
        if frame.empty:
            st.info("No dual-divergence signal found.")
        else:
            shown = frame.copy()
            shown["Trigger Date"] = pd.to_datetime(shown["Trigger Date"]).dt.strftime("%Y-%m-%d")
            st.dataframe(shown.style.format({
                "Pivot Price": "${:.2f}", "Trigger Close": "${:.2f}",
                "Latest Price": "${:.2f}", "Change (%)": "{:+.2f}%",
                "RSI": "{:.1f}", "MACD": "{:.3f}",
            }), use_container_width=True, hide_index=True)


# ============================================================
# Stock analysis: lazy load, cached, and no duplicate requests
# ============================================================
else:
    st.title("⚡ Stock Analysis Platform")
    left, right = st.columns([1, 3])
    with left:
        quick = st.selectbox("Quick switch", ["-- Select --"] + st.session_state.watchlist)
        manual = st.text_input("Ticker", value="NVDA").strip().upper()
        symbol = quick if quick != "-- Select --" else manual
        load = st.button("Load analysis", type="primary", use_container_width=True)
    with right:
        st.info("Data loads only after you click Load analysis. Fundamentals are cached for 12 hours; prices for 30 minutes.")

    if load and symbol:
        st.session_state.analysis_symbol = symbol
    symbol = st.session_state.get("analysis_symbol")

    if not symbol:
        st.stop()

    try:
        with st.spinner(f"Loading {symbol}..."):
            history = fetch_price_history(symbol, "2y")
            fundamentals = fetch_fundamentals(symbol)
        if history.empty:
            st.error(f"No price history returned for {symbol}.")
            st.stop()

        info = fundamentals["info"]
        bs = fundamentals["balance_sheet"]
        fin = fundamentals["financials"]
        cf = fundamentals["cashflow"]
        df = add_technicals(history)
        price = float(df["Close"].dropna().iloc[-1])
        name = info.get("shortName") or info.get("longName") or symbol

        st.subheader(f"{name} ({symbol})")
        cards = st.columns(4)
        cards[0].metric("Current Price", f"${price:,.2f}")
        cards[1].metric("Market Cap", f"${float(info.get('marketCap') or 0)/1e9:,.2f}B")
        pe = info.get("forwardPE") or info.get("trailingPE")
        cards[2].metric("P/E", f"{float(pe):.2f}x" if pe else "N/A")
        target = info.get("targetMeanPrice") or info.get("targetMedianPrice")
        cards[3].metric("Analyst Target", f"${float(target):.2f}" if target else "N/A")

        st.write("---")
        st.subheader("🌐 Market sentiment")
        vix = get_vix_value()
        fear_score, fear_rating = get_cnn_fear_and_greed()
        s1, s2 = st.columns(2)
        s1.metric("VIX", f"{vix:.2f}" if pd.notna(vix) else "N/A")
        s2.metric("CNN Fear & Greed", f"{fear_score} ({fear_rating})" if fear_score is not None else fear_rating)

        st.write("---")
        st.subheader("💰 Five-year DCF cross-check")
        rf = get_risk_free_rate()
        beta = float(info.get("beta") or 1.0)
        discount = min(max(rf + beta * 0.05, 0.06), 0.20)
        dcf = basic_dcf(info, cf, bs, discount)
        if dcf:
            d1, d2, d3 = st.columns(3)
            d1.metric("DCF Value / Share", f"${dcf['value_per_share']:.2f}")
            d2.metric("Discount Rate", f"{discount:.2%}")
            d3.metric("Growth Anchor", f"{dcf['growth']:.2%}")
            projection = pd.DataFrame(dcf["projections"], columns=["Year", "FCF", "PV of FCF"])
            st.dataframe(projection.style.format({"FCF": "${:,.0f}", "PV of FCF": "${:,.0f}"}), hide_index=True, use_container_width=True)
        else:
            st.warning("DCF cannot be calculated because positive FCF or shares outstanding are unavailable.")

        st.write("---")
        st.subheader("🎯 Technical analysis")
        trend = classify_trend(df)
        support, resistance = support_resistance(df)
        rsi = float(df["RSI14"].iloc[-1])
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Trend", trend)
        t2.metric("RSI(14)", f"{rsi:.1f}")
        t3.metric("Support", f"${support:.2f}")
        t4.metric("Resistance", f"${resistance:.2f}")

        chart = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.60, 0.20, 0.20], vertical_spacing=0.025,
            subplot_titles=("Price and Bollinger Bands", "RSI", "MACD"),
        )
        chart.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price"
        ), row=1, col=1)
        for col, color in (("MA10", "#FFD166"), ("MA20", "#00D4FF"), ("MA55", "#7CFF6B"), ("MA250", "#FF5C8A")):
            chart.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=color)), row=1, col=1)
        chart.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper", line=dict(color="#B388FF", width=1)), row=1, col=1)
        chart.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower", line=dict(color="#B388FF", width=1), fill="tonexty", fillcolor="rgba(179,136,255,0.10)"), row=1, col=1)
        chart.add_hline(y=support, line_dash="dash", line_color="green", row=1, col=1)
        chart.add_hline(y=resistance, line_dash="dash", line_color="red", row=1, col=1)
        chart.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14"), row=2, col=1)
        chart.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        chart.add_hline(y=30, line_dash="dash", line_color="blue", row=2, col=1)
        chart.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"), row=3, col=1)
        chart.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal"), row=3, col=1)
        chart.add_trace(go.Bar(x=df.index, y=df["Hist"], name="Histogram"), row=3, col=1)
        chart.update_yaxes(range=[0, 100], row=2, col=1)
        chart.update_layout(height=780, xaxis_rangeslider_visible=False)
        st.plotly_chart(chart, use_container_width=True)

        st.write("---")
        st.subheader("📑 Options analysis")
        if st.button("Load options separately"):
            st.session_state.load_options_for = symbol
        if st.session_state.get("load_options_for") == symbol:
            options = fetch_options_analysis(symbol, price)
            if not options.get("available"):
                st.info(options.get("message", "Options unavailable."))
            else:
                o1, o2, o3 = st.columns(3)
                o1.metric("Put/Call Volume", f"{options['pcr_volume']:.2f}" if pd.notna(options["pcr_volume"]) else "N/A")
                o2.metric("Put/Call OI", f"{options['pcr_oi']:.2f}" if pd.notna(options["pcr_oi"]) else "N/A")
                o3.metric("Max Pain", f"${options['max_pain']:.2f}" if pd.notna(options["max_pain"]) else "N/A")
                unusual = options["unusual"].copy()
                if unusual.empty:
                    st.info("No unusual contracts under the configured rule.")
                else:
                    unusual["Implied Volatility"] *= 100
                    st.dataframe(unusual.style.format({
                        "Strike Price": "${:.2f}", "Volume": "{:,.0f}",
                        "Open Interest": "{:,.0f}", "Vol/OI Ratio": "{:.2f}x",
                        "Implied Volatility": "{:.2f}%",
                        "Estimated Trade Value": "${:,.0f}",
                    }), hide_index=True, use_container_width=True)

    except Exception as error:
        if is_rate_limit_error(error):
            st.error(
                "Yahoo Finance is rate-limiting this IP. The app has stopped further requests. "
                "Use cached data, reduce the watchlist, and avoid repeatedly clearing Streamlit's cache."
            )
        else:
            st.exception(error)

st.caption(
    "Data source: Yahoo Finance through yfinance. This app is an analytical tool, not investment advice. "
    "For production use, connect a licensed market-data provider."
)
