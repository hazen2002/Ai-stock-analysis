import json
import datetime
import urllib.request
from pathlib import Path
import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from ta_utils import calculate_synchronized_macd

# Page Configuration
st.set_page_config(
    page_title="Pro Stock Analysis & Watchlist Platform", page_icon="⚡", layout="wide"
)

DEFAULT_WATCHLIST = """NOK.US
PEP.US
WM.US
PANW.US
AXTI.US
CIEN.US
ATI.US
MCD.US
CRCL.US
RAM.US
.SKEW.US
JEPQ.US
PSI.US
AIS.US
RMBS.US
SPCX.US
CRDO.US
AGX.US
RKLB.US
ROK.US
ENTG.US
IOO.US
FLEX.US
V.US
TMO.US
BTI.US
HPE.US
AMAT.US
WFC.US
FISV.US
ABEV.US
SPYM.US
XYZ.US
ENPH.US
GEV.US
DELL.US
ZETA.US
MUU.US
VSAT.US
STM.US
SNOW.US
VRT.US
PL.US
.SPX.US
EQT.US
VST.US
GLW.US
DRAM.US
NOW.US
BE.US
SOXX.US
TTMI.US
CLS.US
VCX.US
AAOI.US
TEVA.US
LIST2591.US
MRVL.US
CSCO.US
LITE.US
SCCO.US
LIST22865.US
CAT.US
USAR.US
VTI.US
IWO.US
JPM.US
O.US
GILT.US
CVX.US
CLmain.US
SLV.US
VT.US
SYM.US
CVNA.US
P.US
ILMN.US
SNDK.US
EBAY.US
.RUT.US
WDC.US
CLOV.US
RIGL.US
ISRG.US
ONDS.US
DHR.US
SMCI.US
SARO.US
VIXY.US
INFQ.US
MSFT.US
NVO.US
RBRK.US
F.US
DAL.US
MU.US
MS.US
DUK.US
ADBE.US
WOLF.ESC.US
PPL.US
AVGO.US
ASTS.US
CRWD.US
IONQ.US
LEN.US
COST.US
ORCL.US
IREN.US
CRM.US
LLY.US
BWXT.US
RTX.US
INTC.US
JNJ.US
SPMO.US
UPST.US
CORZ.US
RDDT.US
LMND.US
NBIS.US
MP.US
NB.US
AEO.US
AMZN.US
UNH.US
SOFI.US
NRG.US
ARKK.US
JOBY.US
QCOM.US
DLO.US
ASML.US
ARM.US
AMD.US
BBAI.US
NKE.US
META.US
ALAB.US
HOOD.US
DJT.US
BLK.US
LIST20883.US
SPOT.US
APLD.US
QBTS.US
CRWV.US
TQQQ.US
DIS.US
WMT.US
GOOG.US
IAU.US
GLD.US
NFLX.US
BRK.B.US
QQQ.US
PLTR.US
SPYG.US
VOOG.US
VOO.US
TSM.US
PDD.US
NVDA.US
AAPL.US
TSLA.US
FUTU.US
EWJ.US
00005.HK
01888.HK
03121.HK
01810.HK
02259.HK
02382.HK
00522.HK
03454.HK
07709.HK
00992.HK
06869.HK
03277.HK
06288.HK
07747.HK
03317.HK
02513.HK
09880.HK
01768.HK
02050.HK
00100.HK
03986.HK
03696.HK
02693.HK
01024.HK
01347.HK
09888.HK
00981.HK
02432.HK
09866.HK
09868.HK
00116.HK
00939.HK
02631.HK
00288.HK
02577.HK
03448.HK
01788.HK
00002.HK
09618.HK
01929.HK
00941.HK
03750.HK
09992.HK
06181.HK
09988.HK
02097.HK
00700.HK
800000.HK""".splitlines()
WATCHLIST_PATH = Path(__file__).with_name("watchlist.json")
WATCHLIST_REVISION = 2


def to_yfinance_symbol(symbol):
    normalized = symbol.strip().upper()
    if normalized.endswith(".US"):
        normalized = normalized[:-3]
    if normalized.endswith(".HK"):
        hong_kong_code = normalized[:-3]
        if hong_kong_code.isdigit() and len(hong_kong_code) == 5:
            normalized = f"{hong_kong_code[1:]}.HK"
    special_symbols = {
        ".SKEW": "^SKEW",
        ".SPX": "^GSPC",
        ".RUT": "^RUT",
        "CLMAIN": "CL=F",
        "BRK.B": "BRK-B",
    }
    if normalized in special_symbols:
        return special_symbols[normalized]
    if normalized.endswith(".ESC"):
        normalized = normalized[:-4]
    return normalized


def load_watchlist():
    try:
        saved = json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
        if isinstance(saved, list):
            symbols = list(
                dict.fromkeys(
                    value.strip().upper()
                    for value in saved
                    if isinstance(value, str) and value.strip()
                )
            )
            return symbols
    except (OSError, json.JSONDecodeError):
        pass
    return DEFAULT_WATCHLIST.copy()


def save_watchlist(symbols):
    WATCHLIST_PATH.write_text(json.dumps(symbols, indent=2) + "\n", encoding="utf-8")


if st.session_state.get("watchlist_revision") != WATCHLIST_REVISION:
    st.session_state.watchlist = load_watchlist()
    st.session_state.watchlist_revision = WATCHLIST_REVISION
    if not WATCHLIST_PATH.exists():
        save_watchlist(st.session_state.watchlist)

st.sidebar.title("⭐ Watchlist Manager")
with st.sidebar.form("watchlist_add_form"):
    new_symbol = (
        st.text_input("Stock ticker", key="new_watchlist_symbol").strip().upper()
    )
    add_symbol = st.form_submit_button("Add ticker", use_container_width=True)
if add_symbol and new_symbol and new_symbol not in st.session_state.watchlist:
    st.session_state.watchlist.append(new_symbol)
    save_watchlist(st.session_state.watchlist)
    st.rerun()

with st.sidebar.expander(f"Saved tickers ({len(st.session_state.watchlist)})"):
    if st.session_state.watchlist:
        st.dataframe(
            pd.DataFrame({"Ticker": st.session_state.watchlist}),
            height=260,
            hide_index=True,
            use_container_width=True,
        )
        ticker_to_remove = st.selectbox(
            "Ticker to remove", st.session_state.watchlist, key="ticker_to_remove"
        )
        if st.button(
            "Remove selected ticker",
            key="remove_selected_ticker",
            use_container_width=True,
        ):
            st.session_state.watchlist.remove(ticker_to_remove)
            save_watchlist(st.session_state.watchlist)
            st.rerun()
    else:
        st.caption("The watchlist is empty.")

st.sidebar.caption("Only saved tickers are scanned.")


def get_seasonal_roadmap_context(today=None):
    today = today or datetime.date.today()
    if (today.month == 10 and today.day >= 15) or (
        today.month == 11 and today.day <= 6
    ):
        return (
            "warning",
            "Election/FOMC Liquidity Sweep Window: Strict Observation Mode. Do NOT"
            " average down on the left side. Wait for right-side trigger"
            " (Signal_Hist_Turn == True + RSI/CPD Bullish Divergence).",
        )
    if (today.month == 11 and today.day >= 7) or today.month == 12 or (
        today.month == 1 and today.day <= 15
    ):
        return (
            "success",
            "Post-Election & Year-End Rally: V-shaped recovery phase. Hold core"
            " positions. Take 30% profits on high-beta satellites (APLD, NBIS,"
            " BE) around mid-January.",
        )
    if (today.month == 1 and today.day >= 16) or today.month == 2:
        return (
            "info",
            "Feb Seasonal Pullback & Portfolio Rebalancing: Secondary add window"
            " for core quality stocks.",
        )
    if 3 <= today.month <= 7:
        return (
            "success",
            "Spring/Summer Golden Bull Run: Peak seasonal strength. Watch for the"
            " Semiconductor Annual Seasonal Peak around July 18-28.",
        )
    return "warning", "Summer Doldrums / September Effect: Capital preservation mode."


app_view = st.radio(
    "Workspace",
    [
        "Scanner Dashboard",
        "Stock Analysis",
        "💼 Portfolio & Secondary Allocator",
    ],
    horizontal=True,
    key="app_view",
)
selected_from_watchlist = "-- Select --"
if app_view == "Stock Analysis":
    st.title("⚡ Livermore & Lynch 機構級股票分析平台")
    banner_style, banner_message = get_seasonal_roadmap_context()
    getattr(st, banner_style)(banner_message)
    selected_from_watchlist = st.sidebar.selectbox(
        "Quick switch", ["-- Select --"] + st.session_state.watchlist
    )
    col_search, _ = st.columns([1, 2])
    with col_search:
        manual_symbol = (
            st.text_input(
                "輸入股票代碼 (例: NVDA, AAPL, TSLA, 2330.TW):",
                value=st.session_state.get("analysis_symbol", "NVDA"),
                key="manual_symbol_input",
            )
            .strip()
            .upper()
        )
    symbol = (
        selected_from_watchlist
        if selected_from_watchlist != "-- Select --"
        else manual_symbol
    )
else:
    symbol = st.session_state.get("analysis_symbol", "")


# ==========================================
# Caching & Helper Functions
# ==========================================
@st.cache_data(ttl=3600)
def get_cnn_fear_and_greed():
    """Scrape official CNN Fear & Greed Index score and rating"""
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                    " Chrome/140.0 Safari/537.36"
                ),
                "Origin": "https://www.cnn.com",
                "Referer": "https://www.cnn.com/markets/fear-and-greed",
            },
        )
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            score = round(float(data["fear_and_greed"]["score"]))
            rating = str(data["fear_and_greed"]["rating"]).title()
            return score, rating
    except Exception:
        return None, "CNN unavailable"


@st.cache_data(ttl=300)
def get_risk_free_rate():
    """Fetches real-time 10-Year US Treasury Yield (^TNX) as Risk-Free Rate"""
    try:
        tnx = yf.Ticker("^TNX")
        tnx_hist = tnx.history(period="5d")
        if not tnx_hist.empty:
            return float(tnx_hist["Close"].iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.042


@st.cache_data(ttl=300)
def get_vix_value():
    history = yf.Ticker("^VIX").history(period="5d")
    return float(history["Close"].iloc[-1]) if not history.empty else 20.0


@st.cache_data(ttl=3600)
def fetch_ticker_data(symbol_str):
    yf_symbol = to_yfinance_symbol(symbol_str)
    t = yf.Ticker(yf_symbol)

    # 1. Fetch price history
    try:
        hist = t.history(period="2y")
    except Exception:
        hist = pd.DataFrame()

    # 2. Safely retrieve financial statements
    def safe_get(attr_name):
        try:
            val = getattr(t, attr_name)
            return val if isinstance(val, pd.DataFrame) and not val.empty else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    bs = safe_get("balance_sheet")
    fin = safe_get("financials")
    cf = safe_get("cashflow")
    q_fin = safe_get("quarterly_financials")
    q_bs = safe_get("quarterly_balance_sheet")
    q_cf = safe_get("quarterly_cashflow")

    # 3. Fetch primary info dictionary with quick retry
    info = {}
    for _ in range(2):
        try:
            fetched_info = t.info
            if isinstance(fetched_info, dict) and len(fetched_info) > 5:
                info = fetched_info.copy()
                break
        except Exception:
            time.sleep(0.3)

    # 4. Fill price, market cap, and shares from fast_info if t.info was blocked
    curr_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
    try:
        fast_inf = t.fast_info
        if not info.get("currentPrice"):
            info["currentPrice"] = float(getattr(fast_inf, "last_price", curr_price) or curr_price)
        if not info.get("previousClose"):
            info["previousClose"] = float(getattr(fast_inf, "previous_close", curr_price) or curr_price)
        if not info.get("fiftyTwoWeekLow"):
            info["fiftyTwoWeekLow"] = float(getattr(fast_inf, "year_low", curr_price) or curr_price)
        if not info.get("fiftyTwoWeekHigh"):
            info["fiftyTwoWeekHigh"] = float(getattr(fast_inf, "year_high", curr_price) or curr_price)
        if not info.get("marketCap"):
            info["marketCap"] = float(getattr(fast_inf, "market_cap", 0) or 0)
        if not info.get("sharesOutstanding"):
            shares = getattr(fast_inf, "shares", 0)
            if not shares and info.get("marketCap") and info.get("currentPrice"):
                shares = info["marketCap"] / info["currentPrice"]
            info["sharesOutstanding"] = float(shares or 0)
    except Exception:
        pass

    if not info.get("shortName"):
        info["shortName"] = symbol_str

    # 5. Ticker Metadata Fallback for Industry Classification when t.info is blocked
    TICKER_DEFAULTS = {
        "NVDA": {"industry": "Semiconductors", "sector": "Technology"},
        "AAPL": {"industry": "Consumer Electronics", "sector": "Technology"},
        "NKE":  {"industry": "Footwear & Accessories", "sector": "Consumer Cyclical"},
        "TSLA": {"industry": "Auto Manufacturers", "sector": "Consumer Cyclical"},
        "MSFT": {"industry": "Software - Infrastructure", "sector": "Technology"},
        "AMZN": {"industry": "Internet Retail", "sector": "Consumer Cyclical"},
        "GOOG": {"industry": "Internet Content & Information", "sector": "Communication Services"},
        "GOOGL": {"industry": "Internet Content & Information", "sector": "Communication Services"},
        "META": {"industry": "Internet Content & Information", "sector": "Communication Services"},
        "AMD":  {"industry": "Semiconductors", "sector": "Technology"},
        "TSM":  {"industry": "Semiconductors", "sector": "Technology"},
        "AVGO": {"industry": "Semiconductors", "sector": "Technology"},
        "INTC": {"industry": "Semiconductors", "sector": "Technology"},
        "QCOM": {"industry": "Semiconductors", "sector": "Technology"},
        "MU":   {"industry": "Semiconductors", "sector": "Technology"},
        "ARM":  {"industry": "Semiconductors", "sector": "Technology"},
        "SMCI": {"industry": "Computer Hardware", "sector": "Technology"},
        "DELL": {"industry": "Computer Hardware", "sector": "Technology"},
        "PLTR": {"industry": "Software - Infrastructure", "sector": "Technology"},
    }
    clean_sym = symbol_str.replace(".US", "").upper()
    if not info.get("industry") and clean_sym in TICKER_DEFAULTS:
        info["industry"] = TICKER_DEFAULTS[clean_sym]["industry"]
        info["sector"] = TICKER_DEFAULTS[clean_sym]["sector"]

    # 6. Compute missing financial ratios directly from Income Statement & Balance Sheet
    mkt_cap = info.get("marketCap", 0)

    if not fin.empty:
        try:
            latest_fin = fin.iloc[:, 0]

            def get_row(df_col, possible_names):
                for name in possible_names:
                    if name in df_col.index and pd.notna(df_col[name]):
                        return float(df_col[name])
                return 0.0

            revenue = get_row(latest_fin, ["Total Revenue", "Operating Revenue"])
            gross_profit = get_row(latest_fin, ["Gross Profit"])
            op_income = get_row(latest_fin, ["Operating Income"])
            net_income = get_row(latest_fin, ["Net Income", "Net Income Common Stockholders"])

            if revenue > 0:
                info["totalRevenue"] = revenue  # <-- ADD THIS LINE
                if not info.get("grossMargins") and gross_profit > 0:
                    info["grossMargins"] = gross_profit / revenue
                if not info.get("operatingMargins") and op_income != 0:
                    info["operatingMargins"] = op_income / revenue
                if not info.get("profitMargins") and net_income != 0:
                    info["profitMargins"] = net_income / revenue
                if not info.get("priceToSalesTrailing12Months") and mkt_cap > 0:
                    info["priceToSalesTrailing12Months"] = mkt_cap / revenue

            if not info.get("trailingPE") and net_income > 0 and mkt_cap > 0:
                info["trailingPE"] = mkt_cap / net_income

            # Calculate YoY earnings growth for PEG Ratio if missing
            if fin.shape[1] >= 2 and not info.get("earningsGrowth"):
                prev_net_inc = get_row(fin.iloc[:, 1], ["Net Income", "Net Income Common Stockholders"])
                if prev_net_inc > 0 and net_income != 0:
                    info["earningsGrowth"] = (net_income - prev_net_inc) / prev_net_inc
        except Exception:
            pass

    if not bs.empty:
        try:
            latest_bs = bs.iloc[:, 0]

            def get_row_bs(df_col, possible_names):
                for name in possible_names:
                    if name in df_col.index and pd.notna(df_col[name]):
                        return float(df_col[name])
                return 0.0

            equity = get_row_bs(latest_bs, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Total Stockholder Equity"])
            assets = get_row_bs(latest_bs, ["Total Assets"])
            curr_assets = get_row_bs(latest_bs, ["Current Assets"])
            curr_liab = get_row_bs(latest_bs, ["Current Liabilities"])
            inventory = get_row_bs(latest_bs, ["Current Inventory", "Inventory"])

            if not info.get("priceToBook") and equity > 0 and mkt_cap > 0:
                info["priceToBook"] = mkt_cap / equity

            if not info.get("quickRatio") and curr_liab > 0:
                info["quickRatio"] = (curr_assets - inventory) / curr_liab

            if not fin.empty and revenue > 0:
                net_inc_val = get_row(fin.iloc[:, 0], ["Net Income", "Net Income Common Stockholders"])
                if not info.get("returnOnEquity") and equity > 0 and net_inc_val != 0:
                    info["returnOnEquity"] = net_inc_val / equity
                if not info.get("returnOnAssets") and assets > 0 and net_inc_val != 0:
                    info["returnOnAssets"] = net_inc_val / assets
        except Exception:
            pass

    # 7. Compute PEG Ratio dynamically using P/E and calculated earnings growth
    if not info.get("pegRatio") and info.get("trailingPE"):
        earnings_g = info.get("earningsGrowth")
        if isinstance(earnings_g, (int, float)) and earnings_g > 0:
            info["pegRatio"] = info["trailingPE"] / (earnings_g * 100)

    return info, hist, bs, fin, cf, q_fin, q_bs, q_cf


@st.cache_data(ttl=300)
def fetch_scanner_history(symbol_str):
    return yf.Ticker(to_yfinance_symbol(symbol_str)).history(
        period="1y", interval="1d"
    )


@st.cache_data(ttl=900, show_spinner=False)
def fetch_intraday_history(symbol_str):
    return yf.Ticker(to_yfinance_symbol(symbol_str)).history(
        period="60d", interval="1h"
    )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_portfolio_quote(symbol):
    try:
        info = yf.Ticker(to_yfinance_symbol(symbol)).info
        beta = info.get("beta", 1.0)
        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
        beta = float(beta) if beta is not None and np.isfinite(float(beta)) else 1.0
        price = float(price) if price is not None and np.isfinite(float(price)) else 0.0
        return beta, price
    except Exception:
        return 1.0, 0.0


def find_dual_divergences(history, lookback_days=365, window=3):
    """Compare RSI and MACD at matching price pivots within the lookback window.
       Updated lookback_days default to 365 so individual Stock Analysis views
       match and include signals over a full year."""
    required = {"High", "Low", "Close"}
    if history.empty or not required.issubset(history.columns):
        return []

    data = history.copy().dropna(subset=["High", "Low", "Close"]).sort_index()
    data["RSI"] = calculate_rsi(data["Close"], 14)
    data = calculate_synchronized_macd(data)
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
    if getattr(data.index, "tz", None) is not None:
        cutoff = cutoff.tz_localize(data.index.tz)

    lows, highs = [], []
    prices_low = data["Low"].to_numpy(dtype=float)
    prices_high = data["High"].to_numpy(dtype=float)
    for index in range(window, len(data) - window):
        date = data.index[index]
        if date < cutoff:
            continue
        row = data.iloc[index]
        if all(
            prices_low[index] <= prices_low[index - offset]
            for offset in range(1, window + 1)
        ) and all(
            prices_low[index] <= prices_low[index + offset]
            for offset in range(1, window + 1)
        ):
            lows.append(
                (index, date, prices_low[index], float(row["RSI"]), float(row["MACD"]))
            )
        if all(
            prices_high[index] >= prices_high[index - offset]
            for offset in range(1, window + 1)
        ) and all(
            prices_high[index] >= prices_high[index + offset]
            for offset in range(1, window + 1)
        ):
            highs.append(
                (
                    index,
                    date,
                    prices_high[index],
                    float(row["RSI"]),
                    float(row["MACD"]),
                )
            )

    signals = []
    for pivots, bullish in ((lows, True), (highs, False)):
        for first, second in zip(pivots, pivots[1:]):
            index_1, date_1, price_1, rsi_1, macd_1 = first
            index_2, date_2, price_2, rsi_2, macd_2 = second
            price_condition = price_2 < price_1 if bullish else price_2 > price_1
            indicators_condition = (
                (rsi_2 > rsi_1 and macd_2 > macd_1)
                if bullish
                else (rsi_2 < rsi_1 and macd_2 < macd_1)
            )
            if price_condition and indicators_condition:
                trigger_close = float(data["Close"].iloc[index_2])
                latest_price = float(data["Close"].iloc[-1])
                price_change = latest_price - trigger_close
                signals.append({
                    "Signal Type": "Bullish" if bullish else "Bearish",
                    "Trigger Date": date_2,
                    "Price": price_2,
                    "Trigger Close": trigger_close,
                    "Latest Price": latest_price,
                    "Change ($)": price_change,
                    "Change (%)": (
                        (price_change / trigger_close * 100) if trigger_close else np.nan
                    ),
                    "RSI": rsi_2,
                    "MACD": macd_2,
                    "First Date": date_1,
                    "First Price": price_1,
                    "First RSI": rsi_1,
                    "First MACD": macd_1,
                    "Second Price": price_2,
                    "Second RSI": rsi_2,
                    "Second MACD": macd_2,
                })
    return sorted(signals, key=lambda signal: signal["Trigger Date"])


def find_four_indicator_confluences(
    history, lookback_days=90, window=3, max_pivot_gap_days=14
):
    required = {"High", "Low", "Close", "Volume"}
    if history.empty or not required.issubset(history.columns):
        return []

    data, _ = calculate_advanced_indicators_and_signals(history)
    data = data.dropna(subset=["High", "Low", "Close"]).sort_index().copy()
    data["RSI"] = calculate_rsi(data["Close"], 14)
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
    if getattr(data.index, "tz", None) is not None:
        cutoff = cutoff.tz_localize(data.index.tz)

    indicator_columns = {
        "RSI": "RSI",
        "MACD": "MACD",
        "CPD": "CPD",
        "MFI": "MFI14",
    }
    divergences_by_direction = {
        direction: {name: [] for name in indicator_columns}
        for direction in ("Bullish", "Bearish")
    }
    low_prices = data["Low"].to_numpy(dtype=float)
    high_prices = data["High"].to_numpy(dtype=float)
    low_pivots = []
    high_pivots = []

    for index in range(window, len(data) - window):
        if all(
            low_prices[index] <= low_prices[index - offset]
            and low_prices[index] <= low_prices[index + offset]
            for offset in range(1, window + 1)
        ):
            low_pivots.append(index)
        if all(
            high_prices[index] >= high_prices[index - offset]
            and high_prices[index] >= high_prices[index + offset]
            for offset in range(1, window + 1)
        ):
            high_pivots.append(index)

    for indicator_name, column in indicator_columns.items():
        values = data[column].to_numpy(dtype=float)
        for pivots, bullish in ((low_pivots, True), (high_pivots, False)):
            valid_pivots = [index for index in pivots if np.isfinite(values[index])]
            for first, second in zip(valid_pivots, valid_pivots[1:]):
                if second - first > 60 or data.index[second] < cutoff:
                    continue
                first_price = low_prices[first] if bullish else high_prices[first]
                second_price = low_prices[second] if bullish else high_prices[second]
                is_divergence = (
                    second_price < first_price and values[second] > values[first]
                    if bullish
                    else second_price > first_price and values[second] < values[first]
                )
                if is_divergence:
                    direction = "Bullish" if bullish else "Bearish"
                    divergences_by_direction[direction][indicator_name].append({
                        "index": second,
                        "date": data.index[second],
                        "price": float(second_price),
                        "value": float(values[second]),
                    })

    confluences = []
    for direction, by_indicator in divergences_by_direction.items():
        anchor_dates = sorted({
            signal["date"]
            for signals in by_indicator.values()
            for signal in signals
        })
        seen_pivots = set()
        for anchor_date in anchor_dates:
            aligned = {}
            for indicator_name, signals in by_indicator.items():
                recent_signals = [
                    signal
                    for signal in signals
                    if signal["date"] <= anchor_date
                    and (
                        anchor_date.normalize() - signal["date"].normalize()
                    ).days <= max_pivot_gap_days
                ]
                if not recent_signals:
                    break
                aligned[indicator_name] = recent_signals[-1]
            if len(aligned) != len(indicator_columns):
                continue

            pivot_dates = [signal["date"] for signal in aligned.values()]
            pivot_window = (
                max(date.normalize() for date in pivot_dates)
                - min(date.normalize() for date in pivot_dates)
            ).days
            if pivot_window > max_pivot_gap_days:
                continue
            pivot_key = (
                direction,
                tuple(aligned[name]["date"] for name in indicator_columns),
            )
            if pivot_key in seen_pivots:
                continue
            seen_pivots.add(pivot_key)

            trigger = max(aligned.values(), key=lambda signal: signal["date"])
            trigger_index = trigger["index"]
            trigger_close = float(data["Close"].iloc[trigger_index])
            latest_price = float(data["Close"].iloc[-1])
            price_change = latest_price - trigger_close
            confluences.append({
                "Signal Type": direction,
                "Trigger Date": trigger["date"],
                "Price": trigger["price"],
                "Trigger Close": trigger_close,
                "Latest Price": latest_price,
                "Change ($)": price_change,
                "Change (%)": (
                    price_change / trigger_close * 100 if trigger_close else np.nan
                ),
                "RSI": aligned["RSI"]["value"],
                "MACD": aligned["MACD"]["value"],
                "CPD": aligned["CPD"]["value"],
                "MFI": aligned["MFI"]["value"],
                "Pivot Window (days)": pivot_window,
                "Confirmed Indicators": "MACD, RSI, CPD, MFI",
                **{
                    f"{name} Pivot Date": signal["date"]
                    for name, signal in aligned.items()
                },
            })

    return sorted(confluences, key=lambda signal: signal["Trigger Date"])


def calculate_signal_win_rate(history):
    required = {"Close", "High", "Low"}
    if history.empty or not required.issubset(history.columns):
        return 0, 0

    data = calculate_synchronized_macd(history.copy().sort_index())
    signal_indexes = np.flatnonzero(data["Signal_Hist_Turn"].to_numpy())
    highs = data["High"].to_numpy(dtype=float)
    lows = data["Low"].to_numpy(dtype=float)
    closes = data["Close"].to_numpy(dtype=float)
    wins = 0
    evaluated_signals = 0

    for signal_index in signal_indexes:
        if signal_index + 30 >= len(data):
            continue

        entry_price = closes[signal_index]
        stop_price = entry_price * 0.95
        target_price = entry_price * 1.15
        evaluated_signals += 1

        for future_index in range(signal_index + 1, signal_index + 31):
            if lows[future_index] <= stop_price:
                break
            if highs[future_index] >= target_price:
                wins += 1
                break

    return wins, evaluated_signals


def map_symbol_for_tradingview(symbol_input):
    sym = to_yfinance_symbol(symbol_input)
    if sym.endswith(".TW") or sym.endswith(".TWO"):
        code = sym.split(".")[0]
        return f"TPE:{code}"
    elif sym.endswith(".HK"):
        code = sym.split(".")[0].zfill(4)
        return f"HKEX:{code}"
    elif sym.endswith(".SS"):
        code = sym.split(".")[0]
        return f"SSE:{code}"
    elif sym.endswith(".SZ"):
        code = sym.split(".")[0]
        return f"SZSE:{code}"
    elif sym.endswith(".T"):
        code = sym.split(".")[0]
        return f"TSE:{code}"
    return sym.replace(".", "")


def get_granular_industry_benchmarks(industry_str, sector_str):
    ind = (industry_str or "").lower()
    sec = (sector_str or "").lower()

    if "semicon" in ind:
        return "Semiconductors & AI Chips (半導體與AI晶片)", {
            "PE": 35.0,
            "PS": 10.0,
            "PEG": 1.4,
            "GM": 62.0,
            "NM": 28.0,
            "ROE": 25.0,
        }
    elif "software" in ind or "cloud" in ind or "infrastructure" in ind:
        return "Software & Enterprise Cloud (軟體與雲端 SaaS)", {
            "PE": 38.0,
            "PS": 9.0,
            "PEG": 1.6,
            "GM": 72.0,
            "NM": 22.0,
            "ROE": 20.0,
        }
    elif "consumer electronics" in ind or "hardware" in ind:
        return "Consumer Electronics (消費電子/硬體裝置)", {
            "PE": 28.0,
            "PS": 6.5,
            "PEG": 1.8,
            "GM": 44.0,
            "NM": 24.0,
            "ROE": 30.0,
        }
    elif (
        "internet content" in ind
        or "interactive media" in ind
        or "social media" in ind
    ):
        return "Digital Media & Ad Tech (數位媒體與廣告科技)", {
            "PE": 25.0,
            "PS": 7.0,
            "PEG": 1.3,
            "GM": 78.0,
            "NM": 26.0,
            "ROE": 22.0,
        }
    elif "internet retail" in ind or "e-commerce" in ind:
        return "E-Commerce & Digital Retail (電子商務與數位零售)", {
            "PE": 32.0,
            "PS": 2.8,
            "PEG": 1.4,
            "GM": 45.0,
            "NM": 8.0,
            "ROE": 18.0,
        }
    elif "auto" in ind or "ev" in ind:
        return "Electric Vehicles & Auto (電動車與新興汽車)", {
            "PE": 45.0,
            "PS": 5.5,
            "PEG": 1.5,
            "GM": 22.0,
            "NM": 10.0,
            "ROE": 15.0,
        }
    elif "biotech" in ind or "pharmaceutical" in ind:
        return "Biotech & Pharma (生技與製藥)", {
            "PE": 24.0,
            "PS": 5.0,
            "PEG": 1.4,
            "GM": 65.0,
            "NM": 18.0,
            "ROE": 16.0,
        }
    elif "bank" in ind or "financial" in ind:
        return "Banking & Financial Services (銀行與金融服務)", {
            "PE": 12.0,
            "PS": 3.0,
            "PEG": 1.1,
            "GM": 50.0,
            "NM": 22.0,
            "ROE": 12.0,
        }
    elif "energy" in ind or "oil" in ind:
        return "Energy & Clean Tech (能源與潔淨科技)", {
            "PE": 14.0,
            "PS": 1.8,
            "PEG": 1.0,
            "GM": 38.0,
            "NM": 12.0,
            "ROE": 14.0,
        }
    else:
        return f"{sector_str} - General", {
            "PE": 20.0,
            "PS": 2.5,
            "PEG": 1.3,
            "GM": 35.0,
            "NM": 10.0,
            "ROE": 15.0,
        }


def get_industry_growth_profile(industry_str, sector_str):
    ind = (industry_str or "").lower()
    sec = (sector_str or "").lower()

    if "semicon" in ind or "chip" in ind:
        return {"pess": 0.08, "base": 0.14, "opt": 0.22, "terminal": 0.035}
    elif (
        "software" in ind
        or "cloud" in ind
        or "saas" in ind
        or "infrastructure" in ind
    ):
        return {"pess": 0.09, "base": 0.15, "opt": 0.24, "terminal": 0.035}
    elif "consumer electronics" in ind or "hardware" in ind or "device" in ind:
        return {"pess": 0.06, "base": 0.10, "opt": 0.16, "terminal": 0.030}
    elif (
        "internet content" in ind
        or "interactive media" in ind
        or "social media" in ind
        or "advert" in ind
    ):
        return {"pess": 0.08, "base": 0.12, "opt": 0.18, "terminal": 0.030}
    elif "internet retail" in ind or "e-commerce" in ind or "retail" in ind:
        return {"pess": 0.07, "base": 0.11, "opt": 0.17, "terminal": 0.030}
    elif "auto" in ind or "ev" in ind:
        return {"pess": 0.07, "base": 0.12, "opt": 0.18, "terminal": 0.030}
    elif "biotech" in ind or "pharmaceutical" in ind or "health" in ind:
        return {"pess": 0.06, "base": 0.10, "opt": 0.16, "terminal": 0.030}
    elif "bank" in ind or "financial" in ind or "insurance" in ind:
        return {"pess": 0.04, "base": 0.07, "opt": 0.10, "terminal": 0.025}
    elif "energy" in ind or "oil" in ind or "utility" in ind:
        return {"pess": 0.03, "base": 0.05, "opt": 0.09, "terminal": 0.025}
    else:
        return {"pess": 0.06, "base": 0.10, "opt": 0.16, "terminal": 0.030}


def estimate_stock_growth_anchor(info, industry_profile):
    candidates = []
    for key in ["earningsGrowth", "revenueGrowth", "grossMargins", "operatingMargins"]:
        value = info.get(key)
        if isinstance(value, (int, float)) and pd.notna(value):
            if key in ["grossMargins", "operatingMargins"]:
                candidates.append(value / 100.0)
            else:
                candidates.append(float(value))

    if "capacityToGrow" in info:
        maybe = info.get("capacityToGrow")
        if isinstance(maybe, (int, float)) and pd.notna(maybe):
            candidates.append(float(maybe))

    if not candidates:
        return float(industry_profile["base"])

    stock_anchor = float(np.median(candidates))
    lower = float(industry_profile["pess"])
    upper = float(industry_profile["opt"])

    if stock_anchor < 0:
        stock_anchor = lower

    return max(lower, min(stock_anchor, upper))


def estimate_fcf_growth_anchor(fcf_history, industry_profile, fcf_dates=None):
    if fcf_dates and len(fcf_dates) == len(fcf_history):
        dated_fcf = sorted(zip(fcf_dates, fcf_history), key=lambda item: item[0])
        chronological_fcf = [
            float(value) for _, value in dated_fcf if pd.notna(value)
        ]
    else:
        chronological_fcf = [
            float(value) for value in reversed(fcf_history) if pd.notna(value)
        ]

    if len(chronological_fcf) < 4:
        return None, [], 0.0

    recent_fcf = chronological_fcf[-4:]
    if any(value <= 0 for value in recent_fcf):
        return None, [], 0.0

    raw_growth_rates = [
        (current / previous) - 1.0
        for previous, current in zip(recent_fcf, recent_fcf[1:])
        if previous > 0
    ][-3:]
    growth_rates = [max(-0.20, min(rate, 0.60)) for rate in raw_growth_rates]

    if not growth_rates:
        return None, [], 0.0

    if len(growth_rates) >= 2:
        years = np.arange(len(growth_rates), dtype=float)
        slope = float(np.polyfit(years, growth_rates, 1)[0])
    else:
        slope = 0.0

    latest_growth = growth_rates[-1]
    projected_growth = latest_growth + (slope * 0.5)
    projected_growth = max(-0.50, min(projected_growth, 1.00))
    return projected_growth, growth_rates, slope


def estimate_net_income_growth_anchor(financials, info):
    net_income = None
    if not financials.empty:
        for row_name in ["Net Income", "Net Income Common Stockholders"]:
            if row_name in financials.index:
                values = (
                    pd.to_numeric(financials.loc[row_name], errors="coerce")
                    .dropna()
                    .tolist()
                )
                if len(values) >= 2:
                    net_income = [float(value) for value in reversed(values)]
                    break

    growth_rates = []
    if net_income and len(net_income) >= 2:
        for previous, current in zip(net_income[-4:], net_income[-3:]):
            if previous > 0:
                growth_rates.append((current / previous) - 1.0)

    reported_growth = info.get("earningsGrowth")
    if growth_rates:
        latest_growth = growth_rates[-1]
        slope = (
            float(np.polyfit(np.arange(len(growth_rates), dtype=float), growth_rates, 1)[0])
            if len(growth_rates) >= 2
            else 0.0
        )
        return max(-0.50, min(latest_growth + (slope * 0.5), 1.00))
    if isinstance(reported_growth, (int, float)) and pd.notna(reported_growth):
        return max(-0.50, min(float(reported_growth), 1.00))
    return None


def classify_corporate_lifecycle(revenue_growth, fcf_margin, negative_trend=False):
    revenue_growth = float(revenue_growth) if pd.notna(revenue_growth) else 0.0
    fcf_margin = float(fcf_margin) if pd.notna(fcf_margin) else 0.0

    if negative_trend or revenue_growth < 0.05:
        return "Dog", {"fcf": 0.40, "net_income": 0.40, "revenue": 0.20}, -0.12, 0.015
    if revenue_growth > 0.15 and fcf_margin >= 0.10:
        return "Star", {"fcf": 0.20, "net_income": 0.30, "revenue": 0.50}, -0.05, 0.025
    if revenue_growth <= 0.15 and fcf_margin >= 0.10:
        return "Cash Cow", {"fcf": 0.60, "net_income": 0.30, "revenue": 0.10}, -0.02, 0.025
    return "Question Mark", {"fcf": 0.10, "net_income": 0.10, "revenue": 0.80}, 0.00, 0.015


def adjust_growth_weights_for_scale(
    weights, market_cap, net_income_growth, fcf_growth, capex_intensity
):
    market_cap = float(market_cap or 0.0)
    if market_cap >= 200_000_000_000:
        size_label = "Mega Cap (>= $200B)"
        scale_shift = 0.10
    elif market_cap >= 10_000_000_000:
        size_label = "Large Cap ($10B-$200B)"
        scale_shift = 0.06
    elif market_cap >= 2_000_000_000:
        size_label = "Mid Cap ($2B-$10B)"
        scale_shift = 0.02
    else:
        size_label = "Small Cap (< $2B)"
        scale_shift = 0.0

    adjusted = dict(weights)
    earnings_outpace_fcf = (
        net_income_growth is not None
        and fcf_growth is not None
        and float(net_income_growth) > float(fcf_growth) + 0.05
    )
    high_capex = float(capex_intensity or 0.0) >= 0.10

    if scale_shift > 0 and (earnings_outpace_fcf or high_capex):
        shift = scale_shift + (0.05 if earnings_outpace_fcf and high_capex else 0.0)
        shift = min(0.20, shift)
        adjusted["fcf"] = max(0.05, adjusted["fcf"] - shift)
        adjusted["net_income"] += shift * 0.65
        adjusted["revenue"] += shift * 0.35

    total_weight = sum(adjusted.values())
    normalized = {key: value / total_weight for key, value in adjusted.items()}
    return normalized, size_label, earnings_outpace_fcf, high_capex


def get_revenue_growth_anchor(info):
    revenue_growth = info.get("revenueGrowth")
    if isinstance(revenue_growth, (int, float)) and pd.notna(revenue_growth):
        return max(-0.50, min(float(revenue_growth), 1.00))
    return None


def calculate_fcfe_dcf(
    current_stock_price,
    base_revenue,
    revenue_growth_rate,
    target_net_margin,
    cash_flow_conversion_rate,
    forecast_period,
    discount_rate,
    exit_multiple,
    total_shares_outstanding,
    terminal_growth_rate=0.0,
    terminal_value_eligible=True,
):
    revenue = max(float(base_revenue), 0.0)
    growth_rate = max(-0.50, min(float(revenue_growth_rate), 1.00))
    net_margin = max(0.0, min(float(target_net_margin), 1.00))
    conversion_rate = max(0.0, min(float(cash_flow_conversion_rate), 2.00))
    years = max(1, int(forecast_period))
    discount = float(discount_rate)
    if discount > 1.0:
        discount /= 100.0
    discount = max(0.01, min(discount, 0.50))
    shares = max(float(total_shares_outstanding), 0.0)

    yearly_projections = []
    present_value_total = 0.0
    for year in range(1, years + 1):
        revenue *= 1.0 + growth_rate
        net_income = revenue * net_margin
        fcfe = net_income * conversion_rate
        present_value = fcfe / ((1.0 + discount) ** year)
        present_value_total += present_value
        yearly_projections.append({
            "Year": year,
            "Revenue": revenue,
            "Net Income": net_income,
            "FCFE": fcfe,
            "Present Value": present_value,
        })

    final_fcfe = yearly_projections[-1]["FCFE"]
    terminal_growth = max(
        0.0, min(float(terminal_growth_rate), 0.03, discount - 0.01)
    )
    if terminal_value_eligible and final_fcfe > 0 and discount > terminal_growth:
        sustainable_multiple = (1.0 + terminal_growth) / (discount - terminal_growth)
        multiple = min(max(0.0, float(exit_multiple)), sustainable_multiple, 25.0)
        terminal_value = final_fcfe * multiple
    else:
        multiple = 0.0
        terminal_value = 0.0
    present_value_terminal = terminal_value / ((1.0 + discount) ** years)
    total_present_value = present_value_total + present_value_terminal
    value_per_share = total_present_value / shares if shares > 0 else 0.0
    valuation_difference = (
        ((float(current_stock_price) - value_per_share) / value_per_share) * 100
        if value_per_share > 0
        else 0.0
    )
    status = "Overvalued" if valuation_difference >= 0 else "Undervalued"

    return {
        "dcfValuePerShare": value_per_share,
        "valuationStatus": f"{status} by {abs(valuation_difference):.2f}%",
        "yearlyProjections": yearly_projections,
        "terminalValue": terminal_value,
        "presentValueTerminal": present_value_terminal,
        "totalPresentValue": total_present_value,
        "discountRate": discount,
        "exitMultiple": multiple,
        "terminalGrowthRate": terminal_growth,
    }


def _finite_float(value, default=0.0):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default
    return numeric_value if np.isfinite(numeric_value) else default


def calculate_traditional_fcff_dcf(
    current_stock_price,
    base_fcf,
    fcf_history,
    discount_rate,
    net_debt,
    total_shares_outstanding,
    forecast_period=5,
    terminal_growth_rate=0.0225,
):
    base_fcf = max(_finite_float(base_fcf), 0.0)
    shares = max(_finite_float(total_shares_outstanding), 0.0)
    years = max(1, int(forecast_period))
    discount = _finite_float(discount_rate)
    if discount > 1.0:
        discount /= 100.0
    discount = max(0.01, min(discount, 0.50))
    terminal_growth = max(0.0, min(_finite_float(terminal_growth_rate, 0.0225), 0.04))
    if discount <= terminal_growth:
        discount = min(0.50, terminal_growth + 0.01)

    historical_fcf = [
        _finite_float(value, np.nan) for value in (fcf_history or [])
    ]
    historical_growth = [
        current / previous - 1.0
        for previous, current in zip(historical_fcf, historical_fcf[1:])
        if np.isfinite(previous) and np.isfinite(current) and previous > 0 and current > 0
    ]
    initial_growth = (
        float(np.median(historical_growth)) if historical_growth else terminal_growth
    )
    initial_growth = max(-0.10, min(initial_growth, 0.15))

    projections = []
    present_value_total = 0.0
    projected_fcf = base_fcf
    for year in range(1, years + 1):
        fade = (year - 1) / max(years - 1, 1)
        growth = initial_growth + (terminal_growth - initial_growth) * fade
        projected_fcf *= 1.0 + growth
        present_value = projected_fcf / ((1.0 + discount) ** year)
        present_value_total += present_value
        projections.append({
            "Year": year,
            "Growth Rate": growth,
            "FCFF": projected_fcf,
            "Present Value": present_value,
        })

    final_fcf = projections[-1]["FCFF"] if projections else 0.0
    terminal_value = (
        final_fcf * (1.0 + terminal_growth) / (discount - terminal_growth)
        if final_fcf > 0 and discount > terminal_growth
        else 0.0
    )
    present_value_terminal = terminal_value / ((1.0 + discount) ** years)
    enterprise_value = present_value_total + present_value_terminal
    equity_value = max(0.0, enterprise_value - _finite_float(net_debt))
    value_per_share = equity_value / shares if shares > 0 else 0.0
    valuation_difference = (
        ((_finite_float(current_stock_price) - value_per_share) / value_per_share) * 100
        if value_per_share > 0
        else 0.0
    )
    status = "Overvalued" if valuation_difference >= 0 else "Undervalued"
    return {
        "dcfValuePerShare": value_per_share,
        "valuationStatus": f"{status} by {abs(valuation_difference):.2f}%",
        "yearlyProjections": projections,
        "terminalValue": terminal_value,
        "presentValueTerminal": present_value_terminal,
        "totalPresentValue": enterprise_value,
        "equityValue": equity_value,
        "discountRate": discount,
        "terminalGrowthRate": terminal_growth,
        "historicalGrowthRate": initial_growth,
    }


def estimate_roa_history(financials, balance_sheet):
    if financials.empty or balance_sheet.empty:
        return []

    net_income_row = next(
        (
            row
            for row in ["Net Income", "Net Income Common Stockholders"]
            if row in financials.index
        ),
        None,
    )
    assets_row = next(
        (row for row in ["Total Assets"] if row in balance_sheet.index),
        None,
    )
    if net_income_row is None or assets_row is None:
        return []

    balance_sheet_columns = {str(column): column for column in balance_sheet.columns}
    roa_values = []
    for column in financials.columns:
        matching_column = balance_sheet_columns.get(str(column))
        if matching_column is None:
            continue
        net_income = pd.to_numeric(
            financials.at[net_income_row, column], errors="coerce"
        )
        total_assets = pd.to_numeric(
            balance_sheet.at[assets_row, matching_column], errors="coerce"
        )
        if (
            pd.notna(net_income)
            and pd.notna(total_assets)
            and float(total_assets) > 0
        ):
            roa_values.append(float(net_income) / float(total_assets))
    return roa_values


def estimate_industry_roa_benchmark(industry, sector):
    industry_text = (industry or "").lower()
    sector_text = (sector or "").lower()
    if "bank" in industry_text or "bank" in sector_text:
        return 0.012
    if "financial" in industry_text or "financial" in sector_text:
        return 0.015
    if "semiconductor" in industry_text:
        return 0.07
    if "software" in industry_text or "cloud" in industry_text:
        return 0.07
    if "hardware" in industry_text or "electronics" in industry_text:
        return 0.06
    if "utility" in industry_text or "utilities" in sector_text:
        return 0.035
    if any(term in industry_text for term in ["energy", "oil", "gas"]):
        return 0.05
    return 0.05


def calculate_roa_adjusted_dcf(
    current_stock_price,
    base_revenue,
    revenue_growth_rate,
    target_net_margin,
    cash_flow_conversion_rate,
    forecast_period,
    discount_rate,
    exit_multiple,
    total_shares_outstanding,
    terminal_growth_rate,
    terminal_value_eligible,
    roa,
    industry_roa_benchmark,
    historical_roa,
):
    benchmark = max(_finite_float(industry_roa_benchmark, 0.05), 0.005)
    roa_numeric = _finite_float(roa, np.nan)
    valid_roa = roa_numeric if np.isfinite(roa_numeric) else None
    history = [
        value
        for value in (_finite_float(item, np.nan) for item in (historical_roa or []))
        if np.isfinite(value)
    ][-5:]
    if valid_roa is None:
        quality_delta = 0.0
        cash_flow_multiplier = 1.0
        stability_factor = 1.0
    else:
        average_roa = float(np.mean(history)) if history else valid_roa
        roa_volatility = (
            float(np.std(history)) / max(abs(average_roa), benchmark)
            if len(history) >= 2
            else 0.0
        )
        stability_factor = max(0.70, 1.0 - min(roa_volatility, 1.0) * 0.30)
        relative_roa = valid_roa / benchmark
        quality_delta = max(
            -0.02,
            min((relative_roa - 1.0) * 0.015 * stability_factor, 0.02),
        )
        cash_flow_multiplier = max(
            0.85,
            min(1.0 + (relative_roa - 1.0) * 0.15 * stability_factor, 1.15),
        )

    adjusted_discount_rate = max(
        0.01, min(_finite_float(discount_rate, 0.08) - quality_delta, 0.50)
    )
    adjusted_terminal_growth = max(
        0.0,
        min(
            _finite_float(terminal_growth_rate) + quality_delta * 0.25,
            0.03,
            adjusted_discount_rate - 0.01,
        ),
    )
    result = calculate_fcfe_dcf(
        current_stock_price,
        base_revenue,
        revenue_growth_rate,
        target_net_margin,
        cash_flow_conversion_rate * cash_flow_multiplier,
        forecast_period,
        adjusted_discount_rate,
        exit_multiple * cash_flow_multiplier,
        total_shares_outstanding,
        adjusted_terminal_growth,
        terminal_value_eligible,
    )
    result.update({
        "roa": valid_roa,
        "industryRoaBenchmark": benchmark,
        "roaStabilityFactor": stability_factor,
        "cashFlowMultiplier": cash_flow_multiplier,
        "qualityAdjustedDiscountRate": adjusted_discount_rate,
    })
    return result


def classify_trend_status(df):
    if df.empty or len(df) < 55:
        return "資料不足", "無法計算趨勢"

    curr_p = df["Close"].iloc[-1]
    ma10 = df["MA10"].iloc[-1]
    ma20 = df["MA20"].iloc[-1]
    ma55 = df["MA55"].iloc[-1]
    ma20_slope = df["MA20"].iloc[-1] - df["MA20"].iloc[-5] if len(df) >= 5 else 0

    ma250_valid = "MA250" in df.columns and pd.notna(df["MA250"].iloc[-1])
    ma250 = df["MA250"].iloc[-1] if ma250_valid else ma55

    is_perfect_bull = (curr_p > ma10 > ma20 > ma55) and (
        not ma250_valid or ma55 > ma250
    )
    is_perfect_bear = (curr_p < ma10 < ma20 < ma55) and (
        not ma250_valid or ma55 < ma250
    )

    if is_perfect_bull and ma20_slope > 0:
        status = "🚀 強勢多頭 (Strong Bull)"
        desc = (
            f"完美多頭排列 (Price ${curr_p:.2f} > MA10 > MA20 > MA55)，月線向上斜率正向，資金全面控盤。"
        )
    elif curr_p > ma20 and ma20 > ma55 and not is_perfect_bull:
        status = "📈 弱勢多頭 / 回檔整理 (Weak Bull)"
        desc = (
            "中期多頭格局未變 (Price > MA20 > MA55)，但短線 MA10 震盪或股價短踩，屬多頭回檔健康整理。"
        )
    elif is_perfect_bear and ma20_slope < 0:
        status = "🩸 強勢空頭 (Strong Bear)"
        desc = (
            f"完美空頭排列 (Price ${curr_p:.2f} < MA10 < MA20 < MA55)，均線全面下彎且極具下行壓力。"
        )
    elif curr_p < ma20 and ma20 < ma55 and not is_perfect_bear:
        status = "📉 弱勢空頭 / 反彈波 (Weak Bear)"
        desc = (
            "中期趨勢偏空 (Price < MA20 < MA55)，但短線價格接近短期均線，屬空頭架構下的弱勢反彈。"
        )
    else:
        status = "⚖️ 區間震盪整理 (Consolidation)"
        desc = (
            f"均線交錯糾結於 ${ma20:.2f} ~ ${ma55:.2f} 區間，多空動能互相抵消，等待量價突破指引方向。"
        )

    return status, desc


def scan_indicator_divergences(df, indicator_column, window=5, min_gap_days=10):
    if len(df) < (window * 2 + 5) or indicator_column not in df.columns:
        return []

    prices = df["Close"].to_numpy(dtype=float)
    indicator = df[indicator_column].to_numpy(dtype=float)
    dates = df.index
    piv_lows = []
    piv_highs = []
    for index in range(window, len(df) - window):
        if not np.isfinite(indicator[index]):
            continue
        if all(
            prices[index] <= prices[index - offset] for offset in range(1, window + 1)
        ) and all(
            prices[index] <= prices[index + offset] for offset in range(1, window + 1)
        ):
            piv_lows.append((index, dates[index], prices[index], indicator[index]))
        if all(
            prices[index] >= prices[index - offset] for offset in range(1, window + 1)
        ) and all(
            prices[index] >= prices[index + offset] for offset in range(1, window + 1)
        ):
            piv_highs.append((index, dates[index], prices[index], indicator[index]))

    divergences = []
    for pivots, bullish in ((piv_lows, True), (piv_highs, False)):
        for first, second in zip(pivots, pivots[1:]):
            index_1, date_1, price_1, value_1 = first
            index_2, date_2, price_2, value_2 = second
            if index_2 - index_1 > 60:
                continue
            is_divergence = (
                (price_2 < price_1 and value_2 > value_1)
                if bullish
                else (price_2 > price_1 and value_2 < value_1)
            )
            if is_divergence:
                divergences.append({
                    "index": index_2,
                    "date": date_2,
                    "type": (
                        "看多背離 (Bullish)" if bullish else "看空背離 (Bearish)"
                    ),
                    "price": price_2,
                    "indicator": value_2,
                })

    divergences.sort(key=lambda item: item["index"])
    filtered = []
    for divergence in divergences:
        if (
            not filtered
            or divergence["index"] - filtered[-1]["index"] >= min_gap_days
            or divergence["type"] != filtered[-1]["type"]
        ):
            filtered.append(divergence)
        else:
            filtered[-1] = divergence
    return filtered


def find_single_indicator_divergences(history, lookback_days=90):
    required = {"High", "Low", "Close", "Volume"}
    if history.empty or not required.issubset(history.columns):
        return []

    data, _ = calculate_advanced_indicators_and_signals(history)
    data["RSI14"] = calculate_rsi(data["Close"], 14)
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
    if getattr(data.index, "tz", None) is not None:
        cutoff = cutoff.tz_localize(data.index.tz)

    indicator_columns = {
        "RSI": "RSI14",
        "MACD": "MACD",
        "CPD": "CPD",
        "MFI": "MFI14",
    }
    latest_price = float(data["Close"].iloc[-1])
    signals = []
    for indicator_name, column in indicator_columns.items():
        for divergence in scan_indicator_divergences(
            data, column, window=3, min_gap_days=10
        ):
            if divergence["date"] < cutoff:
                continue
            trigger_close = float(data["Close"].iloc[divergence["index"]])
            price_change = latest_price - trigger_close
            direction = (
                "Bullish"
                if "Bullish" in divergence["type"]
                else "Bearish"
            )
            signals.append({
                "Signal Type": direction,
                "Trigger Date": divergence["date"],
                "Price": float(divergence["price"]),
                "Trigger Close": trigger_close,
                "Latest Price": latest_price,
                "Change ($)": price_change,
                "Change (%)": (
                    price_change / trigger_close * 100 if trigger_close else np.nan
                ),
                "Indicator": indicator_name,
                "Indicator Value": float(divergence["indicator"]),
            })
    return sorted(signals, key=lambda signal: signal["Trigger Date"])


def calculate_rsi(series, period=14):
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    average_loss = losses.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(100.0).where(average_gain > 0, 0.0)


def calculate_advanced_indicators_and_signals(data, period=14, window=3):
    required = {"High", "Low", "Close", "Volume"}
    if data.empty or not required.issubset(data.columns):
        return data.copy(), []

    result = data.copy()
    hl_range = result["High"] - result["Low"]
    safe_range = np.where(hl_range == 0, 1e-9, hl_range)
    clv = np.where(
        hl_range == 0,
        0.0,
        (
            (result["Close"] - result["Low"])
            - (result["High"] - result["Close"])
        )
        / safe_range,
    )
    result["Delta"] = result["Volume"] * clv
    result["CPD"] = result["Delta"].cumsum()

    typical_price = (result["High"] + result["Low"] + result["Close"]) / 3.0
    raw_money_flow = typical_price * result["Volume"]
    price_change = typical_price.diff()
    positive_flow = raw_money_flow.where(price_change > 0, 0.0)
    negative_flow = raw_money_flow.where(price_change < 0, 0.0)
    positive_sum = positive_flow.rolling(period, min_periods=period).sum()
    negative_sum = negative_flow.rolling(period, min_periods=period).sum()
    money_ratio = positive_sum / negative_sum.replace(0, np.nan)
    result[f"MFI{period}"] = (100.0 - (100.0 / (1.0 + money_ratio))).where(
        negative_sum > 0, 100.0
    )
    result.loc[(positive_sum == 0) & (negative_sum == 0), f"MFI{period}"] = 50.0
    result = calculate_synchronized_macd(result)
    histogram = result["MACD_Hist"]
    result["Signal_Hist_Exit"] = (
        (histogram > 0)
        & (histogram < histogram.shift(1))
        & (histogram.shift(1) > histogram.shift(2))
    )

    recent_start = max(0, len(result) - 30)
    recent = result.iloc[recent_start:]
    divergences = []
    if len(recent) >= (window * 2 + 3):
        low_pivots = []
        high_pivots = []
        lows = recent["Low"].to_numpy(dtype=float)
        highs = recent["High"].to_numpy(dtype=float)
        for pivot_index in range(window, len(recent) - window):
            if np.isfinite(lows[pivot_index]) and all(
                lows[pivot_index] <= lows[pivot_index + offset]
                and lows[pivot_index] <= lows[pivot_index - offset]
                for offset in range(1, window + 1)
            ):
                low_pivots.append(pivot_index)
            if np.isfinite(highs[pivot_index]) and all(
                highs[pivot_index] >= highs[pivot_index + offset]
                and highs[pivot_index] >= highs[pivot_index - offset]
                for offset in range(1, window + 1)
            ):
                high_pivots.append(pivot_index)

        for indicator in ("CPD", f"MFI{period}"):
            indicator_values = recent[indicator].to_numpy(dtype=float)
            for pivots, bullish in ((low_pivots, True), (high_pivots, False)):
                valid_pivots = [
                    pivot
                    for pivot in pivots
                    if np.isfinite(indicator_values[pivot])
                ]
                if len(valid_pivots) < 2:
                    continue
                first, second = valid_pivots[-2:]
                first_price = lows[first] if bullish else highs[first]
                second_price = lows[second] if bullish else highs[second]
                first_indicator = indicator_values[first]
                second_indicator = indicator_values[second]
                is_divergence = (
                    second_price < first_price and second_indicator > first_indicator
                    if bullish
                    else second_price > first_price
                    and second_indicator < first_indicator
                )
                if is_divergence:
                    absolute_second = recent_start + second
                    divergences.append({
                        "indicator": indicator,
                        "type": "Bullish" if bullish else "Bearish",
                        "first_date": result.index[recent_start + first],
                        "first_price": float(first_price),
                        "first_indicator_value": float(first_indicator),
                        "date": result.index[absolute_second],
                        "price": float(second_price),
                        "indicator_value": float(second_indicator),
                        "delta_at_trigger": float(
                            result["Delta"].iloc[absolute_second]
                        ),
                        "cpd_at_trigger": float(
                            result["CPD"].iloc[absolute_second]
                        ),
                        "mfi_at_trigger": float(
                            result[f"MFI{period}"].iloc[absolute_second]
                        ),
                    })

    return result, divergences


def summarize_bollinger_state(df):
    if df.empty or "BB_Upper" not in df.columns:
        return {
            "squeeze": False,
            "tag": "Neutral",
            "percent_b": np.nan,
            "bandwidth": np.nan,
        }
    latest = df.iloc[-1]
    squeeze = bool(
        latest["BB_Bandwidth"]
        <= df["BB_Bandwidth"].rolling(120, min_periods=20).quantile(0.20).iloc[-1]
    )
    close = float(latest["Close"])
    upper = float(latest["BB_Upper"])
    lower = float(latest["BB_Lower"])
    tolerance = max((upper - lower) * 0.05, close * 0.002)
    if close >= upper - tolerance:
        tag = "Upper Band Tag"
    elif close <= lower + tolerance:
        tag = "Lower Band Tag"
    else:
        tag = "Neutral"
    return {
        "squeeze": squeeze,
        "tag": tag,
        "percent_b": float(latest["BB_%B"]),
        "bandwidth": float(latest["BB_Bandwidth"]),
    }


@st.cache_data(ttl=300)
def fetch_options_analysis(symbol_str, current_price):
    try:
        ticker = yf.Ticker(to_yfinance_symbol(symbol_str))
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        expirations = []
        for expiry in ticker.options or ():
            expiry_date = pd.Timestamp(expiry)
            days_to_expiry = (expiry_date - today).days
            if 0 <= days_to_expiry <= 45:
                expirations.append((expiry, expiry_date))
        if not expirations:
            return {
                "available": False,
                "message": "近 45 天內沒有可用的選擇權到期日。",
            }

        chain_frames = []
        for expiry, expiry_date in expirations:
            chain = ticker.option_chain(expiry)
            for option_type, frame in (("Call", chain.calls), ("Put", chain.puts)):
                if frame is None or frame.empty:
                    continue
                option_frame = frame.copy()
                option_frame["Expiry"] = expiry_date.strftime("%Y-%m-%d")
                option_frame["Option Type"] = option_type
                chain_frames.append(option_frame)
        if not chain_frames:
            return {"available": False, "message": "券商未提供近期期權鏈資料。"}

        options = pd.concat(chain_frames, ignore_index=True)
        for column in [
            "volume",
            "openInterest",
            "impliedVolatility",
            "lastPrice",
            "strike",
        ]:
            if column not in options:
                options[column] = 0.0
            options[column] = pd.to_numeric(
                options[column], errors="coerce"
            ).fillna(0.0)
        options["Vol/OI Ratio"] = np.where(
            options["openInterest"] > 0,
            options["volume"] / options["openInterest"],
            np.inf,
        )
        options["Estimated Trade Value"] = (
            options["volume"] * options["lastPrice"] * 100
        )

        calls = options[options["Option Type"] == "Call"]
        puts = options[options["Option Type"] == "Put"]
        call_volume = calls["volume"].sum()
        put_volume = puts["volume"].sum()
        call_oi = calls["openInterest"].sum()
        put_oi = puts["openInterest"].sum()
        pcr_volume = put_volume / call_volume if call_volume > 0 else np.nan
        pcr_oi = put_oi / call_oi if call_oi > 0 else np.nan

        strikes = sorted(options["strike"].unique())
        max_pain = np.nan
        if strikes:
            pain_by_strike = {}
            for settlement in strikes:
                call_pain = (
                    (settlement - calls["strike"]).clip(lower=0)
                    * calls["openInterest"]
                ).sum()
                put_pain = (
                    (puts["strike"] - settlement).clip(lower=0)
                    * puts["openInterest"]
                ).sum()
                pain_by_strike[settlement] = call_pain + put_pain
            max_pain = min(pain_by_strike, key=pain_by_strike.get)

        atm_distance = (options["strike"] - float(current_price)).abs()
        near_money = options.loc[
            atm_distance <= max(float(current_price) * 0.10, 1.0)
        ]
        call_iv = near_money.loc[
            near_money["Option Type"] == "Call", "impliedVolatility"
        ]
        put_iv = near_money.loc[
            near_money["Option Type"] == "Put", "impliedVolatility"
        ]
        iv_summary = {
            "Call IV": (
                float(call_iv.replace(0, np.nan).median())
                if not call_iv.empty
                else np.nan
            ),
            "Put IV": (
                float(put_iv.replace(0, np.nan).median())
                if not put_iv.empty
                else np.nan
            ),
        }
        iv_summary["Put-Call IV Skew"] = (
            iv_summary["Put IV"] - iv_summary["Call IV"]
            if pd.notna(iv_summary["Put IV"]) and pd.notna(iv_summary["Call IV"])
            else np.nan
        )

        unusual = options[
            (options["volume"] > 2.5 * options["openInterest"])
            & (options["volume"] > 1000)
        ].copy()
        unusual = unusual.sort_values(
            ["Estimated Trade Value", "volume"], ascending=False
        )
        unusual = unusual.rename(
            columns={
                "strike": "Strike Price",
                "impliedVolatility": "Implied Volatility",
                "volume": "Volume",
                "openInterest": "Open Interest",
                "lastPrice": "Last Price",
            }
        )
        display_columns = [
            "Strike Price",
            "Expiry",
            "Option Type",
            "Volume",
            "Open Interest",
            "Vol/OI Ratio",
            "Implied Volatility",
            "Estimated Trade Value",
        ]
        return {
            "available": True,
            "expirations": [expiry for expiry, _ in expirations],
            "pcr_volume": pcr_volume,
            "pcr_oi": pcr_oi,
            "max_pain": max_pain,
            "iv_summary": iv_summary,
            "unusual": unusual[display_columns],
            "call_volume": call_volume,
            "put_volume": put_volume,
        }
    except Exception as error:
        return {"available": False, "message": f"選擇權資料暫時無法取得：{error}"}


def interpret_options_flow(options_summary):
    unusual = options_summary.get("unusual", pd.DataFrame())
    if unusual.empty:
        return "目前沒有符合 Volume > 2.5x OI 且 Volume > 1,000 的異常大單。"
    call_value = unusual.loc[
        unusual["Option Type"] == "Call", "Estimated Trade Value"
    ].sum()
    put_value = unusual.loc[
        unusual["Option Type"] == "Put", "Estimated Trade Value"
    ].sum()
    if call_value > put_value * 1.5:
        return "異常流量以 Call 名目金額為主，顯示智能資金偏向積極做多，但仍需結合 IV 與成交價判斷是否為買方。"
    if put_value > call_value * 1.5:
        return "異常流量以 Put 名目金額為主，較接近下行避險或偏空布局，需觀察 Put IV Skew 是否同步走高。"
    return "Call 與 Put 異常流量相對均衡，可能包含跨式策略或波動率交易，暫不宜解讀為單方向押注。"


def find_smart_support_resistance(df, window=5):
    if df.empty or len(df) < 30:
        fallback_low = df["Low"].min() if not df.empty else 10.0
        fallback_high = df["High"].max() if not df.empty else 100.0
        return fallback_low, fallback_high

    prices_high = df["High"].values
    prices_low = df["Low"].values
    n = len(df)

    swing_highs = []
    swing_lows = []

    for i in range(window, n - window):
        if all(prices_low[i] <= prices_low[i - j] for j in range(1, window + 1)) and all(
            prices_low[i] <= prices_low[i + j] for j in range(1, window + 1)
        ):
            swing_lows.append(prices_low[i])

        if all(
            prices_high[i] >= prices_high[i - j] for j in range(1, window + 1)
        ) and all(
            prices_high[i] >= prices_high[i + j] for j in range(1, window + 1)
        ):
            swing_highs.append(prices_high[i])

    curr_price = df["Close"].iloc[-1]

    valid_supps = [s for s in swing_lows if s < curr_price]
    support = max(valid_supps) if valid_supps else df["Low"].min()

    valid_res = [r for r in swing_highs if r > curr_price]
    resistance = min(valid_res) if valid_res else df["High"].max()

    return support, resistance


def find_volume_poc(df, bins=15):
    if df.empty or len(df) < 30:
        return df["Close"].mean() if not df.empty else 0.0
    hist, bin_edges = np.histogram(df["Close"], bins=bins, weights=df["Volume"])
    max_idx = np.argmax(hist)
    poc_price = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2.0
    return poc_price


def find_volume_value_area(df, bins=15, value_area_fraction=0.70):
    required = {"Close", "Volume"}
    if df.empty or not required.issubset(df.columns):
        return None, None
    prices = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float)
    volumes = pd.to_numeric(df["Volume"], errors="coerce").to_numpy(dtype=float)
    valid = (
        np.isfinite(prices)
        & np.isfinite(volumes)
        & (prices > 0)
        & (volumes > 0)
    )
    if not valid.any():
        return None, None

    price_bins = max(1, min(int(bins), int(valid.sum())))
    profile, edges = np.histogram(
        prices[valid], bins=price_bins, weights=volumes[valid]
    )
    total_volume = float(profile.sum())
    if total_volume <= 0:
        return None, None

    poc_index = int(np.argmax(profile))
    included = {poc_index}
    accumulated_volume = float(profile[poc_index])
    target_volume = total_volume * max(0.0, min(float(value_area_fraction), 1.0))
    lower_index = upper_index = poc_index
    while accumulated_volume < target_volume and (
        lower_index > 0 or upper_index < len(profile) - 1
    ):
        lower_volume = (
            float(profile[lower_index - 1]) if lower_index > 0 else -1.0
        )
        upper_volume = (
            float(profile[upper_index + 1])
            if upper_index < len(profile) - 1
            else -1.0
        )
        if upper_volume > lower_volume:
            upper_index += 1
            accumulated_volume += upper_volume
            included.add(upper_index)
        else:
            lower_index -= 1
            accumulated_volume += lower_volume
            included.add(lower_index)

    return float(edges[min(included)]), float(edges[max(included) + 1])


def calculate_atr_risk_metrics(
    history,
    support,
    resistance,
    current_price,
    atr_period=14,
    atr_multiple=1.5,
    value_area_low=None,
    poc_price=None,
    exhaustion_signal=False,
):
    if history.empty or not {"High", "Low", "Close"}.issubset(history.columns):
        return None
    high = pd.to_numeric(history["High"], errors="coerce")
    low = pd.to_numeric(history["Low"], errors="coerce")
    close = pd.to_numeric(history["Close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(atr_period, min_periods=1).mean()
    atr_value = float(atr.iloc[-1]) if not atr.empty and pd.notna(atr.iloc[-1]) else 0.0
    current_price = float(current_price or 0.0)
    support = float(support or 0.0)
    resistance = float(resistance or 0.0)
    if atr_value <= 0 or current_price <= 0:
        return None

    structural_levels = [
        (name, float(level))
        for name, level in (
            ("Support", support),
            ("VAL", value_area_low),
            ("POC", poc_price),
        )
        if level is not None
        and pd.notna(level)
        and np.isfinite(float(level))
        and 0 < float(level) < current_price
    ]
    structural_source, structural_level = (
        max(structural_levels, key=lambda item: item[1])
        if structural_levels
        else ("ATR fallback", current_price)
    )
    structural_stop = structural_level - atr_value * atr_multiple
    stop_loss = structural_stop
    if stop_loss <= 0 or stop_loss >= current_price:
        stop_loss = current_price - atr_value * atr_multiple
        structural_source = "ATR fallback"
    trailing_multiple = 1.0 if exhaustion_signal else 2.0
    trailing_stop = max(
        stop_loss, current_price - atr_value * trailing_multiple
    )
    risk_per_share = max(current_price - stop_loss, 0.0)
    target_price = resistance if resistance > current_price else None
    reward_per_share = (
        max(target_price - current_price, 0.0) if target_price is not None else 0.0
    )
    risk_reward = (
        reward_per_share / risk_per_share if risk_per_share > 0 else None
    )
    middle = close.ewm(span=20, adjust=False).mean()
    return {
        "atr": atr_value,
        "stop_loss": stop_loss,
        "structural_stop": structural_stop,
        "structural_level": structural_level,
        "structural_source": structural_source,
        "trailing_stop": trailing_stop,
        "trailing_multiple": trailing_multiple,
        "target_price": target_price,
        "exhaustion_signal": bool(exhaustion_signal),
        "risk_reward": risk_reward,
        "channel_middle": middle,
        "channel_upper": middle + (atr * atr_multiple),
        "channel_lower": middle - (atr * atr_multiple),
    }


def find_chart_pattern_setups(
    history,
    support=None,
    resistance=None,
    info=None,
    financials=None,
    min_confidence=45,
):
    required = {"High", "Low", "Close", "Volume"}
    if history.empty or not required.issubset(history.columns):
        return []
    data = (
        history.sort_index()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=list(required))
        .copy()
    )
    if len(data) < 30:
        return []
    high = pd.to_numeric(data["High"], errors="coerce")
    low = pd.to_numeric(data["Low"], errors="coerce")
    close = pd.to_numeric(data["Close"], errors="coerce")
    volume = pd.to_numeric(data["Volume"], errors="coerce")
    last_price = float(close.iloc[-1])
    patterns = []

    def add_pattern(category, name, confidence, status, evidence):
        confidence = max(0, min(100, int(round(confidence))))
        if confidence >= min_confidence:
            patterns.append({
                "類別": category,
                "型態": name,
                "信心分數": confidence,
                "狀態": status,
                "觀察依據": evidence,
            })

    def breakout_status(direction, level):
        if direction == "bullish" and last_price > level:
            return "突破觸發"
        if direction == "bearish" and last_price < level:
            return "突破觸發"
        return "形成中"

    window = 3
    high_values = high.to_numpy(dtype=float)
    low_values = low.to_numpy(dtype=float)
    high_pivots = [
        index
        for index in range(window, len(data) - window)
        if all(
            high_values[index] >= high_values[index - offset]
            and high_values[index] >= high_values[index + offset]
            for offset in range(1, window + 1)
        )
    ]
    low_pivots = [
        index
        for index in range(window, len(data) - window)
        if all(
            low_values[index] <= low_values[index - offset]
            and low_values[index] <= low_values[index + offset]
            for offset in range(1, window + 1)
        )
    ]
    volume_average = volume.rolling(20, min_periods=5).mean()
    volume_ratio = (
        float(volume.iloc[-1] / volume_average.iloc[-1])
        if pd.notna(volume_average.iloc[-1]) and volume_average.iloc[-1] > 0
        else 1.0
    )
    ema20 = close.ewm(span=20, adjust=False).mean()
    prior_high_20 = float(high.iloc[-21:-1].max()) if len(data) > 21 else float(high.max())

    # Livermore pivotal-point retest and fast shakeout/recovery.
    pivot_candidates = [
        float(level)
        for level in (support, resistance)
        if level is not None and pd.notna(level) and np.isfinite(float(level))
    ]
    pivot_candidates.extend(
        float(low_values[index]) for index in low_pivots[-8:]
    )
    if pivot_candidates and len(data) >= 3:
        recent = data.iloc[-3:]
        recovered_pivots = [
            level
            for level in pivot_candidates
            if float(recent["Low"].min()) < level
            and last_price > level
        ]
        if recovered_pivots:
            level = max(recovered_pivots)
            surge_ratio = max(volume_ratio, float(volume.iloc[-3:].max() / max(
                float(volume_average.iloc[-1]) if pd.notna(volume_average.iloc[-1]) else 1.0,
                1.0,
            )))
            confidence = 55 + min(25, max(0, (surge_ratio - 1.0) * 20))
            add_pattern(
                "大師型態",
                "Livermore Pivotal Point / Shakeout",
                confidence,
                "突破觸發",
                f"近 3 根 K 線跌破並收復 ${level:.2f}；量能約 {surge_ratio:.1f}x 均量。",
            )

    # Minervini VCP: successively smaller peak-to-trough contractions and drying volume.
    pullbacks = []
    for peak_position in high_pivots[-8:]:
        trough_positions = [
            position
            for position in low_pivots
            if peak_position < position <= min(peak_position + 20, len(data) - 1)
        ]
        if not trough_positions:
            continue
        trough_position = trough_positions[0]
        peak_value = high_values[peak_position]
        trough_value = low_values[trough_position]
        if peak_value > 0 and trough_value < peak_value:
            pullbacks.append(
                (trough_position, (peak_value - trough_value) / peak_value)
            )
    ordered_pullbacks = [
        magnitude
        for _, magnitude in sorted(pullbacks, key=lambda item: item[0])[-3:]
    ]
    if len(ordered_pullbacks) >= 2:
        contracting = all(
            later < earlier * 0.90
            for earlier, later in zip(ordered_pullbacks, ordered_pullbacks[1:])
        )
        dry_up = (
            float(volume.iloc[-10:].mean()) < float(volume.iloc[-30:-10].mean())
            if len(data) >= 30 and float(volume.iloc[-30:-10].mean()) > 0
            else False
        )
        near_base = last_price >= float(ema20.iloc[-1]) * 0.97
        confidence = (
            15
            + (30 if contracting else 0)
            + (10 if dry_up else 0)
            + (10 if near_base else 0)
        )
        contraction_text = " → ".join(
            f"{value:.0%}" for value in ordered_pullbacks
        )
        add_pattern(
            "大師型態",
            "Minervini Volatility Contraction Pattern (VCP)",
            confidence,
            breakout_status("bullish", prior_high_20),
            f"回撤收斂 {contraction_text}"
            f"；量縮={'是' if dry_up else '否'}；價格位於 EMA20 附近/上方={'是' if near_base else '否'}。",
        )

    # Lynch growth/earnings line: use reported EPS, PEG and stable annual net-income history.
    info = info or {}
    financials = financials if financials is not None else pd.DataFrame()
    income_row = next(
        (
            row
            for row in ("Net Income", "Net Income Common Stockholders")
            if row in financials.index
        ),
        None,
    ) if not financials.empty else None
    income_values = []
    if income_row:
        income_values = [
            float(value)
            for value in pd.to_numeric(
                financials.loc[income_row], errors="coerce"
            ).dropna().tolist()
            if np.isfinite(float(value))
        ]
    income_growth = [
        current / previous - 1.0
        for previous, current in zip(income_values[1:], income_values[:-1])
        if previous > 0 and current > 0
    ]
    earnings_growth = info.get("earningsGrowth")
    if not isinstance(earnings_growth, (int, float, np.number)) or not np.isfinite(float(earnings_growth)):
        earnings_growth = (
            (income_values[0] / income_values[-1]) ** (1 / (len(income_values) - 1)) - 1
            if len(income_values) >= 2 and income_values[-1] > 0 and income_values[0] > 0
            else None
        )
    peg = info.get("pegRatio")
    if not isinstance(peg, (int, float, np.number)) or not np.isfinite(float(peg)) or float(peg) <= 0:
        pe = info.get("trailingPE") or info.get("forwardPE")
        peg = (
            float(pe) / (float(earnings_growth) * 100)
            if isinstance(pe, (int, float, np.number))
            and isinstance(earnings_growth, (int, float, np.number))
            and float(pe) > 0
            and float(earnings_growth) > 0
            else None
        )
    eps = info.get("trailingEps")
    if isinstance(eps, (int, float, np.number)) and np.isfinite(float(eps)) and float(eps) > 0:
        earnings_line = float(eps) * 15
        stable_growth = (
            isinstance(earnings_growth, (int, float, np.number))
            and float(earnings_growth) >= 0.15
            and (
                len(income_growth) < 2
                or (
                    all(growth > 0 for growth in income_growth[-3:])
                    and float(np.std(income_growth[-3:])) <= 0.20
                )
            )
        )
        peg_below_one = isinstance(peg, (int, float, np.number)) and 0 < float(peg) < 1
        confidence = 25 + (35 if stable_growth else 0) + (25 if peg_below_one else 0)
        relative_position = "高於" if last_price >= earnings_line else "低於"
        confidence += 10 if last_price <= earnings_line * 1.10 else 0
        add_pattern(
            "大師型態",
            "Lynch Growth & Earnings Line",
            confidence,
            (
                "突破觸發"
                if stable_growth and peg_below_one and last_price >= earnings_line
                else "形成中"
            ),
            f"EPS×15 參考線 ${earnings_line:.2f}；股價{relative_position}參考線；"
            f"PEG={float(peg):.2f}" if isinstance(peg, (int, float, np.number))
            else f"EPS×15 參考線 ${earnings_line:.2f}；股價{relative_position}參考線；PEG 資料不足。",
        )

    # Head-and-shoulders: five alternating pivots with approximate shoulder symmetry.
    if len(high_pivots) >= 3 and len(low_pivots) >= 2:
        for shoulder_side in ("top", "bottom"):
            candidates = sorted(
                [(i, high_values[i], "H") for i in high_pivots[-12:]]
                + [(i, low_values[i], "L") for i in low_pivots[-12:]],
                key=lambda item: item[0],
            )
            for offset in range(max(0, len(candidates) - 9), max(0, len(candidates) - 4)):
                sequence = candidates[offset:offset + 5]
                expected = ["H", "L", "H", "L", "H"] if shoulder_side == "top" else ["L", "H", "L", "H", "L"]
                if (
                    len(sequence) != 5
                    or [item[2] for item in sequence] != expected
                    or len(data) - sequence[-1][0] > 20
                ):
                    continue
                left, neck1, head, neck2, right = [item[1] for item in sequence]
                shoulders = abs(left - right) / max(abs(left), abs(right), 1e-9)
                neckline = (neck1 + neck2) / 2
                is_shape = (
                    head > max(left, right) * 1.03 and shoulders <= 0.08
                    if shoulder_side == "top"
                    else head < min(left, right) * 0.97 and shoulders <= 0.08
                )
                if is_shape:
                    name = "Head & Shoulders Top" if shoulder_side == "top" else "Inverse Head & Shoulders"
                    direction = "bearish" if shoulder_side == "top" else "bullish"
                    add_pattern(
                        "反轉型態",
                        name,
                        60 + max(0, int((0.08 - shoulders) * 150)),
                        breakout_status(direction, neckline),
                        f"頭肩比例差 {shoulders:.1%}；頸線約 ${neckline:.2f}。",
                    )
                    break

    # Double/triple tops and bottoms from similarly priced local pivots.
    for pivot_list, values, is_top in (
        (high_pivots, high_values, True),
        (low_pivots, low_values, False),
    ):
        recent_pivots = pivot_list[-5:]
        if len(recent_pivots) < 2:
            continue
        for count in (3, 2):
            if len(recent_pivots) < count:
                continue
            group = recent_pivots[-count:]
            if len(data) - group[-1] > 20:
                continue
            prices = [values[position] for position in group]
            dispersion = (
                (max(prices) - min(prices)) / max(abs(float(np.mean(prices))), 1e-9)
            )
            if dispersion > 0.04:
                continue
            if any(
                later - earlier < 5
                for earlier, later in zip(group, group[1:])
            ):
                continue
            if is_top:
                neckline = float(low.iloc[group[0]:group[-1] + 1].min())
                rejection_depth = (
                    (float(np.mean(prices)) - neckline)
                    / max(float(np.mean(prices)), 1e-9)
                )
                name = "Triple Top" if count == 3 else "Double Top"
                direction = "bearish"
                triggered = last_price < neckline
            else:
                neckline = float(high.iloc[group[0]:group[-1] + 1].max())
                rejection_depth = (
                    (neckline - float(np.mean(prices)))
                    / max(float(np.mean(prices)), 1e-9)
                )
                name = "Triple Bottom" if count == 3 else "Double Bottom"
                direction = "bullish"
                triggered = last_price > neckline
            if rejection_depth < 0.03:
                continue
            add_pattern(
                "反轉型態",
                name,
                55 + max(0, int((0.04 - dispersion) * 250)),
                "突破觸發" if triggered else "形成中",
                f"{count} 個波段轉折差異 {dispersion:.1%}；確認位約 ${neckline:.2f}。",
            )
            break

    # Rounding bottom and cup/handle measured on a recent 60-bar window.
    if len(data) >= 60:
        series = close.iloc[-60:].to_numpy(dtype=float)
        start_price = float(series[0])
        end_price = float(series[-1])
        minimum_position = int(np.argmin(series))
        minimum_price = float(series[minimum_position])
        if 15 <= minimum_position <= 44 and min(start_price, end_price) > 0:
            rim_difference = abs(start_price - end_price) / max(start_price, end_price)
            recovery = (end_price - minimum_price) / minimum_price
            if rim_difference <= 0.12 and recovery >= 0.08:
                add_pattern(
                    "反轉型態",
                    "Rounding Bottom",
                    55 + max(0, int((0.12 - rim_difference) * 150)),
                    breakout_status("bullish", max(start_price, end_price)),
                    f"60 根 K 線底部回升 {recovery:.1%}；杯口區約 ${max(start_price, end_price):.2f}。",
                )

        rim = max(float(np.max(series[:12])), float(np.max(series[-18:-8])))
        cup_bottom = float(np.min(series[12:-8]))
        handle_low = float(np.min(series[-8:]))
        cup_depth = (rim - cup_bottom) / rim if rim > 0 else 0.0
        handle_depth = (rim - handle_low) / rim if rim > 0 else 0.0
        if 0.08 <= cup_depth <= 0.35 and 0 < handle_depth < cup_depth * 0.6:
            add_pattern(
                "延續型態",
                "Cup and Handle",
                55 + (15 if end_price >= rim * 0.95 else 0),
                breakout_status("bullish", rim),
                f"杯深 {cup_depth:.1%}；把手回撤 {handle_depth:.1%}；杯口約 ${rim:.2f}。",
            )

    # Bull/bear flags: impulse leg followed by a short, counter-trend consolidation.
    if len(data) >= 30:
        impulse_start = float(close.iloc[-30])
        impulse_end = float(close.iloc[-11])
        consolidation = close.iloc[-10:]
        consolidation_change = (
            float(consolidation.iloc[-1] / consolidation.iloc[0] - 1)
            if consolidation.iloc[0] else 0.0
        )
        consolidation_range = (
            float(consolidation.max() - consolidation.min())
            / max(float(consolidation.mean()), 1e-9)
        )
        impulse_return = impulse_end / impulse_start - 1 if impulse_start else 0.0
        flag_volume_dry = (
            float(volume.iloc[-10:].mean()) < float(volume.iloc[-30:-10].mean())
            if float(volume.iloc[-30:-10].mean()) > 0 else False
        )
        for bullish in (True, False):
            valid_flag = (
                impulse_return >= 0.12 and consolidation_change <= 0.02
                if bullish
                else impulse_return <= -0.12 and consolidation_change >= -0.02
            )
            if valid_flag:
                direction = "bullish" if bullish else "bearish"
                level = float(high.iloc[-11:-1].max()) if bullish else float(low.iloc[-11:-1].min())
                name = "Bull Flag" if bullish else "Bear Flag"
                confidence = 50 + (15 if consolidation_range <= 0.10 else 0) + (10 if flag_volume_dry else 0)
                add_pattern(
                    "延續型態",
                    name,
                    confidence,
                    breakout_status(direction, level),
                    f"旗桿漲跌 {impulse_return:.1%}；整理區間 {consolidation_range:.1%}；量縮={'是' if flag_volume_dry else '否'}。",
                )

    # Rectangular range and triangular convergence from linear slopes of recent highs/lows.
    range_window = min(30, len(data))
    range_high = high.iloc[-range_window:]
    range_low = low.iloc[-range_window:]
    range_mid = float((range_high.max() + range_low.min()) / 2)
    range_width = (
        float(range_high.max() - range_low.min()) / range_mid
        if range_mid > 0 else 1.0
    )
    if range_width <= 0.12:
        triggered = last_price > float(range_high.iloc[:-1].max()) or last_price < float(range_low.iloc[:-1].min())
        add_pattern(
            "延續型態",
            "Rectangle / Darvas Box",
            50 + max(0, int((0.12 - range_width) * 200)),
            "突破觸發" if triggered else "形成中",
            f"近 {range_window} 根 K 線箱體寬度 {range_width:.1%}；上沿 ${range_high.max():.2f}、下沿 ${range_low.min():.2f}。",
        )

    if range_window >= 20:
        x = np.arange(range_window, dtype=float)
        high_slope = float(np.polyfit(x, range_high.to_numpy(dtype=float), 1)[0])
        low_slope = float(np.polyfit(x, range_low.to_numpy(dtype=float), 1)[0])
        price_scale = max(last_price, 1e-9)
        high_trend = high_slope * range_window / price_scale
        low_trend = low_slope * range_window / price_scale
        triangle_specs = [
            ("Ascending Triangle", high_trend <= 0.025 and low_trend >= 0.025, "bullish", float(range_high.max())),
            ("Descending Triangle", high_trend <= -0.025 and low_trend >= -0.025, "bearish", float(range_low.min())),
            ("Symmetrical Triangle", high_trend <= -0.025 and low_trend >= 0.025, "bullish", float(range_high.max())),
        ]
        for name, converging, direction, level in triangle_specs:
            if converging:
                slope_score = min(20, int((abs(high_trend) + abs(low_trend)) * 150))
                add_pattern(
                    "雙向 / 三角型態",
                    name,
                    50 + slope_score,
                    breakout_status(direction, level),
                    f"近 {range_window} 根高低點趨勢收斂；上沿趨勢 {high_trend:.1%}、下沿趨勢 {low_trend:.1%}。",
                )

    return sorted(
        patterns,
        key=lambda item: (
            item["類別"],
            -item["信心分數"],
            item["型態"],
        ),
    )


def calculate_timeframe_confluence(daily_history, hourly_history):
    def timeframe_reading(frame, label):
        required = {"High", "Low", "Close"}
        if frame.empty or not required.issubset(frame.columns) or len(frame) < 20:
            return {
                "時間週期": label,
                "趨勢": "資料不足",
                "動能": "資料不足",
                "確認分數": 0,
            }
        data = frame.copy().sort_index()
        close = pd.to_numeric(data["Close"], errors="coerce")
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        macd = calculate_synchronized_macd(data)
        histogram = macd["MACD_Hist"]
        if pd.isna(close.iloc[-1]) or pd.isna(ema50.iloc[-1]):
            return {
                "時間週期": label,
                "趨勢": "資料不足",
                "動能": "資料不足",
                "確認分數": 0,
            }

        bullish_trend = close.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1]
        bearish_trend = close.iloc[-1] < ema20.iloc[-1] < ema50.iloc[-1]
        trend_score = 1 if bullish_trend else -1 if bearish_trend else 0
        current_hist = histogram.iloc[-1]
        prior_hist = histogram.iloc[-2] if len(histogram) >= 2 else np.nan
        bullish_momentum = (
            pd.notna(current_hist)
            and pd.notna(prior_hist)
            and current_hist > 0
            and current_hist > prior_hist
        )
        bearish_momentum = (
            pd.notna(current_hist)
            and pd.notna(prior_hist)
            and current_hist < 0
            and current_hist < prior_hist
        )
        momentum_score = (
            1 if bullish_momentum else -1 if bearish_momentum else 0
        )
        status_score = trend_score + momentum_score
        return {
            "時間週期": label,
            "趨勢": (
                "多頭" if trend_score > 0 else "空頭" if trend_score < 0 else "中性"
            ),
            "動能": (
                "增強" if momentum_score > 0 else "轉弱" if momentum_score < 0 else "中性"
            ),
            "確認分數": status_score,
        }

    hourly = (
        hourly_history.sort_index()
        if not hourly_history.empty
        else pd.DataFrame()
    )
    four_hour = pd.DataFrame()
    if not hourly.empty and {"Open", "High", "Low", "Close", "Volume"}.issubset(
        hourly.columns
    ):
        four_hour = hourly.resample("4h").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna(subset=["Open", "High", "Low", "Close"])

    readings = [
        timeframe_reading(hourly, "1H"),
        timeframe_reading(four_hour, "4H（1H 重採樣）"),
        timeframe_reading(daily_history, "日線"),
    ]
    valid_scores = [
        reading["確認分數"]
        for reading in readings
        if reading["趨勢"] != "資料不足"
    ]
    aggregate = sum(valid_scores)
    if not valid_scores:
        label = "資料不足"
    elif aggregate >= 3:
        label = "多週期偏多"
    elif aggregate <= -3:
        label = "多週期偏空"
    else:
        label = "訊號混合"
    return pd.DataFrame(readings), aggregate, label


def find_liquidity_sweeps(history, lookback=20, atr_period=14, max_bars=120):
    required = {"High", "Low", "Close"}
    if history.empty or not required.issubset(history.columns):
        return []
    data = history.copy().sort_index()
    high = pd.to_numeric(data["High"], errors="coerce")
    low = pd.to_numeric(data["Low"], errors="coerce")
    close = pd.to_numeric(data["Close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(atr_period, min_periods=1).mean()
    prior_high = high.rolling(lookback, min_periods=lookback).max().shift(1)
    prior_low = low.rolling(lookback, min_periods=lookback).min().shift(1)

    signals = (
        data["Signal_Hist_Turn"].fillna(False).astype(bool)
        if "Signal_Hist_Turn" in data.columns
        else calculate_synchronized_macd(data)["Signal_Hist_Turn"].fillna(False).astype(bool)
    )
    signal_positions = np.flatnonzero(signals.to_numpy())
    start = max(lookback, len(data) - max_bars)
    events = []
    for position in range(start, len(data)):
        if (
            pd.isna(prior_high.iloc[position])
            or pd.isna(prior_low.iloc[position])
            or pd.isna(atr.iloc[position])
            or atr.iloc[position] <= 0
        ):
            continue
        upper_level = float(prior_high.iloc[position])
        lower_level = float(prior_low.iloc[position])
        bar_high = float(high.iloc[position])
        bar_low = float(low.iloc[position])
        bar_close = float(close.iloc[position])
        threshold = float(atr.iloc[position]) * 0.10
        direction = None
        swept_level = None
        if bar_high > upper_level + threshold and bar_close < upper_level:
            direction, swept_level = "Bearish high sweep", upper_level
        elif bar_low < lower_level - threshold and bar_close > lower_level:
            direction, swept_level = "Bullish low sweep", lower_level
        if direction is None:
            continue

        nearby_signals = [
            int(signal_position)
            for signal_position in signal_positions
            if abs(int(signal_position) - position) <= 2
        ]
        matched_position = (
            min(nearby_signals, key=lambda item: abs(item - position))
            if nearby_signals
            else None
        )
        events.append({
            "日期": data.index[position],
            "掃流方向": direction,
            "被掃關鍵位": swept_level,
            "收盤價": bar_close,
            "回收幅度（ATR）": (
                (bar_high - upper_level) / atr.iloc[position]
                if direction == "Bearish high sweep"
                else (lower_level - bar_low) / atr.iloc[position]
            ),
            "Signal_Hist_Turn 配對": matched_position is not None,
            "配對觸發日期": (
                data.index[matched_position] if matched_position is not None else None
            ),
        })
    return events


def generate_technical_narrative(
    symbol,
    curr_price,
    ma10,
    ma20,
    ma55,
    ma250,
    gunshot,
    div_type,
    support,
    resistance,
    fib_618,
    poc_price,
    trend_status,
    trend_desc,
    ma250_valid=True,
    rsi_value=None,
    rsi_div_type="Neutral",
    bollinger_state=None,
    macd_divergence=False,
):
    narrative = []

    narrative.append(
        f"**【均線型態與趨勢等級】**\n當前標的經多因子模型分類為：**{trend_status}**。\n{trend_desc}"
    )

    if div_type == "看多背離 (Bullish)":
        div_desc = (
            "指標面上，近期觸發「看多背離」訊號。雖然價格波段探低，但下行賣壓顯著減弱，機構資金有暗中承接跡象。"
        )
    elif div_type == "看空背離 (Bearish)":
        div_desc = (
            "指標面上，近期出現「看空背離」預警。股價攀高過程中動能未同步放大，提防高檔誘多後的回檔修正。"
        )
    else:
        div_desc = "指標面上，MACD 與價格同步運行，未見明顯動能背離。"
    rsi_value = (
        float(rsi_value)
        if rsi_value is not None and pd.notna(rsi_value)
        else np.nan
    )
    bollinger_state = bollinger_state or {
        "squeeze": False,
        "tag": "Neutral",
        "percent_b": np.nan,
    }
    rsi_state = "超買" if rsi_value > 70 else "超賣" if rsi_value < 30 else "中性"
    rsi_desc = f"RSI(14) 為 {rsi_value:.1f}（{rsi_state}）"
    if rsi_div_type != "Neutral":
        rsi_desc += f"，並出現 {rsi_div_type}。"
    else:
        rsi_desc += "，尚未發現 RSI 背離。"
    narrative.append(f"**【指標動能與背離分析】**\n{div_desc}\n{rsi_desc}")

    dist_to_supp = (
        ((curr_price - support) / curr_price) * 100 if curr_price > 0 else 0
    )
    dist_to_res = (
        ((resistance - curr_price) / curr_price) * 100 if curr_price > 0 else 0
    )
    of_desc = (
        f"波段關鍵支撐位落在 **${support:.2f}** (距今 {dist_to_supp:.1f}%)，關鍵壓力位在"
        f" **${resistance:.2f}** (距今 {dist_to_res:.1f}%)。"
    )
    of_desc += f" 籌碼最大密集區 (POC) 落在 **${poc_price:.2f}**。"
    if curr_price > 0 and abs(curr_price - fib_618) / curr_price < 0.02:
        of_desc += (
            " 值得注意，當前股價接近斐波那契黃金分割支撐 61.8%"
            f" (**${fib_618:.2f}**)，具備機構買盤防守力道。"
        )
    if bollinger_state.get("tag") != "Neutral":
        of_desc += (
            f" Bollinger {bollinger_state['tag']}，%B="
            f"{bollinger_state.get('percent_b', np.nan):.2f}。"
        )
    if bollinger_state.get("squeeze"):
        of_desc += (
            " Bollinger Bandwidth 位於近期低分位，屬波動壓縮狀態，需留意突破。"
        )
    narrative.append(f"**【關鍵關卡與訂單流佈局】**\n{of_desc}")

    confluence_entry = (
        bollinger_state.get("tag") == "Lower Band Tag"
        or (curr_price > 0 and abs(curr_price - fib_618) / curr_price < 0.02)
    ) and (rsi_div_type == "看多背離 (Bullish)" or div_type == "看多背離 (Bullish)")
    if confluence_entry:
        decision = (
            "🎯 **機構策略**：價格位於 Bollinger 下軌或 61.8%"
            " 支撐附近，且 RSI/MACD"
            " 出現看多背離，形成高信念反轉共振，可採分批建倉並以支撐失守控管風險。"
        )
    elif gunshot and (div_type != "看空背離 (Bearish)"):
        decision = (
            "🎯 **機構策略**：符合 Livermore"
            " 第一槍爆發型態（爆量+突破），且無空頭背離，可採取順勢突破建倉策略，將停損設於突破"
            " K 棒低點。"
        )
    elif div_type == "看多背離 (Bullish)" and curr_price <= support * 1.05:
        decision = (
            "🎯 **機構策略**：符合弱勢左側抄底買點（看多背離 +"
            " 近關鍵支撐區），風險報酬比優良，適合分批佈局。"
        )
    elif curr_price >= resistance * 0.98:
        decision = (
            "🎯"
            " **機構策略**：股價逼近前高壓力區，追高風險偏高，建議等待爆量突破後回踩不破再行加碼。"
        )
    else:
        decision = (
            "🎯"
            " **機構策略**：當前處於區間震盪整理，建議維持觀望，或於黃金分割支撐位附近佈局。"
        )
    narrative.append(f"**【綜合實戰決策總結】**\n{decision}")

    return "\n\n".join(narrative)


def open_scanner_signal():
    selected_signal = st.session_state.get("scanner_signal_choice")
    signal = st.session_state.get("scanner_signal_options", {}).get(selected_signal)
    if signal:
        st.session_state.analysis_symbol = signal["Ticker"]
        st.session_state.manual_symbol_input = signal["Ticker"]
        st.session_state.selected_scanner_signal = signal
        st.session_state.app_view = "Stock Analysis"


if app_view == "Scanner Dashboard":
    st.title("📡 Watchlist Divergence Scanner")
    st.caption(
        "Scan saved tickers for confirmed daily divergences. Four-indicator"
        " confluence aligns MACD, RSI, CPD, and MFI divergence pivots within"
        " 14 calendar days."
    )
    scanner_mode = st.radio(
        "Scanner mode",
        [
            "Dual RSI + MACD",
            "Individual indicator alerts",
            "4-indicator confluence",
        ],
        horizontal=True,
        key="scanner_mode",
    )
    scan_rows = []
    individual_rows = []
    confluence_rows = []
    scan_errors = []
    total_win_signals = 0
    total_triggered_signals = 0
    total_tickers = len(st.session_state.watchlist)
    scan_progress = (
        st.progress(0, text="Scanning saved watchlist...")
        if total_tickers
        else None
    )
    for ticker_index, ticker_symbol in enumerate(
        st.session_state.watchlist, start=1
    ):
        try:
            ticker_history = fetch_scanner_history(ticker_symbol)
            ticker_wins, ticker_signals = calculate_signal_win_rate(ticker_history)
            total_win_signals += ticker_wins
            total_triggered_signals += ticker_signals
            signal_win_rate = (
                f"{ticker_wins / ticker_signals * 100:.1f}%"
                if ticker_signals
                else "N/A"
            )
            for signal in find_dual_divergences(ticker_history):
                scan_rows.append({
                    "Ticker": ticker_symbol,
                    **signal,
                    "Signal Win Rate (%)": signal_win_rate,
                })
            for signal in find_single_indicator_divergences(ticker_history):
                individual_rows.append({
                    "Ticker": ticker_symbol,
                    **signal,
                    "Signal Win Rate (%)": signal_win_rate,
                })
            for signal in find_four_indicator_confluences(ticker_history):
                confluence_rows.append({
                    "Ticker": ticker_symbol,
                    **signal,
                    "Signal Win Rate (%)": signal_win_rate,
                })
        except Exception as error:
            scan_errors.append(f"{ticker_symbol}: {error}")
        if scan_progress:
            scan_progress.progress(
                ticker_index / total_tickers,
                text=f"Scanning {ticker_symbol} ({ticker_index}/{total_tickers})",
            )
    if scan_progress:
        scan_progress.empty()

    confluence_frame = pd.DataFrame(confluence_rows)
    scan_frame = pd.DataFrame(
        confluence_rows
        if scanner_mode == "4-indicator confluence"
        else individual_rows
        if scanner_mode == "Individual indicator alerts"
        else scan_rows
    )
    bullish_count = (
        int((scan_frame["Signal Type"] == "Bullish").sum())
        if not scan_frame.empty
        else 0
    )
    bearish_count = (
        int((scan_frame["Signal Type"] == "Bearish").sum())
        if not scan_frame.empty
        else 0
    )
    metric_bull, metric_bear, metric_confluence, metric_total = st.columns(4)
    metric_bull.metric("🚀 Bullish alerts", bullish_count)
    metric_bear.metric("🚨 Bearish alerts", bearish_count)
    metric_confluence.metric("4-indicator confluence alerts", len(confluence_frame))
    metric_total.metric("Watchlist tickers", len(st.session_state.watchlist))
    total_system_win_rate = (
        f"{total_win_signals / total_triggered_signals * 100:.1f}%"
        if total_triggered_signals
        else "N/A"
    )
    st.metric(
        "Total Signal Detector Win Rate (30D | +15% Target | -5% Stop)",
        total_system_win_rate,
    )
    if scan_errors:
        with st.expander(f"Unavailable symbols ({len(scan_errors)})"):
            st.caption(
                "These tickers could not be loaded from Yahoo Finance in this scan."
            )
            st.write("\n".join(scan_errors))

    if scan_frame.empty:
        if scanner_mode == "4-indicator confluence":
            st.info(
                "No MACD, RSI, CPD, and MFI confluence alerts with pivots aligned"
                " within 14 calendar days were found in the saved watchlist during"
                " the last 90 days."
            )
        elif scanner_mode == "Individual indicator alerts":
            st.info(
                "No RSI, MACD, CPD, or MFI divergence alerts were found in the"
                " saved watchlist during the last 90 days."
            )
        else:
            st.info(
                "No same-day dual divergence alerts were found in the saved"
                " watchlist during the last 90 days."
            )
    else:
        st.caption(
            "Price is the divergence pivot; performance is measured from the"
            " trigger-day close to the latest available daily price."
        )
        if scanner_mode == "Individual indicator alerts":
            display_columns = [
                "Ticker",
                "Indicator",
                "Signal Type",
                "Trigger Date",
                "Price",
                "Indicator Value",
                "Trigger Close",
                "Latest Price",
                "Change (%)",
                "Signal Win Rate (%)",
            ]
        else:
            display_columns = [
                "Ticker",
                "Signal Type",
                "Trigger Date",
                "Price",
                "Trigger Close",
                "Latest Price",
                "Change ($)",
                "Change (%)",
                "RSI",
                "MACD",
                "Signal Win Rate (%)",
            ]
        if scanner_mode == "4-indicator confluence":
            display_columns.extend([
                "CPD",
                "MFI",
                "RSI Pivot Date",
                "MACD Pivot Date",
                "CPD Pivot Date",
                "MFI Pivot Date",
                "Pivot Window (days)",
                "Confirmed Indicators",
            ])
        display_frame = scan_frame[display_columns].copy()
        display_frame["Trigger Date"] = display_frame["Trigger Date"].map(
            lambda trigger_date: pd.Timestamp(trigger_date).strftime("%Y-%m-%d")
        )
        if scanner_mode == "4-indicator confluence":
            pivot_date_columns = [
                "RSI Pivot Date",
                "MACD Pivot Date",
                "CPD Pivot Date",
                "MFI Pivot Date",
            ]
            for column in pivot_date_columns:
                display_frame[column] = display_frame[column].map(
                    lambda pivot_date: pd.Timestamp(pivot_date).strftime("%Y-%m-%d")
                )
        st.dataframe(
            display_frame.style.format({
                "Price": "${:.2f}",
                "Trigger Close": "${:.2f}",
                "Latest Price": "${:.2f}",
                "Change ($)": "{:+.2f}",
                "Change (%)": "{:+.2f}%",
                "RSI": "{:.1f}",
                "MACD": "{:.3f}",
                "Indicator Value": "{:,.2f}",
                **(
                    {
                        "CPD": "{:,.0f}",
                        "MFI": "{:.1f}",
                        "Pivot Window (days)": "{:.0f}",
                    }
                    if scanner_mode == "4-indicator confluence"
                    else {}
                ),
            }),
            use_container_width=True,
            hide_index=True,
        )

        signal_options = {}
        for _, signal_row in scan_frame.iterrows():
            trigger_date = pd.Timestamp(signal_row["Trigger Date"]).strftime(
                "%Y-%m-%d"
            )
            label = (
                f"{signal_row['Ticker']} | {signal_row['Signal Type']} |"
                f" {trigger_date}"
            )
            if label in signal_options:
                label += f" | ${signal_row['Price']:.2f}"
            signal_options[label] = signal_row.to_dict()
        st.session_state.scanner_signal_options = signal_options
        st.selectbox(
            "Select Matching Stock",
            options=["-- Select a matching alert --"] + list(signal_options),
            key="scanner_signal_choice",
            on_change=open_scanner_signal,
        )

if app_view == "💼 Portfolio & Secondary Allocator":
    st.title("💼 Portfolio & Secondary Allocator")
    categories = [
        "Defensive Core",
        "High-Beta Satellite",
        "Cyclical",
        "Other",
    ]
    target_tickers = [
        "TSM",
        "AVGO",
        "ARM",
        "CIEN",
        "CLS",
        "APLD",
        "NBIS",
        "BE",
        "RKLB",
        "ONDS",
    ]
    holding_columns = [
        "Ticker",
        "Category",
        "Beta",
        "Quantity",
        "Average Cost ($)",
        "Last Price ($)",
        "Position Value ($)",
        "Unrealized P/L ($)",
    ]
    portfolio_holdings = st.session_state.get("portfolio_holdings")
    if not isinstance(portfolio_holdings, pd.DataFrame):
        portfolio_holdings = pd.DataFrame(columns=holding_columns)
    else:
        portfolio_holdings = portfolio_holdings.copy()
        if "Quantity" not in portfolio_holdings:
            old_values = pd.to_numeric(
                portfolio_holdings.get(
                    "Position Value ($)", pd.Series(0.0, index=portfolio_holdings.index)
                ),
                errors="coerce",
            ).fillna(0.0)
            old_prices = pd.to_numeric(
                portfolio_holdings.get(
                    "Last Price ($)", pd.Series(0.0, index=portfolio_holdings.index)
                ),
                errors="coerce",
            ).fillna(0.0)
            portfolio_holdings["Quantity"] = np.where(
                old_prices > 0, old_values / old_prices, 0.0
            )
        if "Average Cost ($)" not in portfolio_holdings:
            portfolio_holdings["Average Cost ($)"] = portfolio_holdings.get(
                "Last Price ($)", 0.0
            )
        for column, default in (
            ("Ticker", ""),
            ("Category", "Other"),
            ("Beta", 1.0),
            ("Last Price ($)", 0.0),
            ("Position Value ($)", 0.0),
            ("Unrealized P/L ($)", 0.0),
        ):
            if column not in portfolio_holdings:
                portfolio_holdings[column] = default
        portfolio_holdings = portfolio_holdings.reindex(columns=holding_columns)
    st.session_state.portfolio_holdings = portfolio_holdings

    st.session_state.pop("portfolio_total_capital", None)
    cash_balance = st.number_input(
        "Current Cash Balance / Reserve ($USD)",
        min_value=0.0,
        value=100_000.0,
        step=1_000.0,
        key="portfolio_cash_balance",
    )

    st.subheader("Add a position")
    with st.form("portfolio_add_holding_form", clear_on_submit=True):
        ticker_col, category_col = st.columns(2)
        with ticker_col:
            selected_ticker = st.selectbox(
                "Target ticker",
                ["Choose a target ticker"] + target_tickers,
                key="portfolio_target_ticker",
            )
            custom_ticker = st.text_input(
                "Or enter any ticker",
                key="portfolio_custom_ticker",
                placeholder="e.g. NVDA",
            )
        with category_col:
            new_category = st.selectbox(
                "Portfolio category",
                categories,
                index=1,
                key="portfolio_new_holding_category",
            )
            new_quantity = st.number_input(
                "Quantity",
                min_value=0.0,
                step=1.0,
                format="%.4f",
                key="portfolio_new_holding_quantity",
            )
            new_average_cost = st.number_input(
                "Average cost per share ($)",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                help="Leave at $0 to use the latest quoted price as the cost basis.",
                key="portfolio_new_holding_average_cost",
            )
        add_holding = st.form_submit_button(
            "Add position",
            use_container_width=True,
        )

    if add_holding:
        ticker_to_add = (
            custom_ticker.strip().upper()
            or (
                selected_ticker.upper()
                if selected_ticker != "Choose a target ticker"
                else ""
            )
        )
        holdings = st.session_state.portfolio_holdings.copy()
        existing_tickers = (
            holdings["Ticker"].astype(str).str.upper().tolist()
            if not holdings.empty
            else []
        )
        if not ticker_to_add:
            st.warning("Choose a target ticker or enter a ticker symbol.")
        elif ticker_to_add not in existing_tickers:
            beta, last_price = fetch_portfolio_quote(ticker_to_add)
            new_holding = pd.DataFrame([{
                "Ticker": ticker_to_add,
                "Category": new_category,
                "Beta": beta,
                "Quantity": float(new_quantity),
                "Average Cost ($)": (
                    float(new_average_cost) if new_average_cost > 0 else last_price
                ),
                "Last Price ($)": last_price,
                "Position Value ($)": float(new_quantity) * last_price,
                "Unrealized P/L ($)": 0.0,
            }])
            st.session_state.portfolio_holdings = pd.concat(
                [holdings, new_holding], ignore_index=True
            )
            st.rerun()
        else:
            st.info(f"{ticker_to_add} is already in the holdings table.")

    base_holdings = st.session_state.portfolio_holdings.copy()
    st.caption("Edit quantities, cost basis, category, or ticker in the expandable table.")
    with st.expander("Edit holdings", expanded=False):
        edited_holdings = st.data_editor(
            base_holdings,
            key="portfolio_holdings_editor_v2",
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", required=True),
                "Category": st.column_config.SelectboxColumn(
                    "Category", options=categories, required=True
                ),
                "Beta": st.column_config.NumberColumn(
                    "Beta", min_value=-5.0, max_value=10.0, step=0.05, format="%.2f"
                ),
                "Quantity": st.column_config.NumberColumn(
                    "Quantity", min_value=0.0, step=1.0, format="%.4f"
                ),
                "Average Cost ($)": st.column_config.NumberColumn(
                    "Average Cost ($)", min_value=0.0, step=0.01, format="$%.2f"
                ),
                "Last Price ($)": st.column_config.NumberColumn(
                    "Last Price ($)", format="$%.2f", disabled=True
                ),
                "Position Value ($)": st.column_config.NumberColumn(
                    "Position Value ($)", format="$%.2f", disabled=True
                ),
                "Unrealized P/L ($)": st.column_config.NumberColumn(
                    "Unrealized P/L ($)", format="$%.2f", disabled=True
                ),
            },
        )
    edited_holdings = edited_holdings.reindex(columns=holding_columns).copy()
    for row_position in range(len(edited_holdings)):
        raw_ticker = edited_holdings.iloc[row_position]["Ticker"]
        ticker_value = "" if pd.isna(raw_ticker) else str(raw_ticker)
        ticker_value = ticker_value.strip().upper()
        edited_holdings.iat[row_position, 0] = ticker_value
        if not ticker_value:
            continue
        old_ticker = (
            str(base_holdings.iloc[row_position]["Ticker"]).strip().upper()
            if row_position < len(base_holdings)
            else ""
        )
        beta_value = pd.to_numeric(
            pd.Series([edited_holdings.iloc[row_position]["Beta"]]),
            errors="coerce",
        ).iloc[0]
        price_value = pd.to_numeric(
            pd.Series([edited_holdings.iloc[row_position]["Last Price ($)"]]),
            errors="coerce",
        ).iloc[0]
        if ticker_value != old_ticker or pd.isna(beta_value) or pd.isna(price_value):
            beta_value, price_value = fetch_portfolio_quote(ticker_value)
            edited_holdings.at[row_position, "Beta"] = beta_value
            edited_holdings.at[row_position, "Last Price ($)"] = price_value
        category_value = edited_holdings.iloc[row_position]["Category"]
        if pd.isna(category_value) or category_value not in categories:
            edited_holdings.iat[row_position, 1] = "Other"
    edited_holdings["Beta"] = pd.to_numeric(
        edited_holdings["Beta"], errors="coerce"
    ).fillna(1.0)
    edited_holdings["Quantity"] = pd.to_numeric(
        edited_holdings["Quantity"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    edited_holdings["Average Cost ($)"] = pd.to_numeric(
        edited_holdings["Average Cost ($)"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    edited_holdings["Last Price ($)"] = pd.to_numeric(
        edited_holdings["Last Price ($)"], errors="coerce"
    ).fillna(0.0)
    edited_holdings["Position Value ($)"] = (
        edited_holdings["Quantity"] * edited_holdings["Last Price ($)"]
    )
    edited_holdings["Unrealized P/L ($)"] = (
        edited_holdings["Last Price ($)"] - edited_holdings["Average Cost ($)"]
    ) * edited_holdings["Quantity"]
    st.session_state.portfolio_holdings = edited_holdings.reset_index(drop=True)

    positions_total = float(edited_holdings["Position Value ($)"].sum())
    total_capital = positions_total + float(cash_balance)
    allocated_value = total_capital
    holdings_display = edited_holdings.copy()
    holdings_display["Total Percentage (%)"] = (
        holdings_display["Position Value ($)"] / total_capital * 100
        if total_capital > 0
        else 0.0
    )
    st.subheader("Portfolio holdings")
    st.dataframe(
        holdings_display.style.format({
            "Beta": "{:.2f}",
            "Quantity": "{:.4f}",
            "Average Cost ($)": "${:.2f}",
            "Last Price ($)": "${:.2f}",
            "Position Value ($)": "${:,.2f}",
            "Unrealized P/L ($)": "${:+,.2f}",
            "Total Percentage (%)": "{:.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )
    portfolio_beta = (
        float(
            (
                edited_holdings["Position Value ($)"]
                * edited_holdings["Beta"]
            ).sum()
            / allocated_value
        )
        if allocated_value > 0
        else 0.0
    )
    cash_pct = cash_balance / total_capital * 100 if total_capital > 0 else 0.0
    equity_pct = positions_total / total_capital * 100 if total_capital > 0 else 0.0

    portfolio_metric_cols = st.columns(4)
    portfolio_metric_cols[0].metric("Total Capital", f"${total_capital:,.2f}")
    portfolio_metric_cols[1].metric("Cash Reserve", f"{cash_pct:.1f}%")
    portfolio_metric_cols[2].metric("Invested Positions", f"{equity_pct:.1f}%")
    portfolio_metric_cols[3].metric("Weighted Portfolio Beta", f"{portfolio_beta:.2f}")

    core_value = float(
        edited_holdings.loc[
            edited_holdings["Category"] == "Defensive Core", "Position Value ($)"
        ].sum()
    )
    satellite_value = float(
        edited_holdings.loc[
            edited_holdings["Category"] == "High-Beta Satellite",
            "Position Value ($)",
        ].sum()
    )
    core_pct = core_value / positions_total * 100 if positions_total > 0 else 0.0
    satellite_pct = (
        satellite_value / positions_total * 100 if positions_total > 0 else 0.0
    )
    core_col, satellite_col = st.columns(2)
    core_col.metric("Defensive Core / Equity", f"{core_pct:.1f}%", "Target 60-70%")
    satellite_col.metric(
        "High-Beta Satellites / Equity", f"{satellite_pct:.1f}%", "Target 30-40%"
    )
    if positions_total > 0 and not 60.0 <= core_pct <= 70.0:
        core_col.warning("Defensive Core is outside its 60-70% target range.")
    if positions_total > 0 and not 30.0 <= satellite_pct <= 40.0:
        satellite_col.warning("High-Beta Satellites are outside their 30-40% target range.")

    allocation_values = {"Cash": float(cash_balance)}
    for category in categories:
        allocation_values[category] = float(
            edited_holdings.loc[
                edited_holdings["Category"] == category, "Position Value ($)"
            ].sum()
        )
    allocation_values = {
        name: value for name, value in allocation_values.items() if value > 0
    }
    if allocation_values:
        allocation_fig = go.Figure(
            go.Pie(
                labels=list(allocation_values),
                values=list(allocation_values.values()),
                hole=0.58,
                sort=False,
                marker={
                    "colors": [
                        "#B7C7D3",
                        "#2A9D8F",
                        "#E76F51",
                        "#E9C46A",
                        "#7A8C99",
                    ]
                },
                textinfo="label+percent",
            )
        )
        allocation_fig.update_layout(
            title="Current Portfolio Allocation",
            height=390,
            margin=dict(t=55, b=15, l=15, r=15),
            showlegend=True,
        )
        st.plotly_chart(allocation_fig, use_container_width=True)

    st.subheader("Secondary Add Simulation")
    simulation_col, simulation_ticker_col = st.columns(2)
    with simulation_col:
        simulation_amount = st.number_input(
            "Deploy from cash ($)",
            min_value=0.0,
            max_value=float(cash_balance),
            value=0.0,
            step=500.0,
            key="portfolio_simulation_amount",
        )
    ticker_options = sorted(
        {
            ticker
            for ticker in target_tickers + edited_holdings["Ticker"].astype(str).tolist()
            if ticker.strip()
        }
    )
    with simulation_ticker_col:
        simulation_ticker = st.selectbox(
            "Simulation ticker",
            ticker_options,
            key="portfolio_simulation_ticker",
        )
        custom_simulation_ticker = st.text_input(
            "Or enter simulation ticker",
            key="portfolio_custom_simulation_ticker",
            placeholder="e.g. NVDA",
        )
    simulation_ticker = (
        custom_simulation_ticker.strip().upper() or simulation_ticker
    )
    existing_simulation_row = edited_holdings[
        edited_holdings["Ticker"] == simulation_ticker
    ]
    if not existing_simulation_row.empty:
        simulation_beta = float(existing_simulation_row.iloc[0]["Beta"])
    else:
        simulation_beta, _ = fetch_portfolio_quote(simulation_ticker)

    simulated_cash = max(float(cash_balance) - simulation_amount, 0.0)
    simulated_equity = positions_total + float(simulation_amount)
    simulated_nav = simulated_cash + simulated_equity
    simulated_beta_numerator = float(
        (edited_holdings["Position Value ($)"] * edited_holdings["Beta"]).sum()
    ) + float(simulation_amount) * simulation_beta
    simulated_beta = (
        simulated_beta_numerator / simulated_nav if simulated_nav > 0 else 0.0
    )
    simulated_cash_pct = (
        simulated_cash / total_capital * 100 if total_capital > 0 else 0.0
    )
    simulated_equity_pct = (
        simulated_equity / total_capital * 100 if total_capital > 0 else 0.0
    )
    simulation_metrics = st.columns(3)
    simulation_metrics[0].metric(
        "Simulated Cash Reserve", f"{simulated_cash_pct:.1f}%"
    )
    simulation_metrics[1].metric("Simulated Equity", f"{simulated_equity_pct:.1f}%")
    simulation_metrics[2].metric(
        "Simulated Weighted Beta", f"{simulated_beta:.2f}",
        delta=f"{simulated_beta - portfolio_beta:+.2f} vs. current",
    )

if symbol and app_view == "Stock Analysis":
    try:
        info, df_hist, bs, fin, cf_df, q_fin, q_bs, q_cf = fetch_ticker_data(symbol)

        if df_hist.empty and not info.get("currentPrice"):
            st.error(f"❌ 無法取得股票資料或無效代碼: {symbol}")
        else:
            base_val = None
            z_score = None
            z_status = "N/A"
            curr_price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or (df_hist["Close"].iloc[-1] if not df_hist.empty else 0.0)
            )
            price_low = info.get("fiftyTwoWeekLow")
            price_high = info.get("fiftyTwoWeekHigh")
            analyst_target = info.get("targetMeanPrice") or info.get(
                "targetMedianPrice"
            )
            implied_upside = (
                ((analyst_target / curr_price) - 1) * 100
                if analyst_target and curr_price
                else None
            )

            # Formatting Trailing & Forward P/E cleanly without nan
            t_pe = info.get("trailingPE")
            f_pe = info.get("forwardPE")
            t_pe_str = (
                f"{t_pe:.2f}"
                if isinstance(t_pe, (int, float)) and pd.notna(t_pe)
                else "N/A"
            )
            f_pe_str = (
                f"{f_pe:.2f}"
                if isinstance(f_pe, (int, float)) and pd.notna(f_pe)
                else "N/A"
            )
            pe_card_display = (
                f"{t_pe_str} / {f_pe_str}"
                if (t_pe_str != "N/A" or f_pe_str != "N/A")
                else "N/A"
            )

            fundamental_cards = st.columns(4)
            fundamental_cards[0].metric(
                "Current Price", f"${curr_price:.2f}" if curr_price else "N/A"
            )
            fundamental_cards[1].metric("Trailing / Forward P/E", pe_card_display)
            fundamental_cards[2].metric(
                "52-Week Range",
                (
                    f"${price_low:.2f} - ${price_high:.2f}"
                    if price_low and price_high
                    else "N/A"
                ),
            )
            fundamental_cards[3].metric(
                "Analyst Target / Upside",
                f"${analyst_target:.2f}" if analyst_target else "N/A",
                delta=(
                    f"{implied_upside:+.1f}% implied upside"
                    if implied_upside is not None
                    else None
                ),
            )
            gunshot_signal = False
            trend_status = "資料不足 / 無法判斷"
            trend_desc = ""
            div_type = "Neutral"
            all_divergences = []
            support_level = curr_price * 0.9 if curr_price else 0.0
            resistance_level = curr_price * 1.1 if curr_price else 0.0
            poc_price = curr_price
            val_price = vah_price = None
            atr_risk = None
            liquidity_sweeps = []
            pattern_setups = []
            fib_382 = fib_500 = fib_618 = curr_price
            ma250_valid = False
            rsi_value = np.nan
            rsi_div_type = "Neutral"
            rsi_divergences = []
            advanced_divergences = []
            bollinger_state = {
                "squeeze": False,
                "tag": "Neutral",
                "percent_b": np.nan,
                "bandwidth": np.nan,
            }
            ma250_val = 0.0
            m10_val = m20_val = m55_val = curr_price

            # ==========================================
            # 1. 全球大盤情緒：VIX & CNN Fear and Greed 策略買點
            # ==========================================
            st.subheader(
                "🌐 1. 全球市場情緒與 VIX 抄底濾網 (Market Sentiment & VIX Filter)"
            )

            vix_val = get_vix_value()

            cnn_fng_val, cnn_fng_status = get_cnn_fear_and_greed()

            v1, v2, v3 = st.columns(3)
            v1.metric(
                "VIX 恐慌指數",
                f"{vix_val:.2f}",
                delta=">25 為極度恐慌" if vix_val > 25 else "正常區間",
            )
            cnn_display = (
                f"{cnn_fng_val} ({cnn_fng_status})"
                if cnn_fng_val is not None
                else cnn_fng_status
            )
            v2.metric("CNN Fear & Greed Index", cnn_display)

            buy_window = (
                cnn_fng_val is not None and vix_val >= 25 and cnn_fng_val <= 40
            )
            if buy_window:
                v3.success(
                    "🟢 **觸發黃金抄底訊號**：VIX > 25 且 CNN Fear & Greed < 40"
                    " (極度恐慌為長線建倉時機)！"
                )
            else:
                v3.info(
                    "🟡 **市場情緒平穩/偏熱**：未達到 VIX > 25 且 CNN F&G < 40"
                    " 的黃金逆勢買點。"
                )

            st.write("---")

            # ==========================================
            # 2. 精準細分產業、全方位財務比率 (Financial Ratios) 與基本面估值
            # ==========================================
            st.subheader(
                f"🏛️ 2. {info.get('shortName', symbol)} ({symbol}) -"
                " 完整財務比率與基本面指標"
            )

            raw_industry = info.get("industry", "")
            raw_sector = info.get("sector", "")
            ind_category_name, ind_benchmarks = get_granular_industry_benchmarks(
                raw_industry, raw_sector
            )

            st.caption(
                f"📌 **細分產業類別**：`{raw_industry or raw_sector}` ➔ 定義為"
                f" **{ind_category_name}**\n📊 **同業基準參考**：P/E:"
                f" {ind_benchmarks['PE']}x | P/S: {ind_benchmarks['PS']}x | PEG:"
                f" {ind_benchmarks['PEG']}x | 毛利率: {ind_benchmarks['GM']}% |"
                f" 淨利率: {ind_benchmarks['NM']}% | ROE: {ind_benchmarks['ROE']}%"
            )

            if not bs.empty and not fin.empty:
                try:
                    latest_bs = bs.iloc[:, 0]
                    latest_fin = fin.iloc[:, 0]

                    total_assets = latest_bs.get("Total Assets", np.nan)
                    total_liab = latest_bs.get(
                        "Total Liabilities Net Minority Interest",
                        latest_bs.get("Total Liabilities", np.nan),
                    )
                    curr_assets = latest_bs.get("Current Assets", np.nan)
                    curr_liab = latest_bs.get("Current Liabilities", np.nan)
                    retained_earnings = latest_bs.get("Retained Earnings", 0)
                    working_capital = (
                        (curr_assets - curr_liab)
                        if pd.notna(curr_assets) and pd.notna(curr_liab)
                        else np.nan
                    )
                    ebit = latest_fin.get("EBIT", np.nan)
                    sales = latest_fin.get("Total Revenue", np.nan)
                    mkt_cap = info.get("marketCap", np.nan)

                    if (
                        pd.notna(total_assets)
                        and pd.notna(total_liab)
                        and total_liab > 0
                        and pd.notna(working_capital)
                    ):
                        x1 = working_capital / total_assets
                        x2 = retained_earnings / total_assets
                        x3 = ebit / total_assets if pd.notna(ebit) else 0
                        x4 = mkt_cap / total_liab if pd.notna(mkt_cap) else 0
                        x5 = sales / total_assets if pd.notna(sales) else 0

                        z_score = (
                            1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5
                        )

                        if z_score > 2.99:
                            z_status = f"{z_score:.2f} (🟢 安全)"
                        elif z_score >= 1.81:
                            z_status = f"{z_score:.2f} (🟡 灰色地帶)"
                        else:
                            z_status = f"{z_score:.2f} (🔴 高風險)"
                except Exception:
                    z_status = "無法計算 (數據不全)"

            st.markdown("##### 📈 乘數與估值比率 (Valuation Multiples)")
            r1_1, r1_2, r1_3, r1_4, r1_5, r1_6 = st.columns(6)
            pe_val = info.get("forwardPE") or info.get("trailingPE")
            r1_1.metric(
                "Forward P/E",
                f"{pe_val:.2f}" if pe_val else "N/A",
                delta=f"同業 {ind_benchmarks['PE']}x",
                delta_color=(
                    "inverse" if pe_val and pe_val > ind_benchmarks["PE"] else "normal"
                ),
            )

            ps_val = info.get("priceToSalesTrailing12Months")
            r1_2.metric(
                "P/S Ratio",
                f"{ps_val:.2f}" if ps_val else "N/A",
                delta=f"同業 {ind_benchmarks['PS']}x",
                delta_color=(
                    "inverse" if ps_val and ps_val > ind_benchmarks["PS"] else "normal"
                ),
            )

            r1_3.metric(
                "P/B Ratio",
                f"{info.get('priceToBook', 0):.2f}"
                if info.get("priceToBook")
                else "N/A",
            )

            peg_val = info.get("pegRatio")
            r1_4.metric(
                "PEG Ratio",
                f"{peg_val:.2f}" if peg_val else "N/A",
                delta=f"同業 {ind_benchmarks['PEG']}x",
                delta_color=(
                    "inverse"
                    if peg_val and peg_val > ind_benchmarks["PEG"]
                    else "normal"
                ),
            )

            r1_5.metric(
                "Quick Ratio",
                f"{info.get('quickRatio', 0):.2f}"
                if info.get("quickRatio")
                else "N/A",
            )
            r1_6.metric("Altman Z-Score", z_status)

            st.markdown("##### 💵 獲利能力比率 (Profitability & Margins)")
            r2_1, r2_2, r2_3, r2_4, r2_5 = st.columns(5)

            gm_val = info.get("grossMargins")
            gm_disp = f"{gm_val * 100:.2f}%" if gm_val else "N/A"
            r2_1.metric(
                "毛利率 (Gross Margin)",
                gm_disp,
                delta=f"同業 {ind_benchmarks['GM']}%",
                delta_color=(
                    "normal"
                    if gm_val and gm_val * 100 > ind_benchmarks["GM"]
                    else "inverse"
                ),
            )

            op_val = info.get("operatingMargins")
            r2_2.metric(
                "營業利益率 (Op Margin)",
                f"{op_val * 100:.2f}%" if op_val else "N/A",
            )

            nm_val = info.get("profitMargins")
            nm_disp = f"{nm_val * 100:.2f}%" if nm_val else "N/A"
            r2_3.metric(
                "淨利率 (Net Margin)",
                nm_disp,
                delta=f"同業 {ind_benchmarks['NM']}%",
                delta_color=(
                    "normal"
                    if nm_val and nm_val * 100 > ind_benchmarks["NM"]
                    else "inverse"
                ),
            )

            roe_val = info.get("returnOnEquity")
            roe_disp = f"{roe_val * 100:.2f}%" if roe_val else "N/A"
            r2_4.metric(
                "股東權益報酬率 (ROE)",
                roe_disp,
                delta=f"同業 {ind_benchmarks['ROE']}%",
                delta_color=(
                    "normal"
                    if roe_val and roe_val * 100 > ind_benchmarks["ROE"]
                    else "inverse"
                ),
            )

            roa_val = info.get("returnOnAssets")
            r2_5.metric(
                "資產報酬率 (ROA)", f"{roa_val * 100:.2f}%" if roa_val else "N/A"
            )

            # ==========================================
            # 3. 三種 DCF 現金流折現估值模型
            # ==========================================
            st.write("---")
            st.subheader("💰 3. 三種 DCF 現金流折現估值模型 (Comparative DCF)")

            rf_rate = get_risk_free_rate()
            beta = info.get("beta", 1.0)
            if pd.isna(beta) or beta is None:
                beta = 1.0

            ttm_fcf = None
            if not q_cf.empty:
                try:
                    q_cf_t = q_cf.T.head(4)
                    if (
                        "Free Cash Flow" in q_cf_t.columns
                        and q_cf_t["Free Cash Flow"].notna().sum() == 4
                    ):
                        ttm_fcf = float(q_cf_t["Free Cash Flow"].sum())
                    elif (
                        "Operating Cash Flow" in q_cf_t.columns
                        and "Capital Expenditure" in q_cf_t.columns
                    ):
                        ttm_fcf = float(
                            (
                                q_cf_t["Operating Cash Flow"]
                                + q_cf_t["Capital Expenditure"]
                            ).sum()
                        )
                except Exception:
                    ttm_fcf = None

            fcf_history = []
            fcf_dates = []
            if not cf_df.empty:
                for col in cf_df.columns:
                    val = None
                    if "Free Cash Flow" in cf_df.index and pd.notna(
                        cf_df.loc["Free Cash Flow", col]
                    ):
                        val = cf_df.loc["Free Cash Flow", col]
                    elif (
                        "Operating Cash Flow" in cf_df.index
                        and "Capital Expenditure" in cf_df.index
                    ):
                        ocf = cf_df.loc["Operating Cash Flow", col]
                        capex = cf_df.loc["Capital Expenditure", col]
                        if pd.notna(ocf) and pd.notna(capex):
                            val = ocf + capex
                    if val is not None and not pd.isna(val):
                        fcf_history.append(float(val))
                        fcf_dates.append(str(col)[:10])

            mkt_cap_val = float(info.get("marketCap", 0) or 0.0)

            def normalize_absolute_dollars(value, reference_market_cap):
                if value is None or pd.isna(value):
                    return 0.0
                amount = float(value)
                if reference_market_cap >= 1_000_000_000 and 0 < abs(amount) < 1:
                    return amount * 1_000_000_000.0
                if reference_market_cap >= 1_000_000_000 and 0 < abs(amount) < 1_000_000:
                    return amount * 1_000_000.0
                if (
                    reference_market_cap >= 1_000_000_000
                    and abs(amount) >= 1_000_000
                    and abs(amount) / reference_market_cap < 1e-9
                ):
                    scaled_amount = amount * 1_000.0
                    if abs(scaled_amount) <= reference_market_cap * 3.0:
                        return scaled_amount
                return amount

            def normalize_shares_outstanding(value, market_cap, share_price):
                try:
                    shares = float(value or 0.0)
                    expected_shares = float(market_cap) / float(share_price)
                except (TypeError, ValueError, ZeroDivisionError):
                    return 0.0
                if (
                    not np.isfinite(shares)
                    or not np.isfinite(expected_shares)
                    or shares <= 0
                    or expected_shares <= 0
                ):
                    return max(shares, 0.0) if np.isfinite(shares) else 0.0
                raw_error = abs(np.log(shares / expected_shares))
                candidates = [
                    shares * 1_000.0,
                    shares / 1_000.0,
                    shares * 1_000_000.0,
                    shares / 1_000_000.0,
                    shares * 1_000_000_000.0,
                    shares / 1_000_000_000.0,
                ]
                best_candidate = min(
                    candidates,
                    key=lambda candidate: abs(np.log(candidate / expected_shares)),
                )
                adjusted_error = abs(np.log(best_candidate / expected_shares))
                if (
                    raw_error >= np.log(1_000.0) - 1e-9
                    and adjusted_error < np.log(10.0)
                ):
                    return best_candidate
                return shares

            shares_out = normalize_shares_outstanding(
                info.get("sharesOutstanding", 0), mkt_cap_val, curr_price
            )

            net_debt = 0.0
            total_debt = 0.0
            cash_val = 0.0

            if not bs.empty:
                try:
                    latest_bs = bs.iloc[:, 0]
                    tot_debt = latest_bs.get("Total Debt", None)
                    if pd.isna(tot_debt) or tot_debt is None:
                        lt_debt = latest_bs.get("Long Term Debt", 0)
                        st_debt = latest_bs.get(
                            "Current Debt And Capital Lease Obligation",
                            latest_bs.get("Current Debt", 0),
                        )
                        tot_debt = (lt_debt if pd.notna(lt_debt) else 0) + (
                            st_debt if pd.notna(st_debt) else 0
                        )

                    cash = latest_bs.get(
                        "Cash And Cash Equivalents",
                        latest_bs.get(
                            "Cash Cash Equivalents And Short Term Investments",
                            info.get("cash", 0),
                        ),
                    )

                    if pd.notna(tot_debt):
                        total_debt = float(tot_debt)
                        cash_val = float(cash) if pd.notna(cash) else 0.0
                        net_debt = total_debt - cash_val
                except Exception:
                    pass

            total_debt = normalize_absolute_dollars(total_debt, mkt_cap_val)
            cash_val = normalize_absolute_dollars(cash_val, mkt_cap_val)
            net_debt = total_debt - cash_val

            erp = 0.050
            cost_of_equity = rf_rate + (beta * erp)
            debt_spread = 0.020
            if mkt_cap_val > 0 and total_debt > 0:
                cost_of_debt = rf_rate + debt_spread
                total_capital = mkt_cap_val + total_debt
                weight_e = mkt_cap_val / total_capital
                weight_d = total_debt / total_capital
                calc_wacc = (weight_e * cost_of_equity) + (
                    weight_d * cost_of_debt * 0.79
                )
                discount_rate = calc_wacc
                wacc_label = f"WACC {discount_rate*100:.2f}% (Rf: {rf_rate*100:.2f}%)"
            else:
                discount_rate = cost_of_equity
                wacc_label = (
                    f"Cost of Equity {discount_rate*100:.2f}% (Rf:"
                    f" {rf_rate*100:.2f}%)"
                )

            fcf_history = [
                normalize_absolute_dollars(value, mkt_cap_val)
                for value in fcf_history
            ]
            ttm_fcf = normalize_absolute_dollars(ttm_fcf, mkt_cap_val)

            dated_fcf = (
                sorted(zip(fcf_dates, fcf_history), key=lambda item: item[0])
                if fcf_dates
                else []
            )
            chronological_fcf = [
                float(value) for _, value in dated_fcf if pd.notna(value)
            ]
            latest_annual_fcf = chronological_fcf[-1] if chronological_fcf else 0.0
            recent_positive_annual_fcfs = [
                value for value in chronological_fcf[-3:] if value > 0
            ]
            historical_fcf_median = (
                float(np.median(recent_positive_annual_fcfs))
                if recent_positive_annual_fcfs
                else 0.0
            )
            latest_fcf = ttm_fcf if ttm_fcf != 0 else latest_annual_fcf

            latest_net_income = 0.0
            if not fin.empty:
                for income_name in ["Net Income", "Net Income Common Stockholders"]:
                    if income_name in fin.index:
                        income_values = (
                            pd.to_numeric(fin.loc[income_name], errors="coerce")
                            .dropna()
                            .tolist()
                        )
                        if income_values:
                            latest_net_income = normalize_absolute_dollars(
                                float(income_values[0]), mkt_cap_val
                            )
                            break

            net_income_fcf_proxy = (
                latest_net_income * 0.80 if latest_net_income > 0 else 0.0
            )
            if latest_fcf > 0:
                if historical_fcf_median > 0 and not (
                    historical_fcf_median * 0.50
                    <= latest_fcf
                    <= historical_fcf_median * 2.00
                ):
                    base_fcf = historical_fcf_median or net_income_fcf_proxy
                elif net_income_fcf_proxy > 0 and not historical_fcf_median and not (
                    net_income_fcf_proxy * 0.25
                    <= latest_fcf
                    <= net_income_fcf_proxy * 4.00
                ):
                    base_fcf = net_income_fcf_proxy
                else:
                    base_fcf = latest_fcf
            else:
                base_fcf = historical_fcf_median or net_income_fcf_proxy

            recent_annual_fcfs = chronological_fcf[-3:]
            stable_positive_fcf = bool(
                len(recent_annual_fcfs) >= 2
                and sum(value > 0 for value in recent_annual_fcfs) >= 2
                and latest_annual_fcf > 0
                and base_fcf > 0
            )

            if base_fcf > 0 and shares_out > 0:
                industry_growth = get_industry_growth_profile(
                    raw_industry, raw_sector
                )
                (
                    fcf_growth_anchor,
                    recent_fcf_growth,
                    fcf_growth_slope,
                ) = estimate_fcf_growth_anchor(
                    fcf_history, industry_growth, fcf_dates
                )
                net_income_growth_anchor = estimate_net_income_growth_anchor(
                    fin, info
                )
                revenue_growth_anchor = get_revenue_growth_anchor(info)
                annual_revenue = normalize_absolute_dollars(
                    info.get("totalRevenue") or info.get("revenue") or 0.0,
                    mkt_cap_val,
                )
                if not annual_revenue and not fin.empty:
                    for revenue_name in ["Total Revenue", "Operating Revenue"]:
                        if revenue_name in fin.index:
                            annual_revenue = normalize_absolute_dollars(
                                float(
                                    pd.to_numeric(
                                        fin.loc[revenue_name], errors="coerce"
                                    )
                                    .dropna()
                                    .iloc[0]
                                ),
                                mkt_cap_val,
                            )
                            break
                annual_revenue = float(annual_revenue or 0.0)
                fcf_margin = base_fcf / annual_revenue if annual_revenue > 0 else 0.0
                latest_capex = 0.0
                if not cf_df.empty and "Capital Expenditure" in cf_df.index:
                    capex_values = pd.to_numeric(
                        cf_df.loc["Capital Expenditure"], errors="coerce"
                    ).dropna()
                    if not capex_values.empty:
                        latest_capex = abs(float(capex_values.iloc[0]))
                capex_intensity = (
                    latest_capex / annual_revenue if annual_revenue > 0 else 0.0
                )
                negative_fcf_trend = bool(
                    fcf_growth_slope < -0.10
                    or (recent_fcf_growth and recent_fcf_growth[-1] < -0.20)
                )
                investment_heavy_growth = bool(
                    mkt_cap_val >= 10_000_000_000
                    and capex_intensity >= 0.10
                    and (net_income_growth_anchor or 0.0) > 0.0
                    and (revenue_growth_anchor or 0.0) > 0.05
                )
                lifecycle, weights, growth_floor, terminal_g = (
                    classify_corporate_lifecycle(
                        revenue_growth_anchor or 0.0,
                        fcf_margin,
                        negative_fcf_trend and not investment_heavy_growth,
                    )
                )

                lifecycle_growth_ceiling = {
                    "Star": 0.30,
                    "Cash Cow": 0.18,
                    "Question Mark": 0.30,
                    "Dog": 0.12,
                }.get(lifecycle, 0.25)

                signals = {
                    "fcf": fcf_growth_anchor,
                    "net_income": net_income_growth_anchor,
                    "revenue": revenue_growth_anchor,
                }
                if investment_heavy_growth and signals["fcf"] is not None:
                    signals["fcf"] = max(float(signals["fcf"]), 0.0)
                (
                    weights,
                    company_size,
                    earnings_outpace_fcf,
                    high_capex,
                ) = adjust_growth_weights_for_scale(
                    weights,
                    mkt_cap_val,
                    net_income_growth_anchor,
                    fcf_growth_anchor,
                    capex_intensity,
                )
                clamped_signals = {
                    key: max(-0.20, min(float(value), 0.35))
                    for key, value in signals.items()
                    if value is not None and pd.notna(value)
                }
                available_weight = sum(weights[key] for key in clamped_signals)
                growth_est = (
                    sum(clamped_signals[key] * weights[key] for key in clamped_signals)
                    / available_weight
                    if available_weight > 0
                    else 0.0
                )
                growth_est = max(growth_floor, min(growth_est, lifecycle_growth_ceiling))
                company_ceiling = lifecycle_growth_ceiling

                pe_implied_value = (
                    latest_net_income * float(ind_benchmarks["PE"]) / float(shares_out)
                    if latest_net_income > 0 and shares_out > 0
                    else None
                )
                ps_implied_value = (
                    annual_revenue * float(ind_benchmarks["PS"]) / float(shares_out)
                    if annual_revenue > 0 and shares_out > 0
                    else None
                )

                observed_multiple_premiums = []
                observed_pe = info.get("forwardPE") or info.get("trailingPE")
                observed_ps = info.get("priceToSalesTrailing12Months")
                if (
                    isinstance(observed_pe, (int, float))
                    and pd.notna(observed_pe)
                    and observed_pe > 0
                ):
                    observed_multiple_premiums.append(
                        float(observed_pe) / float(ind_benchmarks["PE"])
                    )
                if (
                    isinstance(observed_ps, (int, float))
                    and pd.notna(observed_ps)
                    and observed_ps > 0
                ):
                    observed_multiple_premiums.append(
                        float(observed_ps) / float(ind_benchmarks["PS"])
                    )
                market_multiple_factor = (
                    max(0.75, min(float(np.mean(observed_multiple_premiums)), 1.50))
                    if observed_multiple_premiums
                    else 1.0
                )
                if pe_implied_value is not None:
                    pe_implied_value *= market_multiple_factor
                if ps_implied_value is not None:
                    ps_implied_value *= market_multiple_factor

                if lifecycle == "Star":
                    valuation_weights = {"dcf": 0.80, "pe": 0.10, "ps": 0.10}
                elif lifecycle == "Cash Cow":
                    valuation_weights = {"dcf": 0.80, "pe": 0.15, "ps": 0.05}
                elif lifecycle == "Question Mark":
                    valuation_weights = {"dcf": 0.80, "pe": 0.05, "ps": 0.15}
                else:
                    valuation_weights = {"dcf": 0.80, "pe": 0.15, "ps": 0.05}

                def blend_valuation(dcf_value):
                    dcf_value = max(float(dcf_value), 0.0)
                    if dcf_value <= 0:
                        return 0.0
                    peer_floor = dcf_value * 0.50
                    peer_ceiling = dcf_value * 1.50
                    values = {
                        "dcf": dcf_value,
                        "pe": (
                            max(peer_floor, min(pe_implied_value, peer_ceiling))
                            if pe_implied_value is not None
                            else None
                        ),
                        "ps": (
                            max(peer_floor, min(ps_implied_value, peer_ceiling))
                            if ps_implied_value is not None
                            else None
                        ),
                    }
                    available = {
                        key: value
                        for key, value in values.items()
                        if value is not None and value >= 0
                    }
                    weight_total = sum(valuation_weights[key] for key in available)
                    if weight_total <= 0:
                        return max(float(dcf_value), 0.0)
                    return (
                        sum(available[key] * valuation_weights[key] for key in available)
                        / weight_total
                    )

                target_net_margin = (
                    latest_net_income / annual_revenue
                    if latest_net_income > 0 and annual_revenue > 0
                    else float(info.get("profitMargins") or 0.0)
                )
                cash_flow_conversion_rate = (
                    base_fcf / latest_net_income
                    if latest_net_income > 0 and base_fcf > 0
                    else 0.80
                )
                cash_flow_conversion_rate = max(
                    0.50, min(cash_flow_conversion_rate, 1.50)
                )
                if stable_positive_fcf:
                    benchmark_exit_multiple = (
                        float(ind_benchmarks["PE"]) * market_multiple_factor
                    )
                    exit_multiple = (
                        max(8.0, min(benchmark_exit_multiple, 20.0))
                        if latest_net_income > 0
                        else 10.0
                    )
                else:
                    exit_multiple = 0.0
                forecast_period = 10

                terminal_growth_cap = max(0.0, min(0.03, discount_rate - 0.01))
                base_term_g = (
                    min(max(float(terminal_g), 0.0), terminal_growth_cap)
                    if stable_positive_fcf
                    else 0.0
                )
                pess_term_g = max(0.0, base_term_g - 0.005)
                opt_term_g = min(terminal_growth_cap, base_term_g + 0.005)

                base_growth_rate = growth_est
                pess_growth_rate = max(growth_floor, growth_est - 0.05)
                opt_growth_rate = min(company_ceiling, growth_est + 0.05)
                base_fcfe_result = calculate_fcfe_dcf(
                    curr_price,
                    annual_revenue,
                    base_growth_rate,
                    target_net_margin,
                    cash_flow_conversion_rate,
                    forecast_period,
                    discount_rate,
                    exit_multiple,
                    shares_out,
                    base_term_g,
                    stable_positive_fcf,
                )
                pess_fcfe_result = calculate_fcfe_dcf(
                    curr_price,
                    annual_revenue,
                    pess_growth_rate,
                    target_net_margin,
                    cash_flow_conversion_rate,
                    forecast_period,
                    discount_rate + 0.01,
                    exit_multiple * 0.90,
                    shares_out,
                    pess_term_g,
                    stable_positive_fcf,
                )
                opt_fcfe_result = calculate_fcfe_dcf(
                    curr_price,
                    annual_revenue,
                    opt_growth_rate,
                    target_net_margin,
                    cash_flow_conversion_rate,
                    forecast_period,
                    max(discount_rate - 0.008, 0.05),
                    exit_multiple * 1.10,
                    shares_out,
                    opt_term_g,
                    stable_positive_fcf,
                )

                base_schedule = [base_growth_rate] * 5
                pess_schedule = [pess_growth_rate] * 5
                opt_schedule = [opt_growth_rate] * 5
                base_dcf_val = base_fcfe_result["dcfValuePerShare"]
                pess_dcf_val = pess_fcfe_result["dcfValuePerShare"]
                opt_dcf_val = opt_fcfe_result["dcfValuePerShare"]
                base_val = blend_valuation(base_dcf_val)
                pess_val = blend_valuation(pess_dcf_val)
                opt_val = blend_valuation(opt_dcf_val)

                traditional_dcf_result = calculate_traditional_fcff_dcf(
                    curr_price,
                    base_fcf,
                    chronological_fcf,
                    discount_rate,
                    net_debt,
                    shares_out,
                    forecast_period=5,
                    terminal_growth_rate=0.0225,
                )
                roa_raw = info.get("returnOnAssets")
                roa_value = (
                    float(roa_raw)
                    if isinstance(roa_raw, (int, float, np.number))
                    and pd.notna(roa_raw)
                    and np.isfinite(float(roa_raw))
                    else None
                )
                industry_roa_benchmark = estimate_industry_roa_benchmark(
                    raw_industry, raw_sector
                )
                historical_roa = estimate_roa_history(fin, bs)
                roa_adjusted_result = calculate_roa_adjusted_dcf(
                    curr_price,
                    annual_revenue,
                    base_growth_rate,
                    target_net_margin,
                    cash_flow_conversion_rate,
                    forecast_period,
                    discount_rate,
                    exit_multiple,
                    shares_out,
                    base_term_g,
                    stable_positive_fcf,
                    roa_value,
                    industry_roa_benchmark,
                    historical_roa,
                )
                traditional_dcf_value = traditional_dcf_result[
                    "dcfValuePerShare"
                ]
                roa_adjusted_dcf_value = roa_adjusted_result[
                    "dcfValuePerShare"
                ]

                base_fcfs = [
                    row["FCFE"] for row in base_fcfe_result["yearlyProjections"][:5]
                ]
                base_pvs = [
                    row["Present Value"]
                    for row in base_fcfe_result["yearlyProjections"][:5]
                ]
                base_tv = base_fcfe_result["terminalValue"]
                base_pv_tv = base_fcfe_result["presentValueTerminal"]
                base_ev = base_fcfe_result["totalPresentValue"]
                base_eq = base_ev

                st.markdown("#### 三種方法估值比較")
                method_columns = st.columns(3)
                method_columns[0].metric(
                    "Approach #1｜傳統 FCFF",
                    f"${traditional_dcf_value:.2f}",
                    help="5 年歷史 FCF 成長率淡出預測，WACC 折現並使用 2.25% Gordon 永續成長率；股權價值扣除淨債務。",
                )
                method_columns[1].metric(
                    "Approach #2｜成長加權（現行）",
                    f"${base_val:.2f}",
                    help="保留原有 FCF、淨利與營收加權成長模型及原有 DCF/同業倍數混合估值。",
                )
                method_columns[2].metric(
                    "Approach #3｜ROA 品質調整",
                    f"${roa_adjusted_dcf_value:.2f}",
                    help=(
                        f"ROA {roa_value * 100:.2f}% vs. 同業基準 "
                        f"{industry_roa_benchmark * 100:.2f}%；依 ROA 與穩定度調整現金流轉換率、"
                        f"終值成長及折現率。"
                        if roa_value is not None
                        else "缺少有效 ROA 資料；以中性品質調整呈現原始成長加權 DCF。"
                    ),
                )
                valuation_comparison = pd.DataFrame([
                    {
                        "DCF 方法": "Approach #1 傳統 FCFF（5 年）",
                        "每股估值": traditional_dcf_value,
                        "折現率": traditional_dcf_result["discountRate"],
                        "終值成長率": traditional_dcf_result["terminalGrowthRate"],
                        "ROA": None,
                        "同業 ROA 基準": None,
                        "現金流調整倍數": 1.0,
                    },
                    {
                        "DCF 方法": "Approach #2 成長加權（現行）",
                        "每股估值": base_val,
                        "折現率": discount_rate,
                        "終值成長率": base_term_g,
                        "ROA": roa_value,
                        "同業 ROA 基準": industry_roa_benchmark,
                        "現金流調整倍數": 1.0,
                    },
                    {
                        "DCF 方法": "Approach #3 ROA 品質調整",
                        "每股估值": roa_adjusted_dcf_value,
                        "折現率": roa_adjusted_result[
                            "qualityAdjustedDiscountRate"
                        ],
                        "終值成長率": roa_adjusted_result["terminalGrowthRate"],
                        "ROA": roa_value,
                        "同業 ROA 基準": industry_roa_benchmark,
                        "現金流調整倍數": roa_adjusted_result[
                            "cashFlowMultiplier"
                        ],
                    },
                ])
                st.dataframe(
                    valuation_comparison.style.format({
                        "每股估值": "${:.2f}",
                        "折現率": "{:.2%}",
                        "終值成長率": "{:.2%}",
                        "ROA": "{:.2%}",
                        "同業 ROA 基準": "{:.2%}",
                        "現金流調整倍數": "{:.2f}x",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "Approach #2 保留既有情境估值、估值混合、基本面評分與下方計算明細；"
                    "Approach #1 為 FCFF/WACC 企業價值模型，Approach #3 為獨立的 ROA 調整 FCFE 模型。"
                )

                pess_discount = discount_rate + 0.01
                opt_discount = max(discount_rate - 0.008, 0.05)
                pess_fcfs = [
                    row["FCFE"] for row in pess_fcfe_result["yearlyProjections"][:5]
                ]
                pess_pvs = [
                    row["Present Value"]
                    for row in pess_fcfe_result["yearlyProjections"][:5]
                ]
                pess_tv = pess_fcfe_result["terminalValue"]
                pess_pv_tv = pess_fcfe_result["presentValueTerminal"]
                pess_ev = pess_fcfe_result["totalPresentValue"]
                pess_eq = pess_ev
                opt_fcfs = [
                    row["FCFE"] for row in opt_fcfe_result["yearlyProjections"][:5]
                ]
                opt_pvs = [
                    row["Present Value"]
                    for row in opt_fcfe_result["yearlyProjections"][:5]
                ]
                opt_tv = opt_fcfe_result["terminalValue"]
                opt_pv_tv = opt_fcfe_result["presentValueTerminal"]
                opt_ev = opt_fcfe_result["totalPresentValue"]
                opt_eq = opt_ev

                dcf_col1, dcf_col2, dcf_col3, dcf_col4 = st.columns(4)
                dcf_col1.metric(
                    "🔴 保守情境估值",
                    f"${pess_val:.2f}",
                    help=(
                        f"成長階段: {', '.join(f'{g*100:.1f}%' for g in pess_schedule)}%"
                        f" | 折現率: {pess_discount*100:.1f}% | 終值成長:"
                        f" {pess_term_g*100:.1f}%"
                    ),
                )
                dcf_col2.metric(
                    "🟡 基準情境估值",
                    f"${base_val:.2f}",
                    help=(
                        f"成長階段: {', '.join(f'{g*100:.1f}%' for g in base_schedule)}%"
                        f" | 折現率: {discount_rate*100:.1f}% | 終值成長:"
                        f" {base_term_g*100:.1f}%"
                    ),
                )
                dcf_col3.metric(
                    "🟢 樂觀情境估值",
                    f"${opt_val:.2f}",
                    help=(
                        f"成長階段: {', '.join(f'{g*100:.1f}%' for g in opt_schedule)}%"
                        f" | 折現率: {opt_discount*100:.1f}% | 終值成長:"
                        f" {opt_term_g*100:.1f}%"
                    ),
                )

                margin_of_safety = (
                    ((base_val - curr_price) / base_val) * 100 if base_val > 0 else 0
                )
                dcf_col4.metric(
                    "當前股價 / 安全邊際",
                    f"${curr_price:.2f}",
                    delta=f"安全邊際: {margin_of_safety:.1f}%",
                )

                st.write({
                    "Classification Stage": lifecycle,
                    "Company Size Benchmark": company_size,
                    "AI/Investment Capex Intensity (%)": round(
                        capex_intensity * 100, 2
                    ),
                    "Investment-Heavy Growth Treatment": investment_heavy_growth,
                    "Net Income Growth > FCF Growth": earnings_outpace_fcf,
                    "Growth Weights (FCF/NI/Revenue)": (
                        f"{weights['fcf']:.0%} / {weights['net_income']:.0%} /"
                        f" {weights['revenue']:.0%}"
                    ),
                    "Valuation Weights (DCF/PE/PS)": (
                        f"{valuation_weights['dcf']:.0%} /"
                        f" {valuation_weights['pe']:.0%} /"
                        f" {valuation_weights['ps']:.0%}"
                    ),
                    "Market Multiple Premium Factor": round(
                        market_multiple_factor, 3
                    ),
                    "P/E Implied Value ($/share)": (
                        round(pe_implied_value, 2)
                        if pe_implied_value is not None
                        else None
                    ),
                    "P/S Implied Value ($/share)": (
                        round(ps_implied_value, 2)
                        if ps_implied_value is not None
                        else None
                    ),
                    "Blended Growth Anchor (%)": round(growth_est * 100, 2),
                    "Starting FCF Base ($)": round(base_fcf, 2),
                    "Discount Rate Used (r)": round(discount_rate, 6),
                    "Terminal FCF Eligible": stable_positive_fcf,
                    "Terminal Growth Rate Used": round(base_term_g, 4),
                    "Terminal Exit Multiple Used": round(
                        base_fcfe_result["exitMultiple"], 2
                    ),
                    "DCF Equity Value ($)": round(base_eq, 2),
                    "Total Debt ($)": round(total_debt, 2),
                    "Shares Outstanding": round(float(shares_out), 2),
                })

                scenario_details = [
                    (
                        "🔴 保守情境計算明細",
                        pess_schedule,
                        pess_discount,
                        pess_term_g,
                        pess_val,
                        pess_fcfs,
                        pess_pvs,
                        pess_tv,
                        pess_pv_tv,
                        pess_ev,
                        pess_eq,
                    ),
                    (
                        "🟡 基準情境計算明細",
                        base_schedule,
                        discount_rate,
                        base_term_g,
                        base_val,
                        base_fcfs,
                        base_pvs,
                        base_tv,
                        base_pv_tv,
                        base_ev,
                        base_eq,
                    ),
                    (
                        "🟢 樂觀情境計算明細",
                        opt_schedule,
                        opt_discount,
                        opt_term_g,
                        opt_val,
                        opt_fcfs,
                        opt_pvs,
                        opt_tv,
                        opt_pv_tv,
                        opt_ev,
                        opt_eq,
                    ),
                ]

                for (
                    scenario_title,
                    scenario_growth_schedule,
                    scenario_discount,
                    scenario_terminal_growth,
                    scenario_value,
                    scenario_fcfs,
                    scenario_pvs,
                    scenario_tv,
                    scenario_pv_tv,
                    scenario_ev,
                    scenario_eq,
                ) in scenario_details:
                    with st.expander(f"🧮 {scenario_title}"):
                        stage_text = ", ".join(
                            f"Y{idx}={g * 100:.1f}%"
                            for idx, g in enumerate(
                                scenario_growth_schedule, start=1
                            )
                        )
                        st.caption(
                            f"假設：基準營收 ${annual_revenue / 1e9:.2f}B | 10年 FCFE"
                            f" 成長率: {stage_text} | 折現率"
                            f" {scenario_discount * 100:.2f}% | FCFE 轉換率"
                            f" {cash_flow_conversion_rate * 100:.1f}% | 退出倍數"
                            f" {exit_multiple:.1f}x | 淨債務"
                            f" ${net_debt / 1e9:.2f}B | 流通股數"
                            f" {shares_out / 1e9:.2f}B"
                        )
                        assumptions = pd.DataFrame({
                            "模型假設": [
                                "預測年期",
                                "基準營收",
                                "FCFE 成長假設",
                                "目標淨利率",
                                "FCFE 轉換率",
                                "退出倍數",
                                "折現率",
                                "無風險利率 / Beta / ERP",
                                "股權成本",
                                "總債務 / 淨債務",
                                "流通股數",
                            ],
                            "採用數值": [
                                "10 年",
                                f"${annual_revenue / 1e9:.2f}B",
                                stage_text,
                                f"{target_net_margin * 100:.2f}%",
                                f"{cash_flow_conversion_rate * 100:.2f}%",
                                f"{exit_multiple:.2f}x",
                                f"{scenario_discount * 100:.2f}%",
                                (
                                    f"{rf_rate * 100:.2f}% / {beta:.2f} /"
                                    f" {erp * 100:.2f}%"
                                ),
                                f"{cost_of_equity * 100:.2f}%",
                                (
                                    f"${total_debt / 1e9:.2f}B /"
                                    f" ${net_debt / 1e9:.2f}B"
                                ),
                                f"{shares_out / 1e9:.2f}B",
                            ],
                        })
                        st.dataframe(
                            assumptions, use_container_width=True, hide_index=True
                        )
                        st.code(
                            "Revenue_t = Revenue_(t-1) * (1 + revenue_growth)\n"
                            "Net Income_t = Revenue_t * target_net_margin\n"
                            "FCFE_t = Net Income_t * cash_flow_conversion_rate\n"
                            "PV(FCFE_t) = FCFE_t / (1 + discount_rate)^t\n"
                            "Terminal Value = FCFE_N * exit_multiple\n"
                            "DCF Value / Share = (sum(PV of FCFE) + PV(Terminal"
                            " Value)) / Shares",
                            language="text",
                        )
                        dcf_demo = pd.DataFrame({
                            "年度": [f"Year {year}" for year in range(1, 6)],
                            "營收成長率": [g * 100 for g in scenario_growth_schedule],
                            "預估 FCFE ($B)": [fcfe / 1e9 for fcfe in scenario_fcfs],
                            "折現後 FCFE ($B)": [pv / 1e9 for pv in scenario_pvs],
                        })
                        st.dataframe(
                            dcf_demo.style.format({
                                "營收成長率": "{:.1f}%",
                                "預估 FCFE ($B)": "${:.2f}",
                                "折現後 FCFE ($B)": "${:.2f}",
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )
                        demo1, demo2, demo3, demo4, demo5 = st.columns(5)
                        demo1.metric("退出終值", f"${scenario_tv / 1e9:.2f}B")
                        demo2.metric("終值現值", f"${scenario_pv_tv / 1e9:.2f}B")
                        demo3.metric("企業價值 EV", f"${scenario_ev / 1e9:.2f}B")
                        demo4.metric("股東權益價值", f"${scenario_eq / 1e9:.2f}B")
                        demo5.metric("每股內在價值", f"${scenario_value:.2f}")

            else:
                st.error(
                    "❌ 該公司歷史 Free Cash Flow"
                    " 持續為負數、數據缺失或無流通股數，無法進行 DCF 現金流估值。"
                )

            # ==========================================
            # 3.5 公司自由現金流 (FCF) 與歷年成長率視覺化圖表
            # ==========================================
            st.write("---")
            st.subheader(
                "💵 3.5 公司自由現金流 (Free Cash Flow, FCF) 與年增長率 (YoY Growth Rate)"
                " 趨勢"
            )

            tot_rev_ttm = info.get("totalRevenue") or annual_revenue or 0
            fcf_margin = (base_fcf / tot_rev_ttm * 100) if tot_rev_ttm > 0 else 0.0
            
            fcf_per_share = (base_fcf / shares_out) if shares_out > 0 else 0.0

            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric(
                "近 12 個月 FCF (TTM)",
                (
                    f"${base_fcf/1e9:.2f} B"
                    if base_fcf > 1e9
                    else f"${base_fcf/1e6:.2f} M"
                ),
            )
            fc2.metric("每股自由現金流 (FCF/Share)", f"${fcf_per_share:.2f}")
            fc3.metric("自由現金流利潤率 (FCF Margin)", f"{fcf_margin:.2f}%")
            fc4.metric(
                "市價與 FCF 比率 (P/FCF)",
                f"{mkt_cap_val / base_fcf:.2f}x" if base_fcf > 0 else "N/A",
            )

            if len(fcf_history) >= 2 and fcf_dates:
                fcf_chronological = fcf_history[::-1]
                dates_chronological = fcf_dates[::-1]

                fcf_series = pd.Series(fcf_chronological)
                fcf_growth_rates = fcf_series.pct_change() * 100

                fig_fcf = make_subplots(specs=[[{"secondary_y": True}]])

                fig_fcf.add_trace(
                    go.Bar(
                        x=dates_chronological,
                        y=[f / 1e9 for f in fcf_chronological],
                        name="FCF 金額 ($B)",
                        marker_color="#00CC96",
                        opacity=0.85,
                    ),
                    secondary_y=False,
                )

                fig_fcf.add_trace(
                    go.Scatter(
                        x=dates_chronological,
                        y=fcf_growth_rates,
                        name="FCF 年增率 (YoY %)",
                        mode="lines+markers+text",
                        text=[
                            f"{g:.1f}%" if pd.notna(g) else ""
                            for g in fcf_growth_rates
                        ],
                        textposition="top center",
                        line=dict(color="#FFD700", width=3),
                        marker=dict(size=8, color="#FFD700"),
                    ),
                    secondary_y=True,
                )

                fig_fcf.update_layout(
                    title="公司歷年自由現金流 (FCF) 與 YoY 增長率雙軸趨勢圖",
                    xaxis_title="財報年度",
                    height=360,
                    hovermode="x unified",
                    margin=dict(t=50, b=20),
                )

                fig_fcf.update_yaxes(
                    title_text="<b>FCF 金額 (十億美元)</b>", secondary_y=False
                )
                fig_fcf.update_yaxes(
                    title_text="<b>FCF 年增率 (%)</b>", secondary_y=True
                )

                st.plotly_chart(fig_fcf, use_container_width=True)

            # ==========================================
            # 4. 5年長期財務成長率與永續性
            # ==========================================
            st.write("---")
            st.subheader(
                "📊 4. 財務成長率與公司永續性分析 (Growth Rate & Long-term History)"
            )

            growth_view = st.radio(
                "選擇財務數據維度：",
                ["歷年年度數據 (5年長期視角 - YoY)", "單季數據 (短期動能 - QoQ)"],
                index=0,
                horizontal=True,
            )

            df_fin_show = q_fin.T if "單季" in growth_view else fin.T

            if (
                not df_fin_show.empty
                and "Total Revenue" in df_fin_show.columns
            ):
                df_fin_show = df_fin_show.sort_index()
                revenue = df_fin_show["Total Revenue"] / 1e9
                net_inc = df_fin_show.get(
                    "Net Income",
                    df_fin_show.get("Net Income Common Stockholders"),
                )
                if net_inc is not None:
                    net_inc = net_inc / 1e9

                rev_g = revenue.pct_change() * 100
                inc_g = net_inc.pct_change() * 100 if net_inc is not None else None

                fig_g = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=("總營收與淨利金額 (10億美元)", "增長率趨勢 (%)"),
                )

                fig_g.add_trace(
                    go.Bar(x=revenue.index.astype(str), y=revenue, name="Revenue"),
                    row=1,
                    col=1,
                )
                if net_inc is not None:
                    fig_g.add_trace(
                        go.Bar(
                            x=net_inc.index.astype(str),
                            y=net_inc,
                            name="Net Income",
                        ),
                        row=1,
                        col=1,
                    )

                fig_g.add_trace(
                    go.Scatter(
                        x=rev_g.index.astype(str),
                        y=rev_g,
                        name="Revenue Growth %",
                        mode="lines+markers",
                    ),
                    row=1,
                    col=2,
                )
                if inc_g is not None:
                    fig_g.add_trace(
                        go.Scatter(
                            x=inc_g.index.astype(str),
                            y=inc_g,
                            name="Income Growth %",
                            mode="lines+markers",
                        ),
                        row=1,
                        col=2,
                    )

                fig_g.update_layout(height=350, hovermode="x unified")
                st.plotly_chart(fig_g, use_container_width=True)
            else:
                st.warning("⚠️ 缺乏財務成長率數據。")

            # ==========================================
            # 5. 技術面量價、均線趨勢分類 (Strong/Weak Bull) & MACD 背離
            # ==========================================
            st.write("---")
            st.subheader("🎯 5. 技術面量價、均線趨勢分類與 MACD 背離掃描")

            if df_hist.empty:
                st.warning("⚠️ 歷史交易 K 線資料不足，無法繪製技術面指標。")
            else:
                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                missing_cols = [
                    c for c in required_cols if c not in df_hist.columns
                ]
                if missing_cols:
                    st.warning(
                        f"⚠️ 技術面資料缺少必要欄位：{missing_cols}，無法繪製技術圖表。"
                    )
                else:
                    df_hist = df_hist.dropna(subset=required_cols).copy()
                    if len(df_hist) < 20:
                        st.warning("⚠️ 歷史交易 K 線資料不足，無法繪製技術面指標。")
                    else:
                        df_hist["MA10"] = df_hist["Close"].rolling(10).mean()
                        df_hist["MA20"] = df_hist["Close"].rolling(20).mean()
                        df_hist["MA55"] = df_hist["Close"].rolling(55).mean()
                        df_hist["EMA20"] = (
                            df_hist["Close"].ewm(span=20, adjust=False).mean()
                        )
                        df_hist["EMA50"] = (
                            df_hist["Close"].ewm(span=50, adjust=False).mean()
                        )
                        df_hist["RSI14"] = calculate_rsi(df_hist["Close"], 14)
                        df_hist["BB_Middle"] = df_hist["Close"].rolling(20).mean()
                        bb_std = df_hist["Close"].rolling(20).std()
                        df_hist["BB_Upper"] = df_hist["BB_Middle"] + (2 * bb_std)
                        df_hist["BB_Lower"] = df_hist["BB_Middle"] - (2 * bb_std)
                        df_hist["BB_Bandwidth"] = (
                            df_hist["BB_Upper"] - df_hist["BB_Lower"]
                        ) / df_hist["BB_Middle"].replace(0, np.nan)
                        df_hist["BB_%B"] = (
                            df_hist["Close"] - df_hist["BB_Lower"]
                        ) / (df_hist["BB_Upper"] - df_hist["BB_Lower"]).replace(
                            0, np.nan
                        )

                        if len(df_hist) >= 250:
                            df_hist["MA250"] = df_hist["Close"].rolling(250).mean()
                            ma250_valid = True
                            ma250_val = df_hist["MA250"].iloc[-1]
                        else:
                            ma250_valid = False
                            ma250_val = 0.0

                        df_hist, advanced_divergences = (
                            calculate_advanced_indicators_and_signals(df_hist)
                        )

                        bullish_advanced = [
                            signal
                            for signal in advanced_divergences
                            if signal["type"] == "Bullish"
                        ]
                        bearish_advanced = [
                            signal
                            for signal in advanced_divergences
                            if signal["type"] == "Bearish"
                        ]
                        if bullish_advanced:
                            indicators = ", ".join(
                                sorted({signal["indicator"] for signal in bullish_advanced})
                            )
                            st.success(
                                f"🟢 CPD / MFI Bullish Divergence Detected ({indicators})!"
                                " Institutional accumulation suspected at lows."
                            )
                        if bearish_advanced:
                            indicators = ", ".join(
                                sorted({signal["indicator"] for signal in bearish_advanced})
                            )
                            st.warning(
                                f"🔴 CPD / MFI Bearish Divergence Detected ({indicators})!"
                                " Hidden distribution/selling pressure at highs."
                            )

                        trend_status, trend_desc = classify_trend_status(df_hist)

                        # 🛠️ 【修改重點】讓個股分析直接共用掃描儀的 find_dual_divergences 函數
                        # 同時回傳並統一背離標準與檢測結果
                        all_divergences = find_dual_divergences(df_hist, lookback_days=365)
                        if all_divergences:
                            # 由於 find_dual_divergences 格式包含 Signal Type ("Bullish" / "Bearish")
                            div_type = "看多背離 (Bullish)" if all_divergences[-1]["Signal Type"] == "Bullish" else "看空背離 (Bearish)"
                        else:
                            div_type = "Neutral"

                        rsi_divergences = scan_indicator_divergences(
                            df_hist, "RSI14"
                        )
                        if rsi_divergences:
                            rsi_div_type = rsi_divergences[-1]["type"]
                        rsi_value = float(df_hist["RSI14"].iloc[-1])
                        bollinger_state = summarize_bollinger_state(df_hist)

                        vol_ma20 = df_hist["Volume"].rolling(20).mean()
                        curr_vol = df_hist["Volume"].iloc[-1]
                        curr_ret = (
                            df_hist["Close"].iloc[-1] - df_hist["Open"].iloc[-1]
                        ) / df_hist["Open"].iloc[-1]
                        high_20 = (
                            df_hist["High"].rolling(20).max().shift(1).iloc[-1]
                        )

                        if (
                            curr_vol > vol_ma20.iloc[-1] * 1.5
                            and curr_ret > 0.02
                            and df_hist["Close"].iloc[-1] > high_20
                        ):
                            gunshot_signal = True

                        support_level, resistance_level = (
                            find_smart_support_resistance(df_hist)
                        )
                        poc_price = find_volume_poc(df_hist)
                        val_price, vah_price = find_volume_value_area(df_hist)
                        latest_exhaustion_signal = bool(
                            df_hist["Signal_Hist_Exit"].iloc[-1]
                            if "Signal_Hist_Exit" in df_hist.columns
                            and pd.notna(df_hist["Signal_Hist_Exit"].iloc[-1])
                            else False
                        )
                        latest_rsi = (
                            float(df_hist["RSI14"].iloc[-1])
                            if "RSI14" in df_hist.columns
                            and pd.notna(df_hist["RSI14"].iloc[-1])
                            else None
                        )
                        near_resistance = (
                            resistance_level > curr_price > 0
                            and (resistance_level - curr_price) / curr_price <= 0.02
                        )
                        exhaustion_signal = bool(
                            latest_exhaustion_signal
                            or (latest_rsi is not None and latest_rsi >= 70)
                            or near_resistance
                        )
                        atr_risk = calculate_atr_risk_metrics(
                            df_hist,
                            support_level,
                            resistance_level,
                            curr_price,
                            value_area_low=val_price,
                            poc_price=poc_price,
                            exhaustion_signal=exhaustion_signal,
                        )
                        pattern_setups = find_chart_pattern_setups(
                            df_hist,
                            support=support_level,
                            resistance=resistance_level,
                            info=info,
                            financials=fin,
                        )
                        liquidity_sweeps = find_liquidity_sweeps(df_hist)

                        h_max = df_hist["High"].max()
                        l_min = df_hist["Low"].min()
                        fib_382 = h_max - (h_max - l_min) * 0.382
                        fib_500 = h_max - (h_max - l_min) * 0.500
                        fib_618 = h_max - (h_max - l_min) * 0.618

                        m10_val = df_hist["MA10"].iloc[-1]
                        m20_val = df_hist["MA20"].iloc[-1]
                        m55_val = df_hist["MA55"].iloc[-1]

                        st.info(
                            f"📊 **當前趨勢評級：{trend_status}**\n\n{trend_desc}"
                        )
                        rsi_status = (
                            "超買"
                            if rsi_value > 70
                            else "超賣" if rsi_value < 30 else "中性"
                        )
                        st.caption(
                            f"RSI(14): **{rsi_value:.1f} ({rsi_status})** |"
                            " Bollinger %B:"
                            f" **{bollinger_state['percent_b']:.2f}** |"
                            f" BandWidth: **{bollinger_state['bandwidth']:.2%}**"
                            f" |"
                            f" {'⚡ Squeeze' if bollinger_state['squeeze'] else '波動正常'}"
                            f" | {bollinger_state['tag']}"
                        )

                        if gunshot_signal:
                            st.success(
                                "🔥 **觸發 Livermore 第一槍爆發型態**：今日帶量突破近"
                                " 20 日高點！"
                            )
                        else:
                            st.caption("ℹ️ 今日未觸發第一槍帶量突破訊號。")

                        with st.expander(
                            "🔍 點擊查看近 1 年共振背離觸發時間點明細"
                        ):
                            if all_divergences:
                                div_df = pd.DataFrame(all_divergences)[
                                    ["Trigger Date", "Signal Type", "Price", "MACD"]
                                ].rename(columns={
                                    "Trigger Date": "Trigger Date",
                                    "Signal Type": "Signal",
                                    "Price": "Trigger Price ($)",
                                    "MACD": "MACD at Trigger",
                                })
                                div_df["Trigger Date"] = pd.to_datetime(
                                    div_df["Trigger Date"]
                                ).dt.strftime("%Y-%m-%d")
                                st.dataframe(
                                    div_df,
                                    use_container_width=True,
                                    hide_index=True,
                                )
                            else:
                                st.info("近 1 年區間內未偵測到同步背離點位。")

                        with st.expander("📐 RSI(14) 背離與 Bollinger Band 訊號"):
                            if rsi_divergences:
                                rsi_df = pd.DataFrame(rsi_divergences)[
                                    ["date", "type", "price", "indicator"]
                                ].rename(columns={
                                    "date": "Trigger Date",
                                    "type": "Signal",
                                    "price": "Trigger Price ($)",
                                    "indicator": "RSI at Trigger",
                                })
                                rsi_df["Trigger Date"] = pd.to_datetime(
                                    rsi_df["Trigger Date"]
                                ).dt.strftime("%Y-%m-%d")
                                st.dataframe(
                                    rsi_df,
                                    use_container_width=True,
                                    hide_index=True,
                                )
                            else:
                                st.info("近 2 年區間內未偵測到顯著 RSI 背離。")

                        st.subheader("📊 CPD / MFI / Order Flow Delta Divergence Triggers")
                        if advanced_divergences:
                            advanced_divergence_df = pd.DataFrame([
                                {
                                    "Indicator": signal["indicator"],
                                    "Signal": signal["type"],
                                    "Trigger Date": pd.Timestamp(
                                        signal["date"]
                                    ).strftime("%Y-%m-%d"),
                                    "Trigger Price ($)": signal["price"],
                                    "Delta at Trigger": signal[
                                        "delta_at_trigger"
                                    ],
                                    "CPD at Trigger": signal[
                                        "cpd_at_trigger"
                                    ],
                                    "MFI (14) at Trigger": signal[
                                        "mfi_at_trigger"
                                    ],
                                }
                                for signal in advanced_divergences
                            ])
                            st.dataframe(
                                advanced_divergence_df.style.format({
                                    "Trigger Price ($)": "${:.2f}",
                                    "Delta at Trigger": "{:,.0f}",
                                    "CPD at Trigger": "{:,.0f}",
                                    "MFI (14) at Trigger": "{:.1f}",
                                }),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info(
                                "No CPD or MFI divergence trigger was detected in"
                                " the latest 30 bars."
                            )

                        # ==========================================
                        # 6. 關鍵支撐/壓力位與黃金分割
                        # ==========================================
                        st.write("---")
                        st.subheader(
                            "🛡️ 6. 關鍵支撐/壓力位、籌碼密集區 (POC)"
                            " 與斐波那契黃金分割"
                        )

                        k1, k2, k3, k4, k5 = st.columns(5)
                        k1.metric(
                            "波段壓力位 (Resistance)", f"${resistance_level:.2f}"
                        )
                        k2.metric("波段支撐位 (Support)", f"${support_level:.2f}")
                        k3.metric("籌碼密集區 (POC)", f"${poc_price:.2f}")
                        k4.metric("斐波那契 (61.8%)", f"${fib_618:.2f}")
                        k5.metric("目前最新股價", f"${curr_price:.2f}")

                        k6, k7 = st.columns(2)
                        k6.metric(
                            "70% Value Area Low (VAL)",
                            f"${val_price:.2f}" if val_price is not None else "N/A",
                        )
                        k7.metric(
                            "70% Value Area High (VAH)",
                            f"${vah_price:.2f}" if vah_price is not None else "N/A",
                        )

                        st.markdown("#### 多時間週期趨勢與動能共振")
                        intraday_history = pd.DataFrame()
                        try:
                            intraday_history = fetch_intraday_history(symbol)
                        except Exception as error:
                            st.warning(
                                f"ℹ️ 無法取得 1H 多週期資料：{error}。"
                                "以下仍顯示可用的日線分析。"
                            )
                        timeframe_matrix, confluence_score, confluence_label = (
                            calculate_timeframe_confluence(
                                df_hist, intraday_history
                            )
                        )
                        available_timeframes = int(
                            (timeframe_matrix["趨勢"] != "資料不足").sum()
                        )
                        max_confluence_score = available_timeframes * 2
                        st.metric(
                            "多週期確認",
                            (
                                f"{confluence_score:+d} / "
                                f"{max_confluence_score}"
                                if max_confluence_score
                                else "N/A"
                            ),
                            delta=confluence_label,
                        )
                        st.dataframe(
                            timeframe_matrix,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.markdown("#### ATR 動態停損、波動通道與風險報酬")
                        if atr_risk:
                            risk_col1, risk_col2, risk_col3, risk_col4, risk_col5, risk_col6 = st.columns(6)
                            risk_col1.metric(
                                "ATR (14)",
                                f"${atr_risk['atr']:.2f}",
                                delta=(
                                    f"{atr_risk['atr'] / curr_price:.2%} of price"
                                    if curr_price > 0
                                    else None
                                ),
                            )
                            risk_col2.metric(
                                f"結構停損（{atr_risk['structural_source']}）",
                                f"${atr_risk['stop_loss']:.2f}",
                                delta=(
                                    f"{(atr_risk['stop_loss'] / curr_price - 1) * 100:.2f}%"
                                    if curr_price > 0
                                    else None
                                ),
                                delta_color="inverse",
                                help=(
                                    f"選用最近且低於現價的結構位 "
                                    f"${atr_risk['structural_level']:.2f}，"
                                    f"再扣除 {1.5:.1f} ATR 緩衝。"
                                ),
                            )
                            risk_col3.metric(
                                "結構停利目標",
                                (
                                    f"${atr_risk['target_price']:.2f}"
                                    if atr_risk["target_price"] is not None
                                    else "N/A"
                                ),
                                delta=(
                                    "波段壓力位"
                                    if atr_risk["target_price"] is not None
                                    else "上方阻力位未確認"
                                ),
                            )
                            risk_col4.metric(
                                "風險報酬比",
                                (
                                    f"1 : {atr_risk['risk_reward']:.2f}"
                                    if atr_risk["risk_reward"] is not None
                                    else "N/A"
                                ),
                                help="以現價為進場參考、結構停損為風險、上方波段壓力為目標估算；未計入滑價與交易成本。",
                            )
                            risk_col5.metric(
                                "結構追蹤停損",
                                f"${atr_risk['trailing_stop']:.2f}",
                                delta=(
                                    "耗竭訊號收緊"
                                    if atr_risk["exhaustion_signal"]
                                    else "2 × ATR 追蹤"
                                ),
                                help="以結構停損為底線；一般使用現價 - 2 ATR，偵測 MACD 動能退出、RSI 超買或接近阻力時收緊至 1 ATR。",
                            )
                            risk_col6.metric(
                                "ATR 波動通道",
                                (
                                    f"${atr_risk['channel_lower'].iloc[-1]:.2f}"
                                    f" – ${atr_risk['channel_upper'].iloc[-1]:.2f}"
                                ),
                                help="EMA(20) ± 1.5 × ATR(14)。",
                            )
                            st.caption(
                                f"結構參考位 ${atr_risk['structural_level']:.2f}"
                                f"（{atr_risk['structural_source']}）− 1.5 ATR；"
                                f"耗竭訊號：{'是' if atr_risk['exhaustion_signal'] else '否'}。"
                            )
                            st.caption(
                                "此為基於歷史波動的參考情境，非個人化投資建議；"
                                "跳空與流動性不足可能導致實際成交偏離停損價。"
                            )
                        else:
                            st.info("ATR 資料不足，暫無法估算動態停損與風險報酬。")

                        st.markdown("#### 流動性掃單 / Stop-Hunt 偵測")
                        if liquidity_sweeps:
                            sweep_table = pd.DataFrame(
                                liquidity_sweeps[-10:]
                            ).rename(columns={
                                "日期": "日期",
                                "掃流方向": "掃單方向",
                                "被掃關鍵位": "被掃價位",
                                "收盤價": "收盤價",
                                "回收幅度（ATR）": "掃過幅度（ATR）",
                                "Signal_Hist_Turn 配對": "配對 Signal_Hist_Turn",
                                "配對觸發日期": "MACD 觸發日期",
                            })
                            st.dataframe(
                                sweep_table.style.format({
                                    "被掃價位": "${:.2f}",
                                    "收盤價": "${:.2f}",
                                    "掃過幅度（ATR）": "{:.2f}",
                                }),
                                use_container_width=True,
                                hide_index=True,
                            )
                            paired_sweeps = sum(
                                bool(event["Signal_Hist_Turn 配對"])
                                for event in liquidity_sweeps
                            )
                            st.caption(
                                f"掃描最近 {min(len(df_hist), 120)} 根日線："
                                "價格刺穿前 20 根 K 線高 / 低點至少 0.1 ATR，"
                                "且同根 K 線收回關鍵位內；"
                                f"共 {len(liquidity_sweeps)} 個候選事件，"
                                f"{paired_sweeps} 個在前後 2 根 K 線內配對 "
                                "Signal_Hist_Turn。候選訊號需人工確認。"
                            )
                        else:
                            st.info(
                                "掃描區間內未偵測到符合條件的流動性掃單反轉事件。"
                            )

                        st.markdown("#### Master & Technical Patterns Scanner")
                        if pattern_setups:
                            pattern_table = pd.DataFrame(pattern_setups)
                            st.dataframe(
                                pattern_table.style.format({
                                    "信心分數": "{:.0f}%",
                                }),
                                use_container_width=True,
                                hide_index=True,
                            )
                            forming_count = sum(
                                item["狀態"] == "形成中"
                                for item in pattern_setups
                            )
                            triggered_count = sum(
                                item["狀態"] == "突破觸發"
                                for item in pattern_setups
                            )
                            st.caption(
                                f"掃描到 {len(pattern_setups)} 個符合最低信心門檻的型態；"
                                f"{triggered_count} 個突破觸發、{forming_count} 個形成中。"
                                "型態辨識為 OHLCV 啟發式篩選，需結合基本面及人工覆核。"
                            )
                        else:
                            st.info(
                                "目前資料中沒有信心分數達 45% 的大師型態或經典圖表型態。"
                            )

            # ==========================================
            # 7. 機構級全方位分析報告
            # ==========================================
            st.write("---")
            st.subheader(
                "📝 7. 全方位技術面與機構交易決策報告 (Institutional Analysis"
                " Report)"
            )

            narrative_output = generate_technical_narrative(
                symbol,
                curr_price,
                m10_val,
                m20_val,
                m55_val,
                ma250_val,
                gunshot_signal,
                div_type,
                support_level,
                resistance_level,
                fib_618,
                poc_price,
                trend_status,
                trend_desc,
                ma250_valid=ma250_valid,
                rsi_value=rsi_value,
                rsi_div_type=rsi_div_type,
                bollinger_state=bollinger_state,
                macd_divergence=bool(all_divergences),
            )
            st.markdown(narrative_output)

            # ==========================================
            # 8. 綜合投資評級 Engine
            # ==========================================
            st.write("---")
            st.subheader(
                "🏆 8. 綜合投資評級與實戰決策 (Factor-Based Rating Engine)"
            )

            fund_score = 35 if (base_val and base_val > curr_price) else 20
            tech_score = (
                25
                if "Bull" in trend_status or gunshot_signal or "Bullish" in div_type
                else 15
            )
            total_score = fund_score + tech_score

            if total_score >= 50:
                rating_str = "🟢 買入 / 逢低佈局 (BUY / ACCUMULATE)"
            elif total_score >= 35:
                rating_str = "🟡 觀望 / 保持中立 (HOLD)"
            else:
                rating_str = "🔴 賣出 / 減碼避險 (SELL / REDUCE)"

            st.markdown(f"### 綜合評級：{rating_str}")
            st.markdown(f"#### 機構綜合評分：**{total_score} / 100**")

            st.markdown(f"""
            **【多因素量化評分明細】**
            * **基本面得分 ({fund_score}/50)**：DCF估值 ({'✓ 高於現價' if base_val and base_val > curr_price else '✗ 低於現價'}) | Z-Score ({z_status})
            * **技術面得分 ({tech_score}/50)**：趨勢架構 ({trend_status}) | 背離 ({div_type}) | Livermore第一槍 ({'✓ 觸發' if gunshot_signal else '✗ 未觸發'})
            
            💡 **最終操作建議**：對 {symbol} 給予 **{rating_str}** 評級。關鍵風險控管區間落於支撐位 **${support_level:.2f}**。
            """)

            # ==========================================
            # 9. Interactive Plotly Chart
            # ==========================================
            st.write("---")
            st.subheader(
                "📈 9. Price / MFI / Order Flow Delta / CPD Chart"
            )

            if (
                not df_hist.empty
                and len(df_hist) >= 20
                and {"MFI14", "Delta", "CPD"}.issubset(df_hist.columns)
            ):
                fig_k = make_subplots(
                    rows=4,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.025,
                    row_heights=[0.50, 0.15, 0.15, 0.20],
                    subplot_titles=(
                        "Price / Signal Turns / Divergences",
                        "MFI (14)",
                        "Order Flow Delta",
                        "CPD (Cumulative Volume Delta)",
                    ),
                )
                fig_k.add_trace(
                    go.Candlestick(
                        x=df_hist.index,
                        open=df_hist["Open"],
                        high=df_hist["High"],
                        low=df_hist["Low"],
                        close=df_hist["Close"],
                        name="K Line",
                    ),
                    row=1,
                    col=1,
                )
                fig_k.add_trace(
                    go.Scatter(
                        x=df_hist.index,
                        y=df_hist["BB_Upper"],
                        name="BB Upper",
                        line=dict(color="#B388FF", width=1),
                    ),
                    row=1,
                    col=1,
                )
                fig_k.add_trace(
                    go.Scatter(
                        x=df_hist.index,
                        y=df_hist["BB_Lower"],
                        name="BB Lower",
                        line=dict(color="#B388FF", width=1),
                        fill="tonexty",
                        fillcolor="rgba(179, 136, 255, 0.12)",
                    ),
                    row=1,
                    col=1,
                )
                fig_k.add_trace(
                    go.Scatter(
                        x=df_hist.index,
                        y=df_hist["BB_Middle"],
                        name="BB Middle",
                        line=dict(color="#CE93D8", width=1, dash="dot"),
                    ),
                    row=1,
                    col=1,
                )

                fig_k.add_trace(
                    go.Scatter(
                        x=df_hist.index,
                        y=df_hist["MA10"],
                        name="MA10 (Yellow)",
                        line=dict(color="#FFEB3B", width=1.2),
                    ),
                    row=1,
                    col=1,
                )
                fig_k.add_trace(
                    go.Scatter(
                        x=df_hist.index,
                        y=df_hist["MA20"],
                        name="MA20 (Blue)",
                        line=dict(color="#2196F3", width=1.5),
                    ),
                    row=1,
                    col=1,
                )
                fig_k.add_trace(
                    go.Scatter(
                        x=df_hist.index,
                        y=df_hist["MA55"],
                        name="MA55 (Green)",
                        line=dict(color="#4CAF50", width=1.8),
                    ),
                    row=1,
                    col=1,
                )
                fig_k.add_trace(
                    go.Scatter(
                        x=df_hist.index,
                        y=df_hist["EMA20"],
                        name="EMA20",
                        line=dict(color="#00D4FF", width=1.8),
                    ),
                    row=1,
                    col=1,
                )
                fig_k.add_trace(
                    go.Scatter(
                        x=df_hist.index,
                        y=df_hist["EMA50"],
                        name="EMA50",
                        line=dict(color="#FF9F1C", width=1.8),
                    ),
                    row=1,
                    col=1,
                )

                if ma250_valid:
                    fig_k.add_trace(
                        go.Scatter(
                            x=df_hist.index,
                            y=df_hist["MA250"],
                            name="MA250 (Red)",
                            line=dict(color="#F44336", width=2.2),
                        ),
                        row=1,
                        col=1,
                    )

                if atr_risk:
                    fig_k.add_trace(
                        go.Scatter(
                            x=df_hist.index,
                            y=atr_risk["channel_upper"],
                            name="EMA20 + 1.5 ATR",
                            line=dict(color="rgba(255, 193, 7, 0.45)", width=1),
                        ),
                        row=1,
                        col=1,
                    )
                    fig_k.add_trace(
                        go.Scatter(
                            x=df_hist.index,
                            y=atr_risk["channel_lower"],
                            name="EMA20 - 1.5 ATR",
                            line=dict(color="rgba(255, 193, 7, 0.45)", width=1),
                            fill="tonexty",
                            fillcolor="rgba(255, 193, 7, 0.06)",
                        ),
                        row=1,
                        col=1,
                    )

                fig_k.add_hline(
                    y=support_level,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Support",
                    row=1,
                    col=1,
                )
                fig_k.add_hline(
                    y=resistance_level,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Resistance",
                    row=1,
                    col=1,
                )
                fig_k.add_hline(
                    y=poc_price,
                    line_color="gold",
                    annotation_text="POC Area",
                    row=1,
                    col=1,
                )
                if val_price is not None and vah_price is not None:
                    fig_k.add_hline(
                        y=val_price,
                        line_dash="dot",
                        line_color="#7E57C2",
                        annotation_text="70% VAL",
                        row=1,
                        col=1,
                    )
                    fig_k.add_hline(
                        y=vah_price,
                        line_dash="dot",
                        line_color="#7E57C2",
                        annotation_text="70% VAH",
                        row=1,
                        col=1,
                    )
                if atr_risk:
                    fig_k.add_hline(
                        y=atr_risk["stop_loss"],
                        line_dash="dash",
                        line_color="#FF6B6B",
                        annotation_text="ATR Dynamic Stop",
                        row=1,
                        col=1,
                    )
                    fig_k.add_hline(
                        y=atr_risk["trailing_stop"],
                        line_dash="dot",
                        line_color="#FF8A65",
                        annotation_text="Structural Trailing Stop",
                        row=1,
                        col=1,
                    )
                    if atr_risk["target_price"] is not None:
                        fig_k.add_hline(
                            y=atr_risk["target_price"],
                            line_dash="dot",
                            line_color="#66BB6A",
                            annotation_text="Resistance Target",
                            row=1,
                            col=1,
                        )

                entry_mask = df_hist["Signal_Hist_Turn"].fillna(False).astype(bool)
                exit_mask = df_hist["Signal_Hist_Exit"].fillna(False).astype(bool)
                fig_k.add_trace(
                    go.Scatter(
                        x=df_hist.index[entry_mask],
                        y=df_hist.loc[entry_mask, "Close"],
                        mode="markers",
                        name="Signal_Hist_Turn Entry",
                        marker=dict(color="#2A9D8F", symbol="triangle-up", size=11),
                    ),
                    row=1,
                    col=1,
                )
                fig_k.add_trace(
                    go.Scatter(
                        x=df_hist.index[exit_mask],
                        y=df_hist.loc[exit_mask, "Close"],
                        mode="markers",
                        name="Histogram Exit",
                        marker=dict(color="#E76F51", symbol="triangle-down", size=11),
                    ),
                    row=1,
                    col=1,
                )

                for divergence in advanced_divergences:
                    divergence_color = (
                        "#2A9D8F" if divergence["type"] == "Bullish" else "#E76F51"
                    )
                    fig_k.add_trace(
                        go.Scatter(
                            x=[divergence["first_date"], divergence["date"]],
                            y=[divergence["first_price"], divergence["price"]],
                            mode="lines+markers",
                            name=(
                                f"{divergence['indicator']} "
                                f"{divergence['type']} Divergence"
                            ),
                            line=dict(color=divergence_color, width=2),
                            marker=dict(size=9, symbol="diamond"),
                        ),
                        row=1,
                        col=1,
                    )
                    indicator_row = 4 if divergence["indicator"] == "CPD" else 2
                    fig_k.add_trace(
                        go.Scatter(
                            x=[divergence["first_date"], divergence["date"]],
                            y=[
                                divergence["first_indicator_value"],
                                divergence["indicator_value"],
                            ],
                            mode="lines+markers",
                            name=f"{divergence['indicator']} divergence pivots",
                            line=dict(color=divergence_color, width=2),
                            marker=dict(size=8, symbol="diamond"),
                            showlegend=False,
                        ),
                        row=indicator_row,
                        col=1,
                    )

                for div in all_divergences:
                    color_tag = "green" if div["Signal Type"] == "Bullish" else "red"
                    fig_k.add_annotation(
                        x=div["Trigger Date"],
                        y=div["Price"],
                        text=" Bull" if div["Signal Type"] == "Bullish" else " Bear",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor=color_tag,
                        ax=0,
                        ay=-25 if div["Signal Type"] == "Bullish" else 25,
                        row=1,
                        col=1,
                    )

                fig_k.add_trace(
                    go.Scatter(
                        x=df_hist.index,
                        y=df_hist["MFI14"],
                        name="MFI (14)",
                        line=dict(color="#00BFA5", width=2),
                    ),
                    row=2,
                    col=1,
                )
                fig_k.add_hline(
                    y=80,
                    line_dash="dash",
                    line_color="#EF5350",
                    annotation_text="Overbought 80",
                    row=2,
                    col=1,
                )
                fig_k.add_hline(
                    y=20,
                    line_dash="dash",
                    line_color="#42A5F5",
                    annotation_text="Oversold 20",
                    row=2,
                    col=1,
                )

                fig_k.add_trace(
                    go.Bar(
                        x=df_hist.index,
                        name="Order Flow Delta",
                        y=df_hist["Delta"],
                        marker_color=np.where(
                            df_hist["Delta"] >= 0, "#2A9D8F", "#E76F51"
                        ),
                    ),
                    row=3,
                    col=1,
                )
                fig_k.add_trace(
                    go.Scatter(
                        x=df_hist.index,
                        y=df_hist["CPD"],
                        name="CPD",
                        line=dict(color="#00CFE8", width=2),
                    ),
                    row=4,
                    col=1,
                )

                scanner_signal = st.session_state.get("selected_scanner_signal")
                if scanner_signal and scanner_signal.get("Ticker") == symbol:
                    trigger_date = pd.Timestamp(scanner_signal["Trigger Date"])
                    signal_color = (
                        "green"
                        if scanner_signal["Signal Type"] == "Bullish"
                        else "red"
                    )
                    fig_k.add_vline(
                        x=trigger_date,
                        line_dash="dash",
                        line_color=signal_color,
                        annotation_text=f"{scanner_signal['Signal Type']} trigger",
                        row=1,
                        col=1,
                    )
                    pivot_dates = [
                        pd.Timestamp(scanner_signal["First Date"]),
                        trigger_date,
                    ]
                    fig_k.add_trace(
                        go.Scatter(
                            x=pivot_dates,
                            y=[
                                scanner_signal["First Price"],
                                scanner_signal["Second Price"],
                            ],
                            mode="lines+markers",
                            name="Dual divergence price pivots",
                            line=dict(color=signal_color, width=3),
                            marker=dict(size=8),
                        ),
                        row=1,
                        col=1,
                    )
                fig_k.update_yaxes(range=[0, 100], row=2, col=1)
                fig_k.update_layout(
                    height=900,
                    xaxis_rangeslider_visible=False,
                    margin=dict(t=35, b=20),
                )
                st.plotly_chart(fig_k, use_container_width=True)
            elif not df_hist.empty:
                st.info("Advanced price-volume indicators are unavailable for this ticker.")

            # ==========================================
            # 10. Embedded TradingView Chart
            # ==========================================
            st.write("---")
            st.subheader(
                "📺 10. TradingView 官方進階技術圖表 (SMA 10/20/55/250 & 白色"
                " Volume MA55)"
            )
            st.markdown(
                '<div style="display:flex;gap:18px;flex-wrap:wrap;font-weight:600;">'
                '<span style="color:#FFD166;">● MA10</span>'
                '<span style="color:#00D4FF;">● MA20</span>'
                '<span style="color:#7CFF6B;">● MA55</span>'
                '<span style="color:#FF5C8A;">● MA250</span>'
                "</div>",
                unsafe_allow_html=True,
            )
            tv_symbol = map_symbol_for_tradingview(symbol)
            tv_html = f"""
            <div class="tradingview-widget-container" style="height:550px;width:100%;">
              <div id="tradingview_chart" style="height:550px;width:100%;"></div>
              <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
              <script type="text/javascript">
              new TradingView.widget({{
                "autosize": true,
                "symbol": "{tv_symbol}",
                "interval": "D",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "zh_TW",
                "toolbar_bg": "#f1f3f6",
                "enable_publishing": false,
                "allow_symbol_change": true,
                "container_id": "tradingview_chart",
                "studies": [
                  {{
                    "id": "MASimple@tv-basicstudies",
                    "inputs": {{ "length": 10 }},
                    "overrides": {{ "Plot.color": "#FFD166", "Plot.linewidth": 3 }}
                  }},
                  {{
                    "id": "MASimple@tv-basicstudies",
                    "inputs": {{ "length": 20 }},
                    "overrides": {{ "Plot.color": "#00D4FF", "Plot.linewidth": 3 }}
                  }},
                  {{
                    "id": "MASimple@tv-basicstudies",
                    "inputs": {{ "length": 55 }},
                    "overrides": {{ "Plot.color": "#7CFF6B", "Plot.linewidth": 3 }}
                  }},
                  {{
                    "id": "MASimple@tv-basicstudies",
                    "inputs": {{ "length": 250 }},
                    "overrides": {{ "Plot.color": "#FF5C8A", "Plot.linewidth": 3 }}
                  }},
                  {{
                    "id": "Volume@tv-basicstudies",
                    "inputs": {{ "showMA": true, "maLength": 55 }}
                  }}
                ],
                "studies_overrides": {{
                    "volume.volume ma.color": "#FFFFFF",
                    "volume.volume ma.linewidth": 2
                }}
              }});
              </script>
            </div>
            """
            components.html(tv_html, height=560)

            # ==========================================
            # 11. 選擇權深度分析與大單異常籌碼掃描
            # ==========================================
            st.write("---")
            st.subheader(
                "11. 選擇權深度分析與大單異常籌碼掃描 (Options Analysis & Unusual"
                " Options Flow)"
            )
            options_summary = fetch_options_analysis(symbol, curr_price)
            if not options_summary.get("available"):
                st.info(
                    f"ℹ️ {options_summary.get('message', '此標的沒有可用的選擇權市場資料。')}"
                    " 非美股（例如 .TW、.HK）通常不提供 yfinance"
                    " 可讀取的選擇權鏈；其餘股票分析不受影響。"
                )
            else:
                pcr_volume = options_summary.get("pcr_volume", np.nan)
                pcr_oi = options_summary.get("pcr_oi", np.nan)
                iv_summary = options_summary.get("iv_summary", {})
                overview_1, overview_2, overview_3, overview_4 = st.columns(4)
                overview_1.metric(
                    "Put/Call Ratio (Volume)",
                    (
                        f"{pcr_volume:.2f}"
                        if pd.notna(pcr_volume)
                        else "N/A"
                    ),
                )
                overview_2.metric(
                    "Put/Call Ratio (OI)",
                    f"{pcr_oi:.2f}" if pd.notna(pcr_oi) else "N/A",
                )
                overview_3.metric(
                    "Max Pain Price",
                    (
                        f"${options_summary['max_pain']:.2f}"
                        if pd.notna(options_summary.get("max_pain"))
                        else "N/A"
                    ),
                )
                overview_4.metric(
                    "NTM Put IV - Call IV",
                    (
                        f"{iv_summary.get('Put-Call IV Skew', np.nan) * 100:.2f}%"
                        if pd.notna(iv_summary.get("Put-Call IV Skew"))
                        else "N/A"
                    ),
                )

                st.caption(
                    f"掃描到期日：{', '.join(options_summary['expirations'])} |"
                    f" NTM Call IV:"
                    f" {iv_summary.get('Call IV', np.nan) * 100:.2f}% | NTM Put"
                    f" IV: {iv_summary.get('Put IV', np.nan) * 100:.2f}%"
                    if pd.notna(iv_summary.get("Call IV"))
                    and pd.notna(iv_summary.get("Put IV"))
                    else (
                        f"掃描到期日：{', '.join(options_summary['expirations'])} |"
                        " NTM IV 資料不足"
                    )
                )

                st.markdown(
                    "##### 🔎 Smart Money Unusual Flow（Volume > 2.5 × OI 且 Volume"
                    " > 1,000）"
                )
                unusual_flow = options_summary.get(
                    "unusual", pd.DataFrame()
                ).copy()
                if unusual_flow.empty:
                    st.info(
                        "近 45 天到期選擇權中沒有符合異常大單條件的合約。"
                    )
                else:
                    unusual_flow["Implied Volatility"] = (
                        unusual_flow["Implied Volatility"] * 100
                    )
                    unusual_flow["Vol/OI Ratio"] = unusual_flow[
                        "Vol/OI Ratio"
                    ].replace(np.inf, np.nan)
                    st.dataframe(
                        unusual_flow.style.format({
                            "Strike Price": "${:.2f}",
                            "Volume": "{:.0f}",
                            "Open Interest": "{:.0f}",
                            "Vol/OI Ratio": "{:.2f}x",
                            "Implied Volatility": "{:.2f}%",
                            "Estimated Trade Value": "${:,.0f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.info(
                    "🤖 **AI Flow Interpretation**："
                    f" {interpret_options_flow(options_summary)}"
                )

    except Exception as e:
        st.error(f"分析時發生未預期錯誤: {e}")
