import json
import datetime
import urllib.request
from pathlib import Path
import time
requests = None # placeholder if needed, using urllib
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

    try:
        hist = t.history(period="2y")
    except Exception:
        hist = pd.DataFrame()

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

    info = {}
    for _ in range(2):
        try:
            fetched_info = t.info
            if isinstance(fetched_info, dict) and len(fetched_info) > 5:
                info = fetched_info.copy()
                break
        except Exception:
            time.sleep(0.3)

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
                info["totalRevenue"] = revenue
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
        except Exception:
            pass

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


def scan_all_macd_divergences(df, window=5, min_gap_days=10):
    """獨立掃描 MACD 背離"""
    if len(df) < 60 or "MACD" not in df.columns:
        return []

    raw_divergences = []
    prices = df["Close"].values
    macds = df["MACD"].values
    dates = df.index
    n = len(df)

    piv_lows = []
    piv_highs = []

    for i in range(window, n - window):
        if all(prices[i] <= prices[i - j] for j in range(1, window + 1)) and all(
            prices[i] <= prices[i + j] for j in range(1, window + 1)
        ):
            piv_lows.append((i, dates[i], prices[i], macds[i]))

        if all(prices[i] >= prices[i - j] for j in range(1, window + 1)) and all(
            prices[i] >= prices[i + j] for j in range(1, window + 1)
        ):
            piv_highs.append((i, dates[i], prices[i], macds[i]))

    for k in range(1, len(piv_lows)):
        i2, d2, p2, m2 = piv_lows[k]
        i1, d1, p1, m1 = piv_lows[k - 1]
        if (i2 - i1) <= 60:
            if p2 < p1 and m2 > m1:
                raw_divergences.append({
                    "index": i2,
                    "date": d2,
                    "type": "看多背離 (Bullish)",
                    "price": p2,
                    "macd": m2,
                })

    for k in range(1, len(piv_highs)):
        i2, d2, p2, m2 = piv_highs[k]
        i1, d1, p1, m1 = piv_highs[k - 1]
        if (i2 - i1) <= 60:
            if p2 > p1 and m2 < m1:
                raw_divergences.append({
                    "index": i2,
                    "date": d2,
                    "type": "看空背離 (Bearish)",
                    "price": p2,
                    "macd": m2,
                })

    raw_divergences.sort(key=lambda x: x["index"])

    filtered_divergences = []
    for div in raw_divergences:
        if not filtered_divergences:
            filtered_divergences.append(div)
        else:
            last_div = filtered_divergences[-1]
            if (
                div["index"] - last_div["index"]
            ) >= min_gap_days or div["type"] != last_div["type"]:
                filtered_divergences.append(div)
            else:
                filtered_divergences[-1] = div

    return filtered_divergences


def scan_indicator_divergences(df, indicator_column, window=5, min_gap_days=10):
    """獨立掃描指定的指標背離（如 RSI）"""
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
    result = calculate_synchronized_macd(result)
    return result, []


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
    return (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2.0


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
        return "🚀 強勢多頭 (Strong Bull)", f"完美多頭排列 (Price ${curr_p:.2f} > MA10 > MA20 > MA55)，月線向上斜率正向。"
    elif curr_p > ma20 and ma20 > ma55 and not is_perfect_bull:
        return "📈 弱勢多頭 / 回檔整理 (Weak Bull)", "中期多頭格局未變 (Price > MA20 > MA55)，多頭回檔健康整理。"
    elif is_perfect_bear and ma20_slope < 0:
        return "🩸 強勢空頭 (Strong Bear)", f"完美空頭排列 (Price ${curr_p:.2f} < MA10 < MA20 < MA55)，下行壓力大。"
    elif curr_p < ma20 and ma20 < ma55 and not is_perfect_bear:
        return "📉 弱勢空頭 / 反彈波 (Weak Bear)", "中期趨勢偏空 (Price < MA20 < MA55)，空頭架構下的弱勢反彈。"
    else:
        return "⚖️ 區間震盪整理 (Consolidation)", f"均線交錯糾結於 ${ma20:.2f} ~ ${ma55:.2f} 區間，等待突破。"


def map_symbol_for_tradingview(symbol_input):
    sym = to_yfinance_symbol(symbol_input)
    if sym.endswith(".TW") or sym.endswith(".TWO"):
        return f"TPE:{sym.split('.')[0]}"
    elif sym.endswith(".HK"):
        return f"HKEX:{sym.split('.')[0].zfill(4)}"
    return sym.replace(".", "")


def get_granular_industry_benchmarks(industry_str, sector_str):
    ind = (industry_str or "").lower()
    if "semicon" in ind:
        return "Semiconductors & AI Chips", {"PE": 35.0, "PS": 10.0, "PEG": 1.4, "GM": 62.0, "NM": 28.0, "ROE": 25.0}
    elif "software" in ind or "cloud" in ind:
        return "Software & Cloud", {"PE": 38.0, "PS": 9.0, "PEG": 1.6, "GM": 72.0, "NM": 22.0, "ROE": 20.0}
    return f"{sector_str} - General", {"PE": 20.0, "PS": 2.5, "PEG": 1.3, "GM": 35.0, "NM": 10.0, "ROE": 15.0}


# ==========================================
# Main App Execution
# ==========================================
if symbol and app_view == "Stock Analysis":
    try:
        info, df_hist, bs, fin, cf_df, q_fin, q_bs, q_cf = fetch_ticker_data(symbol)

        if df_hist.empty and not info.get("currentPrice"):
            st.error(f"❌ 無法取得股票資料或無效代碼: {symbol}")
        else:
            curr_price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or (df_hist["Close"].iloc[-1] if not df_hist.empty else 0.0)
            )
            price_low = info.get("fiftyTwoWeekLow")
            price_high = info.get("fiftyTwoWeekHigh")
            analyst_target = info.get("targetMeanPrice") or info.get("targetMedianPrice")
            implied_upside = (
                ((analyst_target / curr_price) - 1) * 100
                if analyst_target and curr_price
                else None
            )

            t_pe = info.get("trailingPE")
            f_pe = info.get("forwardPE")
            t_pe_str = f"{t_pe:.2f}" if isinstance(t_pe, (int, float)) and pd.notna(t_pe) else "N/A"
            f_pe_str = f"{f_pe:.2f}" if isinstance(f_pe, (int, float)) and pd.notna(f_pe) else "N/A"

            fcards = st.columns(4)
            fcards[0].metric("Current Price", f"${curr_price:.2f}" if curr_price else "N/A")
            fcards[1].metric("Trailing / Forward P/E", f"{t_pe_str} / {f_pe_str}")
            fcards[2].metric("52-Week Range", f"${price_low:.2f} - ${price_high:.2f}" if price_low and price_high else "N/A")
            fcards[3].metric("Analyst Target / Upside", f"${analyst_target:.2f}" if analyst_target else "N/A", delta=f"{implied_upside:+.1f}%" if implied_upside else None)

            # 1. Market Sentiment & VIX Filter
            st.subheader("🌐 1. 全球市場情緒與 VIX 抄底濾網 (Market Sentiment & VIX Filter)")
            vix_val = get_vix_value()
            cnn_fng_val, cnn_fng_status = get_cnn_fear_and_greed()
            v1, v2, v3 = st.columns(3)
            v1.metric("VIX 恐慌指數", f"{vix_val:.2f}", delta=">25 極度恐慌" if vix_val > 25 else "正常")
            v2.metric("CNN Fear & Greed Index", f"{cnn_fng_val} ({cnn_fng_status})" if cnn_fng_val else cnn_fng_status)
            if cnn_fng_val is not None and vix_val >= 25 and cnn_fng_val <= 40:
                v3.success("🟢 觸發黃金抄底訊號 (VIX > 25 且 F&G < 40)！")
            else:
                v3.info("🟡 市場情緒平穩，未達極度恐慌黃金買點。")

            st.write("---")

            # 5. Technicals & Independent Divergences
            st.subheader("🎯 5. 技術面量價、均線趨勢分類與 MACD / RSI 獨立背離掃描")
            if not df_hist.empty and len(df_hist) >= 20:
                df_hist["MA10"] = df_hist["Close"].rolling(10).mean()
                df_hist["MA20"] = df_hist["Close"].rolling(20).mean()
                df_hist["MA55"] = df_hist["Close"].rolling(55).mean()
                df_hist["RSI14"] = calculate_rsi(df_hist["Close"], 14)
                df_hist = calculate_synchronized_macd(df_hist)
                
                trend_status, trend_desc = classify_trend_status(df_hist)

                # 獨立掃描 MACD 與 RSI 背離
                all_divergences = scan_all_macd_divergences(df_hist)
                rsi_divergences = scan_indicator_divergences(df_hist, "RSI14")

                st.info(f"📊 **當前趨勢評級：{trend_status}**\n\n{trend_desc}")

                with st.expander("🔍 獨立查看 MACD 與 RSI 背離記錄"):
                    col_mac, col_rsi = st.columns(2)
                    with col_mac:
                        st.markdown("##### MACD Divergences")
                        if all_divergences:
                            st.dataframe(pd.DataFrame(all_divergences)[["date", "type", "price", "macd"]], use_container_width=True, hide_index=True)
                        else:
                            st.caption("無 MACD 背離")
                    with col_rsi:
                        st.markdown("##### RSI(14) Divergences")
                        if rsi_divergences:
                            st.dataframe(pd.DataFrame(rsi_divergences)[["date", "type", "price", "indicator"]], use_container_width=True, hide_index=True)
                        else:
                            st.caption("無 RSI 背離")

            # 9. Interactive Plotly Chart (含雙信號同日共振提示)
            st.write("---")
            st.subheader("📈 9. Price / MACD / RSI / Order Flow Delta Chart")
            if not df_hist.empty and len(df_hist) >= 20:
                fig_k = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.75, 0.25],
                    subplot_titles=("Price & Independent Divergences (Both Hit Highlight)", "RSI (14)"),
                )
                fig_k.add_trace(go.Candlestick(x=df_hist.index, open=df_hist["Open"], high=df_hist["High"], low=df_hist["Low"], close=df_hist["Close"], name="K Line"), row=1, col=1)

                # 🛠️ 日期比對：同日觸發顯示 "⭐ BOTH HIT (MACD+RSI)"
                macd_dates = {pd.Timestamp(d["date"]).normalize(): d for d in all_divergences}
                rsi_dates = {pd.Timestamp(d["date"]).normalize(): d for d in rsi_divergences}

                for div in all_divergences:
                    d_norm = pd.Timestamp(div["date"]).normalize()
                    is_both = d_norm in rsi_dates
                    label = "⭐ BOTH HIT (MACD+RSI)" if is_both else ("MACD Bull" if "看多" in div["type"] else "MACD Bear")
                    color = "green" if "看多" in div["type"] else "red"
                    fig_k.add_annotation(x=div["date"], y=div["price"], text=label, showarrow=True, arrowhead=2, arrowcolor=color, ax=0, ay=-35, row=1, col=1)

                for div in rsi_divergences:
                    d_norm = pd.Timestamp(div["date"]).normalize()
                    if d_norm not in macd_dates:
                        color = "green" if "看多" in div["type"] else "orange"
                        fig_k.add_annotation(x=div["date"], y=div["price"], text="RSI Bull" if "看多" in div["type"] else "RSI Bear", showarrow=True, arrowhead=2, arrowcolor=color, ax=0, ay=-20, row=1, col=1)

                fig_k.add_trace(go.Scatter(x=df_hist.index, y=df_hist["RSI14"], name="RSI (14)", line=dict(color="#00BFA5", width=2)), row=2, col=1)
                fig_k.add_hline(y=70, line_dash="dash", line_color="#EF5350", row=2, col=1)
                fig_k.add_hline(y=30, line_dash="dash", line_color="#42A5F5", row=2, col=1)

                fig_k.update_layout(height=750, xaxis_rangeslider_visible=False, margin=dict(t=35, b=20))
                st.plotly_chart(fig_k, use_container_width=True)

            # 10. TradingView
            st.write("---")
            st.subheader("📺 10. TradingView 官方進階技術圖表")
            tv_symbol = map_symbol_for_tradingview(symbol)
            tv_html = f"""
            <div class="tradingview-widget-container" style="height:550px;width:100%;">
              <div id="tradingview_chart" style="height:550px;width:100%;"></div>
              <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
              <script type="text/javascript">
              new TradingView.widget({{
                "autosize": true, "symbol": "{tv_symbol}", "interval": "D", "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "zh_TW", "container_id": "tradingview_chart"
              }});
              </script>
            </div>
            """
            components.html(tv_html, height=560)

    except Exception as e:
        st.error(f"分析時發生未預期錯誤: {e}")
