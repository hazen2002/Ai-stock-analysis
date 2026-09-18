import json
import urllib.request
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="Pro Stock Analysis & Watchlist Platform",
    page_icon="⚡",
    layout="wide"
)

# ==========================================
# 0. Watchlist (自訂觀察清單 - Session State)
# ==========================================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META"]

st.sidebar.title("⭐ 我的觀察清單 (Watchlist)")

new_symbol = st.sidebar.text_input("新增股票代碼:", "").strip().upper()
if st.sidebar.button("➕ 加入清單"):
    if new_symbol and new_symbol not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_symbol)
        st.sidebar.success(f"已加入 {new_symbol}")

selected_from_watchlist = st.sidebar.selectbox(
    "快速切換觀察清單：",
    options=["-- 請選擇 --"] + st.session_state.watchlist,
    index=0
)

st.sidebar.write("---")
st.sidebar.write("📜 **目前清單股票：**")
for w_sym in list(st.session_state.watchlist):
    c_w1, c_w2 = st.sidebar.columns([3, 1])
    c_w1.write(f"• **{w_sym}**")
    if c_w2.button("❌", key=f"del_{w_sym}"):
        st.session_state.watchlist.remove(w_sym)
        st.rerun()

st.title("⚡ Livermore & Lynch 機構級股票分析平台")

col_search, _ = st.columns([1, 2])
with col_search:
    manual_symbol = st.text_input("輸入股票代碼 (例: NVDA, AAPL, TSLA, 2330.TW):", "NVDA").strip().upper()

symbol = selected_from_watchlist if selected_from_watchlist != "-- 請選擇 --" else manual_symbol

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
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/140.0 Safari/537.36",
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

@st.cache_data(ttl=14400)
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

@st.cache_data(ttl=1800)
def fetch_ticker_data(symbol_str):
    t = yf.Ticker(symbol_str)
    info = t.info or {}
    hist = t.history(period="2y")
    bs = t.balance_sheet
    fin = t.financials
    cf = t.cashflow
    q_fin = t.quarterly_financials
    q_bs = t.quarterly_balance_sheet
    q_cf = t.quarterly_cashflow
    return info, hist, bs, fin, cf, q_fin, q_bs, q_cf

def map_symbol_for_tradingview(symbol_input):
    sym = symbol_input.upper().strip()
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
        return "Semiconductors & AI Chips (半導體與AI晶片)", {"PE": 35.0, "PS": 10.0, "PEG": 1.4, "GM": 62.0, "NM": 28.0, "ROE": 25.0}
    elif "software" in ind or "cloud" in ind or "infrastructure" in ind:
        return "Software & Enterprise Cloud (軟體與雲端 SaaS)", {"PE": 38.0, "PS": 9.0, "PEG": 1.6, "GM": 72.0, "NM": 22.0, "ROE": 20.0}
    elif "consumer electronics" in ind or "hardware" in ind:
        return "Consumer Electronics (消費電子/硬體裝置)", {"PE": 28.0, "PS": 6.5, "PEG": 1.8, "GM": 44.0, "NM": 24.0, "ROE": 30.0}
    elif "internet content" in ind or "interactive media" in ind or "social media" in ind:
        return "Digital Media & Ad Tech (數位媒體與廣告科技)", {"PE": 25.0, "PS": 7.0, "PEG": 1.3, "GM": 78.0, "NM": 26.0, "ROE": 22.0}
    elif "internet retail" in ind or "e-commerce" in ind:
        return "E-Commerce & Digital Retail (電子商務與數位零售)", {"PE": 32.0, "PS": 2.8, "PEG": 1.4, "GM": 45.0, "NM": 8.0, "ROE": 18.0}
    elif "auto" in ind or "ev" in ind:
        return "Electric Vehicles & Auto (電動車與新興汽車)", {"PE": 45.0, "PS": 5.5, "PEG": 1.5, "GM": 22.0, "NM": 10.0, "ROE": 15.0}
    elif "biotech" in ind or "pharmaceutical" in ind:
        return "Biotech & Pharma (生技與製藥)", {"PE": 24.0, "PS": 5.0, "PEG": 1.4, "GM": 65.0, "NM": 18.0, "ROE": 16.0}
    elif "bank" in ind or "financial" in ind:
        return "Banking & Financial Services (銀行與金融服務)", {"PE": 12.0, "PS": 3.0, "PEG": 1.1, "GM": 50.0, "NM": 22.0, "ROE": 12.0}
    elif "energy" in ind or "oil" in ind:
        return "Energy & Clean Tech (能源與潔淨科技)", {"PE": 14.0, "PS": 1.8, "PEG": 1.0, "GM": 38.0, "NM": 12.0, "ROE": 14.0}
    else:
        return f"{sector_str} - General", {"PE": 20.0, "PS": 2.5, "PEG": 1.3, "GM": 35.0, "NM": 10.0, "ROE": 15.0}

def classify_trend_status(df):
    if df.empty or len(df) < 55:
        return "資料不足", "無法計算趨勢"
    curr_p = df['Close'].iloc[-1]
    ma10 = df['MA10'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma55 = df['MA55'].iloc[-1]
    ma20_slope = df['MA20'].iloc[-1] - df['MA20'].iloc[-5] if len(df) >= 5 else 0
    ma250_valid = 'MA250' in df.columns and pd.notna(df['MA250'].iloc[-1])
    ma250 = df['MA250'].iloc[-1] if ma250_valid else ma55

    is_perfect_bull = (curr_p > ma10 > ma20 > ma55) and (not ma250_valid or ma55 > ma250)
    is_perfect_bear = (curr_p < ma10 < ma20 < ma55) and (not ma250_valid or ma55 < ma250)

    if is_perfect_bull and ma20_slope > 0:
        status = "🚀 強勢多頭 (Strong Bull)"
        desc = f"完美多頭排列 (Price ${curr_p:.2f} > MA10 > MA20 > MA55)，月線向上斜率正向，資金全面控盤。"
    elif curr_p > ma20 and ma20 > ma55 and not is_perfect_bull:
        status = "📈 弱勢多頭 / 回檔整理 (Weak Bull)"
        desc = f"中期多頭格局未變 (Price > MA20 > MA55)，但短線 MA10 震盪或股價短踩，屬多頭回檔健康整理。"
    elif is_perfect_bear and ma20_slope < 0:
        status = "🩸 強勢空頭 (Strong Bear)"
        desc = f"完美空頭排列 (Price ${curr_p:.2f} < MA10 < MA20 < MA55)，均線全面下彎且極具下行壓力。"
    elif curr_p < ma20 and ma20 < ma55 and not is_perfect_bear:
        status = "📉 弱勢空頭 / 反彈波 (Weak Bear)"
        desc = f"中期趨勢偏空 (Price < MA20 < MA55)，但短線價格接近短期均線，屬空頭架構下的弱勢反彈。"
    else:
        status = "⚖️ 區間震盪整理 (Consolidation)"
        desc = f"均線交錯糾結於 ${ma20:.2f} ~ ${ma55:.2f} 區間，多空動能互相抵消，等待量價突破指引方向。"
    return status, desc

def scan_all_macd_divergences(df, window=5, min_gap_days=10):
    if len(df) < 60 or "MACD" not in df.columns:
        return []
    raw_divergences = []
    prices = df['Close'].values
    macds = df['MACD'].values
    dates = df.index
    n = len(df)
    piv_lows = []
    piv_highs = []

    for i in range(window, n - window):
        if all(prices[i] <= prices[i-j] for j in range(1, window+1)) and \
           all(prices[i] <= prices[i+j] for j in range(1, window+1)):
            piv_lows.append((i, dates[i], prices[i], macds[i]))

        if all(prices[i] >= prices[i-j] for j in range(1, window+1)) and \
           all(prices[i] >= prices[i+j] for j in range(1, window+1)):
            piv_highs.append((i, dates[i], prices[i], macds[i]))

    for k in range(1, len(piv_lows)):
        i2, d2, p2, m2 = piv_lows[k]
        i1, d1, p1, m1 = piv_lows[k-1]
        if (i2 - i1) <= 60:
            if p2 < p1 and m2 > m1:
                raw_divergences.append({
                    'index': i2, 'date': d2, 'type': '看多背離 (Bullish)', 'price': p2, 'macd': m2
                })

    for k in range(1, len(piv_highs)):
        i2, d2, p2, m2 = piv_highs[k]
        i1, d1, p1, m1 = piv_highs[k-1]
        if (i2 - i1) <= 60:
            if p2 > p1 and m2 < m1:
                raw_divergences.append({
                    'index': i2, 'date': d2, 'type': '看空背離 (Bearish)', 'price': p2, 'macd': m2
                })

    raw_divergences.sort(key=lambda x: x['index'])
    filtered_divergences = []
    for div in raw_divergences:
        if not filtered_divergences:
            filtered_divergences.append(div)
        else:
            last_div = filtered_divergences[-1]
            if (div['index'] - last_div['index']) >= min_gap_days or div['type'] != last_div['type']:
                filtered_divergences.append(div)
            else:
                filtered_divergences[-1] = div
    return filtered_divergences

def find_smart_support_resistance(df, window=5):
    if df.empty or len(df) < 30:
        fallback_low = df['Low'].min() if not df.empty else 10.0
        fallback_high = df['High'].max() if not df.empty else 100.0
        return fallback_low, fallback_high

    prices_high = df['High'].values
    prices_low = df['Low'].values
    n = len(df)
    swing_highs = []
    swing_lows = []

    for i in range(window, n - window):
        if all(prices_low[i] <= prices_low[i-j] for j in range(1, window+1)) and \
           all(prices_low[i] <= prices_low[i+j] for j in range(1, window+1)):
            swing_lows.append(prices_low[i])

        if all(prices_high[i] >= prices_high[i-j] for j in range(1, window+1)) and \
           all(prices_high[i] >= prices_high[i+j] for j in range(1, window+1)):
            swing_highs.append(prices_high[i])

    curr_price = df['Close'].iloc[-1]
    valid_supps = [s for s in swing_lows if s < curr_price]
    support = max(valid_supps) if valid_supps else df['Low'].min()

    valid_res = [r for r in swing_highs if r > curr_price]
    resistance = min(valid_res) if valid_res else df['High'].max()

    return support, resistance

def find_volume_poc(df, bins=15):
    if df.empty or len(df) < 30:
        return df['Close'].mean() if not df.empty else 0.0

    hist, bin_edges = np.histogram(df['Close'], bins=bins, weights=df['Volume'])
    max_idx = np.argmax(hist)
    poc_price = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2.0
    return poc_price

def generate_technical_narrative(symbol, curr_price, ma10, ma20, ma55, ma250, gunshot, div_type, support, resistance, fib_618, poc_price, trend_status, trend_desc, ma250_valid=True):
    narrative = []
    narrative.append(f"**【均線型態與趨勢等級】**\n當前標的經多因子模型分類為：**{trend_status}**。\n{trend_desc}")

    if div_type == "看多背離 (Bullish)":
        div_desc = "指標面上，近期觸發「看多背離」訊號。雖然價格波段探低，但下行賣壓顯著減弱，機構資金有暗中承接跡象。"
    elif div_type == "看空背離 (Bearish)":
        div_desc = "指標面上，近期出現「看空背離」預警。股價攀高過程中動能未同步放大，提防高檔誘多後的回檔修正。"
    else:
        div_desc = "指標面上，MACD 與價格同步運行，未見明顯動能背離。"
    narrative.append(f"**【指標動能與背離分析】**\n{div_desc}")

    dist_to_supp = ((curr_price - support) / curr_price) * 100 if curr_price > 0 else 0
    dist_to_res = ((resistance - curr_price) / curr_price) * 100 if curr_price > 0 else 0

    of_desc = f"波段關鍵支撐位落在 **${support:.2f}** (距今 {dist_to_supp:.1f}%)，關鍵壓力位在 **${resistance:.2f}** (距今 {dist_to_res:.1f}%)。"
    of_desc += f" 籌碼 vested 密集區 (POC) 落在 **${poc_price:.2f}**。"
    if curr_price > 0 and abs(curr_price - fib_618) / curr_price < 0.02:
        of_desc += f" 值得注意，當前股價接近斐波那契黃金分割支撐 61.8% (**${fib_618:.2f}**)，具備機構買盤防守力道。"
    narrative.append(f"**【關鍵關卡與訂單流佈局】**\n{of_desc}")

    if gunshot and (div_type != "看空背離 (Bearish)"):
        decision = "🎯 **機構策略**：符合 Livermore 第一槍爆發型態（爆量+突破），且無空頭背離，可採取順勢突破建倉策略，將停損設於突破 K 棒低點。"
    elif div_type == "看多背離 (Bullish)" and curr_price <= support * 1.05:
        decision = "🎯 **機構策略**：符合弱勢左側抄底買點（看多背離 + 近關鍵支撐區），風險報酬比優良，適合分批佈局。"
    elif curr_price >= resistance * 0.98:
        decision = "🎯 **機構策略**：股價逼近前高壓力區，追高風險偏高，建議等待爆量突破後回踩不破再行加碼。"
    else:
        decision = "🎯 **機構策略**：當前處於區間震盪整理，建議維持觀望，或於黃金分割支撐位附近佈局。"
    narrative.append(f"**【綜合實戰決策總結】**\n{decision}")

    return "\n\n".join(narrative)

# ==========================================
# Main Execution
# ==========================================
if symbol:
    try:
        info, df_hist, bs, fin, cf_df, q_fin, q_bs, q_cf = fetch_ticker_data(symbol)

        if not info or ("shortName" not in info and "longName" not in info and df_hist.empty):
            st.error(f"❌ 無法取得股票資料或無效代碼: {symbol}")
        else:
            base_val = None
            z_score = None
            z_status = "N/A"
            curr_price = info.get("currentPrice") or info.get("regularMarketPrice") or (df_hist["Close"].dropna().iloc[-1] if not df_hist.empty and not df_hist["Close"].dropna().empty else 0.0)
            gunshot_signal = False
            trend_status = "資料不足 / 無法判斷"
            trend_desc = ""
            div_type = "Neutral"
            all_divergences = []
            support_level = curr_price * 0.9 if curr_price else 0.0
            resistance_level = curr_price * 1.1 if curr_price else 0.0
            poc_price = curr_price
            fib_382 = fib_500 = fib_618 = curr_price
            ma250_valid = False

            # ==========================================
            # 1. 全球大盤情緒：VIX & CNN Fear and Greed 策略買點
            # ==========================================
            st.subheader("🌐 1. 全球市場情緒與 VIX 抄底濾網 (Market Sentiment & VIX Filter)")
            vix_ticker = yf.Ticker("^VIX")
            vix_hist = vix_ticker.history(period="5d")
            vix_val = vix_hist["Close"].iloc[-1] if not vix_hist.empty else 20.0
            cnn_fng_val, cnn_fng_status = get_cnn_fear_and_greed()

            v1, v2, v3 = st.columns(3)
            v1.metric("VIX 恐慌指數", f"{vix_val:.2f}", delta=">25 為極度恐慌" if vix_val > 25 else "正常區間")
            cnn_display = f"{cnn_fng_val} ({cnn_fng_status})" if cnn_fng_val is not None else cnn_fng_status
            v2.metric("CNN Fear & Greed Index", cnn_display)

            buy_window = (cnn_fng_val is not None and vix_val >= 25 and cnn_fng_val <= 40)
            if buy_window:
                v3.success("🟢 **觸發黃金抄底訊號**：VIX > 25 且 CNN Fear & Greed < 40 (極度恐慌為長線建倉時機)！")
            else:
                v3.info("🟡 **市場情緒平穩/偏熱**：未達到 VIX > 25 且 CNN F&G < 40 的黃金逆勢買點。")

            st.write("---")

            # ==========================================
            # 2. 精準細分產業、全方位財務比率 (Financial Ratios) 與基本面估值
            # ==========================================
            st.subheader(f"🏛️ 2. {info.get('shortName', symbol)} ({symbol}) - 完整財務比率與基本面指標")
            raw_industry = info.get("industry", "")
            raw_sector = info.get("sector", "")
            ind_category_name, ind_benchmarks = get_granular_industry_benchmarks(raw_industry, raw_sector)
            st.caption(
                f"📌 **細分產業類別**：`{raw_industry or raw_sector}` ➔ 定義為 **{ind_category_name}**\n"
                f"📊 **同業基準參考**：P/E: {ind_benchmarks['PE']}x | P/S: {ind_benchmarks['PS']}x | PEG: {ind_benchmarks['PEG']}x | 毛利率: {ind_benchmarks['GM']}% | 淨利率: {ind_benchmarks['NM']}% | ROE: {ind_benchmarks['ROE']}%"
            )

            if not bs.empty and not fin.empty:
                try:
                    latest_bs = bs.iloc[:, 0]
                    latest_fin = fin.iloc[:, 0]
                    total_assets = latest_bs.get("Total Assets", np.nan)
                    total_liab = latest_bs.get("Total Liabilities Net Minority Interest", latest_bs.get("Total Liabilities", np.nan))
                    curr_assets = latest_bs.get("Current Assets", np.nan)
                    curr_liab = latest_bs.get("Current Liabilities", np.nan)
                    retained_earnings = latest_bs.get("Retained Earnings", 0)
                    working_capital = (curr_assets - curr_liab) if pd.notna(curr_assets) and pd.notna(curr_liab) else np.nan
                    ebit = latest_fin.get("EBIT", np.nan)
                    sales = latest_fin.get("Total Revenue", np.nan)
                    mkt_cap = info.get("marketCap", np.nan)

                    if pd.notna(total_assets) and pd.notna(total_liab) and total_liab > 0 and pd.notna(working_capital):
                        x1 = working_capital / total_assets
                        x2 = retained_earnings / total_assets
                        x3 = ebit / total_assets if pd.notna(ebit) else 0
                        x4 = mkt_cap / total_liab if pd.notna(mkt_cap) else 0
                        x5 = sales / total_assets if pd.notna(sales) else 0
                        z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5
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
            pe_val = info.get('forwardPE') or info.get('trailingPE')
            r1_1.metric("Forward P/E", f"{pe_val:.2f}" if pe_val else "N/A", delta=f"同業 {ind_benchmarks['PE']}x", delta_color="inverse" if pe_val and pe_val > ind_benchmarks['PE'] else "normal")
            ps_val = info.get('priceToSalesTrailing12Months')
            r1_2.metric("P/S Ratio", f"{ps_val:.2f}" if ps_val else "N/A", delta=f"同業 {ind_benchmarks['PS']}x", delta_color="inverse" if ps_val and ps_val > ind_benchmarks['PS'] else "normal")
            r1_3.metric("P/B Ratio", f"{info.get('priceToBook', 0):.2f}" if info.get("priceToBook") else "N/A")
            peg_val = info.get('pegRatio')
            r1_4.metric("PEG Ratio", f"{peg_val:.2f}" if peg_val else "N/A", delta=f"同業 {ind_benchmarks['PEG']}x", delta_color="inverse" if peg_val and peg_val > ind_benchmarks['PEG'] else "normal")
            r1_5.metric("Quick Ratio", f"{info.get('quickRatio', 0):.2f}" if info.get("quickRatio") else "N/A")
            r1_6.metric("Altman Z-Score", z_status)

            st.markdown("##### 💵 獲利能力比率 (Profitability & Margins)")
            r2_1, r2_2, r2_3, r2_4, r2_5 = st.columns(5)
            gm_val = info.get('grossMargins')
            gm_disp = f"{gm_val * 100:.2f}%" if gm_val else "N/A"
            r2_1.metric("毛利率 (Gross Margin)", gm_disp, delta=f"同業 {ind_benchmarks['GM']}%", delta_color="normal" if gm_val and gm_val * 100 > ind_benchmarks['GM'] else "inverse")
            op_val = info.get('operatingMargins')
            r2_2.metric("營業利益率 (Op Margin)", f"{op_val * 100:.2f}%" if op_val else "N/A")
            nm_val = info.get('profitMargins')
            nm_disp = f"{nm_val * 100:.2f}%" if nm_val else "N/A"
            r2_3.metric("淨利率 (Net Margin)", nm_disp, delta=f"同業 {ind_benchmarks['NM']}%", delta_color="normal" if nm_val and nm_val * 100 > ind_benchmarks['NM'] else "inverse")
            roe_val = info.get('returnOnEquity')
            roe_disp = f"{roe_val * 100:.2f}%" if roe_val else "N/A"
            r2_4.metric("股東權益報酬率 (ROE)", roe_disp, delta=f"同業 {ind_benchmarks['ROE']}%", delta_color="normal" if roe_val and roe_val * 100 > ind_benchmarks['ROE'] else "inverse")
            roa_val = info.get('returnOnAssets')
            r2_5.metric("資產報酬率 (ROA)", f"{roa_val * 100:.2f}%" if roa_val else "N/A")

            # ==========================================
            # 3. DCF 現金流折現估值模型 (Pragmatic 5-Year FCFF Model)
            # ==========================================
            st.write("---")
            st.subheader("💰 3. DCF 現金流折現估值模型 (Pragmatic 5-Year FCFF Model)")
            rf_rate = get_risk_free_rate()
            beta = info.get("beta", 1.0)
            if pd.isna(beta) or beta is None:
                beta = 1.0

            ttm_fcf = None
            if not q_cf.empty:
                try:
                    q_cf_t = q_cf.T.head(4)
                    if "Free Cash Flow" in q_cf_t.columns and q_cf_t["Free Cash Flow"].notna().sum() == 4:
                        ttm_fcf = float(q_cf_t["Free Cash Flow"].sum())
                    elif "Operating Cash Flow" in q_cf_t.columns and "Capital Expenditure" in q_cf_t.columns:
                        ttm_fcf = float((q_cf_t["Operating Cash Flow"] + q_cf_t["Capital Expenditure"]).sum())
                except Exception:
                    ttm_fcf = None

            fcf_history = []
            fcf_dates = []
            if not cf_df.empty:
                for col in cf_df.columns:
                    val = None
                    if "Free Cash Flow" in cf_df.index and pd.notna(cf_df.loc["Free Cash Flow", col]):
                        val = cf_df.loc["Free Cash Flow", col]
                    elif "Operating Cash Flow" in cf_df.index and "Capital Expenditure" in cf_df.index:
                        ocf = cf_df.loc["Operating Cash Flow", col]
                        capex = cf_df.loc["Capital Expenditure", col]
                        if pd.notna(ocf) and pd.notna(capex):
                            val = ocf + capex
                    if val is not None and not pd.isna(val):
                        fcf_history.append(float(val))
                        fcf_dates.append(str(col)[:10])

            shares_out = info.get("sharesOutstanding", 0)
            net_debt = 0.0
            total_debt = 0.0
            if not bs.empty:
                try:
                    latest_bs = bs.iloc[:, 0]
                    tot_debt = latest_bs.get("Total Debt", None)
                    if pd.isna(tot_debt) or tot_debt is None:
                        lt_debt = latest_bs.get("Long Term Debt", 0)
                        st_debt = latest_bs.get("Current Debt And Capital Lease Obligation", latest_bs.get("Current Debt", 0))
                        tot_debt = (lt_debt if pd.notna(lt_debt) else 0) + (st_debt if pd.notna(st_debt) else 0)
                    cash = latest_bs.get("Cash And Cash Equivalents", latest_bs.get("Cash Cash Equivalents And Short Term Investments", info.get("cash", 0)))
                    if pd.notna(tot_debt):
                        total_debt = float(tot_debt)
                    cash_val = float(cash) if pd.notna(cash) else 0.0
                    net_debt = total_debt - cash_val
                except Exception:
                    pass

            erp = 0.050
            cost_of_equity = rf_rate + (beta * erp)
            mkt_cap_val = info.get("marketCap", 0) or 0.0
            debt_spread = 0.020

            if mkt_cap_val > 0 and total_debt > 0:
                cost_of_debt = rf_rate + debt_spread
                total_capital = mkt_cap_val + total_debt
                weight_e = mkt_cap_val / total_capital
                weight_d = total_debt / total_capital
                calc_wacc = (weight_e * cost_of_equity) + (weight_d * cost_of_debt * 0.79)
                discount_rate = calc_wacc
            else:
                discount_rate = cost_of_equity

            valid_fcfs = [f for f in fcf_history if f > 0]
            if ttm_fcf and ttm_fcf > 0:
                base_fcf = max(ttm_fcf, max(valid_fcfs) if valid_fcfs else 0)
            elif valid_fcfs:
                base_fcf = max(valid_fcfs)
            else:
                base_fcf = 0

            if base_fcf > 0 and shares_out > 0:
                growth_est = info.get("earningsGrowth") or info.get("revenueGrowth") or 0.15
                if pd.isna(growth_est) or growth_est is None:
                    growth_est = 0.12
                growth_est = max(0.05, min(growth_est, 0.50))

                def run_dcf_5yr_details(b_fcf, init_g, disc_r, term_g=0.025):
                    fcfs = []
                    pvs = []
                    fcf = b_fcf
                    for yr in range(1, 6):
                        fcf *= (1 + init_g)
                        pv = fcf / ((1 + disc_r) ** yr)
                        fcfs.append(fcf)
                        pvs.append(pv)
                    terminal_val = (fcfs[-1] * (1 + term_g)) / (disc_r - term_g) if disc_r > term_g else 0
                    pv_terminal = terminal_val / ((1 + disc_r) ** 5)
                    enterprise_value = sum(pvs) + pv_terminal
                    equity_value = enterprise_value - net_debt
                    intrinsic_value_per_share = equity_value / shares_out
                    return max(intrinsic_value_per_share, 0.0), fcfs, pvs, terminal_val, pv_terminal, enterprise_value, equity_value

                base_val, base_fcfs, base_pvs, base_tv, base_pv_tv, base_ev, base_eq = run_dcf_5yr_details(base_fcf, growth_est, discount_rate, term_g=0.025)
                pess_growth = max(0.03, growth_est * 0.75)
                pess_discount = discount_rate + 0.01
                pess_val, pess_fcfs, pess_pvs, pess_tv, pess_pv_tv, pess_ev, pess_eq = run_dcf_5yr_details(base_fcf, pess_growth, pess_discount, term_g=0.020)
                opt_growth = min(growth_est * 1.25, 0.50)
                opt_discount = max(discount_rate - 0.008, 0.05)
                opt_val, opt_fcfs, opt_pvs, opt_tv, opt_pv_tv, opt_ev, opt_eq = run_dcf_5yr_details(base_fcf, opt_growth, opt_discount, term_g=0.030)

                dcf_col1, dcf_col2, dcf_col3, dcf_col4 = st.columns(4)
                dcf_col1.metric("🔴 保守情境估值", f"${pess_val:.2f}")
                dcf_col2.metric("🟡 基準情境估值", f"${base_val:.2f}")
                dcf_col3.metric("🟢 樂觀情境估值", f"${opt_val:.2f}")
                diff_pct = ((base_val - curr_price) / curr_price) * 100 if curr_price else 0
                dcf_col4.metric("目前價格 / 潛在空間", f"${curr_price:.2f}", delta=f"{diff_pct:+.1f}%")

            # ==========================================
            # 5. 技術面量價與 Plotly 繪圖 (與安全性對策修復)
            # ==========================================
            st.write("---")
            st.subheader("🎯 5. 技術面量價、均線趨勢分類與 MACD 背離掃描")

            if df_hist.empty or len(df_hist.dropna(subset=['Close'])) < 15:
                st.warning("⚠️ 數據不足，無法進行完整技術分析與圖表繪製。")
            else:
                try:
                    df = df_hist.copy()
                    
                    # 進行完整與乾淨的資料清理與指標計算
                    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
                    df = df.sort_index()

                    df['MA10'] = df['Close'].rolling(window=10, min_periods=1).mean()
                    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
                    df['MA55'] = df['Close'].rolling(window=55, min_periods=1).mean()
                    df['MA250'] = df['Close'].rolling(window=250, min_periods=1).mean()

                    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                    df['MACD'] = exp1 - exp2
                    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                    df['Histogram'] = df['MACD'] - df['Signal']

                    df = df.bfill().ffill()

                    trend_status, trend_desc = classify_trend_status(df)
                    all_divergences = scan_all_macd_divergences(df)
                    support_level, resistance_level = find_smart_support_resistance(df)
                    poc_price = find_volume_poc(df)

                    recent_high = df['High'].max()
                    recent_low = df['Low'].min()
                    diff = recent_high - recent_low
                    fib_618 = recent_high - (0.618 * diff) if diff > 0 else curr_price

                    last_div = all_divergences[-1] if all_divergences else None
                    div_type = last_div['type'] if last_div else "Neutral"

                    curr_p = df['Close'].iloc[-1]
                    ma10_val = df['MA10'].iloc[-1]
                    ma20_val = df['MA20'].iloc[-1]
                    ma55_val = df['MA55'].iloc[-1]
                    ma250_val = df['MA250'].iloc[-1]

                    narrative = generate_technical_narrative(
                        symbol, curr_p, ma10_val, ma20_val, ma55_val, ma250_val,
                        gunshot_signal, div_type, support_level, resistance_level,
                        fib_618, poc_price, trend_status, trend_desc
                    )

                    st.markdown(narrative)

                    # 創建兩層軸線子圖 (主價格圖 + MACD 震盪指標)
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=(f'{symbol} Price & Moving Averages', 'MACD Indicator'),
                        row_width=[0.3, 0.7]
                    )

                    # K線圖
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'],
                        name='Price'
                    ), row=1, col=1)

                    # 均線群組
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA10'], line=dict(color='orange', width=1), name='MA10'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='blue', width=1.5), name='MA20'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA55'], line=dict(color='purple', width=2), name='MA55'), row=1, col=1)
                    
                    if not df['MA250'].isnull().all():
                        fig.add_trace(go.Scatter(x=df.index, y=df['MA250'], line=dict(color='red', width=2), name='MA250'), row=1, col=1)

                    # MACD 與 Signal
                    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue', width=1.5), name='MACD'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='orange', width=1.5), name='Signal'), row=2, col=1)

                    # MACD 柱狀圖 (Histogram)
                    colors = ['green' if val >= 0 else 'red' for val in df['Histogram']]
                    fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], marker_color=colors, name='Histogram'), row=2, col=1)

                    # 設定軸範圍與邊界優化
                    fig.update_xaxes(rangeslider_visible=False)
                    fig.update_layout(height=650, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))

                    st.plotly_chart(fig, use_container_width=True)

                    # TradingView Widget 嵌入
                    tv_symbol = map_symbol_for_tradingview(symbol)
                    tv_html = f"""
                    <div class="tradingview-widget-container">
                      <div id="tradingview_chart"></div>
                      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                      <script type="text/javascript">
                      new TradingView.widget({{
                        "width": "100%",
                        "height": 500,
                        "symbol": "{tv_symbol}",
                        "interval": "D",
                        "timezone": "Etc/UTC",
                        "theme": "light",
                        "style": "1",
                        "locale": "zh_TW",
                        "toolbar_bg": "#f1f3f6",
                        "enable_publishing": false,
                        "allow_symbol_change": true,
                        "container_id": "tradingview_chart"
                      }});
                      </script>
                    </div>
                    """
                    components.html(tv_html, height=520)

                except Exception as e:
                    st.error(f"分析時發生未預期錯誤: {e}")

    except Exception as e:
        st.error(f"資料抓取失敗或錯誤: {e}")
