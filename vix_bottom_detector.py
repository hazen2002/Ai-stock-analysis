"""Macro and volatility correlation dashboard."""

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from utils.events import get_event_horizon


_TICKERS = ("^VIX", "^VIX3M", "^VVIX", "SPY", "QQQ", "DIA")
_MACRO_RISKS = (
    ("【小風險】", "FOMC / 財報周", "短期事件波動；關注持倉與流動性。"),
    ("【中型風險】", "個股財報", "個股跳空風險；避免將單一公司事件誤讀為系統風險。"),
    (
        "【大風險】",
        "美國大選 / 中期選舉 De-risking 預警",
        "政策與選舉不確定性升高時，監控波動結構及風險曝險。",
    ),
    (
        "【超大風險】",
        "政權與範式轉移",
        "結構性制度變化；本頁市場指標無法預測此類事件。",
    ),
)


@st.cache_data(ttl=datetime.timedelta(hours=1), show_spinner=False)
def _download_history(ticker):
    """Cache daily history so a rerun does not repeat Yahoo Finance requests."""
    try:
        history = yf.download(
            ticker,
            period="3y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        return pd.DataFrame(), f"{ticker}: {exc}"
    if history.empty:
        return pd.DataFrame(), f"{ticker}: Yahoo Finance returned no data."
    return history, None


def _extract_series(history, field, ticker):
    if isinstance(history.columns, pd.MultiIndex):
        matching_columns = [
            column
            for column in history.columns
            if field.lower() in {str(level).lower() for level in column}
        ]
        if not matching_columns:
            return None, f"{ticker}: no {field} prices were returned."
        values = history[matching_columns[0]]
    elif field in history.columns:
        values = history[field]
    else:
        return None, f"{ticker}: no {field} prices were returned."

    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    values = pd.to_numeric(values, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None, f"{ticker}: no valid {field} prices were returned."
    return values.rename(ticker), None


def _percentage_change(series, periods):
    observations = series.dropna()
    if len(observations) <= periods:
        return None
    prior = float(observations.iloc[-periods - 1])
    latest = float(observations.iloc[-1])
    if not np.isfinite(prior) or not np.isfinite(latest) or prior == 0:
        return None
    return (latest / prior - 1.0) * 100.0


def _base_100(series):
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or float(clean.iloc[0]) == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return series / float(clean.iloc[0]) * 100.0


def _build_fis_and_macd(spy_history):
    fields = {}
    errors = []
    for field in ("High", "Low", "Close"):
        series, error = _extract_series(spy_history, field, "SPY")
        if error:
            errors.append(error)
        else:
            fields[field] = series.rename(field)
    if errors:
        return None, errors

    prices = pd.concat(fields.values(), axis=1).sort_index()
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna()
    prices = prices.loc[
        (prices["High"] > 0)
        & (prices["Low"] > 0)
        & (prices["Close"] > 0)
        & (prices["High"] >= prices["Low"])
    ]
    if prices.empty:
        return None, ["SPY: no valid OHLC observations for ATR calculations."]

    previous_close = prices["Close"].shift(1)
    true_range = pd.concat(
        [
            prices["High"] - prices["Low"],
            (prices["High"] - previous_close).abs(),
            (prices["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()
    prices["FIS_Fast"] = (atr / prices["Close"]) * 100.0
    prices["FIS_Slow"] = prices["FIS_Fast"].rolling(
        20, min_periods=5
    ).mean()

    ema_fast = prices["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = prices["Close"].ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    prices["MACD_Hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    return prices, []


def render_event_horizon(watchlist, liquidity_flush, cash_locked):
    st.markdown("### 📅 Macro & Earnings Event Horizon (未來總經與財報事件排程)")
    days_ahead = st.slider(
        "預視未來天數範圍",
        min_value=3,
        max_value=30,
        value=14,
        key="macro_earnings_event_horizon_days",
    )
    today = datetime.date.today()
    target_date = today + datetime.timedelta(days=days_ahead)
    with st.spinner("載入官方總經排程與 Watchlist 財報日期..."):
        events, errors = get_event_horizon(watchlist, today=today)

    if not events.empty:
        visible = events.loc[
            events["Date"].between(today, target_date, inclusive="both")
        ].copy()
    else:
        visible = events.copy()

    urgent_events = (
        visible.loc[
            visible["Date"].between(
                today,
                today + datetime.timedelta(days=3),
                inclusive="both",
            )
            & (
                visible["Type"].isin(["央行", "財報"])
                | visible["Event"].str.contains(
                    r"FOMC|財報|Earnings",
                    case=False,
                    regex=True,
                )
            )
        ]
        if not visible.empty
        else visible
    )
    if (liquidity_flush or cash_locked) and not urgent_events.empty:
        st.warning(
            "目前處於流動性緊縮，且緊鄰重大總經事件，建議維持現金防守，"
            "切勿盲目抄底。"
        )

    if visible.empty:
        st.info(f"未來 {days_ahead} 天沒有找到已排定的總經、央行或 Watchlist 財報事件。")
    else:
        risk_labels = {
            "高風險": "🔴 高風險",
            "中風險": "🟡 中風險",
            "小風險": "🟢 小風險",
        }
        display = visible.loc[
            :, ["Date", "Event", "Type", "Risk", "Source"]
        ].copy()
        display["Date"] = display["Date"].map(lambda value: value.isoformat())
        display["Risk"] = display["Risk"].map(risk_labels).fillna("⚪ 未分級")
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

    macro_errors = [error for error in errors if "calendar unavailable" in error]
    earnings_errors = [
        error for error in errors if "calendar unavailable" not in error
    ]
    if macro_errors:
        st.caption(
            "部分官方總經日曆暫時無法連線；若有備援事件，來源欄會標示其資料提供者。"
            "未取得的日期不會以估算值代替。"
        )
    if earnings_errors:
        st.caption(
            f"{len(earnings_errors)} 個 Watchlist 標的的財報日無法取得；"
            "可用日期會照常列出。"
        )
    if not visible.empty:
        sources = ", ".join(sorted(visible["Source"].dropna().unique()))
        st.caption(
            f"資料來源：{sources}。公告可能更新，"
            "請以發布機構或公司 IR 最新資訊為準。"
        )


def render_vix_bottom_ui(watchlist=None):
    st.title("⚡ Macro & Volatility Correlation Matrix")
    st.caption(
        "三年期宏觀風險、波動率期限結構與跨資產趨勢追蹤。"
        "指標為規則式觀察工具，不是市場預測或個人化投資建議。"
    )

    histories = {}
    closes = {}
    errors = []
    for ticker in _TICKERS:
        history, download_error = _download_history(ticker)
        if download_error:
            errors.append(download_error)
            continue
        histories[ticker] = history
        close, close_error = _extract_series(history, "Close", ticker)
        if close_error:
            errors.append(close_error)
        else:
            closes[ticker] = close

    required_tickers = set(_TICKERS)
    missing_tickers = required_tickers.difference(closes)
    if missing_tickers:
        details = " ".join(errors) or (
            "Missing tickers: " + ", ".join(sorted(missing_tickers))
        )
        st.error("Unable to load required market data: " + details)
        return

    close_history = pd.concat(closes.values(), axis=1).sort_index()
    close_history["VIX_Ratio"] = (
        close_history["^VIX"] / close_history["^VIX3M"]
    ).replace([np.inf, -np.inf], np.nan)
    close_history.loc[close_history["^VIX3M"] <= 0, "VIX_Ratio"] = np.nan
    ratio = close_history["VIX_Ratio"].dropna()
    if len(ratio) < 5:
        st.error("至少需要五筆有效 VIX / VIX3M 比率資料才能評估跨日動能。")
        return

    spy_metrics, spy_errors = _build_fis_and_macd(histories["SPY"])
    if spy_errors:
        st.error("Unable to calculate SPY volatility metrics: " + " ".join(spy_errors))
        return

    latest_ratio_date = ratio.index[-1]
    latest_vix = float(close_history.at[latest_ratio_date, "^VIX"])
    latest_vix3m = float(close_history.at[latest_ratio_date, "^VIX3M"])
    latest_vvix = float(close_history["^VVIX"].dropna().iloc[-1])
    latest_ratio = float(ratio.iloc[-1])
    five_day_max_vvix = float(close_history["^VVIX"].dropna().tail(5).max())
    ratio_rising = bool(
        len(ratio) >= 3
        and ratio.iloc[-1] > ratio.iloc[-2] > ratio.iloc[-3]
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("VIX", f"{latest_vix:.2f}")
    metric_columns[1].metric("VIX3M", f"{latest_vix3m:.2f}")
    metric_columns[2].metric("VVIX", f"{latest_vvix:.2f}")
    ratio_state = "Backwardation" if latest_ratio >= 1.0 else "Contango"
    metric_columns[3].metric(
        "VIX / VIX3M Ratio",
        f"{latest_ratio:.3f}",
        delta=ratio_state,
        delta_color="inverse" if latest_ratio >= 1.0 else "normal",
    )

    delta_columns = st.columns(3)
    for column, periods in zip(delta_columns, (1, 3, 5)):
        delta = _percentage_change(ratio, periods)
        column.metric(
            f"Ratio Delta T-{periods}",
            f"{delta:+.2f}%" if delta is not None else "資料不足",
            delta_color="inverse",
        )

    if latest_ratio > 1.0 and ratio_rising:
        st.error("🚨 Liquidity Flush & De-risking — 機構去風險化加速中！")

    st.markdown("#### Macro Risk Taxonomy")
    risk_columns = st.columns(4)
    for column, (level, title, description) in zip(risk_columns, _MACRO_RISKS):
        with column:
            st.markdown(f"**{level} {title}**")
            st.caption(description)

    if latest_ratio >= 1.0:
        detector_message = (
            "Liquidity Flush in progress. Maintain elevated cash and avoid "
            "treating a falling market as a confirmed bottom."
        )
        st.error(f"🔴 Liquidity Flush — {detector_message}")
    elif latest_vvix < five_day_max_vvix * 0.9:
        detector_message = (
            "Backwardation resolved and VVIX eased from its five-observation "
            "high; this is a volatility-resolution heuristic, not a buy signal."
        )
        st.success(f"🟢 Volatility Resolution Watch — {detector_message}")
    else:
        detector_message = (
            "Backwardation has cleared, but VVIX has not met the recent "
            "decline condition; wait for independent confirmation."
        )
        st.warning(f"🟡 Bottom Formation Watch — {detector_message}")

    spy_close = spy_metrics["Close"].dropna()
    ma250_series = spy_close.rolling(250, min_periods=250).mean()
    ma250_available = not ma250_series.dropna().empty
    latest_spy = float(spy_close.iloc[-1])
    latest_ma250 = (
        float(ma250_series.dropna().iloc[-1]) if ma250_available else np.nan
    )
    above_ma250 = bool(ma250_available and latest_spy > latest_ma250)

    fis_fast = spy_metrics["FIS_Fast"].dropna()
    fis_slow = spy_metrics["FIS_Slow"].dropna()
    fis_fast_latest = float(fis_fast.iloc[-1]) if not fis_fast.empty else np.nan
    fis_slow_latest = float(fis_slow.iloc[-1]) if not fis_slow.empty else np.nan
    fis_bullish = bool(
        np.isfinite(fis_fast_latest)
        and np.isfinite(fis_slow_latest)
        and fis_fast_latest > fis_slow_latest
    )
    histogram = spy_metrics["MACD_Hist"].dropna()
    macd_hist_rising = bool(
        len(histogram) >= 3
        and histogram.iloc[-1] > histogram.iloc[-2] > histogram.iloc[-3]
    )
    ratio_contango = latest_ratio < 1.0
    green_light = all(
        (ratio_contango, fis_bullish, macd_hist_rising, above_ma250)
    )

    chart_column, signal_column = st.columns([4, 1])
    with chart_column:
        comparison = close_history[
            ["SPY", "QQQ", "DIA", "VIX_Ratio"]
        ].join(spy_metrics[["FIS_Fast"]], how="outer").sort_index()
        figure = go.Figure()
        for series_name, color in (
            ("SPY", "#42A5F5"),
            ("QQQ", "#AB47BC"),
            ("DIA", "#26A69A"),
            ("VIX_Ratio", "#EF5350"),
            ("FIS_Fast", "#FFCA28"),
        ):
            normalized = _base_100(comparison[series_name])
            figure.add_trace(
                go.Scatter(
                    x=normalized.index,
                    y=normalized,
                    name=series_name,
                    mode="lines",
                    line=dict(color=color, width=1.8),
                )
            )
        figure.update_layout(
            title="Three-Year Base-100 Correlation | SPY, QQQ, DIA, VIX Ratio, FIS Fast",
            template="plotly_dark",
            height=560,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1),
            margin=dict(t=90, b=35),
            yaxis_title="Normalized value (first valid observation = 100)",
        )
        st.plotly_chart(figure, use_container_width=True)
        st.caption(
            "FIS Fast = (SPY ATR(14) / SPY Close) × 100; "
            "FIS Slow = 20 日 FIS Fast 均值。圖表以各序列首個有效值歸一化至 100。"
        )
        legacy_figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("VIX / VIX3M Ratio", "VIX vs. VIX3M"),
        )
        legacy_figure.add_trace(
            go.Scatter(
                x=ratio.index,
                y=ratio,
                name="VIX / VIX3M",
                line=dict(color="#42A5F5", width=2),
            ),
            row=1,
            col=1,
        )
        legacy_figure.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="red",
            annotation_text="Backwardation threshold (1.0)",
            row=1,
            col=1,
        )
        for ticker, color in (("^VIX", "#EF5350"), ("^VIX3M", "#26A69A")):
            legacy_figure.add_trace(
                go.Scatter(
                    x=close_history.index,
                    y=close_history[ticker],
                    name=ticker.lstrip("^"),
                    line=dict(color=color, width=2),
                ),
                row=2,
                col=1,
            )
        legacy_figure.update_layout(
            title="VIX Term-Structure Detail",
            template="plotly_dark",
            height=620,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1),
            margin=dict(t=90, b=35),
        )
        legacy_figure.update_yaxes(title_text="Ratio", row=1, col=1)
        legacy_figure.update_yaxes(title_text="Volatility Index", row=2, col=1)
        st.plotly_chart(legacy_figure, use_container_width=True)

    with signal_column:
        st.markdown("#### 右側信號矩陣")
        st.metric("SPY", f"${latest_spy:.2f}")
        st.metric(
            "SPY 250 SMA",
            f"${latest_ma250:.2f}" if ma250_available else "資料不足",
            delta=(
                "站穩年線"
                if above_ma250
                else "⚠️ 弱勢回測 / 假反彈"
                if ma250_available
                else "尚無 250 筆資料"
            ),
            delta_color="normal" if above_ma250 else "inverse",
        )
        matrix = pd.DataFrame(
            [
                ("VIX / VIX3M < 1.0", ratio_contango),
                ("FIS Fast > FIS Slow", fis_bullish),
                ("MACD 柱狀圖連續抽升", macd_hist_rising),
                ("股價 > 250 SMA", above_ma250),
            ],
            columns=["條件", "狀態"],
        )
        matrix["狀態"] = matrix["狀態"].map(
            {True: "🟢 通過", False: "🔴 未通過"}
        )
        st.dataframe(matrix, use_container_width=True, hide_index=True)
        st.metric(
            "FIS Fast / Slow",
            (
                f"{fis_fast_latest:.3f}% / {fis_slow_latest:.3f}%"
                if np.isfinite(fis_fast_latest) and np.isfinite(fis_slow_latest)
                else "資料不足"
            ),
        )
        if green_light:
            st.success("🟢 三重綠燈：所有趨勢條件均符合；仍需獨立風險評估。")
        else:
            st.error("🔒 100% 鎖死現金 — 三重綠燈條件未全部成立。")

    render_event_horizon(
        watchlist=watchlist or (),
        liquidity_flush=latest_ratio >= 1.0,
        cash_locked=not green_light,
    )

    with st.expander("How to interpret this detector"):
        st.markdown(
            f"""
            - **VIX / VIX3M** 比率高於 1.0 代表短天期隱含波動高於三個月期限；
              T-1、T-3、T-5 顯示相對各觀察日前的比率百分比變化。
            - **VVIX** 衡量 VIX 的預期波動；由近期高點回落僅代表波動降溫，
              不保證股價已築底。
            - **250 SMA** 僅在具備至少 250 筆有效 SPY 收盤價時提供判斷；
              資料不足不會被當成站穩年線。
            - **綠燈條件** 必須同時滿足 Ratio < 1、FIS Fast > FIS Slow、
              MACD 柱狀圖連續三筆走升及 SPY > 250 SMA。

            最新比率為 {latest_ratio:.3f}；VVIX {latest_vvix:.2f}；
            最近五筆 VVIX 高點 {five_day_max_vvix:.2f}。
            """
        )
        st.caption(
            "圖表與訊號是規則式市場觀察，不構成投資建議；"
            "模型不代表未來績效，也不應取代個人風險管理。"
        )
