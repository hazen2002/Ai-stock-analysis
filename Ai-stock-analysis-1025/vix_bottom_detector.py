import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


_TICKERS = ("^VIX", "^VIX3M", "^VVIX")


def _download_close(ticker):
    try:
        history = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        return None, f"{ticker}: {exc}"

    if history.empty:
        return None, f"{ticker}: Yahoo Finance returned no data."

    if isinstance(history.columns, pd.MultiIndex):
        close_columns = [column for column in history.columns if "Close" in column]
        if not close_columns:
            return None, f"{ticker}: no Close prices were returned."
        close = history[close_columns[0]]
    elif "Close" in history.columns:
        close = history["Close"]
    else:
        return None, f"{ticker}: no Close prices were returned."

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    if close.empty:
        return None, f"{ticker}: no valid Close prices were returned."
    return close.rename(ticker), None


def render_vix_bottom_ui():
    st.title("⚡ VIX 波動率結構底部探測器")
    st.caption(
        "Track volatility term structure and VVIX momentum for a data-driven "
        "read on potential market bottoms."
    )

    closes = {}
    errors = []
    for ticker in _TICKERS:
        close, error = _download_close(ticker)
        if error:
            errors.append(error)
        elif close is not None:
            closes[ticker] = close

    if errors:
        st.error("Unable to load volatility data: " + " ".join(errors))
        return

    history = pd.concat(closes.values(), axis=1, join="inner").dropna()
    if len(history) < 5:
        st.error(
            "At least five complete daily observations for VIX, VIX3M, and "
            "VVIX are required to evaluate the detector."
        )
        return

    history["VIX_Ratio"] = history["^VIX"] / history["^VIX3M"]
    latest = history.iloc[-1]
    latest_vix = float(latest["^VIX"])
    latest_vix3m = float(latest["^VIX3M"])
    latest_vvix = float(latest["^VVIX"])
    latest_ratio = float(latest["VIX_Ratio"])
    five_day_max_vvix = float(history["^VVIX"].tail(5).max())

    if latest_ratio >= 1.0:
        state = 1
        message = (
            "Liquidity Flush in progress. Maintain 30%+ Cash. Do not catch "
            "falling knives."
        )
    elif latest_vvix < five_day_max_vvix * 0.9:
        state = 3
        message = (
            "VIX Backwardation resolved & VVIX peaking down. Right-side entry "
            "unlocked (70% Core / 30% Satellite)!"
        )
    else:
        state = 2
        message = (
            "IV Panic peaking. Await VIX/VIX3M ratio to stabilize strictly "
            "below 1.0."
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

    if state == 1:
        st.error(f"🔴 Liquidity Flush — {message}")
    elif state == 2:
        st.warning(f"🟡 Bottom Formation — {message}")
    else:
        st.success(f"🟢 Quant Bottom Confirmed / Resolution Cross — {message}")

    with st.expander("How to interpret this detector"):
        st.markdown(
            """
            **What these indicators measure**

            - **VIX** estimates expected S&P 500 volatility over roughly the
              next 30 days. A higher reading means options are pricing in
              larger near-term moves, not a particular market direction.
            - **VIX3M** measures expected volatility over roughly the next
              three months. Comparing VIX with VIX3M puts near-term fear in
              context with the longer-term volatility outlook.
            - **VVIX** tracks expected volatility of the VIX itself. A falling
              VVIX can indicate that demand for volatility protection is
              easing, but it does not confirm that prices have bottomed.

            **How to read the ratio and status**

            - **VIX / VIX3M >= 1.0 — Backwardation:** near-term implied
              volatility is at least as high as three-month implied
              volatility. The detector flags this as a liquidity-flush risk.
            - **Ratio < 1.0 — Contango:** the immediate backwardation
              condition has cleared. This alone is not a bottom signal.
            - **Quant Bottom Confirmed:** the ratio is below 1.0 and the latest
              VVIX is more than 10% below its highest reading in the latest
              five observations. This is the detector's resolution-cross rule.
            - **Bottom Formation:** backwardation has cleared, but VVIX has
              not yet met that five-observation decline condition.

            **Chart guide:** the top panel plots the VIX/VIX3M ratio; the red
            dashed line marks 1.0. The lower panel compares the VIX and VIX3M
            levels over the selected six-month data window.

            **Important:** these are mechanical heuristics, not a forecast or
            a guarantee of a market bottom. They do not account for price
            trends, portfolio needs, transaction costs, or your risk
            tolerance. Validate signals independently and do not treat the
            allocation language above as personalized financial advice.
            """
        )
        st.caption(
            f"Current check: ratio {latest_ratio:.3f} "
            f"({'backwardation' if latest_ratio >= 1.0 else 'contango'}); "
            f"latest VVIX {latest_vvix:.2f}; five-observation VVIX high "
            f"{five_day_max_vvix:.2f}; 90% threshold "
            f"{five_day_max_vvix * 0.9:.2f}."
        )

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("VIX / VIX3M Ratio", "VIX vs. VIX3M"),
    )
    figure.add_trace(
        go.Scatter(
            x=history.index,
            y=history["VIX_Ratio"],
            name="VIX / VIX3M",
            line=dict(color="#42A5F5", width=2),
        ),
        row=1,
        col=1,
    )
    figure.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="red",
        annotation_text="Backwardation threshold (1.0)",
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=history.index,
            y=history["^VIX"],
            name="VIX",
            line=dict(color="#EF5350", width=2),
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=history.index,
            y=history["^VIX3M"],
            name="VIX3M",
            line=dict(color="#26A69A", width=2),
        ),
        row=2,
        col=1,
    )
    figure.update_layout(
        template="plotly_dark",
        height=700,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=90, b=35),
    )
    figure.update_yaxes(title_text="Ratio", row=1, col=1)
    figure.update_yaxes(title_text="Volatility Index", row=2, col=1)
    st.plotly_chart(figure, use_container_width=True)
