"""Vectorized quantitative analytics and historical signal backtests."""

import numpy as np
import pandas as pd


def fit_trade_return_regression(x_values, y_values):
    """Fit a linear OLS trend only when the sample can support a useful fit."""
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("Regression x and y values must have matching lengths.")

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if (
        len(x) < 4
        or np.unique(x).size < 3
        or np.unique(y).size < 2
    ):
        return None

    x_center = float(x.mean())
    x_scale = float(x.std())
    total_sum = float(np.sum((y - y.mean()) ** 2))
    if not np.isfinite(x_scale) or x_scale == 0 or total_sum == 0:
        return None

    scaled_x = (x - x_center) / x_scale
    design = np.column_stack((scaled_x, np.ones_like(scaled_x)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != 2 or not np.isfinite(coefficients).all():
        return None

    fitted_y = design @ coefficients
    residual_sum = float(np.sum((y - fitted_y) ** 2))
    fitted_x = np.linspace(float(x.min()), float(x.max()), 100)
    fitted_scaled_x = (fitted_x - x_center) / x_scale
    fitted_line = coefficients[0] * fitted_scaled_x + coefficients[1]
    return {
        "x": fitted_x,
        "y": fitted_line,
        "r_squared": 1.0 - residual_sum / total_sum,
    }


def rolling_correlation_matrix(price_frames, window=30, min_periods=None):
    """Return the latest rolling correlation matrix for aligned close prices."""
    valid_window = max(2, int(window))
    required = min_periods or max(2, int(valid_window * 0.70))
    close_series = {}
    for symbol, frame in price_frames.items():
        if frame is None or frame.empty or "Close" not in frame.columns:
            continue
        close = pd.to_numeric(frame["Close"], errors="coerce")
        close = close.replace([np.inf, -np.inf], np.nan).dropna()
        if len(close) >= required:
            close_series[symbol] = close.rename(symbol)
    if len(close_series) < 2:
        return pd.DataFrame()
    returns = pd.concat(close_series.values(), axis=1).sort_index().pct_change()
    returns = returns.tail(valid_window).dropna(how="all")
    if returns.shape[0] < required:
        return pd.DataFrame()
    matrix = returns.corr(min_periods=required)
    return matrix.replace([np.inf, -np.inf], np.nan)


def rolling_correlation_history(price_frames, window=30, min_periods=None):
    """Return a date-indexed rolling correlation series for asset pairs."""
    valid_window = max(2, int(window))
    required = min_periods or max(2, int(valid_window * 0.70))
    close_series = {}
    for symbol, frame in price_frames.items():
        if frame is None or frame.empty or "Close" not in frame.columns:
            continue
        close_series[symbol] = pd.to_numeric(
            frame["Close"], errors="coerce"
        ).rename(symbol)
    if len(close_series) < 2:
        return pd.DataFrame()
    returns = pd.concat(close_series.values(), axis=1).sort_index().pct_change()
    pair_values = {}
    symbols = list(returns.columns)
    for left_index, left in enumerate(symbols):
        for right in symbols[left_index + 1:]:
            pair_values[f"{left} / {right}"] = (
                returns[left].rolling(valid_window, min_periods=required)
                .corr(returns[right])
            )
    return pd.DataFrame(pair_values)


def backtest_signal_strategy(
    history,
    signal_mask,
    hold_limit=30,
    target_return=0.15,
    stop_return=0.05,
    vix_rate_of_change=None,
    calm_vix_threshold=-10.0,
    panic_vix_threshold=15.0,
    calm_target_multiplier=1.5,
    panic_stop_multiplier=0.7,
    signal_indicator_values=None,
    vix_overlay_enabled=True,
):
    """Backtest sequential long signals with optional VIX-adjusted exits."""
    required = {"High", "Low", "Close"}
    if history.empty or not required.issubset(history.columns):
        return {
            "trades": pd.DataFrame(),
            "metrics": {
                "trade_count": 0,
                "win_rate": np.nan,
                "average_holding_period": np.nan,
                "max_drawdown": np.nan,
                "total_return": np.nan,
            },
        }
    data = history.sort_index().copy()
    mask = pd.Series(signal_mask, index=data.index).fillna(False).astype(bool)
    if not vix_overlay_enabled or vix_rate_of_change is None:
        vix_roc = pd.Series(np.nan, index=data.index, dtype=float)
    else:
        vix_roc = pd.to_numeric(
            pd.Series(vix_rate_of_change).reindex(data.index), errors="coerce"
        )
    if signal_indicator_values is None:
        signal_values = pd.Series(np.nan, index=data.index, dtype=float)
    else:
        signal_values = pd.to_numeric(
            pd.Series(signal_indicator_values).reindex(data.index),
            errors="coerce",
        )
    high = pd.to_numeric(data["High"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(data["Low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(data["Close"], errors="coerce").to_numpy(dtype=float)
    signal_indexes = np.flatnonzero(mask.to_numpy())
    trades = []
    next_available_index = 0

    for signal_index in signal_indexes:
        if signal_index < next_available_index or signal_index >= len(data) - 1:
            continue
        entry_index = signal_index + 1
        entry_price = close[entry_index]
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        entry_vix_roc = vix_roc.iloc[entry_index]
        if not vix_overlay_enabled:
            vix_regime = "Disabled"
            target_multiplier = 1.0
            stop_multiplier = 1.0
        elif pd.isna(entry_vix_roc):
            vix_regime = "Unavailable"
            target_multiplier = 1.0
            stop_multiplier = 1.0
        elif entry_vix_roc <= calm_vix_threshold:
            vix_regime = "Calm"
            target_multiplier = max(float(calm_target_multiplier), 0.0)
            stop_multiplier = 1.0
        elif entry_vix_roc >= panic_vix_threshold:
            vix_regime = "Panic"
            target_multiplier = 1.0
            stop_multiplier = max(float(panic_stop_multiplier), 0.0)
        else:
            vix_regime = "Normal"
            target_multiplier = 1.0
            stop_multiplier = 1.0
        applied_target = max(float(target_return), 0.0) * target_multiplier
        applied_stop = max(float(stop_return), 0.0) * stop_multiplier
        target_price = entry_price * (1.0 + applied_target)
        stop_price = entry_price * (1.0 - applied_stop)
        final_index = min(len(data) - 1, entry_index + max(1, int(hold_limit)))
        exit_index = final_index
        exit_price = close[final_index]
        if not np.isfinite(exit_price):
            continue
        exit_reason = "Time exit"

        for future_index in range(entry_index, final_index + 1):
            hit_stop = low[future_index] <= stop_price
            hit_target = high[future_index] >= target_price
            if hit_stop:
                exit_index = future_index
                exit_price = stop_price
                exit_reason = "Stop"
                break
            if hit_target:
                exit_index = future_index
                exit_price = target_price
                exit_reason = "Target"
                break
        holding_period = max(1, exit_index - entry_index + 1)
        trade_return = exit_price / entry_price - 1.0
        trades.append({
            "Signal Date": data.index[signal_index],
            "Entry Date": data.index[entry_index],
            "Exit Date": data.index[exit_index],
            "Entry Price": entry_price,
            "Exit Price": exit_price,
            "Holding Period (bars)": holding_period,
            "Return (%)": trade_return * 100.0,
            "Exit Reason": exit_reason,
            "VIX ROC (%)": entry_vix_roc,
            "VIX Regime": vix_regime,
            "VIX Calm": vix_regime == "Calm",
            "VIX Panic": vix_regime == "Panic",
            "Applied Target (%)": applied_target * 100.0,
            "Applied Stop (%)": applied_stop * 100.0,
            "Entry Signal Value": signal_values.iloc[signal_index],
        })
        next_available_index = exit_index + 1

    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        metrics = {
            "trade_count": 0,
            "win_rate": np.nan,
            "average_holding_period": np.nan,
            "max_drawdown": np.nan,
            "total_return": 0.0,
        }
        return {"trades": trade_frame, "metrics": metrics}

    returns = trade_frame["Return (%)"].to_numpy(dtype=float) / 100.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))
    drawdown = equity / peaks[1:] - 1.0
    metrics = {
        "trade_count": int(len(trade_frame)),
        "win_rate": float((returns > 0).mean() * 100.0),
        "average_holding_period": float(
            trade_frame["Holding Period (bars)"].mean()
        ),
        "max_drawdown": float(drawdown.min() * 100.0),
        "total_return": float((equity[-1] - 1.0) * 100.0),
    }
    return {"trades": trade_frame, "metrics": metrics}
