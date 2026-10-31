"""Valuation simulations independent from the Streamlit presentation layer."""

import numpy as np


def _finite_float(value, default=0.0):
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return float(default)
    return converted if np.isfinite(converted) else float(default)


def simulate_dcf_intrinsic_values(
    base_revenue,
    revenue_growth_rate,
    target_net_margin,
    cash_flow_conversion_rate,
    forecast_period,
    discount_rate,
    exit_multiple,
    total_shares_outstanding,
    terminal_growth_rate=0.02,
    terminal_value_eligible=True,
    iterations=1000,
    seed=17,
):
    """Simulate per-share FCFE values with bounded WACC, growth, and terminal rates."""
    shares = _finite_float(total_shares_outstanding)
    revenue = max(_finite_float(base_revenue), 0.0)
    iteration_count = max(0, int(_finite_float(iterations)))
    if shares <= 0 or revenue <= 0 or iteration_count <= 0:
        return {
            "values": np.array([], dtype=float),
            "mean": np.nan,
            "median": np.nan,
            "var_5": np.nan,
            "var_95": np.nan,
            "probability_above_price": np.nan,
        }

    periods = max(1, int(_finite_float(forecast_period, 1)))
    growth_center = float(
        np.clip(_finite_float(revenue_growth_rate), -0.50, 1.0)
    )
    margin = float(np.clip(_finite_float(target_net_margin), 0.0, 1.0))
    conversion = float(
        np.clip(_finite_float(cash_flow_conversion_rate), 0.0, 2.0)
    )
    wacc_center = float(np.clip(_finite_float(discount_rate, 0.09), 0.01, 0.50))
    terminal_center = float(
        np.clip(
            _finite_float(terminal_growth_rate, 0.02),
            0.0,
            min(0.04, wacc_center - 0.01),
        )
    )
    multiple_center = max(_finite_float(exit_multiple), 0.0)
    generator = np.random.default_rng(seed)

    growth_draws = np.clip(
        generator.normal(growth_center, 0.06, iteration_count), -0.50, 0.80
    )
    wacc_draws = np.clip(
        generator.normal(
            wacc_center,
            max(0.005, wacc_center * 0.12),
            iteration_count,
        ),
        0.02,
        0.50,
    )
    terminal_draws = np.clip(
        generator.normal(terminal_center, 0.005, iteration_count),
        0.0,
        np.minimum(0.04, wacc_draws - 0.01),
    )
    multiple_draws = np.clip(
        generator.normal(
            multiple_center,
            max(0.5, multiple_center * 0.15),
            iteration_count,
        ),
        0.0,
        25.0,
    )

    projected_revenue = np.full(iterations, revenue, dtype=float)
    present_value = np.zeros(iterations, dtype=float)
    final_fcfe = np.zeros(iterations, dtype=float)
    for year in range(1, periods + 1):
        projected_revenue *= 1.0 + growth_draws
        final_fcfe = projected_revenue * margin * conversion
        present_value += final_fcfe / ((1.0 + wacc_draws) ** year)

    if terminal_value_eligible:
        sustainable_multiple = (
            (1.0 + terminal_draws) / (wacc_draws - terminal_draws)
        )
        terminal_multiple = np.minimum(
            np.minimum(multiple_draws, sustainable_multiple), 25.0
        )
        terminal_value = final_fcfe * terminal_multiple
        present_value += terminal_value / ((1.0 + wacc_draws) ** periods)

    values = np.maximum(present_value / shares, 0.0)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "values": values,
            "mean": np.nan,
            "median": np.nan,
            "var_5": np.nan,
            "var_95": np.nan,
            "probability_above_price": np.nan,
        }
    return {
        "values": values,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "var_5": float(np.percentile(values, 5)),
        "var_95": float(np.percentile(values, 95)),
        "probability_above_price": None,
    }


def summarize_monte_carlo(values, current_price):
    """Summarize simulation output against the selected ticker's current price."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "mean": np.nan,
            "median": np.nan,
            "var_5": np.nan,
            "var_95": np.nan,
            "probability_above_price": np.nan,
        }
    price = _finite_float(current_price)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "var_5": float(np.percentile(values, 5)),
        "var_95": float(np.percentile(values, 95)),
        "probability_above_price": (
            float(np.mean(values > price)) if price > 0 else np.nan
        ),
    }
