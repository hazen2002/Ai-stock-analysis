"""Fundamental quality and earnings-event helpers."""

import datetime

import numpy as np
import pandas as pd


_INCOME_ROWS = {
    "net_income": (
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income From Continuing Operations",
        "Net Income Continuous Operations",
    ),
    "revenue": ("Total Revenue", "Operating Revenue", "Revenue"),
    "gross_profit": ("Gross Profit",),
}
_BALANCE_ROWS = {
    "assets": ("Total Assets",),
    "current_assets": ("Current Assets",),
    "current_liabilities": ("Current Liabilities",),
    "long_term_debt": (
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation",
        "Long Term Debt And Capital Lease Obligation Current",
    ),
    "shares": (
        "Ordinary Shares Number",
        "Share Issued",
        "Common Stock Shares Outstanding",
    ),
}
_CASH_FLOW_ROWS = {
    "operating_cash_flow": (
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities",
    ),
}


def _period_columns(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    dated_columns = []
    for column in frame.columns:
        try:
            period = pd.Timestamp(column)
        except (TypeError, ValueError):
            continue
        if pd.notna(period):
            dated_columns.append((period, column))
    return [column for _, column in sorted(dated_columns, reverse=True)]


def _row_value(frame, row_names, period_index=0):
    columns = _period_columns(frame)
    if period_index >= len(columns):
        return None
    column = columns[period_index]
    for row_name in row_names:
        if row_name not in frame.index:
            continue
        try:
            value = float(pd.to_numeric(frame.at[row_name, column], errors="coerce"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def calculate_piotroski_f_score(financials, balance_sheet, cashflow):
    """Calculate a 0–9 Piotroski score from the latest two annual periods.

    Unknown criteria are not awarded points. ``coverage`` reports how many of
    the nine signals could be evaluated, so incomplete statements remain visible.
    """
    income = {
        name: (
            _row_value(financials, row_names, 0),
            _row_value(financials, row_names, 1),
        )
        for name, row_names in _INCOME_ROWS.items()
    }
    balance = {
        name: (
            _row_value(balance_sheet, row_names, 0),
            _row_value(balance_sheet, row_names, 1),
        )
        for name, row_names in _BALANCE_ROWS.items()
    }
    cash = {
        name: (
            _row_value(cashflow, row_names, 0),
            _row_value(cashflow, row_names, 1),
        )
        for name, row_names in _CASH_FLOW_ROWS.items()
    }

    net_income, prior_net_income = income["net_income"]
    revenue, prior_revenue = income["revenue"]
    gross_profit, prior_gross_profit = income["gross_profit"]
    assets, prior_assets = balance["assets"]
    current_assets, prior_current_assets = balance["current_assets"]
    current_liabilities, prior_current_liabilities = balance["current_liabilities"]
    long_term_debt, prior_long_term_debt = balance["long_term_debt"]
    shares, prior_shares = balance["shares"]
    operating_cash_flow, _ = cash["operating_cash_flow"]
    if gross_profit is None:
        cost_of_revenue = _row_value(
            financials,
            ("Cost Of Revenue", "Cost Of Goods And Services Sold"),
            0,
        )
        if revenue is not None and cost_of_revenue is not None:
            gross_profit = revenue - cost_of_revenue
    if prior_gross_profit is None:
        prior_cost_of_revenue = _row_value(
            financials,
            ("Cost Of Revenue", "Cost Of Goods And Services Sold"),
            1,
        )
        if prior_revenue is not None and prior_cost_of_revenue is not None:
            prior_gross_profit = prior_revenue - prior_cost_of_revenue

    current_roa = (
        net_income / assets
        if net_income is not None and assets is not None and assets > 0
        else None
    )
    prior_roa = (
        prior_net_income / prior_assets
        if prior_net_income is not None and prior_assets is not None and prior_assets > 0
        else None
    )
    current_debt_ratio = (
        long_term_debt / assets
        if long_term_debt is not None and assets is not None and assets > 0
        else None
    )
    prior_debt_ratio = (
        prior_long_term_debt / prior_assets
        if prior_long_term_debt is not None
        and prior_assets is not None
        and prior_assets > 0
        else None
    )
    current_ratio = (
        current_assets / current_liabilities
        if current_assets is not None
        and current_liabilities is not None
        and current_liabilities > 0
        else None
    )
    prior_current_ratio = (
        prior_current_assets / prior_current_liabilities
        if prior_current_assets is not None
        and prior_current_liabilities is not None
        and prior_current_liabilities > 0
        else None
    )
    gross_margin = (
        gross_profit / revenue
        if gross_profit is not None and revenue is not None and revenue > 0
        else None
    )
    prior_gross_margin = (
        prior_gross_profit / prior_revenue
        if prior_gross_profit is not None
        and prior_revenue is not None
        and prior_revenue > 0
        else None
    )
    asset_turnover = (
        revenue / assets
        if revenue is not None and assets is not None and assets > 0
        else None
    )
    prior_asset_turnover = (
        prior_revenue / prior_assets
        if prior_revenue is not None and prior_assets is not None and prior_assets > 0
        else None
    )

    criteria = {
        "Positive ROA": (
            net_income > 0 and assets > 0
            if net_income is not None and assets is not None
            else None
        ),
        "Positive Operating Cash Flow": (
            operating_cash_flow > 0 if operating_cash_flow is not None else None
        ),
        "Improving ROA": (
            current_roa > prior_roa
            if current_roa is not None and prior_roa is not None
            else None
        ),
        "Cash Flow Exceeds Net Income": (
            operating_cash_flow > net_income
            if operating_cash_flow is not None and net_income is not None
            else None
        ),
        "Lower Long-Term Debt Ratio": (
            current_debt_ratio < prior_debt_ratio
            if current_debt_ratio is not None and prior_debt_ratio is not None
            else None
        ),
        "Improving Current Ratio": (
            current_ratio > prior_current_ratio
            if current_ratio is not None and prior_current_ratio is not None
            else None
        ),
        "No Share Dilution": (
            shares <= prior_shares
            if shares is not None and prior_shares is not None
            else None
        ),
        "Improving Gross Margin": (
            gross_margin > prior_gross_margin
            if gross_margin is not None and prior_gross_margin is not None
            else None
        ),
        "Improving Asset Turnover": (
            asset_turnover > prior_asset_turnover
            if asset_turnover is not None and prior_asset_turnover is not None
            else None
        ),
    }
    evaluated = [value for value in criteria.values() if value is not None]
    return {
        "score": int(sum(value is True for value in criteria.values())),
        "coverage": len(evaluated),
        "criteria": criteria,
        "metrics": {
            "current_roa": current_roa,
            "prior_roa": prior_roa,
            "current_debt_ratio": current_debt_ratio,
            "prior_debt_ratio": prior_debt_ratio,
            "current_ratio": current_ratio,
            "prior_current_ratio": prior_current_ratio,
            "gross_margin": gross_margin,
            "prior_gross_margin": prior_gross_margin,
            "asset_turnover": asset_turnover,
            "prior_asset_turnover": prior_asset_turnover,
        },
    }


def next_earnings_date(candidates, today=None):
    """Return the nearest candidate date on or after today, or ``None``."""
    reference_date = today or datetime.date.today()
    parsed_dates = []
    for candidate in candidates or ():
        if candidate is None:
            continue
        try:
            if isinstance(candidate, (int, float, np.number)):
                numeric_candidate = float(candidate)
                if not np.isfinite(numeric_candidate) or numeric_candidate <= 0:
                    continue
                unit = "ms" if numeric_candidate > 1e12 else "s"
                parsed = pd.Timestamp(numeric_candidate, unit=unit, tz="UTC")
            else:
                parsed = pd.Timestamp(candidate)
            if pd.isna(parsed):
                continue
            if parsed.tzinfo is not None:
                parsed = parsed.tz_convert(None)
            event_date = parsed.date()
        except (TypeError, ValueError, OverflowError):
            continue
        if event_date >= reference_date:
            parsed_dates.append(event_date)
    return min(parsed_dates) if parsed_dates else None
