"""Cached macro and watchlist earnings event data."""

import concurrent.futures
import datetime
import html
import json
import re

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from utils.fundamentals import next_earnings_date


_FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
_BLS_RELEASE_URLS = {
    "CPI": "https://www.bls.gov/schedule/news_release/cpi.htm",
    "NFP": "https://www.bls.gov/schedule/news_release/empsit.htm",
}
_BACKUP_ECONOMIC_CALENDAR_URL = "https://economic-calendar.tradingview.com/events"
_MONTHS = {
    name.lower(): number
    for number, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}
_DATE_PATTERN = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(\d{1,2})"
    r"(?:\s*[-–]\s*(\d{1,2}))?(?:,?\s+(\d{4}))?\b",
    flags=re.IGNORECASE,
)


def _strip_html(document):
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", document, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _parse_fomc_events(document, today):
    text = _strip_html(document)
    events = []
    years = sorted(
        {
            int(year)
            for year in re.findall(r"\b(20\d{2})\s+FOMC Meetings\b", text)
            if int(year) >= today.year
        }
    )
    for year in years:
        section_match = re.search(
            rf"\b{year}\s+FOMC Meetings\b(.*?)(?=\b20\d{{2}}\s+FOMC Meetings\b|$)",
            text,
            flags=re.IGNORECASE,
        )
        if not section_match:
            continue
        for match in _DATE_PATTERN.finditer(section_match.group(1)):
            event_year = int(match.group(4) or year)
            month = _MONTHS[match.group(1).lower()]
            first_day = int(match.group(2))
            second_day = int(match.group(3) or first_day)
            try:
                # FOMC decisions are normally announced on the second meeting day.
                event_date = datetime.date(event_year, month, second_day)
            except ValueError:
                continue
            if event_date >= today:
                events.append({
                    "Date": event_date,
                    "Event": "FOMC 利率決策",
                    "Type": "央行",
                    "Risk": "高風險",
                    "Source": "Federal Reserve",
                })
    return events


def _parse_bls_events(document, release_name, today):
    events = []
    text = _strip_html(document)
    for match in _DATE_PATTERN.finditer(text):
        year = int(match.group(4) or today.year)
        month = _MONTHS[match.group(1).lower()]
        day = int(match.group(2))
        try:
            event_date = datetime.date(year, month, day)
        except ValueError:
            continue
        if event_date < today:
            continue
        events.append({
            "Date": event_date,
            "Event": (
                "CPI 消費者物價指數"
                if release_name == "CPI"
                else "非農就業 NFP / Employment Situation"
            ),
            "Type": "總經",
            "Risk": "高風險" if release_name == "CPI" else "中風險",
            "Source": "U.S. Bureau of Labor Statistics",
        })
    return events


def _parse_backup_economic_events(document, today):
    events = []
    try:
        payload = json.loads(document)
    except (TypeError, ValueError):
        return events
    rows = payload.get("result", []) if isinstance(payload, dict) else []
    for row in rows:
        if not isinstance(row, dict) or str(row.get("country", "")).upper() != "US":
            continue
        title = str(row.get("title") or "")
        normalized_title = re.sub(r"[^a-z]+", " ", title.casefold()).strip()
        if normalized_title in {"cpi", "consumer price index"}:
            event_name = "CPI 消費者物價指數"
            event_type = "總經"
            risk = "高風險"
        elif normalized_title in {
            "non farm payrolls",
            "nonfarm payrolls",
            "nonfarm employment change",
            "employment situation",
        }:
            event_name = "非農就業 NFP / Employment Situation"
            event_type = "總經"
            risk = "高風險"
        elif any(
            phrase in normalized_title
            for phrase in (
                "federal funds rate",
                "fed interest rate decision",
                "fomc",
            )
        ):
            event_name = "FOMC 利率決策"
            event_type = "央行"
            risk = "高風險"
        else:
            continue
        try:
            event_date = pd.Timestamp(row["date"])
            if event_date.tzinfo is not None:
                event_date = event_date.tz_localize(None)
            event_day = event_date.date()
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if event_day < today:
            continue
        events.append({
            "Date": event_day,
            "Event": event_name,
            "Type": event_type,
            "Risk": risk,
            "Source": "TradingView economic calendar fallback",
        })
    return events


@st.cache_data(ttl=datetime.timedelta(hours=12), show_spinner=False)
def fetch_macro_event_schedule(today_iso):
    """Fetch upcoming FOMC, CPI, and employment dates from official calendars."""
    today = datetime.date.fromisoformat(today_iso)
    events = []
    errors = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MarketEventCalendar/1.0)"}

    try:
        response = requests.get(
            _FOMC_CALENDAR_URL,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        events.extend(_parse_fomc_events(response.text, today))
    except (requests.RequestException, ValueError) as error:
        errors.append(f"FOMC calendar unavailable: {error}")

    for release_name, url in _BLS_RELEASE_URLS.items():
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            events.extend(
                _parse_bls_events(response.text, release_name, today)
            )
        except (requests.RequestException, ValueError, ImportError) as error:
            errors.append(f"{release_name} calendar unavailable: {error}")

    missing_release_names = {
        "CPI 消費者物價指數",
        "非農就業 NFP / Employment Situation",
    }.difference(event["Event"] for event in events)
    if "FOMC 利率決策" not in {event["Event"] for event in events}:
        missing_release_names.add("FOMC 利率決策")
    if missing_release_names:
        try:
            start = datetime.datetime.combine(
                today,
                datetime.time.min,
                tzinfo=datetime.timezone.utc,
            )
            end = start + datetime.timedelta(days=30)
            response = requests.get(
                _BACKUP_ECONOMIC_CALENDAR_URL,
                params={
                    "from": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "to": end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "countries": "US",
                },
                headers={
                    **headers,
                    "Origin": "https://www.tradingview.com",
                    "Referer": "https://www.tradingview.com/",
                    "Accept": "application/json",
                },
                timeout=15,
            )
            response.raise_for_status()
            fallback_events = _parse_backup_economic_events(
                response.text,
                today,
            )
            events.extend(
                event
                for event in fallback_events
                if event["Event"] in missing_release_names
            )
        except (requests.RequestException, ValueError) as error:
            errors.append(f"Backup economic calendar unavailable: {error}")

    return events, errors


def _to_yahoo_symbol(symbol):
    normalized = str(symbol).strip().upper()
    if normalized.startswith("."):
        return ""
    if normalized.endswith(".US"):
        return normalized[:-3]
    if normalized.endswith(".HK"):
        code = normalized[:-3]
        if code.isdigit() and len(code) == 5:
            return f"{code[1:]}.HK"
    return normalized


def _fetch_one_earnings_event(symbol, today):
    try:
        yahoo_symbol = _to_yahoo_symbol(symbol)
        if not yahoo_symbol:
            return None, None
        ticker = yf.Ticker(yahoo_symbol)
        calendar = ticker.calendar
        candidates = []
        if isinstance(calendar, pd.DataFrame):
            for index, row in calendar.iterrows():
                if "earn" in str(index).lower():
                    candidates.extend(row.dropna().tolist())
            for column in calendar.columns:
                if "earn" in str(column).lower():
                    candidates.extend(calendar[column].dropna().tolist())
        elif isinstance(calendar, dict):
            for key, value in calendar.items():
                if "earn" not in str(key).lower():
                    continue
                if isinstance(value, (list, tuple, pd.Series)):
                    candidates.extend(value)
                elif value is not None:
                    candidates.append(value)
        event_date = next_earnings_date(candidates, today=today)
        if event_date:
            return {
                "Date": event_date,
                "Event": f"{symbol} 財報",
                "Type": "財報",
                "Risk": "高風險",
                "Source": "Yahoo Finance calendar",
            }, None
        return None, None
    except Exception as error:
        return None, f"{symbol}: {error}"


@st.cache_data(ttl=datetime.timedelta(hours=12), show_spinner=False)
def fetch_watchlist_earnings_events(watchlist, today_iso):
    today = datetime.date.fromisoformat(today_iso)
    symbols = tuple(dict.fromkeys(str(symbol).strip() for symbol in watchlist if symbol))
    events = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_fetch_one_earnings_event, symbol, today): symbol
            for symbol in symbols
        }
        for future in concurrent.futures.as_completed(futures):
            event, error = future.result()
            if event:
                events.append(event)
            if error:
                errors.append(error)
    return events, errors


def get_event_horizon(watchlist, today=None):
    today = today or datetime.date.today()
    macro_events, macro_errors = fetch_macro_event_schedule(today.isoformat())
    earnings_events, earnings_errors = fetch_watchlist_earnings_events(
        tuple(watchlist or ()),
        today.isoformat(),
    )
    events = pd.DataFrame(macro_events + earnings_events)
    if events.empty:
        events = pd.DataFrame(columns=["Date", "Event", "Type", "Risk", "Source"])
    else:
        events["Date"] = pd.to_datetime(events["Date"]).dt.date
        events = (
            events.drop_duplicates(subset=["Date", "Event", "Type"])
            .sort_values(["Date", "Risk", "Event"])
            .reset_index(drop=True)
        )
    return events, macro_errors + earnings_errors
