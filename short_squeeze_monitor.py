#!/usr/bin/env python3
"""
US Stock Short Squeeze Monitor AI Agent.

Pipeline:
1. Scrape/parse Fintel leaderboard data, falling back to a local fintel.html file.
2. Enrich each ticker with yfinance previous completed trading-day volume,
   3-month average volume, and relative volume.
3. Evaluate a multi-factor short squeeze risk score and print high-alert targets.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from io import StringIO
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


DEFAULT_FINTEL_URL = "https://fintel.io"
DEFAULT_LOCAL_HTML = "fintel.html"
REQUEST_TIMEOUT_SECONDS = 30
MARKET_TIMEZONE = ZoneInfo("America/New_York")
MARKET_CLOSE_BUFFER = dt_time(hour=16, minute=10)


@dataclass(frozen=True)
class MonitorConfig:
    fintel_url: str
    local_html_path: Path
    min_sleep_seconds: float
    max_sleep_seconds: float
    limit: int | None
    output_csv_path: Path | None


def log_phase(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"PHASE: {title}")
    print("=" * 80)


def log(message: str) -> None:
    print(f"[ShortSqueezeMonitor] {message}", flush=True)


def browser_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
            "image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "DNT": "1",
        "Pragma": "no-cache",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "sec-ch-ua": '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
    }


def fetch_fintel_html(url: str, local_html_path: Path) -> str:
    """Fetch Fintel HTML directly, then fall back to a browser-saved local file."""
    log(f"Attempting direct HTTP request to {url}")
    session = requests.Session()
    session.headers.update(browser_headers())

    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        log(f"Direct request completed with HTTP status {response.status_code}")
        if response.status_code == 200 and response.text.strip():
            return response.text

        log(
            "Direct request did not return usable HTTP 200 HTML. "
            "Cloudflare or anti-bot protection may be active."
        )
    except requests.RequestException as exc:
        log(f"Direct request failed: {exc}")

    log(f"Falling back to local HTML file: {local_html_path}")
    html = read_local_html(local_html_path)
    log(f"Loaded {len(html):,} characters from local fallback HTML.")
    return html


def read_local_html(local_html_path: Path) -> str:
    if not local_html_path.exists():
        raise FileNotFoundError(f"Local fallback file was not found: {local_html_path}")

    html = local_html_path.read_text(encoding="utf-8", errors="replace")
    if not html.strip():
        raise ValueError(f"Local fallback file is empty: {local_html_path}")
    return html


def clean_column_name(column: object) -> str:
    if isinstance(column, tuple):
        column = " ".join(str(part) for part in column if str(part) != "nan")

    cleaned = re.sub(r"\s+", " ", str(column)).strip()
    cleaned = cleaned.replace("\xa0", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_cell_value(value: object) -> object:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()
    return value


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [clean_column_name(column) for column in normalized.columns]
    normalized = normalized.apply(lambda column: column.map(clean_cell_value))
    normalized = normalized.dropna(axis=0, how="all")
    normalized = normalized.loc[:, [column for column in normalized.columns if column]]
    return normalized.reset_index(drop=True)


def table_score(columns: Iterable[str]) -> int:
    joined = " | ".join(column.lower() for column in columns)
    score = 0
    for keyword in ("security", "ticker", "symbol"):
        if keyword in joined:
            score += 4
    for keyword in ("short", "squeeze", "borrow", "fee", "float"):
        if keyword in joined:
            score += 2
    return score


def parse_leaderboard_table(html: str) -> pd.DataFrame:
    """Extract the most likely Fintel leaderboard table from HTML."""
    log("Parsing HTML with BeautifulSoup and pandas.read_html.")
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    table_html_fragments = [str(table) for table in soup.find_all("table")]
    if not table_html_fragments:
        raise ValueError("No HTML <table> elements were found in the supplied Fintel HTML.")

    candidates: list[tuple[int, int, pd.DataFrame]] = []
    for index, table_html in enumerate(table_html_fragments):
        try:
            parsed_tables = pd.read_html(StringIO(table_html))
        except ValueError:
            continue

        for parsed_df in parsed_tables:
            normalized = normalize_dataframe(parsed_df)
            if normalized.empty:
                continue
            score = table_score(normalized.columns)
            candidates.append((score, index, normalized))

    if not candidates:
        raise ValueError("pandas.read_html could not parse a non-empty leaderboard table.")

    candidates.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    best_score, best_index, best_df = candidates[0]
    log(
        f"Selected table #{best_index + 1} with heuristic score {best_score} "
        f"and shape {best_df.shape[0]} rows x {best_df.shape[1]} columns."
    )

    security_column = find_column(best_df, ["security", "ticker", "symbol"], required=False)
    if security_column is None:
        raise ValueError(
            "Could not identify a Security/Ticker/Symbol column in the selected table. "
            f"Columns found: {list(best_df.columns)}"
        )

    return best_df


def find_column(
    df: pd.DataFrame,
    possible_terms: list[str],
    *,
    required: bool = True,
) -> str | None:
    normalized_terms = [term.lower() for term in possible_terms]
    for column in df.columns:
        lower_column = column.lower()
        if any(term in lower_column for term in normalized_terms):
            return column

    if required:
        raise KeyError(
            f"Could not find a column matching terms {possible_terms}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def extract_ticker(security_value: object) -> str:
    text = str(security_value).strip()
    if "/" in text:
        text = text.split("/", maxsplit=1)[0].strip()
    else:
        text = text.split(maxsplit=1)[0].strip()

    ticker = re.sub(r"[^A-Za-z0-9.\-]", "", text).upper()
    return ticker


def previous_completed_volume(history: pd.DataFrame) -> int:
    if history.empty or "Volume" not in history.columns:
        raise ValueError("No yfinance historical volume data was returned.")

    cleaned_history = history.dropna(subset=["Volume"])
    if cleaned_history.empty:
        raise ValueError("Historical yfinance data contained no valid Volume values.")

    now_et = datetime.now(tz=MARKET_TIMEZONE)
    last_index = cleaned_history.index[-1]
    last_date = pd.Timestamp(last_index).date()

    use_previous_row = last_date == now_et.date() and now_et.time() < MARKET_CLOSE_BUFFER
    if use_previous_row:
        if len(cleaned_history) < 2:
            raise ValueError("Only an in-progress current-day volume row is available.")
        selected_volume = cleaned_history["Volume"].iloc[-2]
    else:
        selected_volume = cleaned_history["Volume"].iloc[-1]

    return int(selected_volume)


def safe_float_from_percent(value: object) -> float:
    if pd.isna(value):
        return 0.0

    text = str(value)
    text = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return 0.0
    return float(match.group(0))


def safe_float(value: object) -> float:
    if pd.isna(value):
        return 0.0

    text = str(value).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return 0.0
    return float(match.group(0))


def enrich_with_volume_data(
    df: pd.DataFrame,
    min_sleep_seconds: float,
    max_sleep_seconds: float,
    limit: int | None,
) -> pd.DataFrame:
    security_column = find_column(df, ["security", "ticker", "symbol"])
    enriched = df.copy() if limit is None else df.iloc[:limit].copy()

    enriched["Ticker"] = ""
    enriched["Yesterday Volume"] = 0
    enriched["3M Average Volume"] = 0
    enriched["RVOL"] = 0.0
    enriched["Volume Fetch Status"] = ""

    rows_to_process = enriched.index
    log(f"Preparing to enrich {len(rows_to_process)} ticker rows via yfinance.")

    for row_number, row_index in enumerate(rows_to_process, start=1):
        security_value = enriched.at[row_index, security_column]
        ticker_symbol = extract_ticker(security_value)
        enriched.at[row_index, "Ticker"] = ticker_symbol

        if not ticker_symbol:
            log(f"Row {row_number}: no ticker could be extracted from Security={security_value!r}; skipping.")
            enriched.at[row_index, "Volume Fetch Status"] = "skipped: no ticker"
            continue

        log(f"Row {row_number}/{len(rows_to_process)}: processing ticker {ticker_symbol}")
        try:
            ticker = yf.Ticker(ticker_symbol)
            history = ticker.history(period="5d", interval="1d", auto_adjust=False)
            yesterday_volume = previous_completed_volume(history)

            info = ticker.info or {}
            average_volume = int(info.get("averageVolume") or 0)
            rvol = (yesterday_volume / average_volume) if average_volume > 0 else 0.0

            enriched.at[row_index, "Yesterday Volume"] = yesterday_volume
            enriched.at[row_index, "3M Average Volume"] = average_volume
            enriched.at[row_index, "RVOL"] = round(rvol, 4)
            enriched.at[row_index, "Volume Fetch Status"] = "ok"

            log(
                f"{ticker_symbol}: yesterday volume={yesterday_volume:,}, "
                f"3M avg volume={average_volume:,}, RVOL={rvol:.2f}x"
            )
        except Exception as exc:
            log(f"{ticker_symbol}: yfinance lookup failed; skipping row. Error: {exc}")
            enriched.at[row_index, "Volume Fetch Status"] = f"error: {exc}"

        sleep_seconds = random.uniform(min_sleep_seconds, max_sleep_seconds)
        log(f"Sleeping {sleep_seconds:.2f}s for rate limiting.")
        time.sleep(sleep_seconds)

    return enriched


def score_row(
    row: pd.Series,
    short_float_column: str | None,
    borrow_fee_column: str | None,
    squeeze_score_column: str | None,
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    short_float = safe_float_from_percent(row.get(short_float_column, 0)) if short_float_column else 0.0
    if short_float > 40:
        score += 40
        reasons.append("Short Float > 40%")
    elif short_float > 20:
        score += 20
        reasons.append("Short Float > 20%")

    borrow_fee = safe_float_from_percent(row.get(borrow_fee_column, 0)) if borrow_fee_column else 0.0
    if borrow_fee > 100:
        score += 30
        reasons.append("Borrow Fee > 100%")
    elif borrow_fee > 30:
        score += 15
        reasons.append("Borrow Fee > 30%")

    rvol = safe_float(row.get("RVOL", 0))
    if rvol > 3.0:
        score += 30
        reasons.append("RVOL > 3.0x")
    elif rvol > 1.5:
        score += 15
        reasons.append("RVOL > 1.5x")

    squeeze_score = safe_float(row.get(squeeze_score_column, 0)) if squeeze_score_column else 0.0
    if squeeze_score > 90:
        score += 10
        reasons.append("Fintel Score > 90")

    return score, reasons


def evaluate_short_squeeze_risk(df: pd.DataFrame) -> pd.DataFrame:
    short_float_column = find_column(df, ["short float"], required=False)
    borrow_fee_column = find_column(df, ["borrow fee", "fee rate"], required=False)
    squeeze_score_column = find_column(df, ["squeeze score", "short squeeze score", "score"], required=False)

    log(f"Short Float column: {short_float_column or 'not found'}")
    log(f"Borrow Fee column: {borrow_fee_column or 'not found'}")
    log(f"Fintel Short Squeeze Score column: {squeeze_score_column or 'not found'}")

    evaluated = df.copy()
    scores: list[int] = []
    reasons_list: list[str] = []

    for _, row in evaluated.iterrows():
        ticker_symbol = row.get("Ticker") or extract_ticker(row.get("Security", ""))
        log(f"Evaluating ticker {ticker_symbol or 'UNKNOWN'}")
        score, reasons = score_row(row, short_float_column, borrow_fee_column, squeeze_score_column)
        scores.append(score)
        reasons_list.append("; ".join(reasons))

    evaluated["Agent Score"] = scores
    evaluated["Alert Reasons"] = reasons_list
    evaluated["High Alert"] = evaluated["Agent Score"] >= 50

    high_alerts = evaluated[evaluated["High Alert"]].copy()
    high_alerts = high_alerts.sort_values(by="Agent Score", ascending=False).reset_index(drop=True)
    log(f"Flagged {len(high_alerts)} high-probability targets with Agent Score >= 50.")
    return high_alerts


def format_int(value: object) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def format_float(value: object, decimals: int = 2) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "No high-alert short squeeze targets found."

    available_columns = [column for column in columns if column in df.columns]
    rows = []
    for _, row in df[available_columns].iterrows():
        rendered_row = []
        for column in available_columns:
            value = row[column]
            if column in {"Yesterday Volume", "3M Average Volume"}:
                rendered_row.append(format_int(value))
            elif column == "RVOL":
                rendered_row.append(f"{format_float(value)}x")
            else:
                rendered_row.append(str(value))
        rows.append(rendered_row)

    widths = [
        max(len(column), *(len(row[index]) for row in rows))
        for index, column in enumerate(available_columns)
    ]
    header = "| " + " | ".join(column.ljust(widths[index]) for index, column in enumerate(available_columns)) + " |"
    separator = "| " + " | ".join("-" * widths[index] for index in range(len(widths))) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def print_high_alert_targets(high_alerts: pd.DataFrame) -> None:
    preferred_columns = [
        "Ticker",
        "Security",
        "Agent Score",
        "Short Float %",
        "Borrow Fee Rate",
        "Short Squeeze Score",
        "Yesterday Volume",
        "3M Average Volume",
        "RVOL",
        "Alert Reasons",
    ]
    print("\nHIGH-ALERT SHORT SQUEEZE TARGETS")
    print(markdown_table(high_alerts, preferred_columns))


def parse_args(argv: list[str]) -> MonitorConfig:
    parser = argparse.ArgumentParser(
        description="Run the US Stock Short Squeeze Monitor AI Agent pipeline."
    )
    parser.add_argument("--url", default=DEFAULT_FINTEL_URL, help="Fintel URL to scrape.")
    parser.add_argument(
        "--local-html",
        default=DEFAULT_LOCAL_HTML,
        help="Fallback local HTML file saved manually from the browser.",
    )
    parser.add_argument(
        "--min-sleep",
        type=float,
        default=0.5,
        help="Minimum random yfinance rate-limit sleep interval in seconds.",
    )
    parser.add_argument(
        "--max-sleep",
        type=float,
        default=1.5,
        help="Maximum random yfinance rate-limit sleep interval in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to enrich. Useful for smoke tests.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to write high-alert results as CSV.",
    )

    args = parser.parse_args(argv)
    if args.min_sleep < 0 or args.max_sleep < 0:
        parser.error("--min-sleep and --max-sleep must be non-negative.")
    if args.min_sleep > args.max_sleep:
        parser.error("--min-sleep cannot exceed --max-sleep.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer when provided.")

    return MonitorConfig(
        fintel_url=args.url,
        local_html_path=Path(args.local_html),
        min_sleep_seconds=args.min_sleep,
        max_sleep_seconds=args.max_sleep,
        limit=args.limit,
        output_csv_path=Path(args.output_csv) if args.output_csv else None,
    )


def run_pipeline(config: MonitorConfig) -> pd.DataFrame:
    log_phase("1. DATA SCRAPING AND PARSING (SCRAPE)")
    html = fetch_fintel_html(config.fintel_url, config.local_html_path)
    try:
        leaderboard_df = parse_leaderboard_table(html)
    except Exception as exc:
        log(f"Initial HTML parse failed: {exc}")
        log("Attempting to parse local fallback HTML in case direct HTTP returned a protected page.")
        local_html = read_local_html(config.local_html_path)
        leaderboard_df = parse_leaderboard_table(local_html)

    log(f"Parsed leaderboard shape: {leaderboard_df.shape[0]} rows x {leaderboard_df.shape[1]} columns.")
    log(f"Parsed columns: {list(leaderboard_df.columns)}")

    log_phase("2. DYNAMIC VOLUME INJECTION (GET VOLUME)")
    enriched_df = enrich_with_volume_data(
        leaderboard_df,
        config.min_sleep_seconds,
        config.max_sleep_seconds,
        config.limit,
    )

    log_phase("3. MULTI-FACTOR QUANT EVALUATION (EVALUATE)")
    high_alerts = evaluate_short_squeeze_risk(enriched_df)
    print_high_alert_targets(high_alerts)

    if config.output_csv_path is not None:
        high_alerts.to_csv(config.output_csv_path, index=False)
        log(f"Wrote high-alert CSV output to {config.output_csv_path}")

    return high_alerts


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        run_pipeline(config)
        return 0
    except Exception as exc:
        log(f"Pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
