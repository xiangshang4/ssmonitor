#!/usr/bin/env python3
"""
US Stock Short Squeeze Monitor AI Agent.

Pipeline:
1. Acquire high-short-float candidates from Finviz, falling back to finviz.html.
2. Enrich each ticker with yfinance previous completed trading-day volume,
   3-month average volume, and relative volume.
3. Evaluate a weighted short-squeeze risk score and print high-alert targets.
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
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


DEFAULT_FINVIZ_URL = "https://finviz.com/screener.ashx?v=131&f=sh_short_o20&ft=4"
DEFAULT_LOCAL_HTML = "finviz.html"
REQUEST_TIMEOUT_SECONDS = 30
MARKET_TIMEZONE = ZoneInfo("America/New_York")
MARKET_CLOSE_BUFFER = dt_time(hour=16, minute=10)


@dataclass(frozen=True)
class MonitorConfig:
    finviz_url: str
    local_html_path: Path
    min_sleep_seconds: float
    max_sleep_seconds: float
    limit: int | None
    output_csv_path: Path | None
    max_pages: int


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
        "Referer": "https://finviz.com/screener.ashx",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "sec-ch-ua": '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
    }


def read_local_html(local_html_path: Path) -> str:
    if not local_html_path.exists():
        raise FileNotFoundError(f"Local fallback file was not found: {local_html_path}")

    html = local_html_path.read_text(encoding="utf-8", errors="replace")
    if not html.strip():
        raise ValueError(f"Local fallback file is empty: {local_html_path}")
    return html


def with_finviz_page(url: str, page_number: int) -> str:
    """Finviz pages are 20 rows wide and use r=1, r=21, r=41, ... offsets."""
    if page_number <= 1:
        return url

    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["r"] = str(1 + ((page_number - 1) * 20))
    return urlunparse(parsed._replace(query=urlencode(query)))


def fetch_single_finviz_page(session: requests.Session, url: str) -> str | None:
    log(f"Attempting direct HTTP request to {url}")
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        log(f"Direct request completed with HTTP status {response.status_code}")
        if response.status_code == 200 and response.text.strip():
            return response.text
    except requests.RequestException as exc:
        log(f"Direct request failed: {exc}")

    log("Direct Finviz request did not return usable HTTP 200 HTML.")
    return None


def fetch_finviz_html_pages(url: str, local_html_path: Path, max_pages: int) -> list[str]:
    """Fetch Finviz screener HTML pages, falling back to one browser-saved file."""
    session = requests.Session()
    session.headers.update(browser_headers())
    session.cookies.set("screenerUrl", url, domain="finviz.com")

    pages: list[str] = []
    for page_number in range(1, max_pages + 1):
        page_url = with_finviz_page(url, page_number)
        html = fetch_single_finviz_page(session, page_url)
        if html is None:
            break
        pages.append(html)

        # If this page has no recognizable screener table, continuing will only
        # multiply blocked/protected pages. Parse validation happens later.
        if not has_html_table(html):
            log("Fetched HTML has no table elements; stopping direct pagination.")
            break

        if page_number < max_pages:
            sleep_seconds = random.uniform(0.5, 1.5)
            log(f"Sleeping {sleep_seconds:.2f}s before next Finviz page.")
            time.sleep(sleep_seconds)

    if pages:
        return pages

    log(f"Falling back to local Finviz HTML file: {local_html_path}")
    html = read_local_html(local_html_path)
    log(f"Loaded {len(html):,} characters from local fallback HTML.")
    return [html]


def has_html_table(html: str) -> bool:
    return bool(BeautifulSoup(html, "html.parser").find("table"))


def clean_column_name(column: object) -> str:
    if isinstance(column, tuple):
        column = " ".join(str(part) for part in column if str(part) != "nan")

    cleaned = str(column).replace("\xa0", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_cell_value(value: object) -> object:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        cleaned = value.replace("\xa0", " ")
        return re.sub(r"\s+", " ", cleaned).strip()
    return value


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [clean_column_name(column) for column in normalized.columns]
    normalized = normalized.apply(lambda column: column.map(clean_cell_value))
    normalized = normalized.dropna(axis=0, how="all")
    normalized = normalized.loc[:, [column for column in normalized.columns if column]]
    return normalized.reset_index(drop=True)


def normalize_column_key(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", column.lower())


def find_column(
    df: pd.DataFrame,
    possible_terms: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    normalized_terms = [normalize_column_key(term) for term in possible_terms]
    for column in df.columns:
        normalized_column = normalize_column_key(column)
        if any(term and term in normalized_column for term in normalized_terms):
            return column

    if required:
        raise KeyError(
            f"Could not find a column matching terms {list(possible_terms)}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def table_score(columns: Iterable[str]) -> int:
    joined = " | ".join(column.lower() for column in columns)
    score = 0
    for keyword in ("ticker", "symbol"):
        if keyword in joined:
            score += 5
    for keyword in ("short float", "shortfloat"):
        if keyword in joined:
            score += 5
    for keyword in ("short ratio", "days to cover", "daystocover"):
        if keyword in joined:
            score += 5
    return score


def extract_tables_from_html(html: str) -> list[pd.DataFrame]:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    tables: list[pd.DataFrame] = []
    for table in soup.find_all("table"):
        try:
            parsed_tables = pd.read_html(StringIO(str(table)))
        except ValueError:
            continue

        for parsed_df in parsed_tables:
            normalized = normalize_dataframe(parsed_df)
            if not normalized.empty:
                tables.append(normalized)
    return tables


def parse_finviz_candidates(html_pages: list[str], local_html_path: Path | None = None) -> pd.DataFrame:
    """Extract Ticker, Short Float, and Days to Cover from Finviz screener HTML."""
    log("Parsing Finviz HTML with BeautifulSoup and pandas.read_html.")
    candidates: list[tuple[int, int, pd.DataFrame]] = []

    for page_index, html in enumerate(html_pages, start=1):
        for table in extract_tables_from_html(html):
            score = table_score(table.columns)
            candidates.append((score, page_index, table))

    if not candidates and local_html_path is not None:
        log("Direct Finviz parse found no tables; retrying local fallback HTML.")
        candidates = [
            (table_score(table.columns), 1, table)
            for table in extract_tables_from_html(read_local_html(local_html_path))
        ]

    if not candidates:
        raise ValueError("No parseable HTML tables were found in the supplied Finviz HTML.")

    candidates.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    best_score = candidates[0][0]
    selected_tables = [table for score, _, table in candidates if score == best_score and score > 0]
    if not selected_tables:
        raise ValueError(
            "Could not identify a Finviz screener table containing Ticker, "
            "Short Float, and Short Ratio/Days to Cover columns."
        )

    raw_df = pd.concat(selected_tables, ignore_index=True)
    log(
        f"Selected {len(selected_tables)} Finviz table(s) with heuristic score "
        f"{best_score}; combined shape {raw_df.shape[0]} rows x {raw_df.shape[1]} columns."
    )
    return normalize_finviz_candidate_columns(raw_df)


def clean_ticker(value: object) -> str:
    text = str(value).strip().upper()
    text = text.split(maxsplit=1)[0]
    ticker = re.sub(r"[^A-Z0-9.\-]", "", text)
    return ticker


def safe_float(value: object) -> float:
    if pd.isna(value):
        return 0.0

    text = str(value).replace(",", "").replace("%", "")
    text = text.replace("x", "").replace("X", "")
    if text.strip() in {"", "-", "N/A", "NA", "nan"}:
        return 0.0

    multiplier = 1.0
    suffix_match = re.search(r"([+-]?\d+(?:\.\d+)?)([KMB])\b", text, flags=re.IGNORECASE)
    if suffix_match:
        suffix = suffix_match.group(2).upper()
        multiplier = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}[suffix]
        return float(suffix_match.group(1)) * multiplier

    match = re.search(r"[+-]?\d+(?:\.\d+)?", text)
    if not match:
        return 0.0
    return float(match.group(0)) * multiplier


def normalize_finviz_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    ticker_column = find_column(df, ["ticker", "symbol"])
    short_float_column = find_column(df, ["short float", "shortfloat"])
    days_to_cover_column = find_column(
        df,
        ["days to cover", "daystocover", "short ratio", "shortratio"],
    )

    normalized = pd.DataFrame(
        {
            "Ticker": df[ticker_column].map(clean_ticker),
            "Short Float": df[short_float_column].map(safe_float),
            "Days to Cover": df[days_to_cover_column].map(safe_float),
        }
    )

    ticker_pattern = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
    normalized = normalized[normalized["Ticker"].map(lambda ticker: bool(ticker_pattern.match(ticker)))]
    normalized = normalized[normalized["Ticker"] != "TICKER"]
    normalized = normalized.drop_duplicates(subset=["Ticker"], keep="first")
    normalized = normalized.reset_index(drop=True)

    if normalized.empty:
        raise ValueError("Finviz table parsing produced no valid ticker rows.")

    log(f"Normalized Finviz candidates: {len(normalized)} ticker rows.")
    return normalized


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


def enrich_with_volume_data(
    df: pd.DataFrame,
    min_sleep_seconds: float,
    max_sleep_seconds: float,
    limit: int | None,
) -> pd.DataFrame:
    enriched = df.copy() if limit is None else df.iloc[:limit].copy()
    enriched["Yesterday Volume"] = 0
    enriched["3M Average Volume"] = 0
    enriched["RVOL"] = 0.0
    enriched["Volume Fetch Status"] = ""

    rows_to_process = enriched.index
    log(f"Preparing to enrich {len(rows_to_process)} ticker rows via yfinance.")

    for row_number, row_index in enumerate(rows_to_process, start=1):
        ticker_symbol = str(enriched.at[row_index, "Ticker"]).strip().upper()
        log(f"Row {row_number}/{len(rows_to_process)}: processing ticker {ticker_symbol}")

        if not ticker_symbol:
            log(f"Row {row_number}: empty ticker; skipping.")
            enriched.at[row_index, "Volume Fetch Status"] = "skipped: no ticker"
            continue

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
            log(f"{ticker_symbol}: yfinance lookup failed; skipping ticker. Error: {exc}")
            enriched.at[row_index, "Volume Fetch Status"] = f"error: {exc}"

        sleep_seconds = random.uniform(min_sleep_seconds, max_sleep_seconds)
        log(f"Sleeping {sleep_seconds:.2f}s for rate limiting.")
        time.sleep(sleep_seconds)

    return enriched


def score_row(row: pd.Series) -> tuple[int, list[str]]:
    volume_fetch_status = str(row.get("Volume Fetch Status", "ok")).strip().lower()
    if volume_fetch_status and volume_fetch_status != "ok":
        return 0, [f"Skipped: volume fetch {volume_fetch_status}"]

    score = 0
    reasons: list[str] = []

    short_float = safe_float(row.get("Short Float", 0))
    if short_float > 40:
        score += 40
        reasons.append("Short Float > 40%")
    elif short_float > 20:
        score += 20
        reasons.append("Short Float > 20%")

    days_to_cover = safe_float(row.get("Days to Cover", 0))
    if days_to_cover > 5:
        score += 20
        reasons.append("Days to Cover > 5")
    elif days_to_cover > 3:
        score += 10
        reasons.append("Days to Cover > 3")

    rvol = safe_float(row.get("RVOL", 0))
    if rvol > 3.0:
        score += 30
        reasons.append("RVOL > 3.0x")
    elif rvol > 1.5:
        score += 15
        reasons.append("RVOL > 1.5x")

    return score, reasons


def evaluate_short_squeeze_risk(df: pd.DataFrame) -> pd.DataFrame:
    evaluated = df.copy()
    scores: list[int] = []
    reasons_list: list[str] = []

    for _, row in evaluated.iterrows():
        ticker_symbol = str(row.get("Ticker", "")).strip().upper() or "UNKNOWN"
        log(f"Evaluating ticker {ticker_symbol}")
        score, reasons = score_row(row)
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
            elif column in {"Short Float", "Days to Cover"}:
                rendered_row.append(format_float(value))
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
        "Agent Score",
        "Short Float",
        "Days to Cover",
        "Yesterday Volume",
        "3M Average Volume",
        "RVOL",
        "Alert Reasons",
    ]
    print("\nHIGH-ALERT SHORT SQUEEZE TARGETS")
    print(markdown_table(high_alerts, preferred_columns))


def parse_args(argv: list[str]) -> MonitorConfig:
    parser = argparse.ArgumentParser(
        description="Run the Finviz-based US Stock Short Squeeze Monitor AI Agent."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_FINVIZ_URL,
        help=(
            "Finviz screener URL to scrape. Default uses ownership view and "
            "Short Float > 20%% filter."
        ),
    )
    parser.add_argument(
        "--local-html",
        default=DEFAULT_LOCAL_HTML,
        help="Fallback local Finviz screener HTML file saved manually from the browser.",
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
        help="Optional maximum number of ticker rows to enrich. Useful for smoke tests.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Maximum Finviz screener pages to request directly before evaluation.",
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
    if args.max_pages <= 0:
        parser.error("--max-pages must be a positive integer.")

    return MonitorConfig(
        finviz_url=args.url,
        local_html_path=Path(args.local_html),
        min_sleep_seconds=args.min_sleep,
        max_sleep_seconds=args.max_sleep,
        limit=args.limit,
        output_csv_path=Path(args.output_csv) if args.output_csv else None,
        max_pages=args.max_pages,
    )


def run_pipeline(config: MonitorConfig) -> pd.DataFrame:
    log_phase("1. DATA ACQUISITION VIA FINVIZ (SCRAPE)")
    html_pages = fetch_finviz_html_pages(config.finviz_url, config.local_html_path, config.max_pages)
    try:
        candidate_df = parse_finviz_candidates(html_pages, config.local_html_path)
    except Exception as exc:
        log(f"Initial Finviz parse failed: {exc}")
        log("Attempting to parse local fallback HTML in case direct HTTP returned a protected page.")
        candidate_df = parse_finviz_candidates([read_local_html(config.local_html_path)])

    log(f"Parsed Finviz candidate shape: {candidate_df.shape[0]} rows x {candidate_df.shape[1]} columns.")
    log(f"Parsed columns: {list(candidate_df.columns)}")

    log_phase("2. DYNAMIC VOLUME AND METRIC INJECTION (GET VOLUME)")
    enriched_df = enrich_with_volume_data(
        candidate_df,
        config.min_sleep_seconds,
        config.max_sleep_seconds,
        config.limit,
    )

    log_phase("3. RISK EVALUATION ENGINE (EVALUATE)")
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
