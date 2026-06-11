#!/usr/bin/env python3
"""
US Stock Short Squeeze Monitor AI Agent.

This version evaluates exactly one ticker at a time from its dedicated Finviz
quote page. It fetches the quote page, parses the snapshot-table2 fundamentals
matrix, normalizes key metrics, and applies a 100-point squeeze-risk scorecard.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup


DEFAULT_TICKER = "CAR"
DEFAULT_LOCAL_HTML = "finviz_quote.html"
REQUEST_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class AgentConfig:
    ticker: str
    local_html_path: Path


@dataclass(frozen=True)
class QuoteMetrics:
    ticker: str
    short_float: float
    short_ratio: float
    institutional_ownership: float
    float_shares: float
    average_volume: float
    current_volume: float
    price: float
    change_percent: float
    raw_snapshot: dict[str, str]

    @property
    def rvol(self) -> float:
        if self.average_volume <= 0:
            return 0.0
        return self.current_volume / self.average_volume


@dataclass(frozen=True)
class EvaluationResult:
    ticker: str
    score: int
    grade: str
    triggered_rules: list[str]
    metrics: QuoteMetrics


def log(message: str) -> None:
    print(f"[Agent] {message}", flush=True)


def log_phase(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


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
        "Referer": "https://finviz.com/",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "sec-ch-ua": '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
    }


def normalize_ticker(ticker: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9.\-]", "", ticker.strip()).upper()
    if not cleaned:
        raise ValueError("Ticker cannot be empty.")
    return cleaned


def build_finviz_quote_url(ticker: str) -> str:
    # Finviz's individual ticker quote page is the dedicated data page for a symbol.
    return f"https://finviz.com/quote.ashx?t={quote_plus(ticker)}&p=d"


def read_local_html(local_html_path: Path) -> str:
    log(f"Reading local fallback HTML from {local_html_path}")
    if not local_html_path.exists():
        raise FileNotFoundError(f"Local fallback file was not found: {local_html_path}")

    html = local_html_path.read_text(encoding="utf-8", errors="replace")
    if not html.strip():
        raise ValueError(f"Local fallback file is empty: {local_html_path}")
    log(f"Loaded {len(html):,} characters from local fallback file.")
    return html


def fetch_finviz_quote_html(ticker: str, local_html_path: Path) -> str:
    url = build_finviz_quote_url(ticker)
    log(f"Fetching URL: {url}")

    session = requests.Session()
    session.headers.update(browser_headers())

    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        log(f"HTTP status: {response.status_code}")
        if response.status_code == 200 and response.text.strip():
            return response.text
        log("Direct request did not return usable HTTP 200 HTML.")
    except requests.RequestException as exc:
        log(f"Direct request failed: {exc}")

    log("Falling back to local finviz_quote.html source.")
    return read_local_html(local_html_path)


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", key.lower())


def parse_snapshot_with_beautifulsoup(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_=lambda value: value and "snapshot-table2" in value)
    if table is None:
        return {}

    snapshot: dict[str, str] = {}
    for row in table.find_all("tr"):
        cells = [clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all("td")]
        for index in range(0, len(cells) - 1, 2):
            key = cells[index]
            value = cells[index + 1]
            if key:
                snapshot[key] = value
    return snapshot


def parse_snapshot_with_pandas(html: str) -> dict[str, str]:
    try:
        tables = pd.read_html(StringIO(html), attrs={"class": "snapshot-table2"})
    except (ImportError, ValueError):
        return {}

    snapshot: dict[str, str] = {}
    for table in tables:
        for _, row in table.iterrows():
            values = [clean_text(value) for value in row.tolist()]
            for index in range(0, len(values) - 1, 2):
                key = values[index]
                value = values[index + 1]
                if key and key.lower() != "nan":
                    snapshot[key] = value
    return snapshot


def parse_snapshot_metrics(html: str) -> dict[str, str]:
    log("Parsing Finviz snapshot metrics from snapshot-table2.")
    snapshot = parse_snapshot_with_beautifulsoup(html)
    if not snapshot:
        log("BeautifulSoup snapshot parse did not find data; trying pandas.read_html.")
        snapshot = parse_snapshot_with_pandas(html)

    if not snapshot:
        raise ValueError("Could not locate or parse Finviz snapshot-table2 metrics.")

    log(f"Extracted {len(snapshot)} raw snapshot key/value pairs.")
    return snapshot


def find_metric(snapshot: dict[str, str], aliases: list[str]) -> str:
    normalized_snapshot = {normalize_key(key): value for key, value in snapshot.items()}
    for alias in aliases:
        normalized_alias = normalize_key(alias)
        if normalized_alias in normalized_snapshot:
            return normalized_snapshot[normalized_alias]
    for alias in aliases:
        normalized_alias = normalize_key(alias)
        for key, value in normalized_snapshot.items():
            if normalized_alias and normalized_alias in key:
                return value
    return "N/A"


def parse_numeric(value: object) -> float:
    """Parse Finviz strings such as '16.33%', '34.77M', '499,938', or 'N/A'."""
    text = clean_text(value)
    if not text or text.upper() in {"N/A", "NA", "NONE", "NULL", "-", "NAN"}:
        return 0.0

    text = text.replace(",", "")
    match = re.search(r"([+-]?\d+(?:\.\d+)?)([KMBT]?)", text, flags=re.IGNORECASE)
    if not match:
        return 0.0

    number = float(match.group(1))
    suffix = match.group(2).upper()
    multiplier = {
        "": 1.0,
        "K": 1_000.0,
        "M": 1_000_000.0,
        "B": 1_000_000_000.0,
        "T": 1_000_000_000_000.0,
    }[suffix]
    return number * multiplier


def extract_quote_metrics(ticker: str, snapshot: dict[str, str]) -> QuoteMetrics:
    log("Normalizing target quote metrics.")

    short_float_raw = find_metric(snapshot, ["Short Float"])
    short_ratio_raw = find_metric(snapshot, ["Short Ratio", "Days to Cover"])
    inst_own_raw = find_metric(snapshot, ["Inst Own", "Institutional Ownership"])
    float_raw = find_metric(snapshot, ["Float", "Shs Float", "Shares Float"])
    avg_volume_raw = find_metric(snapshot, ["Avg Volume", "Avg Vol", "Average Volume"])
    volume_raw = find_metric(snapshot, ["Volume"])
    price_raw = find_metric(snapshot, ["Price"])
    change_raw = find_metric(snapshot, ["Change"])

    metrics = QuoteMetrics(
        ticker=ticker,
        short_float=parse_numeric(short_float_raw),
        short_ratio=parse_numeric(short_ratio_raw),
        institutional_ownership=parse_numeric(inst_own_raw),
        float_shares=parse_numeric(float_raw),
        average_volume=parse_numeric(avg_volume_raw),
        current_volume=parse_numeric(volume_raw),
        price=parse_numeric(price_raw),
        change_percent=parse_numeric(change_raw),
        raw_snapshot=snapshot,
    )

    log(f"Short Float: {short_float_raw} -> {metrics.short_float:.2f}%")
    log(f"Short Ratio / Days to Cover: {short_ratio_raw} -> {metrics.short_ratio:.2f}")
    log(f"Inst Own: {inst_own_raw} -> {metrics.institutional_ownership:.2f}%")
    log(f"Float: {float_raw} -> {metrics.float_shares:,.0f}")
    log(f"Avg Volume: {avg_volume_raw} -> {metrics.average_volume:,.0f}")
    log(f"Volume: {volume_raw} -> {metrics.current_volume:,.0f}")
    log(f"Price: {price_raw} -> {metrics.price:.2f}")
    log(f"Change: {change_raw} -> {metrics.change_percent:.2f}%")
    log(f"RVOL: {metrics.rvol:.2f}x")
    return metrics


def evaluate_squeeze_risk(metrics: QuoteMetrics) -> EvaluationResult:
    log("Evaluating multi-factor short squeeze matrix.")
    score = 0
    triggered_rules: list[str] = []

    if metrics.short_float > 40:
        score += 35
        triggered_rules.append("Fuel Weight: Short Float > 40% (+35)")
    elif metrics.short_float > 20:
        score += 20
        triggered_rules.append("Fuel Weight: Short Float > 20% (+20)")

    if metrics.institutional_ownership > 100:
        score += 25
        triggered_rules.append("Structural Lockup: Inst Own > 100% (+25)")
    elif metrics.institutional_ownership > 80:
        score += 15
        triggered_rules.append("Structural Lockup: Inst Own > 80% (+15)")

    if metrics.short_ratio > 4:
        score += 15
        triggered_rules.append("Exit Trap: Short Ratio > 4 days (+15)")
    elif metrics.short_ratio > 2:
        score += 10
        triggered_rules.append("Exit Trap: Short Ratio > 2 days (+10)")

    if 0 < metrics.float_shares < 20_000_000:
        score += 15
        triggered_rules.append("Float Scale: Float < 20M (+15)")
    elif 0 < metrics.float_shares < 50_000_000:
        score += 10
        triggered_rules.append("Float Scale: Float < 50M (+10)")

    if metrics.rvol > 2.5 and metrics.change_percent > 0:
        score += 10
        triggered_rules.append("Live Trigger: RVOL > 2.5x with positive change (+10)")

    if score >= 75:
        grade = "CRITICAL SQUEEZE RISK"
    elif score >= 50:
        grade = "HIGH SQUEEZE RISK"
    else:
        grade = "LOW/STABLE"

    if not triggered_rules:
        triggered_rules.append("No high-risk criteria triggered.")

    log(f"Final score: {score}/100 -> {grade}")
    return EvaluationResult(
        ticker=metrics.ticker,
        score=score,
        grade=grade,
        triggered_rules=triggered_rules,
        metrics=metrics,
    )


def format_large_number(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.0f}"


def print_report(result: EvaluationResult) -> None:
    metrics = result.metrics
    print("\n# Short Squeeze Risk Report")
    print(f"\n**Ticker:** {result.ticker}")
    print(f"**Final Grade:** {result.grade}")
    print(f"**Squeeze Risk Score:** {result.score}/100")

    print("\n## Normalized Metrics")
    print("| Metric | Value |")
    print("| --- | ---: |")
    print(f"| Short Float | {metrics.short_float:.2f}% |")
    print(f"| Short Ratio / Days to Cover | {metrics.short_ratio:.2f} |")
    print(f"| Institutional Ownership | {metrics.institutional_ownership:.2f}% |")
    print(f"| Float | {format_large_number(metrics.float_shares)} |")
    print(f"| Avg Volume | {format_large_number(metrics.average_volume)} |")
    print(f"| Current Volume | {format_large_number(metrics.current_volume)} |")
    print(f"| RVOL | {metrics.rvol:.2f}x |")
    print(f"| Price | {metrics.price:.2f} |")
    print(f"| Change | {metrics.change_percent:.2f}% |")

    print("\n## Triggered Scorecard Criteria")
    for rule in result.triggered_rules:
        print(f"- {rule}")


def run_pipeline(config: AgentConfig) -> EvaluationResult:
    ticker = normalize_ticker(config.ticker)

    log_phase("STEP 1: TARGET INDIVIDUAL TICKER SCRAPING")
    html = fetch_finviz_quote_html(ticker, config.local_html_path)
    try:
        snapshot = parse_snapshot_metrics(html)
    except Exception as exc:
        log(f"Initial snapshot parse failed: {exc}")
        log("Attempting local fallback HTML in case direct request returned a protected page.")
        snapshot = parse_snapshot_metrics(read_local_html(config.local_html_path))
    metrics = extract_quote_metrics(ticker, snapshot)

    log_phase("STEP 2: MULTI-FACTOR SHORT SQUEEZE EVALUATOR")
    result = evaluate_squeeze_risk(metrics)
    print_report(result)
    return result


def parse_args(argv: list[str]) -> AgentConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate one ticker's Finviz quote page for short squeeze risk."
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default=DEFAULT_TICKER,
        help=f"Ticker symbol to evaluate. Defaults to {DEFAULT_TICKER}.",
    )
    parser.add_argument(
        "--local-html",
        default=DEFAULT_LOCAL_HTML,
        help="Fallback local Finviz quote HTML file saved manually from the browser.",
    )
    args = parser.parse_args(argv)
    return AgentConfig(ticker=args.ticker, local_html_path=Path(args.local_html))


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
