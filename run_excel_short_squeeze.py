#!/usr/bin/env python3
"""
Run short squeeze evaluation for every ticker in an Excel workbook.

By default this reads "mid cap.xlsx", waits 0.5 seconds between Finviz requests,
and outputs only companies with score higher than 60.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import short_squeeze_monitor as monitor


DEFAULT_WORKBOOK = "mid cap.xlsx"
DEFAULT_MIN_SCORE = 60
DEFAULT_DELAY_SECONDS = 0.5
DEFAULT_OUTPUT_CSV = "short_squeeze_score_gt60.csv"
DEFAULT_ALL_SCORES_CSV = "short_squeeze_all_scores.csv"
DEFAULT_ERRORS_CSV = "short_squeeze_errors.csv"


@dataclass(frozen=True)
class BatchConfig:
    workbook_path: Path
    min_score: int
    delay_seconds: float
    output_csv_path: Path
    all_scores_csv_path: Path | None
    errors_csv_path: Path | None
    limit: int | None


def log(message: str) -> None:
    print(f"[BatchAgent] {message}", flush=True)


def load_tickers_from_excel(workbook_path: Path) -> list[str]:
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    df = pd.read_excel(workbook_path, header=None)
    tickers: list[str] = []
    seen: set[str] = set()

    for value in df.iloc[:, 0].dropna().tolist():
        try:
            ticker = monitor.normalize_ticker(str(value))
        except ValueError:
            continue
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)

    if not tickers:
        raise ValueError(f"No ticker symbols found in first column of {workbook_path}")
    return tickers


def result_to_row(result: monitor.EvaluationResult) -> dict[str, str | int]:
    metrics = result.metrics
    return {
        "ticker": result.ticker,
        "score": result.score,
        "grade": result.grade,
        "short_float": f"{metrics.short_float:.2f}",
        "short_ratio": f"{metrics.short_ratio:.2f}",
        "inst_own": f"{metrics.institutional_ownership:.2f}",
        "float": f"{metrics.float_shares:.0f}",
        "avg_volume": f"{metrics.average_volume:.0f}",
        "volume": f"{metrics.current_volume:.0f}",
        "rvol": f"{metrics.rvol:.4f}",
        "price": f"{metrics.price:.2f}",
        "change_percent": f"{metrics.change_percent:.2f}",
        "triggered_rules": "; ".join(result.triggered_rules),
    }


def fieldnames() -> list[str]:
    return [
        "ticker",
        "score",
        "grade",
        "short_float",
        "short_ratio",
        "inst_own",
        "float",
        "avg_volume",
        "volume",
        "rvol",
        "price",
        "change_percent",
        "triggered_rules",
    ]


def evaluate_ticker(ticker: str, fallback_html_path: Path = Path("finviz_quote.html")) -> monitor.EvaluationResult:
    html = monitor.fetch_finviz_quote_html(ticker, fallback_html_path)
    snapshot = monitor.parse_snapshot_metrics(html)
    metrics = monitor.extract_quote_metrics(ticker, snapshot)
    return monitor.evaluate_squeeze_risk(metrics)


def write_markdown_table(rows: list[dict[str, str | int]]) -> None:
    if not rows:
        print("\nNo companies found with score higher than the threshold.")
        return

    print("\nCompanies with score higher than threshold:")
    print("| Ticker | Score | Grade | Short Float | Short Ratio | Inst Own | RVOL | Change |")
    print("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            "| {ticker} | {score} | {grade} | {short_float}% | {short_ratio} | "
            "{inst_own}% | {rvol}x | {change_percent}% |".format(**row)
        )


def run_batch(config: BatchConfig) -> list[dict[str, str | int]]:
    tickers = load_tickers_from_excel(config.workbook_path)
    if config.limit is not None:
        tickers = tickers[: config.limit]

    log(f"Loaded {len(tickers)} ticker(s) from {config.workbook_path}")
    log(f"Output filter: score > {config.min_score}")
    log(f"Delay between tickers: {config.delay_seconds:.2f}s")

    high_score_rows: list[dict[str, str | int]] = []
    errors: list[dict[str, str]] = []

    all_scores_file = config.all_scores_csv_path.open("w", newline="") if config.all_scores_csv_path else None
    high_scores_file = config.output_csv_path.open("w", newline="")
    errors_file = config.errors_csv_path.open("w", newline="") if config.errors_csv_path else None

    try:
        all_writer = csv.DictWriter(all_scores_file, fieldnames=fieldnames()) if all_scores_file else None
        high_writer = csv.DictWriter(high_scores_file, fieldnames=fieldnames())
        error_writer = csv.DictWriter(errors_file, fieldnames=["ticker", "error"]) if errors_file else None

        if all_writer:
            all_writer.writeheader()
        high_writer.writeheader()
        if error_writer:
            error_writer.writeheader()

        for index, ticker in enumerate(tickers, start=1):
            try:
                result = evaluate_ticker(ticker)
                row = result_to_row(result)
                if all_writer:
                    all_writer.writerow(row)
                    all_scores_file.flush()

                if result.score > config.min_score:
                    high_score_rows.append(row)
                    high_writer.writerow(row)
                    high_scores_file.flush()
                    log(f"[{index}/{len(tickers)}] MATCH {ticker}: {result.score} {result.grade}")
                else:
                    log(f"[{index}/{len(tickers)}] {ticker}: {result.score} {result.grade}")
            except Exception as exc:
                error_row = {"ticker": ticker, "error": str(exc)}
                errors.append(error_row)
                if error_writer:
                    error_writer.writerow(error_row)
                    errors_file.flush()
                log(f"[{index}/{len(tickers)}] ERROR {ticker}: {exc}")

            if index < len(tickers):
                time.sleep(config.delay_seconds)
    finally:
        if all_scores_file:
            all_scores_file.close()
        high_scores_file.close()
        if errors_file:
            errors_file.close()

    high_score_rows.sort(key=lambda row: (-int(row["score"]), str(row["ticker"])))
    write_markdown_table(high_score_rows)
    log(f"Matched rows: {len(high_score_rows)}")
    log(f"Errors: {len(errors)}")
    log(f"Filtered CSV written to {config.output_csv_path}")
    return high_score_rows


def parse_args() -> BatchConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate all Excel tickers and output companies with score higher than a threshold."
    )
    parser.add_argument("--workbook", default=DEFAULT_WORKBOOK, help="Excel workbook path.")
    parser.add_argument("--min-score", type=int, default=DEFAULT_MIN_SCORE, help="Strict score threshold.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS, help="Delay between tickers in seconds.")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="CSV for rows above threshold.")
    parser.add_argument("--all-scores-csv", default=DEFAULT_ALL_SCORES_CSV, help="CSV for all scored tickers.")
    parser.add_argument("--errors-csv", default=DEFAULT_ERRORS_CSV, help="CSV for ticker errors.")
    parser.add_argument("--limit", type=int, default=None, help="Optional ticker limit for smoke tests.")
    args = parser.parse_args()

    if args.delay < 0:
        parser.error("--delay must be non-negative.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive when provided.")

    return BatchConfig(
        workbook_path=Path(args.workbook),
        min_score=args.min_score,
        delay_seconds=args.delay,
        output_csv_path=Path(args.output_csv),
        all_scores_csv_path=Path(args.all_scores_csv) if args.all_scores_csv else None,
        errors_csv_path=Path(args.errors_csv) if args.errors_csv else None,
        limit=args.limit,
    )


def main() -> int:
    try:
        run_batch(parse_args())
        return 0
    except Exception as exc:
        log(f"Batch failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
