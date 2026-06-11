# US Stock Short Squeeze Monitor AI Agent

Production-oriented Python pipeline for monitoring potential US stock short squeeze
targets from a Finviz screener export.

## What it does

The agent runs three sequential phases:

1. **Scrape** - Attempts to fetch a Finviz Ownership screener URL with
   browser-like headers. The default URL filters for `Short Float > 20%`. If
   Finviz blocks the request or returns unusable HTML, it falls back to parsing a
   local `finviz.html` file saved manually from your browser.
2. **Get Volume** - Extracts each ticker from the Finviz screener table, queries
   `yfinance`, and appends:
   - Yesterday / previous completed trading-day volume
   - 3-month average volume
   - Relative volume (`RVOL`)
3. **Evaluate** - Scores each ticker with weighted short-float, days-to-cover,
   and RVOL rules, then prints high-alert targets with `Agent Score >= 50` as a
   Markdown table.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Finviz fallback setup

Finviz may block automated scraping. If the direct scrape fails:

1. Open the Finviz screener page in your browser. Use the Ownership view
   (`v=131`) and a short-float filter such as:
   `https://finviz.com/screener.ashx?v=131&f=sh_short_o20&ft=4`
2. Save the raw HTML source as `finviz.html` in this repository directory.
3. Re-run the script.

## Run

```bash
python short_squeeze_monitor.py
```

Optional arguments:

```bash
python short_squeeze_monitor.py \
  --url "https://finviz.com/screener.ashx?v=131&f=sh_short_o20&ft=4" \
  --local-html finviz.html \
  --max-pages 1 \
  --output-csv high_alerts.csv
```

For a quick smoke test against only a few rows:

```bash
python short_squeeze_monitor.py --limit 5 --min-sleep 0.5 --max-sleep 1.5
```

## Notes

- One ticker failure will be logged and skipped without stopping the pipeline.
- Random sleeps between yfinance calls reduce rate-limit and throttling risk.
- The script prints phase banners and per-ticker progress for operational
  visibility.
