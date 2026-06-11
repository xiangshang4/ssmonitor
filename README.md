# US Stock Short Squeeze Monitor AI Agent

Production-oriented Python agent for evaluating exactly one US stock ticker from
its dedicated Finviz quote page.

## What it does

The agent runs two sequential steps:

1. **Scrape** - Fetches the ticker's Finviz quote page using browser-like HTTP
   headers. If Finviz blocks the request or returns unusable HTML, it falls back
   to a local `finviz_quote.html` file saved manually from your browser.
2. **Evaluate** - Parses the `snapshot-table2` fundamentals table, normalizes
   short-interest and liquidity metrics, calculates live RVOL, and applies a
   100-point squeeze-risk matrix.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Direct dependency install:

```bash
pip install pandas beautifulsoup4 lxml requests
```

## Run

Evaluate a ticker:

```bash
python short_squeeze_monitor.py CAR
```

If no ticker is supplied, the script defaults to `CAR`:

```bash
python short_squeeze_monitor.py
```

## Local fallback setup

Finviz may block automated requests. If direct scraping fails:

1. Open the Finviz quote page manually in your browser, for example:
   `https://finviz.com/quote.ashx?t=CAR&p=d`
2. Save the raw HTML source as `finviz_quote.html` in this repository directory.
3. Re-run:

```bash
python short_squeeze_monitor.py CAR --local-html finviz_quote.html
```

## Scoring matrix

- **Fuel Weight / Short Float**: `> 40%` = 35 pts; `> 20%` = 20 pts
- **Structural Lockup / Inst Own**: `> 100%` = 25 pts; `> 80%` = 15 pts
- **Exit Trap / Short Ratio**: `> 4 days` = 15 pts; `> 2 days` = 10 pts
- **Float Scale**: `< 20M` = 15 pts; `< 50M` = 10 pts
- **Live Trigger**: `RVOL > 2.5x` with positive price change = 10 pts

Grades:

- `>= 75`: `CRITICAL SQUEEZE RISK`
- `>= 50`: `HIGH SQUEEZE RISK`
- Otherwise: `LOW/STABLE`
