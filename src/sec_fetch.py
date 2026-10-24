"""Download 10-K filings straight from SEC EDGAR (free, no API key).

SEC asks every client to send a User-Agent with a name and email,
and to stay under 10 requests per second.
"""
import json
import os
import time

import requests

USER_AGENT = os.environ.get("SEC_USER_AGENT", "Kishan Korat kishankorat2444@gmail.com")
CACHE_DIR = os.path.join("data", "sec")
HEADERS = {"User-Agent": USER_AGENT}


def _get(url):
    time.sleep(0.2)  # stay far below SEC's rate limit
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r


def ticker_to_cik(ticker):
    data = _get("https://www.sec.gov/files/company_tickers.json").json()
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10), row["title"]
    raise ValueError(f"Ticker not found on SEC: {ticker}")


def list_10k_filings(cik):
    """Return [(fiscal_year, accession, primary_document)] for every 10-K."""
    sub = _get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    tables = [sub["filings"]["recent"]]
    for extra in sub["filings"].get("files", []):
        tables.append(_get("https://data.sec.gov/submissions/" + extra["name"]).json())

    filings = {}
    for t in tables:
        for form, report, acc, doc in zip(t["form"], t["reportDate"], t["accessionNumber"], t["primaryDocument"]):
            if form != "10-K" or not report:
                continue
            year = int(report[:4])
            filings.setdefault(year, (year, acc, doc))  # newest first, keep first
    return sorted(filings.values())


def download_10k(ticker, years):
    """Download the 10-K HTML for each fiscal year. Cached on disk."""
    cik, name = ticker_to_cik(ticker)
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = {}
    for year, acc, doc in list_10k_filings(cik):
        if year not in years:
            continue
        path = os.path.join(CACHE_DIR, f"{ticker.lower()}_{year}.htm")
        if not os.path.exists(path):
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"
            with open(path, "wb") as f:
                f.write(_get(url).content)
        paths[year] = path
    return name, paths
