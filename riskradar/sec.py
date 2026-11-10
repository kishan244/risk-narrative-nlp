"""Small client for SEC EDGAR (free, no API key).

SEC asks every client to identify itself with a User-Agent (name + email)
and to stay under 10 requests per second.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

import requests

USER_AGENT = os.environ.get("SEC_USER_AGENT", "Kishan Korat kishankorat2444@gmail.com")
_PAUSE = 0.15  # seconds between requests, far below SEC's limit


class SecError(RuntimeError):
    """A problem the user can understand (unknown ticker, no 10-K, SEC down)."""


@dataclass(frozen=True)
class Filing:
    ticker: str
    cik: str
    sec_name: str
    fiscal_year: int
    period_end: str   # YYYY-MM-DD
    filed: str        # YYYY-MM-DD
    accession: str
    url: str

    def to_dict(self) -> dict:
        return asdict(self)


def fiscal_year(period_end: str) -> int:
    """Calendar year the report mostly covers.

    A year that ends in January or February (Walmart, NVIDIA, some J&J years)
    is counted as the year before, so all companies line up on the same axis.
    """
    year, month = int(period_end[:4]), int(period_end[5:7])
    return year - 1 if month <= 2 else year


class SecClient:
    def __init__(self, user_agent: str = USER_AGENT, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._tickers: dict[str, tuple[str, str]] | None = None

    def _get(self, url: str) -> requests.Response:
        last = None
        for attempt in range(3):
            time.sleep(_PAUSE * (1 + 4 * attempt))
            try:
                r = self.session.get(url, timeout=60)
            except requests.RequestException as exc:  # network hiccup: retry
                last = exc
                continue
            if r.status_code == 200:
                return r
            last = SecError(f"SEC answered HTTP {r.status_code} for {url}")
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        raise SecError(f"Could not reach SEC EDGAR: {last}")

    def lookup(self, ticker: str) -> tuple[str, str]:
        """Ticker -> (10-digit CIK, company name as SEC writes it)."""
        if self._tickers is None:
            data = self._get("https://www.sec.gov/files/company_tickers.json").json()
            self._tickers = {
                row["ticker"].upper(): (str(row["cik_str"]).zfill(10), row["title"]) for row in data.values()
            }
        key = ticker.strip().upper().replace(".", "-")
        if key not in self._tickers:
            raise SecError(f"'{ticker}' is not a ticker listed with the SEC.")
        return self._tickers[key]

    def list_10k(self, ticker: str, cik: str | None = None) -> list[Filing]:
        """Every annual report (form 10-K) of a company, oldest first.

        Pass `cik` when a ticker has moved to a new legal entity and the old
        filings still sit under the previous registrant.
        """
        if cik:
            cik = str(cik).zfill(10)
        else:
            cik, _ = self.lookup(ticker)
        sub = self._get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
        name = sub.get("name", ticker.upper())
        tables = [sub["filings"]["recent"]]
        for extra in sub["filings"].get("files", []):
            tables.append(self._get("https://data.sec.gov/submissions/" + extra["name"]).json())

        by_year: dict[int, Filing] = {}
        for t in tables:
            rows = zip(t["form"], t["reportDate"], t["filingDate"], t["accessionNumber"], t["primaryDocument"])
            for form, period, filed, accession, doc in rows:
                if form != "10-K" or not period or not doc:
                    continue
                fy = fiscal_year(period)
                url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{doc}"
                filing = Filing(ticker.upper(), cik, name, fy, period, filed, accession, url)
                # tables are newest first: keep the newest filing for a fiscal year
                by_year.setdefault(fy, filing)
        if not by_year:
            raise SecError(f"{name} has no 10-K filings (foreign companies file form 20-F instead).")
        return [by_year[y] for y in sorted(by_year)]

    def download(self, filing: Filing) -> bytes:
        return self._get(filing.url).content
