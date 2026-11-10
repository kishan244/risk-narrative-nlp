import pytest

from riskradar.sec import SecClient, SecError, fiscal_year

TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
SUBMISSIONS = {
    "name": "Apple Inc.",
    "filings": {"recent": {
        "form": ["10-K/A", "10-K", "8-K", "10-K", "10-K"],
        "reportDate": ["2025-09-27", "2025-09-27", "2025-01-01", "2024-09-28", "2024-01-31"],
        "filingDate": ["2025-12-01", "2025-10-31", "2025-01-02", "2024-11-01", "2024-03-15"],
        "accessionNumber": ["0-25-9", "0-25-1", "0-25-2", "0-24-1", "0-24-0"],
        "primaryDocument": ["a.htm", "k25.htm", "e.htm", "k24.htm", "k23.htm"],
    }, "files": []},
}


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.content = b"<html></html>"

    def json(self):
        return self._payload


@pytest.fixture
def client(monkeypatch):
    c = SecClient()
    monkeypatch.setattr(c, "_get", lambda url: FakeResponse(TICKERS if "tickers" in url else SUBMISSIONS))
    return c


def test_fiscal_year_counts_january_year_ends_as_previous_year():
    assert fiscal_year("2025-09-27") == 2025
    assert fiscal_year("2026-01-25") == 2025   # NVIDIA, Walmart
    assert fiscal_year("2022-01-02") == 2021   # Johnson & Johnson


def test_only_10k_forms_one_per_fiscal_year_oldest_first(client):
    filings = client.list_10k("aapl")
    assert [f.fiscal_year for f in filings] == [2023, 2024, 2025]
    assert filings[-1].url.endswith("/0251/k25.htm") and "320193" in filings[-1].url


def test_unknown_ticker_gives_readable_error(client):
    with pytest.raises(SecError, match="not a ticker"):
        client.list_10k("NOPE")


def test_cik_override_skips_ticker_lookup(client):
    assert client.list_10k("XOM", cik="34088")[0].cik == "0000034088"


def test_company_without_10k_explains_why(client, monkeypatch):
    empty = {"name": "Foreign Co", "filings": {"recent": {k: [] for k in SUBMISSIONS["filings"]["recent"]}, "files": []}}
    monkeypatch.setattr(client, "_get", lambda url: FakeResponse(TICKERS if "tickers" in url else empty))
    with pytest.raises(SecError, match="20-F"):
        client.list_10k("AAPL")
