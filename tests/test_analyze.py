import pandas as pd

from riskradar.analyze import analyze_filing, risk_mix
from riskradar.sec import Filing
from riskradar.taxonomy import RISK_CATEGORIES
from tests.conftest import make_10k_html


class FakeClient:
    def list_10k(self, ticker, cik=None):
        return [
            Filing(ticker, "1", "Test Corp", y, f"{y}-12-31", f"{y + 1}-02-01", "0-0-0", "https://example.com")
            for y in (2023, 2024)
        ]

    def download(self, filing):
        return make_10k_html().encode()


def test_full_pipeline_runs_offline(fake_embedder, tiny_classifier):
    steps = []
    result = analyze_filing("TEST", FakeClient(), fake_embedder, tiny_classifier, on_step=lambda *a: steps.append(a[0]))
    assert result.filing.fiscal_year == 2024                      # newest filing by default
    assert len(steps) == 6 and steps[0] == "Find the filing"
    assert len(result.passages) > 5 and result.words > 1500
    assert abs(result.mix.sum() - 1) < 1e-9
    assert list(result.mix.index) == RISK_CATEGORIES


def test_a_given_year_can_be_requested(fake_embedder, tiny_classifier):
    assert analyze_filing("TEST", FakeClient(), fake_embedder, tiny_classifier, year=2023).filing.fiscal_year == 2023


def test_risk_mix_ignores_introductions():
    mix = risk_mix(pd.Series(["General / Introduction", "Regulatory & Legal", "Regulatory & Legal", "Climate & ESG"]))
    assert mix["Regulatory & Legal"] == 2 / 3 and "General / Introduction" not in mix.index
