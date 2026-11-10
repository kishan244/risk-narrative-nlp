import numpy as np
import pytest

from riskradar.classify import RiskClassifier

RISK_PARAGRAPH = (
    "Our business depends on a small number of suppliers and any disruption could harm our results. "
    "We may not be able to find other sources in time and our costs could rise sharply as a result. "
)


def make_10k_html(risk_paragraphs: int = 40) -> str:
    """A tiny but realistic 10-K: table of contents, a cross-reference, hidden XBRL, real sections."""
    body = "".join(f"<p>{RISK_PARAGRAPH * 3}</p>" for _ in range(risk_paragraphs))
    return f"""<html><head><title>10-K</title><style>p {{}}</style></head><body>
    <div style="display:none">Item 1A. Risk Factors hidden xbrl metadata that must be ignored</div>
    <table>
      <tr><td>Item 1A.</td><td>Risk Factors</td><td>12</td></tr>
      <tr><td>Item 1B.</td><td>Unresolved Staff Comments</td><td>30</td></tr>
      <tr><td>Item 2.</td><td>Properties</td><td>31</td></tr>
    </table>
    <p>Item 1. Business</p>
    <p>We make phones. For a discussion of risks, see Item 1A. Risk Factors below, which matters a lot.</p>
    <p>We also sell services to many customers in many countries around the world every year.</p>
    <p><b>Item 1A. Risk Factors</b></p>
    {body}
    <p>Item 1B. Unresolved Staff Comments</p><p>None.</p>
    <p>Item 2. Properties</p><p>We own a large building in a city.</p>
    </body></html>"""


@pytest.fixture
def ten_k_html() -> str:
    return make_10k_html()


class FakeEmbedder:
    """Deterministic vectors from word hashes, so tests need no model download."""

    def encode(self, texts, batch_size: int = 16):
        vectors = np.zeros((len(texts), 16), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                vectors[i, sum(map(ord, word)) % 16] += 1.0
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def tiny_classifier() -> RiskClassifier:
    rng = np.random.default_rng(0)
    classes = ["Competition & Demand", "Regulatory & Legal", "Supply Chain & Manufacturing"]
    return RiskClassifier(classes, rng.normal(size=(3, 16)), rng.normal(size=3), "fake")
