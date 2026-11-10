"""Run the whole pipeline on one filing. Used by the app's live tab and the CLI."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from riskradar.chunk import split_into_chunks
from riskradar.classify import RiskClassifier
from riskradar.embed import Embedder
from riskradar.extract import risk_section_from_html
from riskradar.sec import Filing, SecClient, SecError
from riskradar.taxonomy import NON_RISK, RISK_CATEGORIES


@dataclass
class Analysis:
    filing: Filing
    passages: pd.DataFrame          # text, category, confidence, second
    mix: pd.Series                  # share of each risk category, sums to 1
    words: int
    timings: dict[str, float] = field(default_factory=dict)


def label_passages(texts: list[str], embeddings: np.ndarray, clf: RiskClassifier) -> pd.DataFrame:
    proba = clf.predict_proba(embeddings)
    order = np.argsort(-proba, axis=1)
    return pd.DataFrame({
        "text": texts,
        "category": [clf.classes[i] for i in order[:, 0]],
        "confidence": proba.max(axis=1).round(3),
        "second": [clf.classes[i] for i in order[:, 1]],
    })


def risk_mix(categories: pd.Series) -> pd.Series:
    """Share of each risk category. Introductions are left out."""
    counts = categories[~categories.isin(NON_RISK)].value_counts()
    return (counts / counts.sum()).reindex(RISK_CATEGORIES, fill_value=0.0)


def analyze_filing(
    ticker: str,
    client: SecClient,
    embedder: Embedder,
    clf: RiskClassifier,
    year: int | None = None,
    on_step: Callable[[str, float, str], None] | None = None,
) -> Analysis:
    """Download -> extract -> split -> embed -> classify, for the newest 10-K (or a given year)."""
    timings: dict[str, float] = {}

    def step(name: str, started: float, detail: str) -> None:
        timings[name] = time.perf_counter() - started
        if on_step:
            on_step(name, timings[name], detail)

    t = time.perf_counter()
    filings = client.list_10k(ticker)
    if year is None:
        filing = filings[-1]
    else:
        match = [f for f in filings if f.fiscal_year == year]
        if not match:
            raise SecError(f"No 10-K for fiscal year {year}. Available: {filings[0].fiscal_year}-{filings[-1].fiscal_year}.")
        filing = match[0]
    step("Find the filing", t, f"{filing.sec_name}, fiscal year {filing.fiscal_year}, filed {filing.filed}")

    t = time.perf_counter()
    raw = client.download(filing)
    step("Download from SEC EDGAR", t, f"{len(raw) / 1e6:.1f} MB of HTML")

    t = time.perf_counter()
    section = risk_section_from_html(raw)
    words = len(section.split())
    step("Extract Item 1A Risk Factors", t, f"{words:,} words")

    t = time.perf_counter()
    chunks = split_into_chunks(section)
    step("Split into passages", t, f"{len(chunks)} passages")

    t = time.perf_counter()
    embeddings = embedder.encode(chunks)
    step("Embed passages", t, f"{embeddings.shape[1]}-dimensional vectors")

    t = time.perf_counter()
    passages = label_passages(chunks, embeddings, clf)
    step("Classify risks", t, f"{passages['category'].nunique()} categories found")

    return Analysis(filing, passages, risk_mix(passages["category"]), words, timings)
