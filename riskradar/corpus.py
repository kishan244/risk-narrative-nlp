"""Build the passage table: every company, every fiscal year, every passage."""
from __future__ import annotations

import json
import logging
import os
import re

import pandas as pd

from riskradar.chunk import split_into_chunks
from riskradar.config import CACHE_DIR, Company
from riskradar.extract import ExtractionError, risk_section_from_html
from riskradar.sec import Filing, SecClient, SecError

log = logging.getLogger(__name__)
SECTION_DIR = CACHE_DIR / "sections"


_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")
_NOT_ALNUM = re.compile(r"[^a-z0-9]")


def sentence_set(section: str) -> set[str]:
    """Sentences of a section, reduced to lower-case letters and digits.

    Dropping spaces and punctuation means that a changed quote mark, or a
    company that starts writing its name as one word, does not make an
    otherwise identical sentence look new.
    """
    sentences = set()
    for line in section.split("\n"):
        for sentence in _SENTENCE_END.split(line):
            norm = _NOT_ALNUM.sub("", sentence.lower())
            if len(norm) >= 40:
                sentences.add(norm)
    return sentences


def _download(client: SecClient, filing: Filing) -> bytes:
    """Download a filing. Set RISKRADAR_KEEP_HTML=1 to keep the raw HTML for debugging."""
    if not os.environ.get("RISKRADAR_KEEP_HTML"):
        return client.download(filing)
    path = CACHE_DIR / "raw" / f"{filing.ticker}_{filing.fiscal_year}.htm"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(client.download(filing))
    return path.read_bytes()


def load_section(client: SecClient, filing: Filing) -> str:
    """Risk section of one filing. Cached on disk so SEC is asked only once."""
    path = SECTION_DIR / f"{filing.ticker}_{filing.fiscal_year}_{filing.accession}.json"
    if path.exists():
        return json.loads(path.read_text())["section"]
    section = risk_section_from_html(_download(client, filing))
    SECTION_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**filing.to_dict(), "section": section}))
    return section


def build_corpus(companies: list[Company], start_year: int, client: SecClient | None = None):
    """Return (passages, filings, problems).

    passages: one row per passage. filings: one row per report.
    problems: reports that could not be read (kept visible, never silently dropped).
    """
    client = client or SecClient()
    rows, reports, problems = [], [], []
    for company in companies:
        try:
            filings = [f for f in client.list_10k(company.ticker, company.cik) if f.fiscal_year >= start_year]
        except SecError as exc:
            problems.append({"ticker": company.ticker, "year": None, "problem": str(exc)})
            continue
        previous: tuple[int, set[str]] | None = None   # (fiscal year, sentences) of the report before
        for filing in filings:
            try:
                section = load_section(client, filing)
            except (SecError, ExtractionError) as exc:
                problems.append({"ticker": company.ticker, "year": filing.fiscal_year, "problem": str(exc)})
                continue
            chunks = split_into_chunks(section)
            # how much of this report is copied word for word from last year's report?
            sentences = sentence_set(section)
            reused = None
            if previous and previous[0] == filing.fiscal_year - 1 and sentences:
                reused = len(sentences & previous[1]) / len(sentences)
            previous = (filing.fiscal_year, sentences)
            reports.append({
                "ticker": company.ticker, "company": company.name, "sector": company.sector,
                "year": filing.fiscal_year, "period_end": filing.period_end, "filed": filing.filed,
                "url": filing.url, "words": len(section.split()), "passages": len(chunks),
                "sentences": len(sentences), "reused_share": reused,
            })
            for i, chunk in enumerate(chunks):
                rows.append({
                    "ticker": company.ticker, "company": company.name, "sector": company.sector,
                    "year": filing.fiscal_year, "chunk_id": i, "text": chunk,
                })
            log.info("%s %s: %s words, %s passages", company.ticker, filing.fiscal_year, f"{len(section.split()):,}", len(chunks))
    return pd.DataFrame(rows), pd.DataFrame(reports), pd.DataFrame(problems, columns=["ticker", "year", "problem"])
