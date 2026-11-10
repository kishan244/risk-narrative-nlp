"""Build everything the app reads: classify all passages and compute the statistics."""
from __future__ import annotations

import json
import logging
from datetime import date

import numpy as np
import pandas as pd

from riskradar import __version__
from riskradar.analyze import label_passages
from riskradar.classify import RiskClassifier
from riskradar.config import APP_DIR, CACHE_DIR, MODEL_DIR, load_companies
from riskradar.corpus import build_corpus
from riskradar.embed import MODEL_NAME, Embedder, EmbeddingStore
from riskradar.sec import SecClient
from riskradar.taxonomy import NON_RISK, RISK_CATEGORIES
from riskradar.themes import THEMES, theme_share

log = logging.getLogger(__name__)


def similarity_to_previous_year(passages: pd.DataFrame, emb: np.ndarray) -> np.ndarray:
    """For each passage: cosine similarity to the closest passage in the same
    company's report of the year before (NaN when there is no such report)."""
    sim = np.full(len(passages), np.nan, dtype=np.float32)
    for _, group in passages.groupby("ticker"):
        years = sorted(group["year"].unique())
        for prev, cur in zip(years[:-1], years[1:]):
            if cur - prev != 1:
                continue
            idx_prev = group.index[group["year"] == prev].to_numpy()
            idx_cur = group.index[group["year"] == cur].to_numpy()
            sim[idx_cur] = (emb[idx_cur] @ emb[idx_prev].T).max(axis=1)
    return sim


def category_shares(passages: pd.DataFrame) -> pd.DataFrame:
    """Share of each risk category per report. Every category is present (0 when absent)."""
    risk = passages[~passages["category"].isin(NON_RISK)]
    counts = risk.groupby(["ticker", "company", "sector", "year", "category"]).size().rename("passages")
    reports = risk[["ticker", "company", "sector", "year"]].drop_duplicates()
    grid = reports.merge(pd.DataFrame({"category": RISK_CATEGORIES}), how="cross")
    share = grid.merge(counts.reset_index(), how="left").fillna({"passages": 0})
    share["passages"] = share["passages"].astype(int)
    share["share"] = share["passages"] / share.groupby(["ticker", "year"])["passages"].transform("sum")
    return share


def complete_years(reports: pd.DataFrame, coverage: float = 0.9) -> list[int]:
    """Years in which nearly every company has filed, so cross-company views are fair."""
    per_year = reports.groupby("year")["ticker"].nunique()
    return sorted(per_year[per_year >= coverage * reports["ticker"].nunique()].index.tolist())


def make_insights(passages: pd.DataFrame, share: pd.DataFrame, reports: pd.DataFrame, years: list[int]) -> list[dict]:
    """Headline findings, computed from the data so they stay true after every refresh."""
    first, last = years[0], years[-1]
    insights = []

    ai = theme_share(passages, THEMES["Artificial intelligence"])
    ai_first, ai_last = ai[ai["year"] == first], ai[ai["year"] == last]
    insights.append({
        "title": "AI became a standard risk",
        "value": f"{(ai_last['mentions'] > 0).sum()} of {len(ai_last)}",
        "text": f"companies mention AI in their {last} risk factors. In {first} it was "
                f"{(ai_first['mentions'] > 0).sum()} of {len(ai_first)}.",
    })

    reuse = reports.dropna(subset=["reused_share"])
    reuse = reuse[reuse["year"].isin(years)]
    by_year = reuse.groupby("year")["reused_share"].mean()
    insights.append({
        "title": "Most of the text is copy-paste",
        "value": f"{reuse['reused_share'].mean():.0%}",
        "text": f"of sentences are repeated word for word from the year before. "
                f"The most rewriting happened in {int(by_year.idxmin())} ({1 - by_year.min():.0%} new sentences).",
    })

    mean = share[share["year"].isin([first, last])].groupby(["category", "year"])["share"].mean().unstack()
    change = (mean[last] - mean[first]).sort_values()
    insights.append({
        "title": f"Biggest riser since {first}",
        "value": change.index[-1],
        "text": f"grew from {mean.loc[change.index[-1], first]:.1%} to {mean.loc[change.index[-1], last]:.1%} "
                f"of the average risk section. Biggest faller: {change.index[0]} "
                f"({mean.loc[change.index[0], first]:.1%} to {mean.loc[change.index[0], last]:.1%}).",
    })
    return insights


def build(embedder: Embedder | None = None, client: SecClient | None = None) -> dict:
    start_year, companies = load_companies()
    passages, reports, problems = build_corpus(companies, start_year, client or SecClient())
    if passages.empty:
        raise RuntimeError("No passages could be built. Is SEC EDGAR reachable?")

    store = EmbeddingStore(CACHE_DIR / "embeddings.npz", embedder or Embedder())
    emb = store.get(passages["text"].tolist())
    clf = RiskClassifier.load()
    if clf.embedding_model != MODEL_NAME:
        raise RuntimeError(f"Classifier was trained on {clf.embedding_model}, embeddings come from {MODEL_NAME}.")

    labelled = label_passages(passages["text"].tolist(), emb, clf)
    passages = passages.assign(
        category=labelled["category"].to_numpy(),
        confidence=labelled["confidence"].to_numpy(),
        second=labelled["second"].to_numpy(),
        similarity_prev=similarity_to_previous_year(passages, emb),
    )
    share = category_shares(passages)
    years = complete_years(reports)
    insights = make_insights(passages, share, reports, years)

    APP_DIR.mkdir(parents=True, exist_ok=True)
    passages.to_parquet(APP_DIR / "passages.parquet", index=False, compression="zstd")
    np.save(APP_DIR / "embeddings.npy", emb.astype(np.float16))
    share.to_csv(APP_DIR / "category_share.csv", index=False)
    reports.to_csv(APP_DIR / "reports.csv", index=False)
    info = {
        "version": __version__,
        "built": date.today().isoformat(),
        "companies": int(reports["ticker"].nunique()),
        "reports": int(len(reports)),
        "passages": int(len(passages)),
        "first_year": int(reports["year"].min()),
        "last_year": int(reports["year"].max()),
        "complete_years": years,
        "embedding_model": MODEL_NAME,
        "problems": problems.to_dict("records"),
        "insights": insights,
        "metrics": json.loads((MODEL_DIR / "metrics.json").read_text()),
    }
    (APP_DIR / "build_info.json").write_text(json.dumps(info, indent=2))
    log.info("Built %s passages from %s reports of %s companies", info["passages"], info["reports"], info["companies"])
    return info
