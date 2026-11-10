"""Checks on the committed data and model: they fail loudly if a build went wrong."""
import json

import numpy as np
import pandas as pd
import pytest

from riskradar.classify import RiskClassifier
from riskradar.config import APP_DIR, EVAL_DIR, load_companies
from riskradar.taxonomy import CATEGORIES


@pytest.fixture(scope="module")
def data():
    return (pd.read_parquet(APP_DIR / "passages.parquet"), pd.read_csv(APP_DIR / "category_share.csv"),
            pd.read_csv(APP_DIR / "reports.csv"), json.loads((APP_DIR / "build_info.json").read_text()))


def test_every_configured_company_has_seven_or_more_reports(data):
    _, _, reports, info = data
    _, companies = load_companies()
    counts = reports.groupby("ticker")["year"].nunique()
    assert set(counts.index) == {c.ticker for c in companies}
    assert counts.min() >= 7 and info["problems"] == []


def test_risk_sections_have_a_plausible_length(data):
    _, _, reports, _ = data
    assert reports["words"].between(2_000, 40_000).all()
    assert (reports["passages"] * 300 >= reports["words"] * 0.85).all()   # passages cover the section


def test_shares_sum_to_one_per_report(data):
    _, share, _, _ = data
    assert np.allclose(share.groupby(["ticker", "year"])["share"].sum(), 1.0)


def test_passages_embeddings_and_labels_line_up(data):
    passages, _, _, info = data
    embeddings = np.load(APP_DIR / "embeddings.npy")
    assert len(passages) == len(embeddings) == info["passages"]
    assert set(passages["category"]) <= set(CATEGORIES)
    assert RiskClassifier.load().coef.shape[1] == embeddings.shape[1]


def test_reuse_share_is_a_fraction(data):
    _, _, reports, _ = data
    assert reports["reused_share"].dropna().between(0, 1).all()


def test_labels_are_valid_and_test_set_is_separate():
    labels = pd.read_csv(EVAL_DIR / "labels.csv")
    assert set(labels["label"]) <= set(CATEGORIES) and set(labels["split"]) == {"train", "test"}
    assert labels["text"].is_unique
    assert not set(labels.loc[labels["split"] == "test", "text"]) & set(labels.loc[labels["split"] == "train", "text"])


def test_reported_accuracy_beats_the_baseline(data):
    metrics = data[3]["metrics"]
    assert metrics["test_accuracy"] > 2 * metrics["baseline_most_common_class"]
    assert metrics["test_passages"] >= 200
