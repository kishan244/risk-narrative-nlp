import json
import os

import pandas as pd
import pytest

APP = "data/app"
pytestmark = pytest.mark.skipif(not os.path.exists(f"{APP}/metrics.json"), reason="run build_results.py first")


def test_shares_add_up_to_one():
    share = pd.read_csv(f"{APP}/category_share.csv")
    totals = share.groupby(["company", "year"])["share"].sum()
    assert ((totals - 1).abs() < 1e-6).all()


def test_every_report_present():
    chunks = pd.read_parquet(f"{APP}/chunks.parquet")
    assert chunks.groupby("company")["year"].nunique().min() >= 7


def test_metrics_are_sane():
    m = json.load(open(f"{APP}/metrics.json"))
    assert 0 < m["test_accuracy"] <= 1
    assert m["test_accuracy"] > m["test_baseline"]
