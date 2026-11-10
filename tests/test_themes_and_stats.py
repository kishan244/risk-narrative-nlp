import numpy as np
import pandas as pd

from riskradar.build import category_shares, complete_years, similarity_to_previous_year
from riskradar.corpus import sentence_set
from riskradar.themes import THEMES, custom_pattern
from riskradar.train import bootstrap_interval, near_copy_groups


def test_ai_theme_needs_upper_case_ai():
    ai = THEMES["Artificial intelligence"]
    assert ai.search("We use AI and machine learning.")
    assert not ai.search("We said that regenerative braking may fail and we maintain airbags.")


def test_custom_pattern_is_safe_and_case_insensitive():
    assert custom_pattern("C++ (beta)").search("we ship c++ (beta) soon")
    assert not custom_pattern("art").search("smart devices")      # word start only


def test_sentence_set_ignores_spacing_and_punctuation():
    a = sentence_set("JPMorgan Chase’s results could be hurt by “higher” interest rates in future periods.")
    b = sentence_set('JPMorganChase\'s results could be hurt by "higher" interest rates in future periods.')
    assert a == b and len(a) == 1


def _passages():
    rows = [("A", "Alpha", "Tech", 2023, "Regulatory & Legal"), ("A", "Alpha", "Tech", 2023, "Climate & ESG"),
            ("A", "Alpha", "Tech", 2024, "Regulatory & Legal"), ("A", "Alpha", "Tech", 2024, "General / Introduction")]
    return pd.DataFrame(rows, columns=["ticker", "company", "sector", "year", "category"])


def test_shares_sum_to_one_and_cover_every_category():
    share = category_shares(_passages())
    totals = share.groupby(["ticker", "year"])["share"].sum()
    assert np.allclose(totals, 1.0)
    assert share.groupby(["ticker", "year"])["category"].nunique().eq(13).all()
    assert share.query("year == 2024 and category == 'Regulatory & Legal'")["share"].item() == 1.0


def test_similarity_to_previous_year():
    passages = _passages()
    emb = np.eye(4, dtype=np.float32)
    emb[2] = emb[0]                                   # 2024 passage copies a 2023 passage
    sim = similarity_to_previous_year(passages, emb)
    assert np.isnan(sim[0]) and np.isnan(sim[1])      # first year has nothing to compare with
    assert sim[2] == 1.0 and sim[3] == 0.0


def test_complete_years_need_most_companies():
    reports = pd.DataFrame({"ticker": list("ABCDEFGHIJ") * 2 + ["A"], "year": [2024] * 10 + [2025] * 10 + [2026]})
    assert complete_years(reports) == [2024, 2025]


def test_near_copies_share_a_group():
    v = np.array([[1, 0], [0.99, 0.14], [0, 1]], dtype=float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    groups = near_copy_groups(v)
    assert groups[0] == groups[1] != groups[2]


def test_bootstrap_interval_contains_the_mean():
    correct = np.array([1.0] * 70 + [0.0] * 30)
    low, high = bootstrap_interval(correct)
    assert low < 0.7 < high and high - low < 0.25
