import numpy as np
from sklearn.linear_model import LogisticRegression

from riskradar.classify import RiskClassifier


def test_matches_scikit_learn_exactly():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(120, 8)), rng.integers(0, 4, size=120).astype(str)
    model = LogisticRegression(max_iter=500).fit(X, y)
    clf = RiskClassifier(list(model.classes_), model.coef_, model.intercept_, "test")
    np.testing.assert_allclose(clf.predict_proba(X), model.predict_proba(X), atol=1e-6)
    assert clf.predict(X)[0] == list(model.predict(X))


def test_probabilities_sum_to_one(tiny_classifier):
    proba = tiny_classifier.predict_proba(np.random.default_rng(2).normal(size=(10, 16)))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_save_and_load_round_trip(tiny_classifier, tmp_path):
    path = tmp_path / "model.npz"
    tiny_classifier.save(path)
    loaded = RiskClassifier.load(path)
    assert loaded.classes == tiny_classifier.classes and loaded.embedding_model == "fake"
    np.testing.assert_allclose(loaded.coef, tiny_classifier.coef, atol=1e-6)
