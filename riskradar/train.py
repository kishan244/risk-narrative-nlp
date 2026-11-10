"""Train the risk classifier and measure it honestly.

Protocol
1. `data/eval/labels.csv` holds labelled passages, split into train and test.
   The test passages were drawn at random and are never used for any choice.
2. 10-K filings repeat paragraphs year after year. Any training passage that is
   a near-copy of a test passage (cosine >= 0.9) is removed from training.
3. The regularisation strength C is chosen by cross-validation on the training
   set only (near-copies stay in the same fold).
4. The model is fitted on the training set and scored once on the test set.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, cross_val_predict

from riskradar.classify import MODEL_PATH, RiskClassifier
from riskradar.config import CACHE_DIR, EVAL_DIR, MODEL_DIR
from riskradar.embed import MODEL_NAME, Embedder, EmbeddingStore
from riskradar.taxonomy import CATEGORIES

NEAR_COPY = 0.9
C_GRID = (4, 8, 16, 32, 64, 128)


def near_copy_groups(vectors: np.ndarray, threshold: float = NEAR_COPY) -> np.ndarray:
    """Group id per row: rows whose cosine similarity is >= threshold share a group."""
    parent = list(range(len(vectors)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sim = vectors @ vectors.T
    for i, j in zip(*np.where(np.triu(sim >= threshold, k=1))):
        parent[find(j)] = find(i)
    return np.array([find(i) for i in range(len(vectors))])


def bootstrap_interval(correct: np.ndarray, rounds: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval for accuracy by resampling the test passages."""
    rng = np.random.default_rng(seed)
    means = rng.choice(correct, size=(rounds, len(correct)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def train(embedder: Embedder | None = None, save: bool = True) -> dict:
    labels = pd.read_csv(EVAL_DIR / "labels.csv")
    unknown = set(labels["label"]) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"labels.csv has categories that are not in the taxonomy: {unknown}")

    store = EmbeddingStore(CACHE_DIR / "embeddings.npz", embedder or Embedder())
    X = store.get(labels["text"].tolist())
    is_test = (labels["split"] == "test").to_numpy()
    X_test, y_test = X[is_test], labels.loc[is_test, "label"].to_numpy()
    X_train, y_train = X[~is_test], labels.loc[~is_test, "label"].to_numpy()

    # step 2: no training passage may be a near-copy of a test passage
    leak = (X_train @ X_test.T).max(axis=1) >= NEAR_COPY
    X_train, y_train = X_train[~leak], y_train[~leak]

    # step 3: choose C on the training set only
    groups = near_copy_groups(X_train)
    folds = GroupKFold(n_splits=5)
    cv_scores = {}
    for c in C_GRID:
        pred = cross_val_predict(LogisticRegression(C=c, max_iter=5000), X_train, y_train, cv=folds, groups=groups)
        cv_scores[c] = float(np.mean(pred == y_train))
    best_c = max(cv_scores, key=cv_scores.get)

    # step 4: fit once, score once
    model = LogisticRegression(C=best_c, max_iter=5000).fit(X_train, y_train)
    clf = RiskClassifier(list(model.classes_), model.coef_, model.intercept_, MODEL_NAME)
    proba = clf.predict_proba(X_test)
    order = np.argsort(-proba, axis=1)
    pred = np.array([clf.classes[i] for i in order[:, 0]], dtype=object)
    second = np.array([clf.classes[i] for i in order[:, 1]], dtype=object)
    correct = pred == y_test

    names = [c for c in CATEGORIES if c in set(y_test) | set(pred)]
    precision, recall, f1, support = precision_recall_fscore_support(y_test, pred, labels=names, zero_division=0)
    low, high = bootstrap_interval(correct.astype(float))
    test_meta = labels.loc[is_test].reset_index(drop=True)
    metrics = {
        "embedding_model": MODEL_NAME,
        "classifier": f"multinomial logistic regression, C={best_c}",
        "train_passages": int(len(y_train)),
        "train_removed_as_near_copies_of_test": int(leak.sum()),
        "test_passages": int(len(y_test)),
        "test_accuracy": float(correct.mean()),
        "test_accuracy_ci95": [low, high],
        "test_top2_accuracy": float(np.mean(correct | (second == y_test))),
        "test_macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "baseline_most_common_class": float(max(Counter(y_test).values()) / len(y_test)),
        "cv_accuracy_by_C": cv_scores,
        "per_category": [
            {"category": n, "precision": float(p), "recall": float(r), "f1": float(f), "test_passages": int(s)}
            for n, p, r, f, s in zip(names, precision, recall, f1, support)
        ],
        "per_sector_accuracy": {
            sector: float(correct[(test_meta["sector"] == sector).to_numpy()].mean())
            for sector in sorted(test_meta["sector"].dropna().unique())
        },
        "confusion": {"labels": names, "matrix": confusion_matrix(y_test, pred, labels=names).tolist()},
    }
    if save:
        clf.save(MODEL_PATH)
        (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
