"""Step 3: classify every chunk, measure, and write the small files the app reads.

Run after build_chunks.py and embed_all.py.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

from src.classify import MODEL_NAME, load_model
from src.prototypes import PROTOTYPES
from src.taxonomy import CATEGORIES, NON_RISK

OUT = "data/app"
NAMES = list(CATEGORIES)


def prototype_scores(emb, model):
    protos = [model.encode(PROTOTYPES[n] + [CATEGORIES[n]], normalize_embeddings=True) for n in NAMES]
    return np.stack([(emb @ p.T).max(1) for p in protos], axis=1)


def near_duplicate_groups(vectors, threshold=0.9):
    sim = vectors @ vectors.T
    group = list(range(len(vectors)))

    def find(i):
        while group[i] != i:
            i = group[i]
        return i

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if sim[i, j] >= threshold:
                group[find(j)] = find(i)
    return [find(i) for i in range(len(vectors))]


def accuracy(pred, truth):
    return float(np.mean(np.asarray(pred, dtype=object) == np.asarray(truth, dtype=object)))


def main():
    chunks = pd.read_parquet("data/processed/chunks.parquet")
    emb = np.load("data/processed/embeddings.npy")
    model = load_model()
    scores = prototype_scores(emb, model)
    zero_shot = np.array([NAMES[i] for i in scores.argmax(1)], dtype=object)

    key = ["ticker", "year", "chunk_id"]
    idx = chunks.reset_index()[key + ["index"]]
    dev = pd.read_csv("data/eval/dev_labels.csv").merge(idx, on=key)
    test = pd.read_csv("data/eval/test_labels.csv").merge(idx, on=key)

    # 1) zero-shot, scored on dev (used for tuning) and test (never used for tuning)
    zs_dev = accuracy(zero_shot[dev["index"]], dev["label"])
    zs_test = accuracy(zero_shot[test["index"]], test["label"])

    # 2) small trained model (logistic regression) on all 119 labels,
    #    scored with 10-fold cross-validation so every passage is predicted by a model that never saw it
    labelled = pd.concat([dev, test], ignore_index=True)
    X = np.hstack([emb[labelled["index"]], scores[labelled["index"]] * 3])
    y = labelled["label"].to_numpy(dtype=object)
    # near-identical passages (same paragraph repeated in another year) must stay in the same fold,
    # otherwise the model is tested on text it has already seen
    groups = near_duplicate_groups(emb[labelled["index"]])
    cv_pred = cross_val_predict(LogisticRegression(C=4, max_iter=5000), X, y, cv=GroupKFold(10), groups=groups)
    lr_cv = accuracy(cv_pred, y)

    use_lr = lr_cv > zs_test
    if use_lr:
        clf = LogisticRegression(C=4, max_iter=5000).fit(X, y)
        labels = clf.predict(np.hstack([emb, scores * 3]))
        method, headline, headline_n = "logistic regression on embeddings", lr_cv, len(labelled)
    else:
        labels = zero_shot
        method, headline, headline_n = "zero-shot prototypes", zs_test, len(test)

    top2 = np.argsort(-scores, 1)[:, :2]
    top2_hit = [test["label"].iloc[i] in [NAMES[j] for j in top2[r]] for i, r in enumerate(test["index"])]

    chunks["category"] = labels
    chunks["company"] = chunks["ticker"].map({"AAPL": "Apple", "JPM": "JPMorgan Chase", "TSLA": "Tesla"})

    # share of each risk category per report (introductions left out)
    risk = chunks[~chunks["category"].isin(NON_RISK)]
    share = risk.groupby(["company", "year", "category"]).size().rename("chunks").reset_index()
    # every company/year/category combination, with 0 where a category is absent
    full = pd.MultiIndex.from_product(
        [share["company"].unique(), sorted(share["year"].unique()), sorted(share["category"].unique())],
        names=["company", "year", "category"],
    )
    share = share.set_index(["company", "year", "category"]).reindex(full, fill_value=0).reset_index()
    share["share"] = share["chunks"] / share.groupby(["company", "year"])["chunks"].transform("sum")

    # new language: passages in the latest report least similar to anything in the first report
    rows = []
    for company, g in risk.groupby("company"):
        first, last = g["year"].min(), g["year"].max()
        a, b = g[g["year"] == first], g[g["year"] == last]
        sim = emb[b.index] @ emb[a.index].T
        novelty = 1 - sim.max(1)
        b = b.assign(novelty=novelty).sort_values("novelty", ascending=False).head(6)
        rows.append(b[["company", "year", "category", "novelty", "text"]])
    new = pd.concat(rows)

    metrics = {
        "reports": int(chunks.groupby(["ticker", "year"]).ngroups),
        "chunks": int(len(chunks)),
        "categories": len(NAMES) - len(NON_RISK),
        "model": MODEL_NAME,
        "method": method,
        "test_accuracy": headline,
        "test_n": headline_n,
        "dup_groups": len(set(groups)),
        "test_baseline": float(labelled["label"].value_counts(normalize=True).iloc[0]),
        "test_top2": float(np.mean(top2_hit)),
        "dev_accuracy": zs_dev,
        "zero_shot_test_accuracy": zs_test,
        "lr_cv_accuracy": lr_cv,
    }

    os.makedirs(OUT, exist_ok=True)
    chunks[["company", "ticker", "year", "chunk_id", "category", "text"]].to_parquet(f"{OUT}/chunks.parquet", index=False)
    share.to_csv(f"{OUT}/category_share.csv", index=False)
    new.to_csv(f"{OUT}/new_language.csv", index=False)
    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
