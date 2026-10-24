"""Step 2: embed every chunk once and cache it (slow part, ~5 min on CPU)."""
import os

import numpy as np
import pandas as pd
from src.classify import load_model

PATH = "data/processed/embeddings.npy"
TEXTS = "data/processed/embedded_texts.parquet"
df = pd.read_parquet("data/processed/chunks.parquet")
model = load_model()
# reuse vectors for passages that were already embedded
cache = {}
if os.path.exists(PATH) and os.path.exists(TEXTS):
    old = np.load(PATH)
    cache = dict(zip(pd.read_parquet(TEXTS)["text"], old))
todo = [t for t in df["text"].unique() if t not in cache]
if todo:
    new = model.encode(todo, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
    cache.update(zip(todo, new))
emb = np.stack([cache[t] for t in df["text"]])
df[["text"]].to_parquet(TEXTS, index=False)
print("embedded", len(todo), "new passages")
np.save(PATH, emb)
print("saved", emb.shape)
