"""Step 2: embed every chunk once and cache it (slow part, ~5 min on CPU)."""
import numpy as np
import pandas as pd
from src.classify import load_model

df = pd.read_parquet("data/processed/chunks.parquet")
model = load_model()
emb = model.encode(df["text"].tolist(), normalize_embeddings=True, batch_size=16, show_progress_bar=False)
np.save("data/processed/embeddings.npy", emb)
print("saved", emb.shape)
