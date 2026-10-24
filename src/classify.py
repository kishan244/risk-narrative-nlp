"""Zero-shot risk classification with sentence embeddings.

Each category has a few example sentences. A chunk gets the category whose
closest example sentence is most similar (cosine similarity).
No training data is needed, and every company is judged by the same rules.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from src.prototypes import PROTOTYPES
from src.taxonomy import CATEGORIES

MODEL_NAME = "BAAI/bge-base-en-v1.5"  # picked on the dev set, see EVALUATION.md


def load_model():
    return SentenceTransformer(MODEL_NAME)


def classify(texts, model):
    names = list(CATEGORIES)
    protos = [model.encode(PROTOTYPES[n] + [CATEGORIES[n]], normalize_embeddings=True) for n in names]
    emb = model.encode(list(texts), normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    scores = np.stack([(emb @ p.T).max(1) for p in protos], axis=1)
    best = scores.argmax(1)
    labels = [names[i] for i in best]
    confidence = scores.max(1)
    return labels, confidence, emb
