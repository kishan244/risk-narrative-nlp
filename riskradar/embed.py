"""Sentence embeddings with a small ONNX model (no PyTorch needed).

The same model is used for the offline build and for the live analysis in the
app, so a passage always gets the same vector in both places.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 22M parameters, reads the first 256 tokens


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


class Embedder:
    """Lazy wrapper: the model (about 90 MB) is downloaded on first use."""

    def __init__(self, model_name: str = MODEL_NAME, threads: int | None = None):
        self.model_name = model_name
        self.threads = threads
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name, threads=self.threads)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Unit-length vectors, one row per text."""
        if not len(texts):
            return np.zeros((0, 0), dtype=np.float32)
        vectors = np.asarray(list(self._load().embed(list(texts), batch_size=batch_size)), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class EmbeddingStore:
    """Disk cache keyed by the passage text, so a re-run only embeds new passages."""

    def __init__(self, path: Path, embedder: Embedder):
        self.path = Path(path)
        self.embedder = embedder
        self._vectors: dict[str, np.ndarray] = {}
        if self.path.exists():
            saved = np.load(self.path, allow_pickle=False)
            self._vectors = dict(zip(saved["keys"].tolist(), saved["vectors"]))

    def get(self, texts: list[str]) -> np.ndarray:
        keys = [text_key(t) for t in texts]
        missing = {k: t for k, t in zip(keys, texts) if k not in self._vectors}
        if missing:
            new = self.embedder.encode(list(missing.values()))
            self._vectors.update(zip(missing.keys(), new))
            self.save()
        return np.stack([self._vectors[k] for k in keys])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._vectors.keys()))
        np.savez(self.path, keys=keys, vectors=np.stack(list(self._vectors.values())).astype(np.float32))
