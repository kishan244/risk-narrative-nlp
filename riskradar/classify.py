"""The trained risk classifier: multinomial logistic regression on embeddings.

Only the weights are stored (a small .npz file), so predicting needs NumPy
alone: probabilities = softmax(embedding @ W.T + b).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from riskradar.config import MODEL_DIR

MODEL_PATH = MODEL_DIR / "risk_classifier.npz"


@dataclass
class RiskClassifier:
    classes: list[str]
    coef: np.ndarray        # (n_classes, n_features)
    intercept: np.ndarray   # (n_classes,)
    embedding_model: str

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        logits = embeddings @ self.coef.T + self.intercept
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, embeddings: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Return (label, confidence) for every row."""
        proba = self.predict_proba(embeddings)
        return [self.classes[i] for i in proba.argmax(axis=1)], proba.max(axis=1)

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, classes=np.array(self.classes), coef=self.coef.astype(np.float32),
            intercept=self.intercept.astype(np.float32), embedding_model=np.array(self.embedding_model),
        )

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> RiskClassifier:
        saved = np.load(path, allow_pickle=False)
        return cls(saved["classes"].tolist(), saved["coef"], saved["intercept"], str(saved["embedding_model"]))
