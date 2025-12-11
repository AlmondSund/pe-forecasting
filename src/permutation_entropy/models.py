"""Scikit-learn forecaster for permutation-entropy features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


@dataclass
class LogisticConfig:
    C: float = 1.0
    max_iter: int = 1000
    class_weight: str | dict | None = "balanced"
    solver: str = "lbfgs"


def train_logistic(
    X: Sequence[Sequence[float]], y: Sequence[int], cfg: LogisticConfig | None = None
) -> LogisticRegression:
    cfg = cfg or LogisticConfig()
    model = LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        class_weight=cfg.class_weight,
        solver=cfg.solver,
    )
    model.fit(X, y)
    return model


def evaluate(
    model: LogisticRegression, X: Sequence[Sequence[float]], y: Sequence[int]
) -> Tuple[float, float]:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs) if len(set(y)) > 1 else float("nan")
    return acc, auc


def predict_proba(
    model: LogisticRegression, X: Sequence[Sequence[float]]
) -> List[float]:
    return model.predict_proba(X)[:, 1].tolist()
