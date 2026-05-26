from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from sklearn.base import clone


def ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3:
        n, t, f = X.shape
        return X.reshape(n, t * f)
    if X.ndim == 2:
        return X
    raise ValueError(f"Unsupported X ndim: {X.ndim}")


def predict_hard_labels_from_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return (proba >= 0.5).astype(int)
    if proba.shape[1] == 1:
        return (proba[:, 0] >= 0.5).astype(int)
    return np.argmax(proba, axis=1).astype(int)


def estimator_predict_proba(estimator, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    X = ensure_2d(X)
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        proba = np.asarray(proba, dtype=np.float64)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        if proba.shape[1] != len(classes):
            full = np.zeros((len(X), len(classes)), dtype=np.float64)
            seen = getattr(estimator, "classes_", classes)
            for j, cls in enumerate(seen):
                target_idx = int(np.where(classes == cls)[0][0])
                full[:, target_idx] = proba[:, j]
            row_sums = full.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0.0] = 1.0
            full /= row_sums
            return full
        return proba
    labels = np.asarray(estimator.predict(X))
    out = np.zeros((len(X), len(classes)), dtype=np.float64)
    for i, label in enumerate(labels):
        idx = int(np.where(classes == label)[0][0])
        out[i, idx] = 1.0
    return out


def partial_fit_or_fit(estimator, X: np.ndarray, y: np.ndarray, classes: np.ndarray, sample_weight=None):
    X = ensure_2d(X)
    y = np.asarray(y).reshape(-1)
    if hasattr(estimator, "partial_fit"):
        kwargs = {"classes": classes}
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight
        return estimator.partial_fit(X, y, **kwargs)
    if sample_weight is not None and "sample_weight" in estimator.fit.__code__.co_varnames:
        return estimator.fit(X, y, sample_weight=sample_weight)
    return estimator.fit(X, y)


class StreamEnsembleBase(ABC):
    def __init__(self, base_estimator, classes: Iterable[int], random_state: Optional[int] = None):
        self.base_estimator = base_estimator
        self.classes = np.asarray(sorted(list(classes)))
        self.random_state = np.random.RandomState(random_state)
        self.is_initialized = False

    def _new_estimator(self):
        return clone(self.base_estimator)

    @abstractmethod
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return predict_hard_labels_from_proba(self.predict_proba(X))
