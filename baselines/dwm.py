from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .stream_base import (
    StreamEnsembleBase,
    ensure_2d,
    estimator_predict_proba,
    partial_fit_or_fit,
)


@dataclass
class _Expert:
    estimator: object
    weight: float = 1.0


class DWMEnsemble(StreamEnsembleBase):
    """
    Lightweight Dynamic Weighted Majority for chunk-based streaming evaluation.

    Notes:
    - Predictions are produced before updating on the current chunk (prequential).
    - Weights are decayed for weak experts after each chunk.
    - New experts are spawned when the ensemble chunk error exceeds theta.
    """

    def __init__(
        self,
        base_estimator,
        classes,
        beta: float = 0.5,
        theta: float = 0.20,
        min_weight: float = 0.01,
        max_experts: int = 16,
        random_state: int | None = None,
    ):
        super().__init__(base_estimator=base_estimator, classes=classes, random_state=random_state)
        self.beta = float(beta)
        self.theta = float(theta)
        self.min_weight = float(min_weight)
        self.max_experts = int(max_experts)
        self.experts: list[_Expert] = []

    def _bootstrap(self, X: np.ndarray, y: np.ndarray):
        est = self._new_estimator()
        partial_fit_or_fit(est, X, y, classes=self.classes)
        self.experts = [_Expert(estimator=est, weight=1.0)]
        self.is_initialized = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = ensure_2d(X)
        if not self.experts:
            return np.full((len(X), len(self.classes)), 1.0 / len(self.classes))
        total = np.zeros((len(X), len(self.classes)), dtype=np.float64)
        weight_sum = 0.0
        for expert in self.experts:
            total += expert.weight * estimator_predict_proba(expert.estimator, X, self.classes)
            weight_sum += expert.weight
        if weight_sum <= 0:
            return np.full((len(X), len(self.classes)), 1.0 / len(self.classes))
        return total / weight_sum

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        X = ensure_2d(X)
        y = np.asarray(y).reshape(-1)
        if not self.is_initialized:
            self._bootstrap(X, y)
            return self

        ensemble_pred = self.predict(X)
        ensemble_error = float(np.mean(ensemble_pred != y))

        updated: list[_Expert] = []
        for expert in self.experts:
            pred = np.asarray(expert.estimator.predict(X)).reshape(-1)
            expert_error = float(np.mean(pred != y))
            if expert_error > 0.5:
                expert.weight *= self.beta
            partial_fit_or_fit(expert.estimator, X, y, classes=self.classes)
            if expert.weight >= self.min_weight:
                updated.append(expert)

        self.experts = updated or self.experts[:1]

        if ensemble_error > self.theta and len(self.experts) < self.max_experts:
            fresh = self._new_estimator()
            partial_fit_or_fit(fresh, X, y, classes=self.classes)
            self.experts.append(_Expert(estimator=fresh, weight=1.0))

        norm = sum(ex.weight for ex in self.experts)
        if norm > 0:
            for ex in self.experts:
                ex.weight /= norm
        return self
