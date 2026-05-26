from __future__ import annotations

import numpy as np

from .stream_base import (
    StreamEnsembleBase,
    ensure_2d,
    estimator_predict_proba,
    partial_fit_or_fit,
)


class LeveragingBaggingEnsemble(StreamEnsembleBase):
    """
    Practical Leveraging Bagging approximation for chunk-level experiments.

    This version uses stronger Poisson resampling (lambda > 1) and optional
    hard-example emphasis, which is usually sufficient as a review-facing
    adaptive baseline in chunked intrusion-detection streams.
    """

    def __init__(
        self,
        base_estimator,
        classes,
        n_estimators: int = 10,
        poisson_lambda: float = 6.0,
        hard_example_boost: float = 2.0,
        random_state: int | None = None,
    ):
        super().__init__(base_estimator=base_estimator, classes=classes, random_state=random_state)
        self.n_estimators = int(n_estimators)
        self.poisson_lambda = float(poisson_lambda)
        self.hard_example_boost = float(hard_example_boost)
        self.estimators = [self._new_estimator() for _ in range(self.n_estimators)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = ensure_2d(X)
        probs = [estimator_predict_proba(est, X, self.classes) for est in self.estimators]
        return np.mean(probs, axis=0)

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        X = ensure_2d(X)
        y = np.asarray(y).reshape(-1)

        if self.is_initialized:
            ensemble_pred = self.predict(X)
            difficulty = 1.0 + self.hard_example_boost * (ensemble_pred != y).astype(np.float64)
        else:
            difficulty = np.ones(len(y), dtype=np.float64)

        for est in self.estimators:
            k = self.random_state.poisson(lam=self.poisson_lambda, size=len(y)).astype(np.float64)
            weights = np.maximum(0.0, k * difficulty)
            if np.sum(weights) == 0:
                weights[self.random_state.randint(0, len(weights))] = 1.0
            partial_fit_or_fit(est, X, y, classes=self.classes, sample_weight=weights)

        self.is_initialized = True
        return self
