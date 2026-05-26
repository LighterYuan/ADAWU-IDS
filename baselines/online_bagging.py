from __future__ import annotations

import numpy as np

from .stream_base import (
    StreamEnsembleBase,
    ensure_2d,
    estimator_predict_proba,
    partial_fit_or_fit,
)


class OnlineBaggingEnsemble(StreamEnsembleBase):
    """
    Oza-style online bagging with chunked updates.
    """

    def __init__(
        self,
        base_estimator,
        classes,
        n_estimators: int = 10,
        poisson_lambda: float = 1.0,
        random_state: int | None = None,
    ):
        super().__init__(base_estimator=base_estimator, classes=classes, random_state=random_state)
        self.n_estimators = int(n_estimators)
        self.poisson_lambda = float(poisson_lambda)
        self.estimators = [self._new_estimator() for _ in range(self.n_estimators)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = ensure_2d(X)
        probs = [estimator_predict_proba(est, X, self.classes) for est in self.estimators]
        return np.mean(probs, axis=0)

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        X = ensure_2d(X)
        y = np.asarray(y).reshape(-1)
        for est in self.estimators:
            k = self.random_state.poisson(lam=self.poisson_lambda, size=len(y)).astype(np.float64)
            if np.sum(k) == 0:
                k[self.random_state.randint(0, len(k))] = 1.0
            partial_fit_or_fit(est, X, y, classes=self.classes, sample_weight=k)
        self.is_initialized = True
        return self
