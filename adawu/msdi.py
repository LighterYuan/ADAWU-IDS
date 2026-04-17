from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class MSDIConfig:
    eta: float = 0.5
    eps: float = 1e-8
    max_classes: int = 2


class MultiScaleDriftIndex:
    """
    A practical MSDI implementation for chunk-wise evaluation.

    It combines:
    1) feature-group distribution drift
    2) class distribution drift
    and normalizes the final score into [0, 1].
    """

    def __init__(self, config: MSDIConfig):
        self.config = config

    @staticmethod
    def _flatten_if_needed(X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            n, t, f = X.shape
            return X.reshape(n, t * f)
        if X.ndim == 2:
            return X
        raise ValueError(f"Unsupported X ndim: {X.ndim}")

    def _safe_hist_distance(self, a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
        lo = min(float(np.min(a)), float(np.min(b)))
        hi = max(float(np.max(a)), float(np.max(b)))
        if abs(hi - lo) < self.config.eps:
            return 0.0

        hist_a, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
        hist_b, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)

        hist_a = hist_a / (hist_a.sum() + self.config.eps)
        hist_b = hist_b / (hist_b.sum() + self.config.eps)

        # simple 1D Wasserstein-like cumulative distance approximation
        cdf_a = np.cumsum(hist_a)
        cdf_b = np.cumsum(hist_b)
        return float(np.mean(np.abs(cdf_a - cdf_b)))

    def _feature_group_score(
        self,
        X_ref: np.ndarray,
        X_cur: np.ndarray,
        group_slices: List[Tuple[int, int]] | None = None,
    ) -> float:
        X_ref = self._flatten_if_needed(X_ref)
        X_cur = self._flatten_if_needed(X_cur)

        d = X_ref.shape[1]
        if group_slices is None:
            step = max(1, d // 4)
            group_slices = []
            s = 0
            while s < d:
                e = min(d, s + step)
                group_slices.append((s, e))
                s = e

        scores = []
        for s, e in group_slices:
            block_ref = X_ref[:, s:e]
            block_cur = X_cur[:, s:e]
            feat_scores = []
            for j in range(block_ref.shape[1]):
                feat_scores.append(self._safe_hist_distance(block_ref[:, j], block_cur[:, j]))
            scores.append(float(np.mean(feat_scores)) if feat_scores else 0.0)

        return float(np.mean(scores)) if scores else 0.0

    def _class_distribution_score(self, y_ref: np.ndarray, y_cur: np.ndarray) -> float:
        max_classes = max(self.config.max_classes, int(max(np.max(y_ref), np.max(y_cur))) + 1)

        p_ref = np.bincount(y_ref.astype(int), minlength=max_classes).astype(np.float64)
        p_cur = np.bincount(y_cur.astype(int), minlength=max_classes).astype(np.float64)

        p_ref = p_ref / (p_ref.sum() + self.config.eps)
        p_cur = p_cur / (p_cur.sum() + self.config.eps)

        return float(np.mean(np.abs(np.cumsum(p_ref) - np.cumsum(p_cur))))

    def compute(
        self,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        X_cur: np.ndarray,
        y_cur: np.ndarray,
        group_slices: List[Tuple[int, int]] | None = None,
    ) -> Dict[str, float]:
        feat_score = self._feature_group_score(X_ref, X_cur, group_slices=group_slices)
        class_score = self._class_distribution_score(y_ref, y_cur)

        msdi = self.config.eta * feat_score + (1.0 - self.config.eta) * class_score
        msdi = float(np.clip(msdi, 0.0, 1.0))

        return {
            "msdi": msdi,
            "feature_score": float(np.clip(feat_score, 0.0, 1.0)),
            "class_score": float(np.clip(class_score, 0.0, 1.0)),
        }
