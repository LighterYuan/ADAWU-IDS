from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ADAWUWeightConfig:
    alpha: float = 0.60
    beta: float = 0.25
    gamma: float = 0.15
    lam: float = 0.10
    min_weight: float = 0.05
    eps: float = 1e-12


class ADAWUWeightUpdater:
    """
    Implements the weight update rule described in the manuscript:

    w_tilde = alpha * Perf + beta * (1 - MSDI) + gamma * T_t * (1 - C_t)
    then threshold by min_weight and normalize.
    """

    def __init__(self, config: ADAWUWeightConfig):
        self.config = config

    def update(
        self,
        perf: np.ndarray,
        msdi: float,
        confidence: float,
        time_index: int,
    ) -> Dict[str, np.ndarray | float]:
        perf = np.asarray(perf, dtype=np.float64)

        temporal_decay = float(np.exp(-self.config.lam * float(time_index)))
        reliability_term = self.config.beta * (1.0 - msdi) + self.config.gamma * temporal_decay * (1.0 - confidence)

        raw = self.config.alpha * perf + reliability_term
        clipped = np.maximum(self.config.min_weight, raw)
        norm = clipped / max(clipped.sum(), self.config.eps)

        return {
            "raw_weights": raw,
            "clipped_weights": clipped,
            "weights": norm,
            "temporal_decay": temporal_decay,
            "reliability_term": float(reliability_term),
        }
