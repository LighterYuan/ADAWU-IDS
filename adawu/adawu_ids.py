from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from baselines.common import (
    build_train_corpus,
    create_base_model,
    evaluate_segment_by_chunks,
    fit_preprocessor,
    predict_labels,
    predict_proba,
    transform_with_preprocessor,
)
from adawu.msdi import MSDIConfig, MultiScaleDriftIndex
from adawu.weight_updater import ADAWUWeightConfig, ADAWUWeightUpdater


@dataclass
class ADAWUConfig:
    n_estimators: int = 5
    init_epochs: int = 5
    update_epochs_moderate: int = 2
    update_epochs_severe: int = 4
    batch_size: int = 256
    random_state: int = 42
    chunk_size: int = 5000

    alpha: float = 0.50
    beta: float = 0.35
    gamma: float = 0.15
    lam: float = 0.05
    min_weight: float = 0.02

    eta: float = 0.60

    mild_threshold: float = 0.20
    moderate_threshold: float = 0.35
    severe_threshold: float = 0.50

    reference_chunks: int = 5

    severe_boost_factor: float = 1.8
    moderate_boost_factor: float = 1.2


class ADAWUIDS:
    """
    Stronger ADAWU-IDS implementation for chunk-wise evaluation.
    """

    def __init__(self, config: ADAWUConfig):
        self.config = config
        self.learners: List = []
        self.weights = None

        self.imputer = None
        self.scaler = None
        self.shape_info = None
        self.num_classes = 2

        self.msdi = MultiScaleDriftIndex(MSDIConfig(eta=config.eta, max_classes=2))
        self.weight_updater = ADAWUWeightUpdater(
            ADAWUWeightConfig(
                alpha=config.alpha,
                beta=config.beta,
                gamma=config.gamma,
                lam=config.lam,
                min_weight=config.min_weight,
            )
        )

        self.reference_memory: List[Tuple[np.ndarray, np.ndarray]] = []
        self.time_index = 0

    def fit_initial(self, train_segments: List[str]) -> None:
        X_train_full, y_train_full = build_train_corpus(train_segments)
        self.imputer, self.scaler, self.shape_info, X_train_full = fit_preprocessor(X_train_full)
        self.num_classes = int(max(2, len(np.unique(y_train_full))))

        self.learners = []
        rng = np.random.default_rng(self.config.random_state)

        for i in range(self.config.n_estimators):
            idx = rng.choice(len(y_train_full), size=len(y_train_full), replace=True)
            X_boot = X_train_full[idx]
            y_boot = y_train_full[idx]

            model = create_base_model(
                X_boot,
                y_boot,
                epochs=self.config.init_epochs,
                batch_size=self.config.batch_size,
                random_state=self.config.random_state + i,
            )
            self.learners.append(model)

        self.weights = np.ones(len(self.learners), dtype=np.float64) / len(self.learners)

        tail_n = min(len(y_train_full), self.config.reference_chunks * self.config.chunk_size)
        self.reference_memory = [(X_train_full[-tail_n:], y_train_full[-tail_n:])]

    def _get_reference_state(self):
        X_ref = np.vstack([m[0] for m in self.reference_memory])
        y_ref = np.concatenate([m[1] for m in self.reference_memory])
        return X_ref, y_ref

    def _predict_each_proba(self, X_chunk_t: np.ndarray) -> List[np.ndarray]:
        return [predict_proba(model, X_chunk_t) for model in self.learners]

    def _predict_each_label(self, X_chunk_t: np.ndarray) -> List[np.ndarray]:
        return [predict_labels(model, X_chunk_t) for model in self.learners]

    def predict(self, X_chunk: np.ndarray) -> np.ndarray:
        X_chunk_t = transform_with_preprocessor(X_chunk, self.imputer, self.scaler, self.shape_info)
        probas = self._predict_each_proba(X_chunk_t)

        weighted = np.zeros_like(probas[0], dtype=np.float64)
        for w, p in zip(self.weights, probas):
            weighted += w * p

        return np.argmax(weighted, axis=1).astype(int)

    def _learner_perf(self, X_chunk_t: np.ndarray, y_chunk: np.ndarray) -> np.ndarray:
        perfs = []
        for learner in self.learners:
            pred = predict_labels(learner, X_chunk_t)
            acc = float(np.mean(pred == y_chunk))
            perfs.append(acc)
        return np.asarray(perfs, dtype=np.float64)

    def _confidence_from_predictions(self, X_chunk_t: np.ndarray) -> float:
        probas = self._predict_each_proba(X_chunk_t)
        weighted = np.zeros_like(probas[0], dtype=np.float64)
        for w, p in zip(self.weights, probas):
            weighted += w * p
        max_conf = np.max(weighted, axis=1)
        return float(np.mean(max_conf))

    def _severity(self, msdi: float) -> str:
        if msdi >= self.config.severe_threshold:
            return "severe"
        if msdi >= self.config.moderate_threshold:
            return "moderate"
        if msdi >= self.config.mild_threshold:
            return "mild"
        return "stable"

    def _incremental_update_all(self, X_chunk_t: np.ndarray, y_chunk: np.ndarray, epochs: int):
        for learner in self.learners:
            for _ in range(max(1, epochs)):
                learner.adaptive_update(X_chunk_t, y_chunk)

    def _replace_weakest(self, X_chunk_t: np.ndarray, y_chunk: np.ndarray, learner_perf: np.ndarray) -> int:
        weakest = int(np.argmin(learner_perf))
        new_model = create_base_model(
            X_chunk_t,
            y_chunk,
            epochs=self.config.update_epochs_severe,
            batch_size=self.config.batch_size,
            random_state=self.config.random_state + 1000 + self.time_index + weakest,
        )
        self.learners[weakest] = new_model
        return weakest

    def _boost_weights(self, weakest_idx: int | None, severity: str):
        if self.weights is None:
            return

        w = np.asarray(self.weights, dtype=np.float64).copy()

        if weakest_idx is not None and 0 <= weakest_idx < len(w):
            if severity == "severe":
                w[weakest_idx] *= self.config.severe_boost_factor
            elif severity == "moderate":
                w[weakest_idx] *= self.config.moderate_boost_factor

        w = np.maximum(w, self.config.min_weight)
        w = w / max(w.sum(), 1e-12)
        self.weights = w

    def _reweight_from_current_state(
        self,
        learner_perf: np.ndarray,
        msdi_value: float,
        confidence: float,
    ):
        weight_info = self.weight_updater.update(
            perf=learner_perf,
            msdi=msdi_value,
            confidence=confidence,
            time_index=self.time_index,
        )
        self.weights = np.asarray(weight_info["weights"], dtype=np.float64)
        return weight_info

    def update(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> Dict[str, float]:
        X_chunk_t = transform_with_preprocessor(X_chunk, self.imputer, self.scaler, self.shape_info)

        X_ref, y_ref = self._get_reference_state()
        drift_info = self.msdi.compute(X_ref, y_ref, X_chunk_t, y_chunk)
        msdi_value = drift_info["msdi"]

        learner_perf_before = self._learner_perf(X_chunk_t, y_chunk)
        confidence_before = self._confidence_from_predictions(X_chunk_t)

        weight_info = self._reweight_from_current_state(
            learner_perf=learner_perf_before,
            msdi_value=msdi_value,
            confidence=confidence_before,
        )

        severity = self._severity(msdi_value)
        weakest_idx = None

        if severity == "stable":
            response = "none"

        elif severity == "mild":
            response = "reweight_only"

        elif severity == "moderate":
            self._incremental_update_all(
                X_chunk_t,
                y_chunk,
                epochs=self.config.update_epochs_moderate,
            )

            learner_perf_after = self._learner_perf(X_chunk_t, y_chunk)
            confidence_after = self._confidence_from_predictions(X_chunk_t)

            weight_info = self._reweight_from_current_state(
                learner_perf=learner_perf_after,
                msdi_value=msdi_value,
                confidence=confidence_after,
            )
            self._boost_weights(weakest_idx=None, severity="moderate")
            response = "strong_incremental_update"

        else:
            weakest_idx = self._replace_weakest(X_chunk_t, y_chunk, learner_perf_before)

            self._incremental_update_all(
                X_chunk_t,
                y_chunk,
                epochs=self.config.update_epochs_severe,
            )

            learner_perf_after = self._learner_perf(X_chunk_t, y_chunk)
            confidence_after = self._confidence_from_predictions(X_chunk_t)

            weight_info = self._reweight_from_current_state(
                learner_perf=learner_perf_after,
                msdi_value=msdi_value,
                confidence=confidence_after,
            )
            self._boost_weights(weakest_idx=weakest_idx, severity="severe")
            response = "replace_and_strong_update"

        self.reference_memory.append((X_chunk_t, y_chunk))
        self.reference_memory = self.reference_memory[-self.config.reference_chunks:]
        self.time_index += 1

        learner_perf_final = self._learner_perf(X_chunk_t, y_chunk)

        return {
            "msdi": float(msdi_value),
            "feature_score": float(drift_info["feature_score"]),
            "class_score": float(drift_info["class_score"]),
            "confidence_before": float(confidence_before),
            "severity": severity,
            "response": response,
            "temporal_decay": float(weight_info["temporal_decay"]),
            "reliability_term": float(weight_info["reliability_term"]),
            "num_learners": int(len(self.learners)),
            "weakest_replaced": int(weakest_idx) if weakest_idx is not None else -1,
            "mean_perf_before": float(np.mean(learner_perf_before)),
            "mean_perf_after": float(np.mean(learner_perf_final)),
            "w0": float(self.weights[0]) if len(self.weights) > 0 else np.nan,
            "w1": float(self.weights[1]) if len(self.weights) > 1 else np.nan,
            "w2": float(self.weights[2]) if len(self.weights) > 2 else np.nan,
            "w3": float(self.weights[3]) if len(self.weights) > 3 else np.nan,
            "w4": float(self.weights[4]) if len(self.weights) > 4 else np.nan,
        }

    def evaluate_stream(self, test_segments: List[str], load_xy_fn):
        y_true_all, y_pred_all, seg_all = [], [], []
        trace = []
        global_chunk_offset = 0

        for seg in test_segments:
            X_seg, y_seg = load_xy_fn(seg)
            seg_name = seg.replace(".pcap_ISCX", "")

            y_true_seg, y_pred_seg, seg_arr, chunk_trace, global_chunk_offset = evaluate_segment_by_chunks(
                model=self,
                X_seg=X_seg,
                y_seg=y_seg,
                segment_name=seg_name,
                chunk_size=self.config.chunk_size,
                global_chunk_offset=global_chunk_offset,
            )

            y_true_all.append(y_true_seg)
            y_pred_all.append(y_pred_seg)
            seg_all.append(seg_arr)
            trace.extend(chunk_trace)

        return (
            np.concatenate(y_true_all),
            np.concatenate(y_pred_all),
            np.concatenate(seg_all),
            trace,
        )
