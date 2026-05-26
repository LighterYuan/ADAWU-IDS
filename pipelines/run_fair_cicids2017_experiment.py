#!/usr/bin/env python3
"""
Fair, reproducible CICIDS2017 chronological streaming experiment for ADAWU-IDS.

This script fixes the main protocol problems found in the earlier project scripts:
1. No train/test segment overlap.
2. One shared preprocessing pipeline fitted on the initial training data only.
3. One shared chronological test stream and chunk size for all methods.
4. Prediction is always made before the current chunk label is used for adaptation.
5. Adaptive baselines and ADAWU-IDS are evaluated under the same stream.
6. All per-chunk predictions, metrics, weights, drift scores, and protocol metadata are saved.

Expected processed input format:
  <data_dir>/<segment>_X.npy
  <data_dir>/<segment>_y.npy
where segment names match CICIDS2017 processed segment stems.

Default protocol:
  train: Tuesday + Wednesday
  validation: Thursday WebAttacks (reserved, not reported as test by default)
  test: Thursday Infiltration + Friday Morning + Friday PortScan

Example:
  python pipelines/run_fair_cicids2017_experiment.py \
    --data-dir datasets/processed \
    --output-dir results/fair_cicids2017 \
    --seeds 42 52 62 72 82 \
    --chunk-size 5000
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import wasserstein_distance as scipy_wasserstein_distance
except Exception:  # pragma: no cover
    scipy_wasserstein_distance = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if (PROJECT_ROOT / "baselines").exists() and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from baselines import DWMEnsemble, LeveragingBaggingEnsemble, OnlineBaggingEnsemble
except Exception:
    DWMEnsemble = None
    LeveragingBaggingEnsemble = None
    OnlineBaggingEnsemble = None

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, LSTM
    from tensorflow.keras.optimizers import Adam
except Exception:
    tf = None


DEFAULT_TRAIN_SEGMENTS = [
    "Tuesday-WorkingHours.pcap_ISCX",
    "Wednesday-workingHours.pcap_ISCX",
]

DEFAULT_VAL_SEGMENTS = [
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX",
]

DEFAULT_TEST_SEGMENTS = [
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX",
    "Friday-WorkingHours-Morning.pcap_ISCX",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX",
]

ALL_CICIDS_SEGMENTS_IN_ORDER = [
    "Tuesday-WorkingHours.pcap_ISCX",
    "Wednesday-workingHours.pcap_ISCX",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX",
    "Friday-WorkingHours-Morning.pcap_ISCX",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX",
]


@dataclass
class ADAWUConfig:
    n_models: int = 3
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    adaptation_rate: float = 1e-4
    dropout_rate: float = 0.2
    lstm_units_1: int = 64
    lstm_units_2: int = 32
    dense_units: int = 64
    alpha: float = 0.60
    beta: float = 0.25
    gamma: float = 0.15
    lambda_decay: float = 0.10
    min_weight: float = 0.05
    eta_feature: float = 0.70
    mild_threshold: float = 0.30
    moderate_threshold: float = 0.50
    severe_threshold: float = 0.70
    reference_max_samples: int = 50000
    update_on_mild: bool = False
    update_on_moderate: bool = True
    update_on_severe: bool = True


@dataclass
class BaselineConfig:
    sgd_alpha: float = 1e-4
    sgd_eta0: float = 0.01
    dwm_beta: float = 0.5
    dwm_theta: float = 0.20
    dwm_min_weight: float = 0.01
    dwm_max_experts: int = 16
    bagging_estimators: int = 10
    online_bagging_lambda: float = 1.0
    leveraging_bagging_lambda: float = 6.0
    leveraging_hard_boost: float = 2.0


# ---------------------------
# Basic utilities
# ---------------------------


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)


def display_segment_name(segment: str) -> str:
    return segment.replace(".pcap_ISCX", "")


def load_xy(data_dir: Path, segment: str) -> Tuple[np.ndarray, np.ndarray]:
    x_path = data_dir / f"{segment}_X.npy"
    y_path = data_dir / f"{segment}_y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing processed pair for segment '{segment}'. Expected:\n"
            f"  {x_path}\n  {y_path}"
        )
    X = np.asarray(np.load(x_path, allow_pickle=True), dtype=np.float32)
    y = np.asarray(np.load(y_path, allow_pickle=True)).reshape(-1).astype(int)
    if len(X) != len(y):
        raise ValueError(f"Length mismatch in {segment}: X={len(X)}, y={len(y)}")
    return X, y


def ensure_3d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        return X[:, None, :]
    if X.ndim == 3:
        return X
    raise ValueError(f"Unsupported X dimension: {X.ndim}")


def flatten_3d_to_2d(X: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    X = ensure_3d(X)
    n, t, f = X.shape
    return X.reshape(n, t * f), (t, f)


def restore_2d_to_3d(X2: np.ndarray, shape_info: Tuple[int, int]) -> np.ndarray:
    t, f = shape_info
    return np.asarray(X2, dtype=np.float32).reshape(len(X2), t, f)


def stack_segments(data_dir: Path, segments: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    X_parts, y_parts = [], []
    for seg in segments:
        X, y = load_xy(data_dir, seg)
        X_parts.append(ensure_3d(X))
        y_parts.append(y)
    return np.vstack(X_parts), np.concatenate(y_parts)


def iter_segment_chunks(
    data_dir: Path,
    segments: Sequence[str],
    chunk_size: int,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    shape_info: Tuple[int, int],
):
    global_chunk = 0
    for seg in segments:
        X_raw, y = load_xy(data_dir, seg)
        X_raw = ensure_3d(X_raw)
        X2, _ = flatten_3d_to_2d(X_raw)
        X2 = scaler.transform(imputer.transform(X2))
        X3 = restore_2d_to_3d(X2, shape_info)
        seg_name = display_segment_name(seg)
        for start in range(0, len(y), chunk_size):
            end = min(start + chunk_size, len(y))
            yield {
                "chunk_id": global_chunk,
                "segment": seg_name,
                "segment_stem": seg,
                "start": int(start),
                "end": int(end),
                "X2": X2[start:end],
                "X3": X3[start:end],
                "y": y[start:end],
            }
            global_chunk += 1


def assert_no_overlap(train_segments: Sequence[str], val_segments: Sequence[str], test_segments: Sequence[str]) -> None:
    train = set(train_segments)
    val = set(val_segments)
    test = set(test_segments)
    if train & test:
        raise ValueError(f"Train/test segment overlap is not allowed: {sorted(train & test)}")
    if val & test:
        raise ValueError(f"Validation/test segment overlap is not allowed: {sorted(val & test)}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    labels = sorted(np.unique(np.concatenate([y_true.reshape(-1), y_pred.reshape(-1)])).tolist())
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if set(labels).issubset({0, 1}) and 1 in labels:
        out.update(
            {
                "attack_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
                "attack_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
                "attack_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            }
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        out.update(
            {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
            }
        )
    return out


def safe_json_dump(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------------------------
# Drift score: Wasserstein MSDI
# ---------------------------


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    if scipy_wasserstein_distance is not None:
        return float(scipy_wasserstein_distance(a, b))
    qs = np.linspace(0.0, 1.0, 101)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def feature_groups(feature_dim: int, n_groups: int = 5) -> List[np.ndarray]:
    idx = np.arange(feature_dim)
    return [g.astype(int) for g in np.array_split(idx, min(n_groups, feature_dim)) if len(g) > 0]


def msdi_score(
    X_ref_3d: np.ndarray,
    X_cur_3d: np.ndarray,
    y_ref: Optional[np.ndarray],
    y_cur: Optional[np.ndarray],
    classes: np.ndarray,
    eta_feature: float = 0.70,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Compute a normalized feature-group + class-conditional drift score.

    Labels are optional. In this script the score is computed after chunk prediction,
    so label-dependent components cannot affect the already-recorded prediction.
    """
    X_ref_3d = ensure_3d(X_ref_3d)
    X_cur_3d = ensure_3d(X_cur_3d)

    # Feature-level view: compare original feature dimensions across all time steps.
    ref_ft = X_ref_3d.reshape(-1, X_ref_3d.shape[-1])
    cur_ft = X_cur_3d.reshape(-1, X_cur_3d.shape[-1])
    groups = feature_groups(ref_ft.shape[1], n_groups=5)
    group_scores = []
    for group in groups:
        vals = []
        for f in group:
            std_ref = float(np.std(ref_ft[:, f])) + eps
            vals.append(min(1.0, wasserstein_1d(ref_ft[:, f], cur_ft[:, f]) / std_ref))
        group_scores.append(float(np.mean(vals)) if vals else 0.0)
    s_feature = float(np.mean(group_scores)) if group_scores else 0.0

    # Class-conditional view: compare sequence-level class means.
    s_class_values = []
    if y_ref is not None and y_cur is not None:
        ref_2d = X_ref_3d.reshape(len(X_ref_3d), -1)
        cur_2d = X_cur_3d.reshape(len(X_cur_3d), -1)
        y_ref = np.asarray(y_ref).reshape(-1)
        y_cur = np.asarray(y_cur).reshape(-1)
        for c in classes:
            ref_mask = y_ref == c
            cur_mask = y_cur == c
            if np.sum(ref_mask) < 2 or np.sum(cur_mask) < 2:
                continue
            ref_mean = np.mean(ref_2d[ref_mask], axis=0)
            cur_mean = np.mean(cur_2d[cur_mask], axis=0)
            score = np.linalg.norm(cur_mean - ref_mean) / (np.linalg.norm(ref_mean) + eps)
            s_class_values.append(min(1.0, float(score)))
    s_class = float(np.mean(s_class_values)) if s_class_values else 0.0

    total = eta_feature * s_feature + (1.0 - eta_feature) * s_class
    total = float(np.clip(total, 0.0, 1.0))
    return {"msdi": total, "feature_drift": s_feature, "class_drift": s_class}


def severity_from_msdi(msdi: float, cfg: ADAWUConfig) -> str:
    if msdi >= cfg.severe_threshold:
        return "severe"
    if msdi >= cfg.moderate_threshold:
        return "moderate"
    if msdi >= cfg.mild_threshold:
        return "mild"
    return "none"


class ReferenceBuffer:
    def __init__(self, X_init_3d: np.ndarray, y_init: np.ndarray, max_samples: int):
        self.max_samples = int(max_samples)
        self.X = ensure_3d(X_init_3d)[-self.max_samples :].copy()
        self.y = np.asarray(y_init).reshape(-1)[-self.max_samples :].copy()

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X, self.y

    def update(self, X_new_3d: np.ndarray, y_new: np.ndarray) -> None:
        X_new_3d = ensure_3d(X_new_3d)
        y_new = np.asarray(y_new).reshape(-1)
        self.X = np.vstack([self.X, X_new_3d])[-self.max_samples :]
        self.y = np.concatenate([self.y, y_new])[-self.max_samples :]


# ---------------------------
# LSTM methods
# ---------------------------


def require_tensorflow(method_name: str) -> None:
    if tf is None:
        raise RuntimeError(f"TensorFlow is required for {method_name}, but it is not installed/importable.")


def build_lstm_model(input_shape: Tuple[int, int], cfg: ADAWUConfig, seed: int):
    require_tensorflow("LSTM-based methods")
    set_global_seed(seed)
    model = Sequential(
        [
            LSTM(cfg.lstm_units_1, return_sequences=True, input_shape=input_shape, dropout=cfg.dropout_rate),
            BatchNormalization(),
            LSTM(cfg.lstm_units_2, return_sequences=False, dropout=cfg.dropout_rate),
            BatchNormalization(),
            Dense(cfg.dense_units, activation="relu"),
            Dropout(cfg.dropout_rate),
            BatchNormalization(),
            Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fit_lstm(model, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray], y_val: Optional[np.ndarray], cfg: ADAWUConfig):
    callbacks = []
    if X_val is not None and y_val is not None and len(y_val) > 0:
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        ]
        val_data = (X_val, y_val)
    else:
        val_data = None
    model.fit(
        X_train,
        y_train,
        validation_data=val_data,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    return model


def predict_proba_lstm(model, X3: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict(X3, verbose=0), dtype=np.float64)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    if proba.shape[1] == 1:
        proba = np.column_stack([1.0 - proba[:, 0], proba[:, 0]])
    return proba


def adaptive_lstm_update(model, X3: np.ndarray, y: np.ndarray, cfg: ADAWUConfig, epochs: int = 1) -> None:
    model.compile(
        optimizer=Adam(learning_rate=cfg.adaptation_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(X3, y, epochs=epochs, batch_size=cfg.batch_size, verbose=0)


def bootstrap_sample(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=len(y))
    return X[idx], y[idx]


def update_adawu_weights(
    weights_before: np.ndarray,
    perf_scores: Sequence[float],
    msdi: float,
    confidence: float,
    chunk_id: int,
    cfg: ADAWUConfig,
    weighting_enabled: bool = True,
) -> np.ndarray:
    if not weighting_enabled:
        return np.ones_like(weights_before) / len(weights_before)
    perf = np.asarray(perf_scores, dtype=np.float64)
    temporal_decay = math.exp(-cfg.lambda_decay * max(0, chunk_id))
    raw = cfg.alpha * perf + cfg.beta * (1.0 - msdi) + cfg.gamma * temporal_decay * (1.0 - confidence)
    raw = np.maximum(cfg.min_weight, raw)
    if np.sum(raw) <= 0:
        return np.ones_like(weights_before) / len(weights_before)
    return raw / np.sum(raw)


def run_static_lstm(
    seed: int,
    cfg: ADAWUConfig,
    data_dir: Path,
    test_segments: Sequence[str],
    chunk_size: int,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    shape_info: Tuple[int, int],
    X_train_3d: np.ndarray,
    y_train: np.ndarray,
    X_val_3d: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
) -> Dict[str, object]:
    model = build_lstm_model(X_train_3d.shape[1:], cfg, seed)
    fit_lstm(model, X_train_3d, y_train, X_val_3d, y_val, cfg)

    records, y_true_all, y_pred_all, chunk_all, seg_all = [], [], [], [], []
    for chunk in iter_segment_chunks(data_dir, test_segments, chunk_size, imputer, scaler, shape_info):
        t0 = time.perf_counter()
        proba = predict_proba_lstm(model, chunk["X3"])
        pred = np.argmax(proba, axis=1).astype(int)
        latency = time.perf_counter() - t0
        metrics = compute_metrics(chunk["y"], pred)
        records.append(
            {
                "seed": seed,
                "method": "static_lstm",
                "chunk_id": chunk["chunk_id"],
                "segment": chunk["segment"],
                "start": chunk["start"],
                "end": chunk["end"],
                "n_samples": int(len(chunk["y"])),
                "latency_predict_s": latency,
                **metrics,
            }
        )
        y_true_all.append(chunk["y"])
        y_pred_all.append(pred)
        chunk_all.extend([chunk["chunk_id"]] * len(chunk["y"]))
        seg_all.extend([chunk["segment"]] * len(chunk["y"]))
    return pack_method_result("static_lstm", seed, records, y_true_all, y_pred_all, chunk_all, seg_all)


def run_adawu_variant(
    seed: int,
    cfg: ADAWUConfig,
    variant_name: str,
    data_dir: Path,
    test_segments: Sequence[str],
    chunk_size: int,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    shape_info: Tuple[int, int],
    X_train_3d: np.ndarray,
    y_train: np.ndarray,
    X_val_3d: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    classes: np.ndarray,
    use_msdi: bool = True,
    use_weighting: bool = True,
    use_hierarchy: bool = True,
) -> Dict[str, object]:
    require_tensorflow("ADAWU-IDS")
    models = []
    base_seeds = [seed + 101 * (i + 1) for i in range(cfg.n_models)]
    for model_seed in base_seeds:
        X_boot, y_boot = bootstrap_sample(X_train_3d, y_train, model_seed)
        model = build_lstm_model(X_train_3d.shape[1:], cfg, model_seed)
        fit_lstm(model, X_boot, y_boot, X_val_3d, y_val, cfg)
        models.append(model)

    weights = np.ones(cfg.n_models, dtype=np.float64) / cfg.n_models
    ref_buffer = ReferenceBuffer(X_train_3d, y_train, max_samples=cfg.reference_max_samples)
    records, y_true_all, y_pred_all, chunk_all, seg_all = [], [], [], [], []

    for chunk in iter_segment_chunks(data_dir, test_segments, chunk_size, imputer, scaler, shape_info):
        weights_before = weights.copy()

        t0 = time.perf_counter()
        per_model_proba = [predict_proba_lstm(model, chunk["X3"]) for model in models]
        ensemble_proba = np.zeros_like(per_model_proba[0], dtype=np.float64)
        for w, p in zip(weights_before, per_model_proba):
            ensemble_proba += float(w) * p
        pred = np.argmax(ensemble_proba, axis=1).astype(int)
        predict_latency = time.perf_counter() - t0

        # Prediction has now been recorded conceptually. The following uses current labels
        # only for adaptation that affects later chunks.
        X_ref_3d, y_ref = ref_buffer.get()
        if use_msdi:
            drift = msdi_score(
                X_ref_3d=X_ref_3d,
                X_cur_3d=chunk["X3"],
                y_ref=y_ref,
                y_cur=chunk["y"],
                classes=classes,
                eta_feature=cfg.eta_feature,
            )
        else:
            drift = {"msdi": 0.0, "feature_drift": 0.0, "class_drift": 0.0}
        severity = severity_from_msdi(float(drift["msdi"]), cfg) if use_msdi else "none"

        per_model_f1 = []
        for p in per_model_proba:
            p_label = np.argmax(p, axis=1).astype(int)
            per_model_f1.append(float(f1_score(chunk["y"], p_label, average="weighted", zero_division=0)))
        confidence = float(np.mean(np.max(ensemble_proba, axis=1)))
        weights_after = update_adawu_weights(
            weights_before=weights_before,
            perf_scores=per_model_f1,
            msdi=float(drift["msdi"]),
            confidence=confidence,
            chunk_id=int(chunk["chunk_id"]),
            cfg=cfg,
            weighting_enabled=use_weighting,
        )

        retrain_event = False
        adapt_latency = 0.0
        if use_hierarchy:
            should_update = (
                (severity == "mild" and cfg.update_on_mild)
                or (severity == "moderate" and cfg.update_on_moderate)
                or (severity == "severe" and cfg.update_on_severe)
            )
            if should_update:
                t1 = time.perf_counter()
                if severity == "moderate":
                    # Update the currently most reliable model.
                    idx = int(np.argmax(weights_after))
                    adaptive_lstm_update(models[idx], chunk["X3"], chunk["y"], cfg, epochs=1)
                elif severity == "severe":
                    # Stronger response: update all ensemble members.
                    for model in models:
                        adaptive_lstm_update(model, chunk["X3"], chunk["y"], cfg, epochs=1)
                else:
                    idx = int(np.argmax(weights_after))
                    adaptive_lstm_update(models[idx], chunk["X3"], chunk["y"], cfg, epochs=1)
                adapt_latency = time.perf_counter() - t1
                retrain_event = True

        metrics = compute_metrics(chunk["y"], pred)
        records.append(
            {
                "seed": seed,
                "method": variant_name,
                "chunk_id": chunk["chunk_id"],
                "segment": chunk["segment"],
                "start": chunk["start"],
                "end": chunk["end"],
                "n_samples": int(len(chunk["y"])),
                "latency_predict_s": predict_latency,
                "latency_adapt_s": adapt_latency,
                "msdi": float(drift["msdi"]),
                "feature_drift": float(drift["feature_drift"]),
                "class_drift": float(drift["class_drift"]),
                "severity": severity,
                "retrain_event": bool(retrain_event),
                "confidence": confidence,
                "weights_before": [float(x) for x in weights_before],
                "weights_after": [float(x) for x in weights_after],
                "per_model_weighted_f1": [float(x) for x in per_model_f1],
                **metrics,
            }
        )
        y_true_all.append(chunk["y"])
        y_pred_all.append(pred)
        chunk_all.extend([chunk["chunk_id"]] * len(chunk["y"]))
        seg_all.extend([chunk["segment"]] * len(chunk["y"]))

        weights = weights_after
        ref_buffer.update(chunk["X3"], chunk["y"])

    return pack_method_result(variant_name, seed, records, y_true_all, y_pred_all, chunk_all, seg_all)


# ---------------------------
# SGD streaming baselines
# ---------------------------


def build_sgd(seed: int, cfg: BaselineConfig) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        alpha=cfg.sgd_alpha,
        learning_rate="optimal",
        eta0=cfg.sgd_eta0,
        random_state=seed,
    )


def make_project_adaptive_baselines(seed: int, classes: np.ndarray, cfg: BaselineConfig):
    if DWMEnsemble is None or OnlineBaggingEnsemble is None or LeveragingBaggingEnsemble is None:
        raise RuntimeError("Project baseline classes are unavailable. Run this script from the project root or keep baselines/ on PYTHONPATH.")
    base = build_sgd(seed, cfg)
    return {
        "dwm": DWMEnsemble(
            base_estimator=clone(base),
            classes=classes,
            beta=cfg.dwm_beta,
            theta=cfg.dwm_theta,
            min_weight=cfg.dwm_min_weight,
            max_experts=cfg.dwm_max_experts,
            random_state=seed,
        ),
        "online_bagging": OnlineBaggingEnsemble(
            base_estimator=clone(base),
            classes=classes,
            n_estimators=cfg.bagging_estimators,
            poisson_lambda=cfg.online_bagging_lambda,
            random_state=seed,
        ),
        "leveraging_bagging": LeveragingBaggingEnsemble(
            base_estimator=clone(base),
            classes=classes,
            n_estimators=cfg.bagging_estimators,
            poisson_lambda=cfg.leveraging_bagging_lambda,
            hard_example_boost=cfg.leveraging_hard_boost,
            random_state=seed,
        ),
    }


def run_sgd_streaming_methods(
    seed: int,
    baseline_cfg: BaselineConfig,
    data_dir: Path,
    test_segments: Sequence[str],
    chunk_size: int,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    shape_info: Tuple[int, int],
    X_train_2d: np.ndarray,
    y_train: np.ndarray,
    classes: np.ndarray,
    methods: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    models = {}
    if "static_sgd" in methods:
        static = build_sgd(seed, baseline_cfg)
        static.partial_fit(X_train_2d, y_train, classes=classes)
        models["static_sgd"] = static

    adaptive_names = {"dwm", "online_bagging", "leveraging_bagging"} & set(methods)
    if adaptive_names:
        all_adaptive = make_project_adaptive_baselines(seed, classes, baseline_cfg)
        for name in sorted(adaptive_names):
            models[name] = all_adaptive[name]
            models[name].partial_fit(X_train_2d, y_train)

    buffers = {
        name: {"records": [], "y_true": [], "y_pred": [], "chunks": [], "segments": []}
        for name in models
    }

    for chunk in iter_segment_chunks(data_dir, test_segments, chunk_size, imputer, scaler, shape_info):
        X2, y = chunk["X2"], chunk["y"]
        for name, model in models.items():
            t0 = time.perf_counter()
            pred = np.asarray(model.predict(X2)).reshape(-1).astype(int)
            predict_latency = time.perf_counter() - t0
            metrics = compute_metrics(y, pred)

            adapt_latency = 0.0
            if name != "static_sgd":
                t1 = time.perf_counter()
                model.partial_fit(X2, y)
                adapt_latency = time.perf_counter() - t1

            buffers[name]["records"].append(
                {
                    "seed": seed,
                    "method": name,
                    "chunk_id": chunk["chunk_id"],
                    "segment": chunk["segment"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "n_samples": int(len(y)),
                    "latency_predict_s": predict_latency,
                    "latency_adapt_s": adapt_latency,
                    **metrics,
                }
            )
            buffers[name]["y_true"].append(y)
            buffers[name]["y_pred"].append(pred)
            buffers[name]["chunks"].extend([chunk["chunk_id"]] * len(y))
            buffers[name]["segments"].extend([chunk["segment"]] * len(y))

    return {
        name: pack_method_result(
            name,
            seed,
            buf["records"],
            buf["y_true"],
            buf["y_pred"],
            buf["chunks"],
            buf["segments"],
        )
        for name, buf in buffers.items()
    }


# ---------------------------
# Result packing and summaries
# ---------------------------


def pack_method_result(
    method: str,
    seed: int,
    records: List[Dict[str, object]],
    y_true_all: Sequence[np.ndarray],
    y_pred_all: Sequence[np.ndarray],
    chunk_all: Sequence[int],
    seg_all: Sequence[str],
) -> Dict[str, object]:
    y_true = np.concatenate(y_true_all) if y_true_all else np.asarray([], dtype=int)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.asarray([], dtype=int)
    overall = compute_metrics(y_true, y_pred) if len(y_true) else {}
    return {
        "method": method,
        "seed": seed,
        "overall": overall,
        "records": records,
        "arrays": {
            "y_true": y_true,
            "y_pred": y_pred,
            "chunk_ids": np.asarray(chunk_all, dtype=int),
            "segments": np.asarray(seg_all, dtype=object),
        },
    }


def ci95(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / math.sqrt(len(arr)))


def aggregate_seed_results(all_results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for result in all_results:
        grouped.setdefault(str(result["method"]), []).append(result)

    rows = []
    for method, items in grouped.items():
        metrics = sorted(items[0]["overall"].keys()) if items and items[0]["overall"] else []
        row = {"method": method, "n_seeds": len(items)}
        for metric in metrics:
            vals = [float(item["overall"].get(metric, np.nan)) for item in items]
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_ci95"] = ci95(vals)
        rows.append(row)
    rows.sort(key=lambda r: str(r["method"]))
    return rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_result(result: Dict[str, object], output_dir: Path) -> None:
    method = str(result["method"])
    seed = int(result["seed"])
    trace_path = output_dir / "traces" / f"{method}_seed{seed}.json"
    case_path = output_dir / "cases" / f"{method}_seed{seed}.npz"
    summary_path = output_dir / "summaries" / f"{method}_seed{seed}.json"

    safe_json_dump({"method": method, "seed": seed, "records": result["records"]}, trace_path)
    safe_json_dump({"method": method, "seed": seed, "overall": result["overall"]}, summary_path)

    case_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = result["arrays"]
    np.savez_compressed(
        case_path,
        y_true=arrays["y_true"],
        y_pred=arrays["y_pred"],
        chunk_ids=arrays["chunk_ids"],
        segments=arrays["segments"],
    )


def parse_segments(text: Optional[str], default: Sequence[str]) -> List[str]:
    if text is None or text.strip() == "":
        return list(default)
    return [x.strip() for x in text.split(",") if x.strip()]


def build_preprocessor(data_dir: Path, train_segments: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int], SimpleImputer, StandardScaler]:
    X_train_raw, y_train = stack_segments(data_dir, train_segments)
    X_train_2d_raw, shape_info = flatten_3d_to_2d(X_train_raw)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(imputer.fit_transform(X_train_2d_raw))
    X_train_3d = restore_2d_to_3d(X_train_2d, shape_info)
    return X_train_2d, X_train_3d, y_train, shape_info, imputer, scaler


def prepare_validation(
    data_dir: Path,
    val_segments: Sequence[str],
    imputer: SimpleImputer,
    scaler: StandardScaler,
    shape_info: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not val_segments:
        return None, None
    X_val_raw, y_val = stack_segments(data_dir, val_segments)
    X_val_2d, _ = flatten_3d_to_2d(X_val_raw)
    X_val_2d = scaler.transform(imputer.transform(X_val_2d))
    X_val_3d = restore_2d_to_3d(X_val_2d, shape_info)
    return X_val_3d, y_val


def run_one_seed(
    seed: int,
    args,
    train_segments: Sequence[str],
    val_segments: Sequence[str],
    test_segments: Sequence[str],
    adawu_cfg: ADAWUConfig,
    baseline_cfg: BaselineConfig,
) -> List[Dict[str, object]]:
    set_global_seed(seed)
    data_dir = Path(args.data_dir)
    X_train_2d, X_train_3d, y_train, shape_info, imputer, scaler = build_preprocessor(data_dir, train_segments)
    X_val_3d, y_val = prepare_validation(data_dir, val_segments, imputer, scaler, shape_info)
    classes = np.asarray(sorted(np.unique(y_train).tolist()))
    if not set(classes.tolist()).issubset({0, 1}):
        raise ValueError(f"This script expects binary labels 0/1. Found classes: {classes.tolist()}")

    methods = set(args.methods)
    results: List[Dict[str, object]] = []

    sgd_methods = sorted(methods & {"static_sgd", "dwm", "online_bagging", "leveraging_bagging"})
    if sgd_methods:
        results.extend(
            run_sgd_streaming_methods(
                seed=seed,
                baseline_cfg=baseline_cfg,
                data_dir=data_dir,
                test_segments=test_segments,
                chunk_size=args.chunk_size,
                imputer=imputer,
                scaler=scaler,
                shape_info=shape_info,
                X_train_2d=X_train_2d,
                y_train=y_train,
                classes=classes,
                methods=sgd_methods,
            ).values()
        )

    if "static_lstm" in methods:
        results.append(
            run_static_lstm(
                seed=seed,
                cfg=adawu_cfg,
                data_dir=data_dir,
                test_segments=test_segments,
                chunk_size=args.chunk_size,
                imputer=imputer,
                scaler=scaler,
                shape_info=shape_info,
                X_train_3d=X_train_3d,
                y_train=y_train,
                X_val_3d=X_val_3d,
                y_val=y_val,
            )
        )

    if "adawu" in methods:
        results.append(
            run_adawu_variant(
                seed=seed,
                cfg=adawu_cfg,
                variant_name="adawu",
                data_dir=data_dir,
                test_segments=test_segments,
                chunk_size=args.chunk_size,
                imputer=imputer,
                scaler=scaler,
                shape_info=shape_info,
                X_train_3d=X_train_3d,
                y_train=y_train,
                X_val_3d=X_val_3d,
                y_val=y_val,
                classes=classes,
                use_msdi=True,
                use_weighting=True,
                use_hierarchy=True,
            )
        )

    if args.include_ablation:
        if "adawu_no_msdi" not in methods:
            results.append(
                run_adawu_variant(
                    seed=seed,
                    cfg=adawu_cfg,
                    variant_name="adawu_no_msdi",
                    data_dir=data_dir,
                    test_segments=test_segments,
                    chunk_size=args.chunk_size,
                    imputer=imputer,
                    scaler=scaler,
                    shape_info=shape_info,
                    X_train_3d=X_train_3d,
                    y_train=y_train,
                    X_val_3d=X_val_3d,
                    y_val=y_val,
                    classes=classes,
                    use_msdi=False,
                    use_weighting=True,
                    use_hierarchy=True,
                )
            )
        if "adawu_no_weighting" not in methods:
            results.append(
                run_adawu_variant(
                    seed=seed,
                    cfg=adawu_cfg,
                    variant_name="adawu_no_weighting",
                    data_dir=data_dir,
                    test_segments=test_segments,
                    chunk_size=args.chunk_size,
                    imputer=imputer,
                    scaler=scaler,
                    shape_info=shape_info,
                    X_train_3d=X_train_3d,
                    y_train=y_train,
                    X_val_3d=X_val_3d,
                    y_val=y_val,
                    classes=classes,
                    use_msdi=True,
                    use_weighting=False,
                    use_hierarchy=True,
                )
            )
        if "adawu_no_hierarchy" not in methods:
            results.append(
                run_adawu_variant(
                    seed=seed,
                    cfg=adawu_cfg,
                    variant_name="adawu_no_hierarchy",
                    data_dir=data_dir,
                    test_segments=test_segments,
                    chunk_size=args.chunk_size,
                    imputer=imputer,
                    scaler=scaler,
                    shape_info=shape_info,
                    X_train_3d=X_train_3d,
                    y_train=y_train,
                    X_val_3d=X_val_3d,
                    y_val=y_val,
                    classes=classes,
                    use_msdi=True,
                    use_weighting=True,
                    use_hierarchy=False,
                )
            )

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fair CICIDS2017 chronological streaming experiment")
    p.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "datasets" / "processed")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "fair_cicids2017")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--train-segments", type=str, default=None, help="Comma-separated segment stems")
    p.add_argument("--val-segments", type=str, default=None, help="Comma-separated segment stems. Empty string disables validation.")
    p.add_argument("--test-segments", type=str, default=None, help="Comma-separated segment stems")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["static_lstm", "adawu", "dwm", "online_bagging", "leveraging_bagging"],
        choices=[
            "static_lstm",
            "static_sgd",
            "adawu",
            "dwm",
            "online_bagging",
            "leveraging_bagging",
        ],
    )
    p.add_argument("--include-ablation", action="store_true", help="Also run ADAWU no-MSDI/no-weighting/no-hierarchy variants")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--adaptation-rate", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.60)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.15)
    p.add_argument("--lambda-decay", type=float, default=0.10)
    p.add_argument("--mild-threshold", type=float, default=0.30)
    p.add_argument("--moderate-threshold", type=float, default=0.50)
    p.add_argument("--severe-threshold", type=float, default=0.70)
    p.add_argument("--reference-max-samples", type=int, default=50000)
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    train_segments = parse_segments(args.train_segments, DEFAULT_TRAIN_SEGMENTS)
    val_segments = parse_segments(args.val_segments, DEFAULT_VAL_SEGMENTS)
    test_segments = parse_segments(args.test_segments, DEFAULT_TEST_SEGMENTS)
    assert_no_overlap(train_segments, val_segments, test_segments)

    adawu_cfg = ADAWUConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        adaptation_rate=args.adaptation_rate,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        lambda_decay=args.lambda_decay,
        mild_threshold=args.mild_threshold,
        moderate_threshold=args.moderate_threshold,
        severe_threshold=args.severe_threshold,
        reference_max_samples=args.reference_max_samples,
    )
    baseline_cfg = BaselineConfig()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "script": Path(__file__).name,
        "dataset": "CICIDS2017",
        "protocol": "chronological_streaming_prequential",
        "fairness_rules": [
            "preprocessor fitted only on train segments",
            "no train/test overlap",
            "no validation/test overlap",
            "all methods share identical test chunks",
            "prediction is recorded before current chunk labels are used for online update",
        ],
        "train_segments": list(train_segments),
        "validation_segments": list(val_segments),
        "test_segments": list(test_segments),
        "chunk_size": args.chunk_size,
        "seeds": list(args.seeds),
        "methods": list(args.methods),
        "include_ablation": bool(args.include_ablation),
        "adawu_config": asdict(adawu_cfg),
        "baseline_config": asdict(baseline_cfg),
    }
    safe_json_dump(manifest, output_dir / "protocol_manifest.json")

    all_results: List[Dict[str, object]] = []
    for seed in args.seeds:
        print(f"[RUN] seed={seed}", flush=True)
        seed_results = run_one_seed(
            seed=seed,
            args=args,
            train_segments=train_segments,
            val_segments=val_segments,
            test_segments=test_segments,
            adawu_cfg=copy.deepcopy(adawu_cfg),
            baseline_cfg=copy.deepcopy(baseline_cfg),
        )
        for result in seed_results:
            save_result(result, output_dir)
            all_results.append(result)
            print(f"  [OK] {result['method']} seed={seed} overall={result['overall']}", flush=True)

    aggregate_rows = aggregate_seed_results(all_results)
    write_csv(aggregate_rows, output_dir / "tables" / "overall_mean_std_ci95.csv")
    safe_json_dump(aggregate_rows, output_dir / "summaries" / "overall_mean_std_ci95.json")

    print("[DONE] Fair CICIDS2017 experiment completed")
    print(f"[OUT] {output_dir}")


if __name__ == "__main__":
    main()
