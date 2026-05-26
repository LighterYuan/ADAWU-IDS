#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_calibrated_cicids2017_experiment.py

Purpose:
    Minimal, fair, attack-sensitive calibration experiment for CICIDS2017.

Key protocol:
    Train:
        Tuesday
        Wednesday

    Validation:
        Thursday WebAttacks

    Test:
        Thursday Infiltration
        Friday Morning
        Friday PortScan

Main goal:
    Determine whether ADAWU-IDS-Calibrated can improve IDS-sensitive metrics:
        - attack_recall
        - attack_f1
        - false_negative_rate

Outputs:
    output_dir/
        calibration/
            selected_candidate.json
            candidate_validation_summary.csv
        per_seed/
            <method>_seed<seed>_summary.json
            <method>_seed<seed>_chunks.csv
        tables/
            overall_mean_std_ci95.csv
            paired_statistical_tests.csv
        protocol_manifest.json
"""

import argparse
import gc
import json
import math
import os
import random
import re
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# TensorFlow / Keras
# ---------------------------------------------------------------------

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras import layers, models, optimizers
except Exception:
    TF_AVAILABLE = False


# ---------------------------------------------------------------------
# Segment definitions
# ---------------------------------------------------------------------

TRAIN_SEGMENTS = ["Tuesday", "Wednesday"]
VALIDATION_SEGMENTS = ["Thursday WebAttacks"]
TEST_SEGMENTS = ["Thursday Infiltration", "Friday Morning", "Friday PortScan"]

SEGMENT_ALIASES = {
    "Tuesday": [
        "Tuesday",
        "Tuesday-WorkingHours",
    ],
    "Wednesday": [
        "Wednesday",
        "Wednesday-workingHours",
        "Wednesday-WorkingHours",
    ],
    "Thursday WebAttacks": [
        "Thursday WebAttacks",
        "Thursday_WebAttacks",
        "WebAttacks",
        "Thursday-WorkingHours-Morning-WebAttacks",
        "Thursday-WorkingHours-Afternoon-WebAttacks",
    ],
    "Thursday Infiltration": [
        "Thursday Infiltration",
        "Thursday_Infiltration",
        "Infiltration",
        "Infilteration",
        "Thursday-WorkingHours-Afternoon-Infilteration",
        "Thursday-WorkingHours-Afternoon-Infiltration",
    ],
    "Friday Morning": [
        "Friday Morning",
        "Friday_Morning",
        "Friday-WorkingHours-Morning",
    ],
    "Friday PortScan": [
        "Friday PortScan",
        "Friday_PortScan",
        "PortScan",
        "Friday-WorkingHours-Afternoon-PortScan",
    ],
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    if TF_AVAILABLE:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def json_dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def ci95(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) <= 1:
        return 0.0
    return 1.96 * np.std(values, ddof=1) / math.sqrt(len(values))


def create_sgd(seed):
    try:
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            random_state=seed,
            max_iter=1000,
            tol=1e-3,
        )
    except Exception:
        return SGDClassifier(
            loss="log",
            penalty="l2",
            alpha=1e-4,
            random_state=seed,
            max_iter=1000,
            tol=1e-3,
        )


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

@dataclass
class SegmentData:
    name: str
    X: np.ndarray
    y: np.ndarray


def is_x_file(path: Path) -> bool:
    stem = path.stem.lower()
    tokens = re.split(r"[^a-z0-9]+", stem)
    return ("x" in tokens) or ("features" in tokens) or ("feature" in tokens)


def is_y_file(path: Path) -> bool:
    stem = path.stem.lower()
    tokens = re.split(r"[^a-z0-9]+", stem)
    return (
        "y" in tokens
        or "label" in tokens
        or "labels" in tokens
        or "target" in tokens
        or "targets" in tokens
    )


def binarize_labels(y):
    y = np.asarray(y).reshape(-1)

    if y.dtype.kind in ["U", "S", "O"]:
        out = []
        for v in y:
            s = str(v).strip().lower()
            if s in ["benign", "normal", "0", "false"]:
                out.append(0)
            else:
                out.append(1)
        return np.asarray(out, dtype=np.int64)

    y = y.astype(float)
    return (y > 0).astype(np.int64)
    
def ensure_2d_features(X):
    """
    Convert feature arrays to 2D: (n_samples, n_features).

    If the processed data are already sequence-shaped, e.g.
    (n_samples, time_steps, n_features), flatten the last two dimensions.

    This is required because SimpleImputer and StandardScaler expect 2D input.
    """
    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    elif X.ndim == 2:
        pass

    elif X.ndim >= 3:
        X = X.reshape(X.shape[0], -1)

    X = X.astype(np.float32)

    # Replace inf with nan; SimpleImputer will handle nan.
    X[~np.isfinite(X)] = np.nan

    return X


def load_npz_segment(path: Path):
    obj = np.load(path, allow_pickle=True)
    keys = list(obj.keys())

    x_key = None
    y_key = None

    for k in keys:
        lk = k.lower()
        if lk in ["x", "features", "data"]:
            x_key = k
        if lk in ["y", "label", "labels", "target", "targets"]:
            y_key = k

    if x_key is None:
        for k in keys:
            if obj[k].ndim == 2:
                x_key = k
                break

    if y_key is None:
        for k in keys:
            if obj[k].ndim == 1:
                y_key = k
                break

    if x_key is None or y_key is None:
        raise ValueError(f"Cannot identify X/y in {path}")

    X = np.asarray(obj[x_key])
    y = binarize_labels(obj[y_key])
    return X, y


def load_csv_segment(path: Path):
    df = pd.read_csv(path)

    label_candidates = [
        "Label",
        "label",
        "labels",
        "target",
        "Target",
        "class",
        "Class",
        "y",
    ]

    label_col = None
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        raise ValueError(f"No label column found in {path}")

    y = binarize_labels(df[label_col].values)
    X_df = df.drop(columns=[label_col])

    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    X = X_df.values.astype(float)
    return X, y


def load_npy_pair(x_path: Path, y_path: Path):
    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    return np.asarray(X), binarize_labels(y)


def find_segment_files(data_dir: Path, segment_name: str):
    aliases = SEGMENT_ALIASES[segment_name]
    alias_norms = [normalize_name(a) for a in aliases]

    all_files = []
    for ext in ["*.npz", "*.npy", "*.csv", "*.parquet"]:
        all_files.extend(list(data_dir.rglob(ext)))

    # Prefer .npz
    npz_matches = []
    for p in all_files:
        if p.suffix.lower() != ".npz":
            continue
        n = normalize_name(p.stem)
        if any(a in n for a in alias_norms):
            npz_matches.append(p)

    if npz_matches:
        return ("npz", sorted(npz_matches)[0], None)

    # Then CSV / parquet
    table_matches = []
    for p in all_files:
        if p.suffix.lower() not in [".csv", ".parquet"]:
            continue
        n = normalize_name(p.stem)
        if any(a in n for a in alias_norms):
            table_matches.append(p)

    if table_matches:
        return ("table", sorted(table_matches)[0], None)

    # Then .npy X/y pairs
    x_matches = []
    y_matches = []

    for p in all_files:
        if p.suffix.lower() != ".npy":
            continue

        n = normalize_name(p.stem)
        if not any(a in n for a in alias_norms):
            continue

        if is_x_file(p):
            x_matches.append(p)
        elif is_y_file(p):
            y_matches.append(p)

    if x_matches and y_matches:
        return ("npy_pair", sorted(x_matches)[0], sorted(y_matches)[0])

    raise FileNotFoundError(
        f"Could not find data for segment '{segment_name}' in {data_dir}"
    )


def load_segment(data_dir: Path, segment_name: str):
    kind, p1, p2 = find_segment_files(data_dir, segment_name)

    if kind == "npz":
        X, y = load_npz_segment(p1)
    elif kind == "table":
        if p1.suffix.lower() == ".csv":
            X, y = load_csv_segment(p1)
        else:
            df = pd.read_parquet(p1)
            temp_csv = p1.with_suffix(".tmp.csv")
            df.to_csv(temp_csv, index=False)
            X, y = load_csv_segment(temp_csv)
            temp_csv.unlink(missing_ok=True)
    elif kind == "npy_pair":
        X, y = load_npy_pair(p1, p2)
    else:
        raise RuntimeError(f"Unknown segment kind: {kind}")

    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch for {segment_name}: {len(X)} vs {len(y)}")


        
    X = ensure_2d_features(X)
    return SegmentData(segment_name, X, y.astype(np.int64))


def load_all_segments(data_dir):
    data_dir = Path(data_dir)

    segments = {}
    for name in TRAIN_SEGMENTS + VALIDATION_SEGMENTS + TEST_SEGMENTS:
        print(f"[LOAD] {name}")
        segments[name] = load_segment(data_dir, name)
        print(
            f"       X={segments[name].X.shape}, "
            f"positive={int(segments[name].y.sum())}, "
            f"total={len(segments[name].y)}"
        )

    return segments


def concat_segments(segments, names):
    X = np.vstack([segments[n].X for n in names])
    y = np.concatenate([segments[n].y for n in names])
    return X, y


def fit_preprocessor(X_train):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_imp)

    return imputer, scaler, X_scaled


def transform_preprocessor(X, imputer, scaler):
    return scaler.transform(imputer.transform(X))


def preprocess_segments(segments, max_train_samples=0, seed=42):
    X_train_raw, y_train = concat_segments(segments, TRAIN_SEGMENTS)

    if max_train_samples and max_train_samples > 0 and len(X_train_raw) > max_train_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train_raw), size=max_train_samples, replace=False)
        X_train_raw = X_train_raw[idx]
        y_train = y_train[idx]

    imputer, scaler, X_train = fit_preprocessor(X_train_raw)

    processed = {}

    processed["train"] = SegmentData("train", X_train, y_train)

    for name, seg in segments.items():
        processed[name] = SegmentData(
            name,
            transform_preprocessor(seg.X, imputer, scaler),
            seg.y,
        )

    return processed, imputer, scaler


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    labels = [0, 1]

    acc = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    attack_precision = precision_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )
    attack_recall = recall_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )
    attack_f1 = f1_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "attack_precision": float(attack_precision),
        "attack_recall": float(attack_recall),
        "attack_f1": float(attack_f1),
        "false_negative_rate": float(false_negative_rate),
        "false_positive_rate": float(false_positive_rate),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_stream(processed_segments, segment_names, chunk_size):
    for seg_name in segment_names:
        seg = processed_segments[seg_name]
        X = seg.X
        y = seg.y

        n = len(y)
        chunk_id = 0

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            yield {
                "segment": seg_name,
                "chunk_id": chunk_id,
                "start": start,
                "end": end,
                "X": X[start:end],
                "y": y[start:end],
            }
            chunk_id += 1


def summarize_chunks(chunks):
    y_true = []
    y_pred = []

    for c in chunks:
        y_true.extend(c["y_true"])
        y_pred.extend(c["y_pred"])

    return compute_metrics(np.asarray(y_true), np.asarray(y_pred))


# ---------------------------------------------------------------------
# LSTM models
# ---------------------------------------------------------------------

def build_lstm_model(n_features, seed=42, units=32, lr=1e-3):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available.")

    set_global_seed(seed)

    model = models.Sequential(
        [
            layers.Input(shape=(1, n_features)),
            layers.LSTM(units),
            layers.Dropout(0.20),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def to_lstm_input(X):
    return X.reshape((X.shape[0], 1, X.shape[1]))


def train_lstm_model(model, X, y, epochs=3, batch_size=512, verbose=0):
    model.fit(
        to_lstm_input(X),
        y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        shuffle=True,
    )
    return model


def predict_lstm_proba(model, X):
    p = model.predict(to_lstm_input(X), verbose=0).reshape(-1)
    return np.clip(p, 1e-6, 1 - 1e-6)


def clone_compiled_model(model):
    new_model = models.clone_model(model)
    new_model.set_weights(model.get_weights())
    new_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return new_model


def train_base_lstm_models(X_train, y_train, seed, epochs, batch_size):
    n_features = X_train.shape[1]

    configs = [
        {"units": 32, "lr": 1e-3, "seed_offset": 0},
        {"units": 48, "lr": 8e-4, "seed_offset": 11},
        {"units": 24, "lr": 1.2e-3, "seed_offset": 23},
    ]

    models_out = []
    for cfg in configs:
        m = build_lstm_model(
            n_features=n_features,
            seed=seed + cfg["seed_offset"],
            units=cfg["units"],
            lr=cfg["lr"],
        )
        train_lstm_model(m, X_train, y_train, epochs=epochs, batch_size=batch_size)
        models_out.append(m)

    return models_out


def evaluate_static_lstm(base_model, stream):
    chunks = []

    for ch in stream:
        t0 = time.time()
        p = predict_lstm_proba(base_model, ch["X"])
        y_pred = (p >= 0.5).astype(int)
        latency = time.time() - t0

        m = compute_metrics(ch["y"], y_pred)

        chunks.append(
            {
                "segment": ch["segment"],
                "chunk_id": int(ch["chunk_id"]),
                "n": int(len(ch["y"])),
                "latency_sec": float(latency),
                "y_true": ch["y"].astype(int).tolist(),
                "y_pred": y_pred.astype(int).tolist(),
                **m,
            }
        )

    return chunks, summarize_chunks(chunks)


# ---------------------------------------------------------------------
# MSDI and ADAWU
# ---------------------------------------------------------------------

@dataclass
class ADAWUConfig:
    name: str
    alpha: float = 0.60
    beta: float = 0.25
    gamma: float = 0.15
    lambda_decay: float = 0.10
    thresholds: tuple = (0.30, 0.50, 0.70)
    use_msdi: bool = True
    use_weighting: bool = True
    use_hierarchy: bool = True


def msdi_score(X_chunk, ref_mean, ref_std):
    eps = 1e-6

    c_mean = np.nanmean(X_chunk, axis=0)
    c_std = np.nanstd(X_chunk, axis=0)

    mean_shift = np.nanmean(np.abs(c_mean - ref_mean) / (ref_std + eps))
    std_shift = np.nanmean(np.abs(c_std - ref_std) / (ref_std + eps))

    raw = 0.70 * mean_shift + 0.30 * std_shift

    score = raw / (1.0 + raw)
    score = float(np.clip(score, 0.0, 1.0))

    return score


def severity_from_msdi(score, thresholds):
    t1, t2, t3 = thresholds

    if score < t1:
        return "none"
    if score < t2:
        return "mild"
    if score < t3:
        return "moderate"
    return "severe"


def normalize_weights(w):
    w = np.asarray(w, dtype=float)
    w = np.maximum(w, 1e-8)
    return w / np.sum(w)


def evaluate_adawu(
    base_models,
    X_train,
    stream,
    config: ADAWUConfig,
    online_epochs=1,
    batch_size=512,
):
    models_local = [clone_compiled_model(m) for m in base_models]

    n_models = len(models_local)
    weights = np.ones(n_models, dtype=float) / n_models

    ref_mean = np.nanmean(X_train, axis=0)
    ref_std = np.nanstd(X_train, axis=0) + 1e-6

    chunks = []

    for ch in stream:
        Xc = ch["X"]
        yc = ch["y"]

        t0 = time.time()

        model_probs = []
        for m in models_local:
            model_probs.append(predict_lstm_proba(m, Xc))
        model_probs = np.vstack(model_probs)

        if config.use_weighting:
            ens_proba = np.average(model_probs, axis=0, weights=weights)
        else:
            ens_proba = np.mean(model_probs, axis=0)

        y_pred = (ens_proba >= 0.5).astype(int)

        score = msdi_score(Xc, ref_mean, ref_std) if config.use_msdi else 0.0
        severity = severity_from_msdi(score, config.thresholds)

        metrics = compute_metrics(yc, y_pred)

        # Update weights after labels become available.
        retrain_event = "none"

        if config.use_weighting:
            perf = []
            for i in range(n_models):
                yp_i = (model_probs[i] >= 0.5).astype(int)
                perf_i = f1_score(yc, yp_i, average="weighted", zero_division=0)
                perf.append(perf_i)
            perf = np.asarray(perf, dtype=float)

            msdi_term = 1.0 - score if config.use_msdi else 1.0

            raw_new = (
                config.alpha * perf
                + config.beta * msdi_term
                + config.gamma * weights
            )

            weights = normalize_weights(
                (1.0 - config.lambda_decay) * weights
                + config.lambda_decay * raw_new
            )

        # Hierarchical response after prediction.
        if config.use_hierarchy:
            if severity == "moderate":
                worst = int(np.argmin(weights))
                train_lstm_model(
                    models_local[worst],
                    Xc,
                    yc,
                    epochs=online_epochs,
                    batch_size=batch_size,
                    verbose=0,
                )
                retrain_event = "moderate_update_worst_model"

            elif severity == "severe":
                for m in models_local:
                    train_lstm_model(
                        m,
                        Xc,
                        yc,
                        epochs=online_epochs,
                        batch_size=batch_size,
                        verbose=0,
                    )
                retrain_event = "severe_update_all_models"

        latency = time.time() - t0

        chunks.append(
            {
                "segment": ch["segment"],
                "chunk_id": int(ch["chunk_id"]),
                "n": int(len(yc)),
                "latency_sec": float(latency),
                "msdi": float(score),
                "severity": severity,
                "weights": weights.tolist(),
                "retrain_event": retrain_event,
                "y_true": yc.astype(int).tolist(),
                "y_pred": y_pred.astype(int).tolist(),
                **metrics,
            }
        )

    try:
        for m in models_local:
            del m
        gc.collect()
        if TF_AVAILABLE:
            K.clear_session()
    except Exception:
        pass

    return chunks, summarize_chunks(chunks)


# ---------------------------------------------------------------------
# Adaptive baselines
# ---------------------------------------------------------------------

class OnlineEnsembleBaseline:
    def __init__(self, name, n_estimators=10, poisson_lambda=1.0, seed=42):
        self.name = name
        self.n_estimators = n_estimators
        self.poisson_lambda = poisson_lambda
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.models = [create_sgd(seed + i * 17) for i in range(n_estimators)]
        self.weights = np.ones(n_estimators, dtype=float) / n_estimators
        self.classes = np.asarray([0, 1], dtype=int)

    def fit(self, X, y):
        n = len(y)

        for i, clf in enumerate(self.models):
            idx = self.rng.choice(n, size=n, replace=True)
            clf.partial_fit(X[idx], y[idx], classes=self.classes)

    def predict_proba(self, X):
        probs = []

        for clf in self.models:
            if hasattr(clf, "predict_proba"):
                p = clf.predict_proba(X)[:, 1]
            else:
                s = clf.decision_function(X)
                p = 1.0 / (1.0 + np.exp(-s))
            probs.append(p)

        probs = np.vstack(probs)
        return np.average(probs, axis=0, weights=self.weights)

    def update(self, X, y):
        perfs = []

        for i, clf in enumerate(self.models):
            yp = clf.predict(X)
            perf = f1_score(y, yp, average="weighted", zero_division=0)
            perfs.append(perf)

            if self.name.lower() == "dwm":
                clf.partial_fit(X, y)
            else:
                sw = self.rng.poisson(self.poisson_lambda, size=len(y)).astype(float)
                if np.sum(sw) <= 0:
                    sw[:] = 1.0
                try:
                    clf.partial_fit(X, y, sample_weight=sw)
                except TypeError:
                    clf.partial_fit(X, y)

        perfs = np.asarray(perfs, dtype=float)

        if self.name.lower() == "dwm":
            self.weights *= np.maximum(perfs, 1e-4)
            self.weights = normalize_weights(self.weights)
        else:
            self.weights = normalize_weights(0.90 * self.weights + 0.10 * perfs)


def evaluate_online_baseline(name, X_train, y_train, stream, seed):
    if name == "DWM":
        model = OnlineEnsembleBaseline(
            name="DWM",
            n_estimators=5,
            poisson_lambda=1.0,
            seed=seed,
        )
    elif name == "Online Bagging":
        model = OnlineEnsembleBaseline(
            name="Online Bagging",
            n_estimators=10,
            poisson_lambda=1.0,
            seed=seed,
        )
    elif name == "Leveraging Bagging":
        model = OnlineEnsembleBaseline(
            name="Leveraging Bagging",
            n_estimators=10,
            poisson_lambda=6.0,
            seed=seed,
        )
    else:
        raise ValueError(name)

    model.fit(X_train, y_train)

    chunks = []

    for ch in stream:
        t0 = time.time()
        p = model.predict_proba(ch["X"])
        y_pred = (p >= 0.5).astype(int)
        latency = time.time() - t0

        m = compute_metrics(ch["y"], y_pred)

        model.update(ch["X"], ch["y"])

        chunks.append(
            {
                "segment": ch["segment"],
                "chunk_id": int(ch["chunk_id"]),
                "n": int(len(ch["y"])),
                "latency_sec": float(latency),
                "y_true": ch["y"].astype(int).tolist(),
                "y_pred": y_pred.astype(int).tolist(),
                **m,
            }
        )

    return chunks, summarize_chunks(chunks)


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------

def compute_selection_score(metrics, selection_metric="attack_sensitive"):
    """
    IDS-oriented validation score.

    This score avoids selecting configurations that achieve high weighted F1
    only because benign samples dominate the validation set.
    """
    weighted_f1 = float(metrics.get("weighted_f1", 0.0))
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    attack_recall = float(metrics.get("attack_recall", 0.0))
    attack_f1 = float(metrics.get("attack_f1", 0.0))
    false_negative_rate = float(
        metrics.get("false_negative_rate", 1.0 - attack_recall)
    )

    if selection_metric == "weighted_f1":
        return weighted_f1

    if selection_metric == "attack_recall":
        return attack_recall

    if selection_metric == "attack_f1":
        return attack_f1

    if selection_metric == "ids_score":
        return (
            0.25 * weighted_f1
            + 0.30 * attack_f1
            + 0.35 * attack_recall
            + 0.10 * macro_f1
            - 0.15 * false_negative_rate
        )

    if selection_metric == "attack_sensitive":
        score = (
            0.20 * weighted_f1
            + 0.35 * attack_f1
            + 0.35 * attack_recall
            + 0.10 * macro_f1
            - 0.20 * false_negative_rate
        )

        # Hard penalty for configurations that miss most attacks.
        if attack_recall < 0.60:
            score -= 2.0 * (0.60 - attack_recall)

        if attack_f1 < 0.50:
            score -= 1.5 * (0.50 - attack_f1)

        return score

    raise ValueError(f"Unknown selection_metric: {selection_metric}")


def get_candidate_grid(grid_name):
    candidates = []

    candidates.append(
        ADAWUConfig(
            name="weight_only_default",
            alpha=0.60,
            beta=0.25,
            gamma=0.15,
            lambda_decay=0.10,
            thresholds=(0.30, 0.50, 0.70),
            use_msdi=True,
            use_weighting=True,
            use_hierarchy=False,
        )
    )

    candidates.append(
        ADAWUConfig(
            name="weight_only_attack_sensitive",
            alpha=0.50,
            beta=0.20,
            gamma=0.30,
            lambda_decay=0.20,
            thresholds=(0.25, 0.45, 0.65),
            use_msdi=True,
            use_weighting=True,
            use_hierarchy=False,
        )
    )

    candidates.append(
        ADAWUConfig(
            name="full_default",
            alpha=0.60,
            beta=0.25,
            gamma=0.15,
            lambda_decay=0.10,
            thresholds=(0.30, 0.50, 0.70),
            use_msdi=True,
            use_weighting=True,
            use_hierarchy=True,
        )
    )

    candidates.append(
        ADAWUConfig(
            name="full_conservative",
            alpha=0.50,
            beta=0.20,
            gamma=0.30,
            lambda_decay=0.10,
            thresholds=(0.45, 0.65, 0.85),
            use_msdi=True,
            use_weighting=True,
            use_hierarchy=True,
        )
    )

    if grid_name in ["medium", "large"]:
        candidates.append(
            ADAWUConfig(
                name="full_aggressive",
                alpha=0.70,
                beta=0.20,
                gamma=0.10,
                lambda_decay=0.20,
                thresholds=(0.20, 0.40, 0.60),
                use_msdi=True,
                use_weighting=True,
                use_hierarchy=True,
            )
        )

        candidates.append(
            ADAWUConfig(
                name="no_msdi_weight_only",
                alpha=0.70,
                beta=0.00,
                gamma=0.30,
                lambda_decay=0.15,
                thresholds=(0.30, 0.50, 0.70),
                use_msdi=False,
                use_weighting=True,
                use_hierarchy=False,
            )
        )

    return candidates


def config_to_dict(cfg):
    return {
        "name": cfg.name,
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "gamma": cfg.gamma,
        "lambda_decay": cfg.lambda_decay,
        "thresholds": list(cfg.thresholds),
        "use_msdi": cfg.use_msdi,
        "use_weighting": cfg.use_weighting,
        "use_hierarchy": cfg.use_hierarchy,
    }


def dict_to_config(d):
    return ADAWUConfig(
        name=d["name"],
        alpha=float(d["alpha"]),
        beta=float(d["beta"]),
        gamma=float(d["gamma"]),
        lambda_decay=float(d["lambda_decay"]),
        thresholds=tuple(d["thresholds"]),
        use_msdi=bool(d["use_msdi"]),
        use_weighting=bool(d["use_weighting"]),
        use_hierarchy=bool(d["use_hierarchy"]),
    )


def calibrate_adawu(processed, args, out_dir):
    ensure_dir(out_dir)

    X_train = processed["train"].X
    y_train = processed["train"].y

    candidates = get_candidate_grid(args.candidate_grid)
    rows = []

    for cfg in candidates:
        print(f"[CALIBRATE] candidate={cfg.name}")

        seed_metrics = []
        seed_scores = []

        for seed in args.calibration_seeds:
            print(f"            seed={seed}")

            set_global_seed(seed)

            base_models = train_base_lstm_models(
                X_train,
                y_train,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )

            val_stream = make_stream(
                processed,
                VALIDATION_SEGMENTS,
                args.chunk_size,
            )

            chunks, overall = evaluate_adawu(
                base_models=base_models,
                X_train=X_train,
                stream=val_stream,
                config=cfg,
                online_epochs=args.online_epochs,
                batch_size=args.batch_size,
            )

            score = compute_selection_score(overall, args.selection_metric)

            seed_metrics.append(overall)
            seed_scores.append(score)

            del base_models
            gc.collect()
            if TF_AVAILABLE:
                K.clear_session()

        row = {
            "candidate": cfg.name,
            "selection_metric": args.selection_metric,
            "selection_score_mean": float(np.mean(seed_scores)),
            "selection_score_std": float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else 0.0,
        }

        metric_keys = [
            "accuracy",
            "weighted_f1",
            "macro_f1",
            "attack_precision",
            "attack_recall",
            "attack_f1",
            "false_negative_rate",
            "false_positive_rate",
        ]

        for k in metric_keys:
            vals = [m[k] for m in seed_metrics]
            row[f"{k}_mean"] = float(np.mean(vals))
            row[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        row.update(config_to_dict(cfg))
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("selection_score_mean", ascending=False)

    df.to_csv(Path(out_dir) / "candidate_validation_summary.csv", index=False)

    selected_row = df.iloc[0].to_dict()
    selected_cfg = None

    for cfg in candidates:
        if cfg.name == selected_row["candidate"]:
            selected_cfg = cfg
            break

    selected = {
        "selected_candidate": selected_cfg.name,
        "selection_metric": args.selection_metric,
        "selection_score_mean": selected_row["selection_score_mean"],
        "config": config_to_dict(selected_cfg),
    }

    json_dump(selected, Path(out_dir) / "selected_candidate.json")

    print(f"[SELECTED] {selected_cfg.name} by {args.selection_metric}")

    return selected_cfg, df


# ---------------------------------------------------------------------
# Saving and summary
# ---------------------------------------------------------------------

def save_method_result(out_dir, method_name, seed, chunks, overall):
    per_seed_dir = Path(out_dir) / "per_seed"
    ensure_dir(per_seed_dir)

    summary_path = per_seed_dir / f"{method_name}_seed{seed}_summary.json"
    chunks_path = per_seed_dir / f"{method_name}_seed{seed}_chunks.csv"

    json_dump(
        {
            "method": method_name,
            "seed": int(seed),
            "overall": overall,
        },
        summary_path,
    )

    rows = []
    for c in chunks:
        row = {k: v for k, v in c.items() if k not in ["y_true", "y_pred", "weights"]}
        if "weights" in c:
            row["weights_json"] = json.dumps(c["weights"])
        rows.append(row)

    pd.DataFrame(rows).to_csv(chunks_path, index=False)


def aggregate_results(out_dir):
    per_seed_dir = Path(out_dir) / "per_seed"
    table_dir = Path(out_dir) / "tables"
    ensure_dir(table_dir)

    records = []

    for p in sorted(per_seed_dir.glob("*_summary.json")):
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)

        method = obj["method"]
        seed = obj["seed"]
        overall = obj["overall"]

        row = {"method": method, "seed": seed}
        row.update(overall)
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(table_dir / "per_seed_overall_metrics.csv", index=False)

    metric_cols = [
        "accuracy",
        "weighted_f1",
        "macro_f1",
        "attack_precision",
        "attack_recall",
        "attack_f1",
        "false_negative_rate",
        "false_positive_rate",
    ]

    summary_rows = []

    for method, g in df.groupby("method"):
        row = {
            "method": method,
            "n_seeds": int(len(g)),
        }

        for m in metric_cols:
            vals = g[m].astype(float).values
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{m}_ci95"] = float(ci95(vals))

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("weighted_f1_mean", ascending=False)
    summary.to_csv(table_dir / "overall_mean_std_ci95.csv", index=False)

    make_paired_tests(df, table_dir)

    return summary


def make_paired_tests(df, table_dir):
    try:
        from scipy.stats import ttest_rel, wilcoxon
        scipy_ok = True
    except Exception:
        scipy_ok = False

    baseline = "static_lstm"

    metric_cols = [
        "weighted_f1",
        "attack_recall",
        "attack_f1",
        "false_negative_rate",
    ]

    rows = []

    if baseline not in set(df["method"]):
        pd.DataFrame(rows).to_csv(table_dir / "paired_statistical_tests.csv", index=False)
        return

    base_df = df[df["method"] == baseline][["seed"] + metric_cols]

    for method in sorted(set(df["method"])):
        if method == baseline:
            continue

        mdf = df[df["method"] == method][["seed"] + metric_cols]
        merged = pd.merge(base_df, mdf, on="seed", suffixes=("_static", f"_{method}"))

        for metric in metric_cols:
            a = merged[f"{metric}_static"].astype(float).values
            b = merged[f"{metric}_{method}"].astype(float).values

            if len(a) < 2:
                t_p = float("nan")
                w_p = float("nan")
            elif scipy_ok:
                try:
                    t_p = float(ttest_rel(b, a).pvalue)
                except Exception:
                    t_p = float("nan")
                try:
                    w_p = float(wilcoxon(b, a).pvalue)
                except Exception:
                    w_p = float("nan")
            else:
                t_p = float("nan")
                w_p = float("nan")

            rows.append(
                {
                    "method": method,
                    "baseline": baseline,
                    "metric": metric,
                    "n_pairs": int(len(a)),
                    "method_mean": float(np.mean(b)) if len(b) else float("nan"),
                    "baseline_mean": float(np.mean(a)) if len(a) else float("nan"),
                    "mean_diff_method_minus_baseline": float(np.mean(b - a)) if len(a) else float("nan"),
                    "paired_t_p": t_p,
                    "wilcoxon_p": w_p,
                }
            )

    pd.DataFrame(rows).to_csv(table_dir / "paired_statistical_tests.csv", index=False)


# ---------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------

def make_ablation_configs(selected_cfg):
    cfgs = []

    no_msdi = deepcopy(selected_cfg)
    no_msdi.name = "adawu_calibrated_no_msdi"
    no_msdi.use_msdi = False
    cfgs.append(no_msdi)

    no_weighting = deepcopy(selected_cfg)
    no_weighting.name = "adawu_calibrated_no_weighting"
    no_weighting.use_weighting = False
    cfgs.append(no_weighting)

    no_hierarchy = deepcopy(selected_cfg)
    no_hierarchy.name = "adawu_calibrated_no_hierarchy"
    no_hierarchy.use_hierarchy = False
    cfgs.append(no_hierarchy)

    return cfgs


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def run_final_experiment(processed, selected_cfg, args):
    out_dir = Path(args.output_dir)

    X_train = processed["train"].X
    y_train = processed["train"].y

    default_full_cfg = ADAWUConfig(
        name="full_default",
        alpha=0.60,
        beta=0.25,
        gamma=0.15,
        lambda_decay=0.10,
        thresholds=(0.30, 0.50, 0.70),
        use_msdi=True,
        use_weighting=True,
        use_hierarchy=True,
    )

    for seed in args.seeds:
        print(f"[TEST] seed={seed}")

        set_global_seed(seed)

        base_models = train_base_lstm_models(
            X_train,
            y_train,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        # Static LSTM
        print("      method=static_lstm")
        static_stream = make_stream(processed, TEST_SEGMENTS, args.chunk_size)
        chunks, overall = evaluate_static_lstm(base_models[0], static_stream)
        save_method_result(out_dir, "static_lstm", seed, chunks, overall)
        print(f"      [OK] static_lstm seed={seed} overall={overall}")

        # ADAWU calibrated
        print("      method=adawu_calibrated")
        test_stream = make_stream(processed, TEST_SEGMENTS, args.chunk_size)
        chunks, overall = evaluate_adawu(
            base_models=base_models,
            X_train=X_train,
            stream=test_stream,
            config=selected_cfg,
            online_epochs=args.online_epochs,
            batch_size=args.batch_size,
        )
        save_method_result(out_dir, "adawu_calibrated", seed, chunks, overall)
        print(f"      [OK] adawu_calibrated seed={seed} overall={overall}")

        # ADAWU default full
        print("      method=adawu_default_full")
        test_stream = make_stream(processed, TEST_SEGMENTS, args.chunk_size)
        chunks, overall = evaluate_adawu(
            base_models=base_models,
            X_train=X_train,
            stream=test_stream,
            config=default_full_cfg,
            online_epochs=args.online_epochs,
            batch_size=args.batch_size,
        )
        save_method_result(out_dir, "adawu_default_full", seed, chunks, overall)
        print(f"      [OK] adawu_default_full seed={seed} overall={overall}")

        # Optional ablations
        if args.include_ablation:
            for ab_cfg in make_ablation_configs(selected_cfg):
                print(f"      method={ab_cfg.name}")
                test_stream = make_stream(processed, TEST_SEGMENTS, args.chunk_size)
                chunks, overall = evaluate_adawu(
                    base_models=base_models,
                    X_train=X_train,
                    stream=test_stream,
                    config=ab_cfg,
                    online_epochs=args.online_epochs,
                    batch_size=args.batch_size,
                )
                save_method_result(out_dir, ab_cfg.name, seed, chunks, overall)
                print(f"      [OK] {ab_cfg.name} seed={seed} overall={overall}")

        del base_models
        gc.collect()
        if TF_AVAILABLE:
            K.clear_session()

        # Adaptive baselines
        for method in ["DWM", "Online Bagging", "Leveraging Bagging"]:
            print(f"      method={method}")
            test_stream = make_stream(processed, TEST_SEGMENTS, args.chunk_size)
            chunks, overall = evaluate_online_baseline(
                name=method,
                X_train=X_train,
                y_train=y_train,
                stream=test_stream,
                seed=seed,
            )

            method_key = method.lower().replace(" ", "_")
            save_method_result(out_dir, method_key, seed, chunks, overall)
            print(f"      [OK] {method} seed={seed} overall={overall}")


def save_protocol_manifest(args, selected_cfg, out_dir):
    manifest = {
        "train_segments": TRAIN_SEGMENTS,
        "validation_segments": VALIDATION_SEGMENTS,
        "test_segments": TEST_SEGMENTS,
        "chunk_size": args.chunk_size,
        "seeds": args.seeds,
        "calibration_seeds": args.calibration_seeds,
        "selection_metric": args.selection_metric,
        "candidate_grid": args.candidate_grid,
        "epochs": args.epochs,
        "online_epochs": args.online_epochs,
        "batch_size": args.batch_size,
        "max_train_samples": args.max_train_samples,
        "selected_adawu_config": config_to_dict(selected_cfg),
        "tensorflow_available": TF_AVAILABLE,
        "protocol_note": (
            "Train, validation, and test segments are disjoint. "
            "All online methods predict first and update only after current chunk labels are available."
        ),
    }

    json_dump(manifest, Path(out_dir) / "protocol_manifest.json")


# ---------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing processed CICIDS2017 segment files.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for experiment outputs.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 52],
        help="Test seeds.",
    )

    parser.add_argument(
        "--calibration-seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Seeds used only for validation-based calibration.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Chronological stream chunk size.",
    )

    parser.add_argument(
        "--candidate-grid",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Candidate grid size for ADAWU calibration.",
    )

    parser.add_argument(
        "--selection-metric",
        type=str,
        default="attack_sensitive",
        choices=[
            "ids_score",
            "attack_sensitive",
            "weighted_f1",
            "attack_recall",
            "attack_f1",
        ],
        help="Metric used to select the calibrated ADAWU configuration on validation data.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Initial LSTM training epochs.",
    )

    parser.add_argument(
        "--online-epochs",
        type=int,
        default=1,
        help="Online fine-tuning epochs for hierarchical response.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="LSTM batch size.",
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Optional cap for training samples. 0 means use all.",
    )

    parser.add_argument(
        "--include-ablation",
        action="store_true",
        help="Run additional ablation variants. For quick test, do not enable this.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    if not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is required for the LSTM-based ADAWU experiment. "
            "Please install TensorFlow or run in the original project environment."
        )

    ensure_dir(args.output_dir)
    ensure_dir(Path(args.output_dir) / "calibration")
    ensure_dir(Path(args.output_dir) / "per_seed")
    ensure_dir(Path(args.output_dir) / "tables")

    print("[INFO] Loading data")
    segments = load_all_segments(args.data_dir)

    print("[INFO] Preprocessing data")
    processed, _, _ = preprocess_segments(
        segments,
        max_train_samples=args.max_train_samples,
        seed=args.seeds[0],
    )

    print("[INFO] Calibration")
    selected_cfg, _ = calibrate_adawu(
        processed=processed,
        args=args,
        out_dir=Path(args.output_dir) / "calibration",
    )

    print("[INFO] Final test")
    run_final_experiment(
        processed=processed,
        selected_cfg=selected_cfg,
        args=args,
    )

    print("[INFO] Aggregating results")
    summary = aggregate_results(args.output_dir)

    save_protocol_manifest(args, selected_cfg, args.output_dir)

    print("[DONE] Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
