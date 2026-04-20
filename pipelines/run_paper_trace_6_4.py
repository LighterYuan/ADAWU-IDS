from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from models.lstm_model import LSTMIDModel

DATA_DIR = PROJECT_ROOT / "datasets" / "processed"
OUTPUT_TRACE = PROJECT_ROOT / "results" / "traces" / "paper_trace_6_4.json"

INITIAL_TRAIN_SEGMENTS = [
    "Tuesday-WorkingHours.pcap_ISCX",
    "Wednesday-workingHours.pcap_ISCX",
]

EVAL_SEGMENTS = [
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX",
    "Friday-WorkingHours-Morning.pcap_ISCX",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX",
]

ALL_SEGMENTS = INITIAL_TRAIN_SEGMENTS + EVAL_SEGMENTS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_xy(stem: str):
    x_path = DATA_DIR / f"{stem}_X.npy"
    y_path = DATA_DIR / f"{stem}_y.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing file: {y_path}")

    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).reshape(-1).astype(int)
    return X, y


def ensure_3d(X: np.ndarray) -> np.ndarray:
    if X.ndim == 2:
        return X[:, None, :]
    if X.ndim == 3:
        return X
    raise ValueError(f"Unsupported X ndim: {X.ndim}")


def flatten_3d_to_2d(X: np.ndarray):
    X = ensure_3d(X)
    n, t, f = X.shape
    return X.reshape(n, t * f), (t, f)


def restore_2d_to_3d(X2: np.ndarray, shape_info):
    t, f = shape_info
    n = X2.shape[0]
    return X2.reshape(n, t, f)


def js_divergence_from_hist(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)

    hist_range = (
        min(float(np.min(a)), float(np.min(b))),
        max(float(np.max(a)), float(np.max(b))),
    )
    if hist_range[0] == hist_range[1]:
        return 0.0

    pa, _ = np.histogram(a, bins=bins, range=hist_range, density=True)
    pb, _ = np.histogram(b, bins=bins, range=hist_range, density=True)

    pa = pa + 1e-12
    pb = pb + 1e-12
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    m = 0.5 * (pa + pb)

    kl1 = np.sum(pa * np.log(pa / m))
    kl2 = np.sum(pb * np.log(pb / m))
    js = 0.5 * (kl1 + kl2)
    return float(js)


def compute_msdi_score(X_ref: np.ndarray, X_new: np.ndarray) -> float:
    if X_ref.ndim == 3:
        X_ref = X_ref.reshape(X_ref.shape[0], -1)
    if X_new.ndim == 3:
        X_new = X_new.reshape(X_new.shape[0], -1)

    ref_mean = np.mean(X_ref, axis=0)
    new_mean = np.mean(X_new, axis=0)
    ref_std = np.std(X_ref, axis=0) + 1e-6

    z_shift = np.mean(np.abs((new_mean - ref_mean) / ref_std))
    js = js_divergence_from_hist(ref_mean, new_mean, bins=20)

    z_score = 1.0 - np.exp(-z_shift / 3.0)
    js_score = min(1.0, js * 5.0)

    msdi = 0.7 * z_score + 0.3 * js_score
    return float(np.clip(msdi, 0.0, 1.0))


def severity_from_msdi(msdi_score: float) -> str:
    if msdi_score > 0.35:
        return "severe"
    if msdi_score > 0.18:
        return "moderate"
    if msdi_score > 0.10:
        return "mild"
    return "none"


def fit_model(model: LSTMIDModel, X_train, y_train):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train if len(np.unique(y_train)) > 1 else None,
    )
    model.train(X_tr, y_tr, X_val, y_val, epochs=5, batch_size=256)


def bootstrap_sample(X: np.ndarray, y: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), size=len(X))
    return X[idx], y[idx]


def predict_probs(model: LSTMIDModel, X: np.ndarray) -> np.ndarray:
    probs = np.asarray(model.predict(X))
    if probs.ndim == 1:
        probs = np.stack([1.0 - probs, probs], axis=1)
    if probs.ndim == 2 and probs.shape[1] == 1:
        probs = np.concatenate([1.0 - probs, probs], axis=1)
    return probs


def metrics_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted"))
    return acc, f1


def update_weights(old_w, perf_scores, severity, perf_histories):
    perf_scores = np.asarray(perf_scores, dtype=float)

    hist_scores = []
    for i, s in enumerate(perf_scores):
        hist = perf_histories[i]
        if len(hist) == 0:
            hist_scores.append(float(s))
        else:
            hist_scores.append(0.7 * float(s) + 0.3 * float(np.mean(hist[-3:])))
    hist_scores = np.asarray(hist_scores, dtype=float)

    target = hist_scores + 1e-8
    target = target / target.sum()

    if severity == "severe":
        eta = 0.70
    elif severity == "moderate":
        eta = 0.50
    elif severity == "mild":
        eta = 0.30
    else:
        eta = 0.10

    new_w = (1.0 - eta) * np.asarray(old_w, dtype=float) + eta * target
    new_w = np.clip(new_w, 1e-6, None)
    new_w = new_w / new_w.sum()

    delta = new_w - np.asarray(old_w, dtype=float)
    delta_l1 = float(np.sum(np.abs(delta)))
    delta_l2 = float(np.sqrt(np.sum(delta ** 2)))
    return new_w.tolist(), delta_l1, delta_l2


def main():
    # ===== initial data =====
    X_init_list, y_init_list = [], []
    for seg in INITIAL_TRAIN_SEGMENTS:
        X, y = load_xy(seg)
        X = ensure_3d(X)
        X_init_list.append(X)
        y_init_list.append(y)

    X_init = np.vstack(X_init_list)
    y_init = np.concatenate(y_init_list)

    X_init_2d, shape_info = flatten_3d_to_2d(X_init)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_init_2d = imputer.fit_transform(X_init_2d)
    X_init_2d = scaler.fit_transform(X_init_2d)
    X_init_3d = restore_2d_to_3d(X_init_2d, shape_info)

    # ===== build 3 base models =====
    models = []
    base_seeds = [11, 22, 33]
    for seed in base_seeds:
        set_seed(seed)
        X_boot, y_boot = bootstrap_sample(X_init_3d, y_init, seed=seed)
        model = LSTMIDModel(input_shape=X_boot.shape[1:], num_classes=2)
        model.build_adaptive_model()
        fit_model(model, X_boot, y_boot)
        models.append(model)

    weights = [1 / 3, 1 / 3, 1 / 3]
    perf_histories = [[] for _ in range(3)]
    X_ref_2d = X_init_2d.copy()
    trace_records = []

    for chunk_id, seg in enumerate(ALL_SEGMENTS):
        X_raw, y = load_xy(seg)
        X_raw = ensure_3d(X_raw)
        X_2d, _ = flatten_3d_to_2d(X_raw)
        X_2d = imputer.transform(X_2d)
        X_2d = scaler.transform(X_2d)
        X_3d = restore_2d_to_3d(X_2d, shape_info)

        msdi_score = compute_msdi_score(X_ref_2d, X_2d)
        severity = severity_from_msdi(msdi_score)
        drift_detected = severity != "none"

        weights_before = list(weights)

        per_model_probs = []
        per_model_acc = []
        per_model_f1 = []

        for i, model in enumerate(models):
            probs = predict_probs(model, X_3d)
            pred = np.argmax(probs, axis=1).astype(int)
            acc, wf1 = metrics_from_pred(y, pred)

            per_model_probs.append(probs)
            per_model_acc.append(round(acc, 4))
            per_model_f1.append(round(wf1, 4))
            perf_histories[i].append(wf1)

        ensemble_probs = np.zeros_like(per_model_probs[0], dtype=float)
        for w, p in zip(weights_before, per_model_probs):
            ensemble_probs += w * p
        ensemble_pred = np.argmax(ensemble_probs, axis=1).astype(int)

        ensemble_acc = float(accuracy_score(y, ensemble_pred))
        ensemble_weighted_f1 = float(f1_score(y, ensemble_pred, average="weighted"))
        ensemble_macro_f1 = float(f1_score(y, ensemble_pred, average="macro"))

        weights_after, delta_l1, delta_l2 = update_weights(
            old_w=weights_before,
            perf_scores=per_model_f1,
            severity=severity,
            perf_histories=perf_histories,
        )

        dominant_before = int(np.argmax(weights_before))
        dominant_after = int(np.argmax(weights_after))

        record = {
            "chunk_id": chunk_id,
            "segment": seg.replace(".pcap_ISCX", ""),
            "msdi_score": round(msdi_score, 4),
            "drift_detected": bool(drift_detected),
            "drift_severity": severity,
            "weights_before": [round(w, 4) for w in weights_before],
            "weights_after": [round(w, 4) for w in weights_after],
            "weight_delta_l1": round(delta_l1, 4),
            "weight_delta_l2": round(delta_l2, 4),
            "dominant_model_before": dominant_before,
            "dominant_model_after": dominant_after,
            "per_model_accuracy": per_model_acc,
            "per_model_f1": per_model_f1,
            "ensemble_accuracy": round(ensemble_acc, 4),
            "ensemble_weighted_f1": round(ensemble_weighted_f1, 4),
            "ensemble_macro_f1": round(ensemble_macro_f1, 4),
        }
        trace_records.append(record)

        weights = list(weights_after)

        if seg in EVAL_SEGMENTS:
            for model in models:
                model.adaptive_update(X_3d, y)

    OUTPUT_TRACE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_TRACE.open("w", encoding="utf-8") as f:
        json.dump({"chunks": trace_records}, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {OUTPUT_TRACE}")


if __name__ == "__main__":
    main()
