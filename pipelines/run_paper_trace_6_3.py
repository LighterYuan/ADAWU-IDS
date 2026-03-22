from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from models.lstm_model import LSTMIDModel

DATA_DIR = PROJECT_ROOT / "datasets" / "processed"
OUTPUT_PRED = PROJECT_ROOT / "results" / "cases" / "adawu_predictions_6_3.npz"
OUTPUT_TRACE = PROJECT_ROOT / "results" / "traces" / "paper_trace_6_3.json"

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


def predict_labels(model: LSTMIDModel, X: np.ndarray) -> np.ndarray:
    probs = np.asarray(model.predict(X))
    if probs.ndim == 2 and probs.shape[1] > 1:
        return np.argmax(probs, axis=1).astype(int)
    if probs.ndim == 2 and probs.shape[1] == 1:
        return (probs.reshape(-1) >= 0.5).astype(int)
    if probs.ndim == 1:
        uniq = np.unique(probs)
        if np.all(np.isin(uniq, [0, 1])):
            return probs.astype(int)
        return (probs >= 0.5).astype(int)
    raise RuntimeError(f"Unsupported prediction shape: {probs.shape}")


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


def compute_msdi_score(X_ref: np.ndarray, X_new: np.ndarray) -> tuple[float, dict]:
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
    msdi = float(np.clip(msdi, 0.0, 1.0))

    return msdi, {
        "z_shift_score": float(z_score),
        "js_shift_score": float(js_score),
    }


def make_detector_bundle(msdi_score: float, current_f1: float, prev_f1: float | None):
    msdi_vote = msdi_score > 0.12
    msdi_conf = float(np.clip(msdi_score, 0.0, 1.0))

    if prev_f1 is None:
        f1_drop_amount = 0.0
    else:
        f1_drop_amount = max(0.0, prev_f1 - current_f1)

    f1_vote = f1_drop_amount > 0.08
    f1_conf = float(np.clip(f1_drop_amount / 0.25, 0.0, 1.0))

    dist_vote = msdi_score > 0.18
    dist_conf = float(np.clip((msdi_score - 0.05) / 0.35, 0.0, 1.0))

    votes = {
        "msdi": bool(msdi_vote),
        "f1_drop": bool(f1_vote),
        "distribution": bool(dist_vote),
    }

    confidences = {
        "msdi": round(msdi_conf, 4),
        "f1_drop": round(f1_conf, 4),
        "distribution": round(dist_conf, 4),
    }

    positive = sum(votes.values())
    drift_detected = positive >= 2

    if msdi_score > 0.35 or f1_drop_amount > 0.25:
        severity = "severe"
    elif msdi_score > 0.18 or f1_drop_amount > 0.12:
        severity = "moderate"
    elif drift_detected:
        severity = "mild"
    else:
        severity = "none"

    return votes, confidences, drift_detected, severity


def main():
    X_hist_list, y_hist_list = [], []
    for seg in INITIAL_TRAIN_SEGMENTS:
        X, y = load_xy(seg)
        X = ensure_3d(X)
        X_hist_list.append(X)
        y_hist_list.append(y)

    X_init = np.vstack(X_hist_list)
    y_init = np.concatenate(y_hist_list)

    X_init_2d, shape_info = flatten_3d_to_2d(X_init)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_init_2d = imputer.fit_transform(X_init_2d)
    X_init_2d = scaler.fit_transform(X_init_2d)
    X_init_3d = restore_2d_to_3d(X_init_2d, shape_info)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_init_3d,
        y_init,
        test_size=0.1,
        random_state=42,
        stratify=y_init if len(np.unique(y_init)) > 1 else None,
    )

    model = LSTMIDModel(
        input_shape=X_tr.shape[1:],
        num_classes=max(2, len(np.unique(y_init))),
    )
    model.build_adaptive_model()
    model.train(X_tr, y_tr, X_val, y_val, epochs=5, batch_size=256)

    X_ref_2d = X_init_2d.copy()

    y_true_all = []
    y_pred_all = []
    seg_labels_all = []
    trace_records = []

    prev_f1 = None

    for chunk_id, seg in enumerate(ALL_SEGMENTS):
        X_raw, y = load_xy(seg)
        X_raw = ensure_3d(X_raw)

        X_2d, _ = flatten_3d_to_2d(X_raw)
        X_2d = imputer.transform(X_2d)
        X_2d = scaler.transform(X_2d)
        X_3d = restore_2d_to_3d(X_2d, shape_info)

        y_pred = predict_labels(model, X_3d)

        acc = float(accuracy_score(y, y_pred))
        weighted_f1 = float(f1_score(y, y_pred, average="weighted"))
        macro_f1 = float(f1_score(y, y_pred, average="macro"))

        msdi_score, msdi_groups = compute_msdi_score(X_ref_2d, X_2d)
        votes, confidences, drift_detected, severity = make_detector_bundle(
            msdi_score=msdi_score,
            current_f1=weighted_f1,
            prev_f1=prev_f1,
        )

        true_drift = chunk_id >= len(INITIAL_TRAIN_SEGMENTS)

        trace_records.append({
            "chunk_id": chunk_id,
            "segment": seg.replace(".pcap_ISCX", ""),
            "ensemble_accuracy": round(acc, 4),
            "ensemble_weighted_f1": round(weighted_f1, 4),
            "ensemble_macro_f1": round(macro_f1, 4),
            "msdi_score": round(msdi_score, 4),
            "msdi_group_scores": msdi_groups,
            "drift_detected": bool(drift_detected),
            "drift_severity": severity,
            "detector_votes": votes,
            "detector_confidences": confidences,
            "true_drift": bool(true_drift),
        })

        y_true_all.append(y)
        y_pred_all.append(y_pred)
        seg_labels_all.extend([seg.replace(".pcap_ISCX", "")] * len(y))

        if seg in EVAL_SEGMENTS:
            model.adaptive_update(X_3d, y)

        prev_f1 = weighted_f1

    OUTPUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PRED,
        y_true=np.concatenate(y_true_all),
        y_pred=np.concatenate(y_pred_all),
        segments=np.array(seg_labels_all),
    )

    OUTPUT_TRACE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_TRACE.open("w", encoding="utf-8") as f:
        json.dump({"chunks": trace_records}, f, ensure_ascii=False, indent=2)

    print("[OK] predictions saved:", OUTPUT_PRED)
    print("[OK] trace saved:", OUTPUT_TRACE)


if __name__ == "__main__":
    main()
