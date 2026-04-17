from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drift.baselines import build_baseline_ensemble
from models.lstm_model import LSTMIDModel

DATA_DIR = PROJECT_ROOT / "datasets" / "processed"
TRACE_DIR = PROJECT_ROOT / "results" / "traces"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries"

TRAIN_SEGMENTS = [
    "Tuesday-WorkingHours.pcap_ISCX",
    "Wednesday-workingHours.pcap_ISCX",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX",
]
TEST_SEGMENTS = [
    "Tuesday-WorkingHours.pcap_ISCX",
    "Wednesday-workingHours.pcap_ISCX",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX",
    "Friday-WorkingHours-Morning.pcap_ISCX",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX",
]


def load_xy(stem: str) -> Tuple[np.ndarray, np.ndarray]:
    x_path = DATA_DIR / f"{stem}_X.npy"
    y_path = DATA_DIR / f"{stem}_y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing dataset pair for {stem}")
    X = np.asarray(np.load(x_path, allow_pickle=True), dtype=np.float32)
    y = np.asarray(np.load(y_path, allow_pickle=True)).reshape(-1).astype(int)
    return X, y


def ensure_3d(X: np.ndarray) -> np.ndarray:
    if X.ndim == 2:
        return X[:, None, :]
    if X.ndim == 3:
        return X
    raise ValueError(f"Unsupported X ndim: {X.ndim}")


def flatten_3d_to_2d(X: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    X = ensure_3d(X)
    n, t, f = X.shape
    return X.reshape(n, t * f), (t, f)


def restore_2d_to_3d(X2: np.ndarray, shape_info: Tuple[int, int]) -> np.ndarray:
    t, f = shape_info
    return X2.reshape(X2.shape[0], t, f)


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


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    acc = float(np.mean(y_true == y_pred))
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    supports = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)
        supports.append(np.sum(y_true == cls))
    f1s = np.asarray(f1s, dtype=float)
    supports = np.asarray(supports, dtype=float)
    macro_f1 = float(np.mean(f1s)) if len(f1s) else 0.0
    weighted_f1 = float(np.sum(f1s * supports) / (np.sum(supports) + 1e-12)) if len(f1s) else 0.0
    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def build_lstm_factory(input_shape: Tuple[int, ...], num_classes: int):
    def _factory() -> LSTMIDModel:
        model = LSTMIDModel(input_shape=input_shape, num_classes=num_classes)
        model.build_model()
        return model
    return _factory


def split_stream_into_chunks(X: np.ndarray, y: np.ndarray, chunk_size: int):
    for start in range(0, len(X), chunk_size):
        end = min(start + chunk_size, len(X))
        yield start, end, X[start:end], y[start:end]


def prepare_data():
    X_train_list, y_train_list = [], []
    for seg in TRAIN_SEGMENTS:
        X, y = load_xy(seg)
        X_train_list.append(ensure_3d(X))
        y_train_list.append(y)
    X_train_full = np.vstack(X_train_list)
    y_train_full = np.concatenate(y_train_list)

    X_train_2d, shape_info = flatten_3d_to_2d(X_train_full)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_2d = imputer.fit_transform(X_train_2d)
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_train_full = restore_2d_to_3d(X_train_2d, shape_info)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.1,
        random_state=42,
        stratify=y_train_full if len(np.unique(y_train_full)) > 1 else None,
    )
    return X_tr, X_val, y_tr, y_val, imputer, scaler, shape_info


def run_static_baseline(
    model_factory,
    X_tr,
    y_tr,
    X_val,
    y_val,
    imputer,
    scaler,
    shape_info,
):
    model = model_factory()
    model.train(X_tr, y_tr, X_val, y_val, epochs=5, batch_size=256)
    rows = []
    y_true_all = []
    y_pred_all = []
    for seg in TEST_SEGMENTS:
        X, y = load_xy(seg)
        X = ensure_3d(X)
        X_2d, _ = flatten_3d_to_2d(X)
        X_2d = scaler.transform(imputer.transform(X_2d))
        X = restore_2d_to_3d(X_2d, shape_info)
        y_pred = predict_labels(model, X)
        mb = metric_bundle(y, y_pred)
        rows.append({"method": "static_lstm", "chunk_id": seg, **mb, "n_samples": int(len(y))})
        y_true_all.append(y)
        y_pred_all.append(y_pred)
    return rows, np.concatenate(y_true_all), np.concatenate(y_pred_all)


def run_online_baseline(
    method: str,
    model_factory,
    X_tr,
    y_tr,
    X_val,
    y_val,
    imputer,
    scaler,
    shape_info,
    chunk_size: int,
    n_models: int,
):
    ensemble = build_baseline_ensemble(
        method,
        model_factory=model_factory,
        n_models=n_models,
        n_classes=max(2, len(np.unique(y_tr))),
        random_state=42,
    )
    ensemble.initialize_fit(X_tr, y_tr, X_val, y_val, epochs=5, batch_size=256)
    rows = []
    all_true = []
    all_pred = []
    all_seg = []
    for seg in TEST_SEGMENTS:
        X, y = load_xy(seg)
        X = ensure_3d(X)
        X_2d, _ = flatten_3d_to_2d(X)
        X_2d = scaler.transform(imputer.transform(X_2d))
        X = restore_2d_to_3d(X_2d, shape_info)
        for start, end, X_chunk, y_chunk in split_stream_into_chunks(X, y, chunk_size=chunk_size):
            chunk_id = f"{seg}::{start}-{end}"
            y_pred = ensemble.predict(X_chunk)
            mb = metric_bundle(y_chunk, y_pred)
            rows.append({"method": ensemble.method_name, "chunk_id": chunk_id, **mb, "n_samples": int(len(y_chunk))})
            all_true.append(y_chunk)
            all_pred.append(y_pred)
            all_seg.extend([seg] * len(y_chunk))
            ensemble.partial_fit(X_chunk, y_chunk, chunk_id=chunk_id)
    return ensemble, rows, np.concatenate(all_true), np.concatenate(all_pred), np.asarray(all_seg)


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            values = [json.dumps(row.get(col, ""), ensure_ascii=False) if isinstance(row.get(col), (dict, list)) else str(row.get(col, "")) for col in columns]
            f.write(",".join(values) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run static and adaptive baseline comparison.")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--n-models", type=int, default=5)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["static_lstm", "dwm", "online_bagging", "leveraging_bagging"],
    )
    args = parser.parse_args()

    X_tr, X_val, y_tr, y_val, imputer, scaler, shape_info = prepare_data()
    model_factory = build_lstm_factory(
        input_shape=X_tr.shape[1:],
        num_classes=max(2, len(np.unique(y_tr))),
    )

    summary_rows = []
    chunk_rows = []

    for method in args.methods:
        start_time = time.perf_counter()
        if method == "static_lstm":
            rows, y_true, y_pred = run_static_baseline(
                model_factory, X_tr, y_tr, X_val, y_val, imputer, scaler, shape_info
            )
            chunk_rows.extend(rows)
            summary = metric_bundle(y_true, y_pred)
            summary_rows.append({
                "method": "static_lstm",
                **summary,
                "elapsed_sec": round(time.perf_counter() - start_time, 4),
                "n_samples": int(len(y_true)),
            })
            continue

        ensemble, rows, y_true, y_pred, segments = run_online_baseline(
            method=method,
            model_factory=model_factory,
            X_tr=X_tr,
            y_tr=y_tr,
            X_val=X_val,
            y_val=y_val,
            imputer=imputer,
            scaler=scaler,
            shape_info=shape_info,
            chunk_size=args.chunk_size,
            n_models=args.n_models,
        )
        chunk_rows.extend(rows)
        ensemble.save_trace(TRACE_DIR / f"{ensemble.method_name}_trace.jsonl")
        summary = metric_bundle(y_true, y_pred)
        summary_rows.append({
            "method": ensemble.method_name,
            **summary,
            "elapsed_sec": round(time.perf_counter() - start_time, 4),
            "n_samples": int(len(y_true)),
            "final_weights": json.dumps([round(w, 6) for w in ensemble.weights.tolist()]),
        })

    save_csv(chunk_rows, TABLE_DIR / "baseline_chunk_metrics.csv")
    save_csv(summary_rows, SUMMARY_DIR / "baseline_summary.csv")
    print("[OK] baseline comparison finished")
    print(f"[OK] summary -> {SUMMARY_DIR / 'baseline_summary.csv'}")
    print(f"[OK] chunk metrics -> {TABLE_DIR / 'baseline_chunk_metrics.csv'}")
    print(f"[OK] traces -> {TRACE_DIR}")


if __name__ == "__main__":
    main()
