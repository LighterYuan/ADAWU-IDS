from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.lstm_model import LSTMIDModel

DATA_DIR = PROJECT_ROOT / "datasets" / "processed"
OUTPUT = PROJECT_ROOT / "results" / "cases" / "static_predictions.npz"

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


def main():
    X_train_list, y_train_list = [], []
    for seg in TRAIN_SEGMENTS:
        X, y = load_xy(seg)
        X = ensure_3d(X)
        X_train_list.append(X)
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

    model = LSTMIDModel(
        input_shape=X_tr.shape[1:],
        num_classes=max(2, len(np.unique(y_train_full))),
    )
    model.build_model()
    model.train(X_tr, y_tr, X_val, y_val, epochs=5, batch_size=256)

    y_true_all = []
    y_pred_all = []
    segment_labels_all = []

    for seg in TEST_SEGMENTS:
        X, y = load_xy(seg)
        X = ensure_3d(X)

        X_2d, _ = flatten_3d_to_2d(X)
        X_2d = imputer.transform(X_2d)
        X_2d = scaler.transform(X_2d)
        X = restore_2d_to_3d(X_2d, shape_info)

        y_pred = predict_labels(model, X)

        y_true_all.append(y)
        y_pred_all.append(y_pred)
        segment_labels_all.extend([seg.replace(".pcap_ISCX", "")] * len(y))

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    segment_labels_all = np.array(segment_labels_all)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT,
        y_true=y_true_all,
        y_pred=y_pred_all,
        segments=segment_labels_all,
    )

    print("[OK] baseline predictions saved:", OUTPUT)


if __name__ == "__main__":
    main()
