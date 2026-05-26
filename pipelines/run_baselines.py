from __future__ import annotations

import sys
from pathlib import Path
import argparse
import random
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.lstm_model import LSTMIDModel

try:
    import tensorflow as tf
except Exception:
    tf = None


DATA_DIR = PROJECT_ROOT / "datasets" / "processed"

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run static LSTM baseline with reproducible seeds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible runs.")
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Optional extra tag appended to output filename, e.g. exp1.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="Path to processed dataset directory.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass


def build_output_path(seed: int, output_tag: str) -> Path:
    suffix = f"_seed{seed}"
    if output_tag:
        suffix += f"_{output_tag}"
    return PROJECT_ROOT / "results" / "cases" / f"static_predictions{suffix}.npz"


def load_xy(data_dir: Path, stem: str):
    x_path = data_dir / f"{stem}_X.npy"
    y_path = data_dir / f"{stem}_y.npy"

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
    args = parse_args()
    seed = args.seed
    output_tag = args.output_tag.strip()
    data_dir = Path(args.data_dir)
    output_path = build_output_path(seed, output_tag)

    set_global_seed(seed)

    print(f"[INFO] Seed: {seed}")
    print(f"[INFO] Data dir: {data_dir}")
    print(f"[INFO] Output: {output_path}")

    X_train_list, y_train_list = [], []
    for seg in TRAIN_SEGMENTS:
        X, y = load_xy(data_dir, seg)
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

    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.1,
            random_state=seed,
            stratify=y_train_full if len(np.unique(y_train_full)) > 1 else None,
        )
    except ValueError:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.1,
            random_state=seed,
        )

    model = LSTMIDModel(
        input_shape=X_tr.shape[1:],
        num_classes=max(2, len(np.unique(y_train_full))),
    )
    model.build_model()
    model.train(
        X_tr,
        y_tr,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    y_true_all = []
    y_pred_all = []
    segment_labels_all = []

    for seg in TEST_SEGMENTS:
        X, y = load_xy(data_dir, seg)
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        y_true=y_true_all,
        y_pred=y_pred_all,
        segments=segment_labels_all,
        seed=np.array([seed]),
    )

    print("[OK] baseline predictions saved:", output_path)


if __name__ == "__main__":
    main()
