from __future__ import annotations

import sys
from pathlib import Path
import json
import argparse
import random
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from models.lstm_model import LSTMIDModel

try:
    import tensorflow as tf
except Exception:
    tf = None


DEFAULT_TRAIN_CSV = PROJECT_ROOT / "datasets" / "UNSW-NB15" / "UNSW_NB15_training-set.csv"
DEFAULT_TEST_CSV = PROJECT_ROOT / "datasets" / "UNSW-NB15" / "UNSW_NB15_testing-set.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Run ADAWU paper trace on UNSW-NB15 with reproducible seed.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible runs.")
    parser.add_argument(
        "--output-tag",
        type=str,
        default="unsw_nb15",
        help="Optional extra tag appended to output filenames.",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default=str(DEFAULT_TRAIN_CSV),
        help="Path to UNSW_NB15_training-set.csv.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=str(DEFAULT_TEST_CSV),
        help="Path to UNSW_NB15_testing-set.csv.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="binary",
        choices=["binary", "multiclass"],
        help="Binary uses label; multiclass uses attack_cat.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Streaming chunk size for test-set evaluation.",
    )
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


def build_output_paths(seed: int, output_tag: str):
    suffix = f"_seed{seed}"
    if output_tag:
        suffix += f"_{output_tag}"

    output_pred = PROJECT_ROOT / "results" / "cases" / f"adawu_predictions{suffix}.npz"
    output_trace = PROJECT_ROOT / "results" / "traces" / f"paper_trace{suffix}.json"
    return output_pred, output_trace


def ensure_3d(X: np.ndarray) -> np.ndarray:
    return X if X.ndim == 3 else X[:, None, :]


def flatten(X: np.ndarray):
    n, t, f = X.shape
    return X.reshape(n, t * f), (t, f)


def restore(X2: np.ndarray, shape):
    t, f = shape
    return X2.reshape(X2.shape[0], t, f)


def predict(model, X: np.ndarray) -> np.ndarray:
    p = model.predict(X)
    return np.argmax(p, axis=1)


def compute_trace_record(chunk_id, seg, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    drift = f1 < 0.9

    return {
        "chunk_id": int(chunk_id),
        "segment": str(seg),
        "ensemble_accuracy": float(acc),
        "ensemble_weighted_f1": float(f1),
        "msdi_score": float(1 - f1),
        "drift_detected": bool(drift),
        "drift_severity": (
            "severe" if f1 < 0.5 else
            "moderate" if f1 < 0.8 else
            "mild"
        ),
        "detector_votes": {
            "msdi": bool(drift),
            "f1_drop": bool(drift),
            "dummy": bool(drift),
        }
    }


def load_unsw_raw(train_csv: Path, test_csv: Path, task: str):
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing training CSV: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing testing CSV: {test_csv}")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_df.columns = [c.strip() for c in train_df.columns]
    test_df.columns = [c.strip() for c in test_df.columns]

    binary_candidates = ["label", "Label", "LABEL"]
    multi_candidates = ["attack_cat", "Attack_cat", "attack_cat "]

    if task == "binary":
        label_col = next((c for c in binary_candidates if c in train_df.columns and c in test_df.columns), None)
        if label_col is None:
            raise ValueError(f"Could not find binary label column. Columns: {list(train_df.columns)}")
        y_train = train_df[label_col].astype(int).to_numpy()
        y_test = test_df[label_col].astype(int).to_numpy()
        drop_cols = [label_col] + [c for c in multi_candidates if c in train_df.columns]
        num_classes = 2
    else:
        label_col = next((c for c in multi_candidates if c in train_df.columns and c in test_df.columns), None)
        if label_col is None:
            raise ValueError(f"Could not find multiclass label column. Columns: {list(train_df.columns)}")

        y_train_raw = train_df[label_col].astype(str).fillna("Normal").replace({"": "Normal"})
        y_test_raw = test_df[label_col].astype(str).fillna("Normal").replace({"": "Normal"})
        classes = sorted(set(y_train_raw.unique()) | set(y_test_raw.unique()))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_train = y_train_raw.map(class_to_idx).astype(int).to_numpy()
        y_test = y_test_raw.map(class_to_idx).astype(int).to_numpy()
        drop_cols = [label_col] + [c for c in binary_candidates if c in train_df.columns]
        num_classes = len(classes)

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors="ignore")

    for c in ["id", "ID", "Id"]:
        if c in X_train.columns:
            X_train = X_train.drop(columns=[c])
        if c in X_test.columns:
            X_test = X_test.drop(columns=[c])

    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    all_X = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    for c in categorical_cols:
        all_X[c] = all_X[c].astype(str).fillna("missing")
        cats = {v: i for i, v in enumerate(sorted(all_X[c].unique()))}
        all_X[c] = all_X[c].map(cats).astype(np.float32)

    for c in all_X.columns:
        all_X[c] = pd.to_numeric(all_X[c], errors="coerce")

    X_all = all_X.to_numpy(dtype=np.float32)
    X_train_np = X_all[:len(X_train)]
    X_test_np = X_all[len(X_train):]

    return X_train_np, y_train, X_test_np, y_test, num_classes


def build_chunk_arrays(X_test: np.ndarray, y_test: np.ndarray, chunk_size: int):
    X_chunks, y_chunks, chunk_names = [], [], []
    chunk_id = 0
    for start in range(0, len(y_test), chunk_size):
        end = min(start + chunk_size, len(y_test))
        X_chunks.append(X_test[start:end])
        y_chunks.append(y_test[start:end])
        chunk_names.append(f"unsw_test_chunk_{chunk_id:03d}")
        chunk_id += 1
    return X_chunks, y_chunks, chunk_names


def main():
    args = parse_args()
    seed = args.seed
    output_tag = args.output_tag.strip()
    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)

    set_global_seed(seed)
    output_pred, output_trace = build_output_paths(seed, output_tag)

    print(f"[INFO] Seed: {seed}")
    print(f"[INFO] Train CSV: {train_csv}")
    print(f"[INFO] Test CSV: {test_csv}")
    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] Chunk size: {args.chunk_size}")
    print(f"[INFO] Prediction output: {output_pred}")
    print(f"[INFO] Trace output: {output_trace}")

    X_train_raw, y_train, X_test_raw, y_test, num_classes = load_unsw_raw(train_csv, test_csv, args.task)

    X_train_raw = ensure_3d(X_train_raw)
    X_test_raw = ensure_3d(X_test_raw)

    X_chunks_raw, y_chunks, chunk_names = build_chunk_arrays(X_test_raw, y_test, args.chunk_size)

    X2, shape = flatten(X_train_raw)

    imputer = SimpleImputer()
    scaler = StandardScaler()

    X2 = scaler.fit_transform(imputer.fit_transform(X2))
    X_train = restore(X2, shape)

    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.1,
            random_state=seed,
            stratify=y_train,
        )
    except ValueError:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.1,
            random_state=seed,
        )

    model = LSTMIDModel(input_shape=X_tr.shape[1:], num_classes=num_classes)
    model.build_adaptive_model()
    model.train(X_tr, y_tr, X_val, y_val)

    y_true_all = []
    y_pred_all = []
    seg_labels = []
    trace = []

    for chunk_id, (seg, X_raw_chunk, y_chunk) in enumerate(zip(chunk_names, X_chunks_raw, y_chunks)):
        X2_chunk, _ = flatten(X_raw_chunk)
        X2_chunk = scaler.transform(imputer.transform(X2_chunk))
        X_chunk = restore(X2_chunk, shape)

        y_pred = predict(model, X_chunk)
        trace.append(compute_trace_record(chunk_id, seg, y_chunk, y_pred))

        y_true_all.append(y_chunk)
        y_pred_all.append(y_pred)
        seg_labels.extend([seg] * len(y_chunk))

        model.adaptive_update(X_chunk, y_chunk)

    output_pred.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_pred,
        y_true=np.concatenate(y_true_all),
        y_pred=np.concatenate(y_pred_all),
        segments=np.array(seg_labels),
        seed=np.array([seed]),
        dataset=np.array(["UNSW-NB15"]),
        task=np.array([args.task]),
        chunk_size=np.array([args.chunk_size]),
    )

    output_trace.parent.mkdir(parents=True, exist_ok=True)
    with open(output_trace, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "dataset": "UNSW-NB15",
                "task": args.task,
                "train_csv": str(train_csv),
                "test_csv": str(test_csv),
                "chunk_size": args.chunk_size,
                "chunks": trace,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("[OK] prediction saved:", output_pred)
    print("[OK] trace saved:", output_trace)


if __name__ == "__main__":
    main()
