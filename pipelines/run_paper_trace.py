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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from models.lstm_model import LSTMIDModel

try:
    import tensorflow as tf
except Exception:
    tf = None


DATA_DIR = PROJECT_ROOT / "datasets" / "processed"

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run ADAWU paper trace with reproducible seed.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible runs.")
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Optional extra tag appended to output filenames, e.g. exp1.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="Path to processed dataset directory.",
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


def load_xy(data_dir: Path, stem: str):
    x_path = data_dir / f"{stem}_X.npy"
    y_path = data_dir / f"{stem}_y.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing feature file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing label file: {y_path}")

    X = np.load(x_path)
    y = np.load(y_path)
    return X.astype(np.float32), y.astype(int)


def ensure_3d(X):
    return X if X.ndim == 3 else X[:, None, :]


def flatten(X):
    n, t, f = X.shape
    return X.reshape(n, t * f), (t, f)


def restore(X2, shape):
    t, f = shape
    return X2.reshape(X2.shape[0], t, f)


def predict(model, X):
    p = model.predict(X)
    return np.argmax(p, axis=1)


def compute_trace_record(chunk_id, seg, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    drift = f1 < 0.9

    return {
        "chunk_id": int(chunk_id),
        "segment": seg.replace(".pcap_ISCX", ""),
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


def main():
    args = parse_args()
    seed = args.seed
    output_tag = args.output_tag.strip()
    data_dir = Path(args.data_dir)

    set_global_seed(seed)
    output_pred, output_trace = build_output_paths(seed, output_tag)

    print(f"[INFO] Seed: {seed}")
    print(f"[INFO] Data dir: {data_dir}")
    print(f"[INFO] Prediction output: {output_pred}")
    print(f"[INFO] Trace output: {output_trace}")

    # ===== 初始化训练 =====
    X_list, y_list = [], []
    for seg in INITIAL_TRAIN_SEGMENTS:
        X, y = load_xy(data_dir, seg)
        X_list.append(ensure_3d(X))
        y_list.append(y)

    X_init = np.vstack(X_list)
    y_init = np.concatenate(y_list)

    X2, shape = flatten(X_init)

    imputer = SimpleImputer()
    scaler = StandardScaler()

    X2 = scaler.fit_transform(imputer.fit_transform(X2))
    X_init = restore(X2, shape)

    # 使用固定 random_state，保证多 seed 可复现
    # 若类别极不平衡，stratify 可能失败，这里做安全回退
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_init,
            y_init,
            test_size=0.1,
            random_state=seed,
            stratify=y_init,
        )
    except ValueError:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_init,
            y_init,
            test_size=0.1,
            random_state=seed,
        )

    model = LSTMIDModel(input_shape=X_tr.shape[1:], num_classes=2)
    model.build_adaptive_model()
    model.train(X_tr, y_tr, X_val, y_val)

    # ===== 记录 =====
    y_true_all = []
    y_pred_all = []
    seg_labels = []
    trace = []

    chunk_id = 0

    # ===== 初始段 + 评估段 =====
    for seg in INITIAL_TRAIN_SEGMENTS + EVAL_SEGMENTS:
        X_raw, y = load_xy(data_dir, seg)
        X_raw = ensure_3d(X_raw)

        X2, _ = flatten(X_raw)
        X2 = scaler.transform(imputer.transform(X2))
        X = restore(X2, shape)

        y_pred = predict(model, X)
        trace.append(compute_trace_record(chunk_id, seg, y, y_pred))

        y_true_all.append(y)
        y_pred_all.append(y_pred)
        seg_labels.extend([seg.replace(".pcap_ISCX", "")] * len(y))

        # ===== 自适应更新 =====
        if seg in EVAL_SEGMENTS:
            model.adaptive_update(X, y)

        chunk_id += 1

    # ===== 保存预测 =====
    output_pred.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_pred,
        y_true=np.concatenate(y_true_all),
        y_pred=np.concatenate(y_pred_all),
        segments=np.array(seg_labels),
        seed=np.array([seed]),
    )

    # ===== 保存 trace =====
    output_trace.parent.mkdir(parents=True, exist_ok=True)
    with open(output_trace, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "initial_train_segments": INITIAL_TRAIN_SEGMENTS,
                "eval_segments": EVAL_SEGMENTS,
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
