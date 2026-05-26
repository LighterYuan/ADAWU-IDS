#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNSW-NB15 ablation experiments for processed train/test files.

Variants:
  full_adawu_ids
  w_o_msdi
  w_o_dynamic_weighting
  w_o_hierarchical_response
  static_lstm_or_static_sgd
  all

Run:
  export OMP_NUM_THREADS=1
  python -W ignore pipelines/run_unsw_ablation.py --data-dir datasets/processed --dataset UNSW-NB15 --variant all --seed 42
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

VARIANTS = [
    "full_adawu_ids",
    "w_o_msdi",
    "w_o_dynamic_weighting",
    "w_o_hierarchical_response",
    "static_lstm_or_static_sgd",
]


def set_threads():
    for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[k] = str(os.environ.get(k, "1"))


def find_train_test_files(data_dir, dataset):
    roots = [Path(data_dir) / dataset, Path(data_dir) / dataset.replace("-", "_"), Path(data_dir)]
    train_file, test_file = None, None
    candidates = []

    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            name = p.name.lower()
            if not name.endswith((".csv", ".parquet", ".pkl", ".pickle")):
                continue
            if "unsw" in name or "nb15" in name or "train" in name or "test" in name:
                candidates.append(p)

    for p in candidates:
        name = p.name.lower()
        if ("train" in name or "training" in name) and train_file is None:
            train_file = p
        elif ("test" in name or "testing" in name) and test_file is None:
            test_file = p


    if train_file is None or test_file is None:
        raise FileNotFoundError(
            "Cannot find UNSW-NB15 train/test files. "
            "Expected filenames containing train/test under datasets/processed/UNSW-NB15."
        )

    return str(train_file), str(test_file)


def read_table(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith((".pkl", ".pickle")):
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported file type: {path}")


def choose_label_column(df):
    for c in ["label", "Label", "target", "class", "attack_cat", "Attack_cat"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find label column. Columns: {df.columns.tolist()}")


def load_unsw_stream(data_dir, dataset, label_col=None, max_samples=None, task="binary"):
    train_file, test_file = find_train_test_files(data_dir, dataset)
    print(f"[INFO] train file: {train_file}")
    print(f"[INFO] test file : {test_file}")

    train_df = read_table(train_file)
    test_df = read_table(test_file)

    if label_col is None:
        label_col = "attack_cat" if task == "multiclass" and "attack_cat" in train_df.columns else choose_label_column(train_df)

    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise ValueError(f"label_col={label_col} not found in both train and test files.")

    drop_cols = [label_col]
    for c in ["id", "Id", "ID"]:
        if c in train_df.columns:
            drop_cols.append(c)

    y_train_raw = train_df[label_col].astype(str).values
    y_test_raw = test_df[label_col].astype(str).values

    X_train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    all_X = pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True)
    all_X = all_X.replace([np.inf, -np.inf], np.nan)
    all_X = pd.get_dummies(all_X, dummy_na=True)
    all_X = all_X.fillna(all_X.median(numeric_only=True)).fillna(0)

    n_train = len(X_train_df)
    X_train_df = all_X.iloc[:n_train]
    X_test_df = all_X.iloc[n_train:]

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df).astype(np.float32)
    X_test = scaler.transform(X_test_df).astype(np.float32)

    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    boundary = len(y_train)

    if max_samples is not None and max_samples > 0 and max_samples < len(y):
        train_keep = min(boundary, max_samples // 2)
        test_keep = max_samples - train_keep
        X = np.vstack([X_train[:train_keep], X_test[:test_keep]])
        y = np.concatenate([y_train[:train_keep], y_test[:test_keep]])
        boundary = train_keep

    meta = {
        "train_file": train_file,
        "test_file": test_file,
        "label_col": label_col,
        "classes": le.classes_.tolist(),
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "train_test_boundary": int(boundary),
    }
    return X, y, meta


def make_chunks(X, y, chunk_size):
    chunks = []
    for start in range(0, len(y), chunk_size):
        end = min(start + chunk_size, len(y))
        if end - start >= max(10, chunk_size // 10):
            chunks.append((X[start:end], y[start:end], start, end))
    return chunks


def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def compute_msdi(ref_X, cur_X, max_features=30):
    if len(ref_X) == 0 or len(cur_X) == 0:
        return 0.0

    n_features = ref_X.shape[1]
    idx = np.linspace(0, n_features - 1, min(max_features, n_features)).astype(int)
    ref = ref_X[:, idx]
    cur = cur_X[:, idx]

    ks_vals = []
    for j in range(ref.shape[1]):
        try:
            ks_vals.append(float(ks_2samp(ref[:, j], cur[:, j], method="asymp").statistic))
        except Exception:
            ks_vals.append(0.0)

    eps = 1e-8
    mean_shift = np.mean(np.abs(np.mean(cur, axis=0) - np.mean(ref, axis=0)) / (np.std(ref, axis=0) + eps))
    var_shift = np.mean(np.abs(np.var(cur, axis=0) - np.var(ref, axis=0)) / (np.var(ref, axis=0) + eps))

    return float(0.5 * np.mean(ks_vals) + 0.3 * min(mean_shift, 10.0) / 10.0 + 0.2 * min(var_shift, 10.0) / 10.0)


def get_class_weight(y, classes, enabled=True):
    if not enabled:
        return None

    present_classes = np.unique(y)

    # 如果当前训练窗口没有包含全部类别，则不给 class_weight
    # 避免 compute_class_weight 报错
    if len(present_classes) < len(classes):
        return None

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    return {c: w for c, w in zip(classes, weights)}


def new_model(seed, classes, y_init, dynamic_weighting=True):
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        random_state=seed,
        class_weight=get_class_weight(y_init, classes, dynamic_weighting),
        max_iter=1,
        tol=None,
    )


def run_static(chunks, classes, args):
    X0 = np.vstack([c[0] for c in chunks[:args.initial_train_chunks]])
    y0 = np.concatenate([c[1] for c in chunks[:args.initial_train_chunks]])

    model = new_model(args.seed, classes, y0, dynamic_weighting=False)
    model.partial_fit(X0, y0, classes=classes)

    rows = []
    for t, (Xc, yc, start, end) in enumerate(chunks[args.initial_train_chunks:], start=args.initial_train_chunks):
        yp = model.predict(Xc)
        rows.append({
            "chunk": t, "start": start, "end": end,
            "accuracy": accuracy_score(yc, yp),
            "weighted_f1": weighted_f1(yc, yp),
            "msdi": 0.0, "drift_level": "none", "updated": False,
        })
    return pd.DataFrame(rows)


def run_variant(chunks, classes, args, variant):
    use_msdi = variant != "w_o_msdi"
    use_dynamic_weighting = variant != "w_o_dynamic_weighting"
    use_hierarchical = variant != "w_o_hierarchical_response"

    X0 = np.vstack([c[0] for c in chunks[:args.initial_train_chunks]])
    y0 = np.concatenate([c[1] for c in chunks[:args.initial_train_chunks]])

    model = new_model(args.seed, classes, y0, dynamic_weighting=use_dynamic_weighting)
    model.partial_fit(X0, y0, classes=classes)

    ref_X = np.vstack([c[0] for c in chunks[:args.reference_chunks]])
    recent_X, recent_y = X0.copy(), y0.copy()

    rows = []
    for t, (Xc, yc, start, end) in enumerate(chunks[args.initial_train_chunks:], start=args.initial_train_chunks):
        yp = model.predict(Xc)
        acc = accuracy_score(yc, yp)
        f1 = weighted_f1(yc, yp)

        drift_score = compute_msdi(ref_X, Xc, args.msdi_features) if use_msdi else max(0.0, 1.0 - f1)

        if drift_score >= args.high_drift_threshold:
            drift_level = "high"
        elif drift_score >= args.low_drift_threshold:
            drift_level = "low"
        else:
            drift_level = "none"

        updated = False
        if drift_level != "none":
            if use_hierarchical and drift_level == "high":
                mem_X = np.vstack([recent_X[-args.memory_size:], Xc])
                mem_y = np.concatenate([recent_y[-args.memory_size:], yc])
                model = new_model(args.seed + t, classes, mem_y, dynamic_weighting=use_dynamic_weighting)
                model.partial_fit(mem_X, mem_y, classes=classes)
            else:
                model.partial_fit(Xc, yc)
            updated = True

        recent_X = np.vstack([recent_X, Xc])
        recent_y = np.concatenate([recent_y, yc])
        if len(recent_y) > args.memory_size * 2:
            recent_X = recent_X[-args.memory_size * 2:]
            recent_y = recent_y[-args.memory_size * 2:]

        rows.append({
            "chunk": t, "start": start, "end": end,
            "accuracy": acc, "weighted_f1": f1,
            "msdi": drift_score, "drift_level": drift_level, "updated": updated,
        })

    return pd.DataFrame(rows)


def summarize(df, variant, meta):
    if df.empty:
        return {"variant": variant}

    drift_idx = df.index[df["drift_level"].isin(["low", "high"])].tolist()
    d0 = drift_idx[0] if drift_idx else max(1, len(df) // 2)

    pre = df.iloc[:max(1, d0)]
    post = df.iloc[d0:]

    pre_f1 = float(pre["weighted_f1"].mean()) if len(pre) else float(df["weighted_f1"].iloc[0])
    post_min = float(post["weighted_f1"].min()) if len(post) else float(df["weighted_f1"].min())
    final_f1 = float(df["weighted_f1"].tail(min(3, len(df))).mean())
    relative_drop = float((pre_f1 - post_min) / max(pre_f1, 1e-8))

    recovery_steps = np.nan
    if len(post):
        threshold = 0.95 * pre_f1
        for i, v in enumerate(post["weighted_f1"].values):
            if v >= threshold:
                recovery_steps = i
                break

    return {
        "variant": variant,
        "accuracy": float(df["accuracy"].mean()),
        "weighted_f1": float(df["weighted_f1"].mean()),
        "pre_drift_f1": pre_f1,
        "post_drift_min_f1": post_min,
        "final_window_f1": final_f1,
        "relative_drop": relative_drop,
        "recovery_steps": recovery_steps,
        "n_updates": int(df["updated"].sum()),
        "n_drift_chunks": int(df["drift_level"].isin(["low", "high"]).sum()),
        "n_samples": meta["n_samples"],
        "n_features": meta["n_features"],
        "label_col": meta["label_col"],
    }


def run_one(args, variant):
    X, y, meta = load_unsw_stream(args.data_dir, args.dataset, args.label_col, args.max_samples, args.task)
    classes = np.unique(y)
    chunks = make_chunks(X, y, args.chunk_size)

    if len(chunks) <= args.initial_train_chunks:
        raise ValueError("Too few chunks. Reduce --chunk-size or --initial-train-chunks.")

    print(f"[INFO] variant={variant}, samples={len(y)}, features={X.shape[1]}, classes={len(classes)}, chunks={len(chunks)}")

    if variant == "static_lstm_or_static_sgd":
        df = run_static(chunks, classes, args)
    else:
        df = run_variant(chunks, classes, args, variant)

    out_dir = Path(args.output_dir) / args.dataset / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = out_dir / f"{variant}_chunk_metrics.csv"
    summary_path = out_dir / f"{variant}_summary.json"

    df.to_csv(detail_path, index=False)
    summary = summarize(df, variant, meta)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] saved {detail_path}")
    print(f"[OK] saved {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_combined_summary(args):
    out_dir = Path(args.output_dir) / args.dataset / f"seed_{args.seed}"
    rows = []
    for p in out_dir.glob("*_summary.json"):
        with open(p, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    if rows:
        df = pd.DataFrame(rows)
        order = {v: i for i, v in enumerate(VARIANTS)}
        df["order"] = df["variant"].map(order).fillna(99)
        df = df.sort_values("order").drop(columns=["order"])
        out = out_dir / "ablation_summary.csv"
        df.to_csv(out, index=False)
        print(f"[OK] combined summary saved to {out}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", default="UNSW-NB15")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", default="all", choices=VARIANTS + ["all"])
    parser.add_argument("--output-dir", default="results/ablation")
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--label-col", default=None)

    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--reference-chunks", type=int, default=2)
    parser.add_argument("--initial-train-chunks", type=int, default=2)
    parser.add_argument("--memory-size", type=int, default=20000)
    parser.add_argument("--msdi-features", type=int, default=30)

    parser.add_argument("--low-drift-threshold", type=float, default=0.08)
    parser.add_argument("--high-drift-threshold", type=float, default=0.16)
    return parser.parse_args()


def main():
    set_threads()
    args = parse_args()
    np.random.seed(args.seed)

    variants = VARIANTS if args.variant == "all" else [args.variant]
    for v in variants:
        run_one(args, v)

    build_combined_summary(args)


if __name__ == "__main__":
    main()
