#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def ci95(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(1.96 * arr.std(ddof=1) / math.sqrt(arr.size))


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    return pd.Series(values).rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_unsw(train_csv: Path, test_csv: Path, task: str):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_df.columns = [c.strip() for c in train_df.columns]
    test_df.columns = [c.strip() for c in test_df.columns]

    binary_candidates = ["label", "Label", "LABEL"]
    multi_candidates = ["attack_cat", "Attack_cat", "attack_cat "]

    if task == "binary":
        label_col = next((c for c in binary_candidates if c in train_df.columns and c in test_df.columns), None)
        if label_col is None:
            raise ValueError(f"Binary label column not found. Columns: {list(train_df.columns)}")
        y_train = train_df[label_col].copy()
        y_test = test_df[label_col].copy()
        drop_cols = [label_col] + [c for c in multi_candidates if c in train_df.columns]
    else:
        label_col = next((c for c in multi_candidates if c in train_df.columns and c in test_df.columns), None)
        if label_col is None:
            raise ValueError(f"Multiclass label column not found. Columns: {list(train_df.columns)}")
        y_train = train_df[label_col].astype(str).fillna("Normal").replace({"": "Normal"})
        y_test = test_df[label_col].astype(str).fillna("Normal").replace({"": "Normal"})
        drop_cols = [label_col] + [c for c in binary_candidates if c in train_df.columns]

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    for c in ["id", "ID", "Id"]:
        if c in X_train.columns:
            X_train = X_train.drop(columns=[c])
        if c in X_test.columns:
            X_test = X_test.drop(columns=[c])
    return X_train, y_train, X_test, y_test


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)], remainder="drop")


def encode_labels(y_train: pd.Series, y_test: pd.Series):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train.astype(str))
    y_test_enc = le.transform(y_test.astype(str))
    classes = np.arange(len(le.classes_), dtype=int)
    return y_train_enc, y_test_enc, classes, le.classes_.tolist()


def iter_chunks(X: np.ndarray, y: np.ndarray, chunk_size: int):
    chunk_id = 0
    for start in range(0, len(y), chunk_size):
        end = min(start + chunk_size, len(y))
        yield chunk_id, X[start:end], y[start:end]
        chunk_id += 1


class DWM:
    def __init__(self, classes: np.ndarray, random_state: int = 42, beta: float = 0.5, theta: float = 0.01):
        self.classes = classes
        self.beta = beta
        self.theta = theta
        self.rng = np.random.RandomState(random_state)
        self.experts = []
        self.weights = []
        self._add_expert(random_state)

    def _new_expert(self, seed: int):
        return SGDClassifier(loss="log_loss", alpha=1e-4, random_state=seed, max_iter=1, tol=None)

    def _add_expert(self, seed: int):
        self.experts.append(self._new_expert(seed))
        self.weights.append(1.0)

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        if len(y) == 0:
            return
        for i, clf in enumerate(self.experts):
            if not hasattr(clf, "classes_"):
                clf.partial_fit(X[:1], y[:1], classes=self.classes)

        preds = [clf.predict(X) for clf in self.experts]
        for i, pred in enumerate(preds):
            err = float(np.mean(pred != y))
            self.weights[i] *= self.beta ** err

        alive = [(clf, w) for clf, w in zip(self.experts, self.weights) if w >= self.theta]
        if not alive:
            alive = [(self._new_expert(self.rng.randint(0, 1_000_000)), 1.0)]
        self.experts = [x[0] for x in alive]
        self.weights = [x[1] for x in alive]

        ensemble_err = float(np.mean(self.predict(X) != y))
        if ensemble_err > 0.5:
            self._add_expert(self.rng.randint(0, 1_000_000))
            self.experts[-1].partial_fit(X, y, classes=self.classes)

        for clf in self.experts:
            clf.partial_fit(X, y)
        s = sum(self.weights)
        if s > 0:
            self.weights = [w / s for w in self.weights]

    def predict(self, X: np.ndarray) -> np.ndarray:
        votes = np.zeros((X.shape[0], len(self.classes)), dtype=float)
        for clf, w in zip(self.experts, self.weights):
            if not hasattr(clf, "classes_"):
                continue
            pred = clf.predict(X)
            votes[np.arange(X.shape[0]), pred] += w
        return votes.argmax(axis=1)


class OnlineBagging:
    def __init__(self, n_estimators: int, classes: np.ndarray, random_state: int = 42, lam: float = 1.0):
        self.classes = classes
        self.lam = lam
        self.rng = np.random.RandomState(random_state)
        self.models = [
            SGDClassifier(loss="log_loss", alpha=1e-4, random_state=random_state + i, max_iter=1, tol=None)
            for i in range(n_estimators)
        ]

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        for clf in self.models:
            if not hasattr(clf, "classes_"):
                clf.partial_fit(X[:1], y[:1], classes=self.classes)
        for clf in self.models:
            k = self.rng.poisson(self.lam, size=len(y))
            idx = np.repeat(np.arange(len(y)), k)
            if idx.size:
                clf.partial_fit(X[idx], y[idx])

    def predict(self, X: np.ndarray) -> np.ndarray:
        votes = np.zeros((X.shape[0], len(self.classes)), dtype=float)
        for clf in self.models:
            if not hasattr(clf, "classes_"):
                continue
            pred = clf.predict(X)
            votes[np.arange(X.shape[0]), pred] += 1.0
        return votes.argmax(axis=1)


class LeveragingBagging(OnlineBagging):
    def __init__(self, n_estimators: int, classes: np.ndarray, random_state: int = 42):
        super().__init__(n_estimators=n_estimators, classes=classes, random_state=random_state, lam=6.0)


@dataclass
class RunResult:
    method: str
    seed: int
    trace: pd.DataFrame
    summary: Dict[str, float]


def evaluate_streaming_method(method_name: str, model, X_train, y_train, X_test, y_test, classes, chunk_size, pretrain_batch=10000):
    if method_name == "Static MLP":
        model.fit(X_train, y_train)
    else:
        for start in range(0, len(y_train), pretrain_batch):
            end = min(start + pretrain_batch, len(y_train))
            model.partial_fit(X_train[start:end], y_train[start:end])

    rows = []
    all_true, all_pred = [], []
    for chunk_id, X_chunk, y_chunk in iter_chunks(X_test, y_test, chunk_size):
        y_pred = model.predict(X_chunk)
        acc = accuracy_score(y_chunk, y_pred)
        wf1 = f1_score(y_chunk, y_pred, average="weighted", zero_division=0)
        mf1 = f1_score(y_chunk, y_pred, average="macro", zero_division=0)
        prec, rec, _, _ = precision_recall_fscore_support(y_chunk, y_pred, average="macro", zero_division=0)
        rows.append({
            "chunk_id": chunk_id,
            "accuracy": float(acc),
            "weighted_f1": float(wf1),
            "macro_f1": float(mf1),
            "macro_precision": float(prec),
            "macro_recall": float(rec),
        })
        all_true.append(y_chunk)
        all_pred.append(y_pred)
        if method_name != "Static MLP":
            model.partial_fit(X_chunk, y_chunk)

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    summary = {
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "weighted_f1": float(f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
        "n_chunks": int(len(rows)),
    }
    return RunResult(method=method_name, seed=-1, trace=pd.DataFrame(rows), summary=summary)


def run_one_seed(seed, X_train, y_train, X_test, y_test, classes, chunk_size, hidden_size):
    results = []
    static_model = MLPClassifier(
        hidden_layer_sizes=(hidden_size, max(32, hidden_size // 2)),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=1024,
        learning_rate_init=1e-3,
        max_iter=20,
        random_state=seed,
    )
    for name, model in [
        ("Static MLP", static_model),
        ("DWM", DWM(classes=classes, random_state=seed)),
        ("Online Bagging", OnlineBagging(n_estimators=10, classes=classes, random_state=seed)),
        ("Leveraging Bagging", LeveragingBagging(n_estimators=10, classes=classes, random_state=seed)),
    ]:
        rr = evaluate_streaming_method(name, model, X_train, y_train, X_test, y_test, classes, chunk_size)
        rr.seed = seed
        results.append(rr)
    return results


def aggregate_results(run_results):
    summary_rows = []
    for rr in run_results:
        row = {"method": rr.method, "seed": rr.seed}
        row.update(rr.summary)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    agg_rows = []
    for method, sdf in summary_df.groupby("method"):
        row = {"method": method, "n_seeds": int(len(sdf))}
        for metric in ["accuracy", "weighted_f1", "macro_f1"]:
            vals = sdf[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_ci95"] = ci95(vals)
        agg_rows.append(row)
    return summary_df, pd.DataFrame(agg_rows).sort_values("method").reset_index(drop=True)


def build_temporal_agg(run_results):
    frames = []
    for rr in run_results:
        df = rr.trace.copy()
        df["method"] = rr.method
        df["seed"] = rr.seed
        frames.append(df)
    long_df = pd.concat(frames, ignore_index=True)
    rows = []
    for method, sdf in long_df.groupby("method"):
        agg = sdf.groupby("chunk_id")["weighted_f1"].agg(["mean", "std", "count"]).reset_index()
        agg["ci95"] = [0.0 if c <= 1 or pd.isna(s) else 1.96 * float(s) / math.sqrt(int(c)) for s, c in zip(agg["std"], agg["count"])]
        agg["method"] = method
        rows.append(agg)
    return long_df, pd.concat(rows, ignore_index=True)


def plot_temporal(temporal_agg, out_path, smooth_window=1):
    order = ["Static MLP", "DWM", "Online Bagging", "Leveraging Bagging"]
    plt.figure(figsize=(12, 6))
    for method in order:
        sdf = temporal_agg[temporal_agg["method"] == method].sort_values("chunk_id")
        if sdf.empty:
            continue
        x = sdf["chunk_id"].to_numpy()
        y = smooth(sdf["mean"].to_numpy(), smooth_window)
        plt.plot(x, y, label=method, linewidth=2.0)
    plt.title("UNSW-NB15 Temporal Weighted F1 Curves")
    plt.xlabel("Chunk ID")
    plt.ylabel("Weighted F1")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_summary(agg_df, out_path):
    order_map = {"Static MLP": 0, "DWM": 1, "Online Bagging": 2, "Leveraging Bagging": 3}
    plot_df = agg_df.copy()
    plot_df["order"] = plot_df["method"].map(order_map)
    plot_df = plot_df.sort_values("order")
    x = np.arange(len(plot_df))
    plt.figure(figsize=(8, 5))
    plt.bar(x, plot_df["weighted_f1_mean"].to_numpy())
    plt.errorbar(x, plot_df["weighted_f1_mean"].to_numpy(), yerr=plot_df["weighted_f1_ci95"].to_numpy(), fmt="none", capsize=4)
    plt.xticks(x, plot_df["method"].tolist(), rotation=18)
    plt.ylabel("Weighted F1")
    plt.ylim(0.0, 1.05)
    plt.title("UNSW-NB15 Overall Weighted F1")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Standalone UNSW-NB15 experiment pipeline.")
    parser.add_argument("--train", type=str, required=True, help="Path to UNSW_NB15_training-set.csv")
    parser.add_argument("--test", type=str, required=True, help="Path to UNSW_NB15_testing-set.csv")
    parser.add_argument("--outdir", type=str, default="results/unsw_nb15")
    parser.add_argument("--task", type=str, default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--smooth-window", type=int, default=1)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    traces_dir = outdir / "traces"
    ensure_dir(outdir)
    ensure_dir(traces_dir)

    X_train_df, y_train_raw, X_test_df, y_test_raw = load_unsw(Path(args.train), Path(args.test), args.task)
    preprocessor = build_preprocessor(X_train_df)
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    y_train, y_test, classes, class_names = encode_labels(y_train_raw, y_test_raw)

    run_results = []
    for seed in args.seeds:
        seed_results = run_one_seed(seed, X_train, y_train, X_test, y_test, classes, args.chunk_size, args.hidden_size)
        for rr in seed_results:
            rr.trace.to_csv(traces_dir / f"{rr.method.lower().replace(' ', '_')}_seed{seed}.csv", index=False)
        run_results.extend(seed_results)

    summary_df, agg_df = aggregate_results(run_results)
    long_df, temporal_agg = build_temporal_agg(run_results)

    summary_df.to_csv(outdir / "summary_per_seed.csv", index=False)
    agg_df.to_csv(outdir / "summary_aggregated.csv", index=False)
    long_df.to_csv(outdir / "temporal_weighted_f1_long.csv", index=False)
    temporal_agg.to_csv(outdir / "temporal_weighted_f1_agg.csv", index=False)

    plot_temporal(temporal_agg, outdir / "temporal_weighted_f1.png", smooth_window=args.smooth_window)
    plot_summary(agg_df, outdir / "overall_weighted_f1.png")

    meta = {
        "train": args.train,
        "test": args.test,
        "task": args.task,
        "chunk_size": args.chunk_size,
        "seeds": args.seeds,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
        "n_classes": int(len(classes)),
        "class_names": class_names,
        "methods": ["Static MLP", "DWM", "Online Bagging", "Leveraging Bagging"],
    }
    (outdir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[OK] UNSW-NB15 experiment completed.")
    print(f"[OK] Outputs saved to: {outdir}")
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
