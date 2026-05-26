#!/usr/bin/env python3
"""
Component-wise ablation runner for ADAWU-IDS.

This file is intentionally self-contained, so it can be dropped into the current
project as pipelines/run_ablation.py and run without modifying the existing
ADAWU implementation.

It evaluates five variants:
1) full_adawu_ids: MSDI + ADAWU dynamic weighting + hierarchical response
2) w_o_msdi: removes MSDI and uses only KS-style distribution drift signal
3) w_o_dynamic_weighting: keeps drift detection/response but uses uniform weights
4) w_o_hierarchical_response: keeps MSDI and dynamic weights, but uses one mild update only
5) static_lstm_or_static_sgd: no drift detection, no dynamic weighting, no adaptation

Expected processed data format:
- one or more *_X.npy and matching *_y.npy files in --data-dir, or
- a single X.npy and y.npy pair.
X may be 2D [n_samples, n_features] or 3D [n_samples, timesteps, features].
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


VARIANTS = [
    "full_adawu_ids",
    "w_o_msdi",
    "w_o_dynamic_weighting",
    "w_o_hierarchical_response",
    "static_lstm_or_static_sgd",
]


@dataclass
class ChunkRecord:
    variant: str
    dataset: str
    seed: int
    chunk_id: int
    start: int
    end: int
    accuracy: float
    weighted_f1: float
    drift_detected: bool
    drift_confidence: float
    msdi_score: float
    ks_score: float
    response: str
    weights: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing processed *_X.npy/*_y.npy files.")
    parser.add_argument("--dataset", type=str, default="CICIDS2017")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--reference-chunks", type=int, default=2)
    parser.add_argument("--initial-train-chunks", type=int, default=3)
    parser.add_argument("--variant", choices=VARIANTS + ["all"], default="all")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "results" / "traces" / "ablations"))
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for quick debugging. 0 means use all samples.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def find_xy_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    if (data_dir / "X.npy").exists() and (data_dir / "y.npy").exists():
        return [(data_dir / "X.npy", data_dir / "y.npy")]
    pairs = []
    for x_path in sorted(data_dir.glob("*_X.npy")):
        y_path = data_dir / x_path.name.replace("_X.npy", "_y.npy")
        if y_path.exists():
            pairs.append((x_path, y_path))
    if not pairs:
        raise FileNotFoundError(f"No X/y npy pairs found in {data_dir}")
    return pairs


def load_stream(data_dir: Path, max_samples: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, segs = [], [], []
    for x_path, y_path in find_xy_pairs(data_dir):
        X = np.asarray(np.load(x_path, allow_pickle=True), dtype=np.float32)
        y = np.asarray(np.load(y_path, allow_pickle=True)).reshape(-1).astype(int)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        elif X.ndim != 2:
            raise ValueError(f"Unsupported X shape {X.shape} in {x_path}")
        n = min(len(X), len(y))
        X, y = X[:n], y[:n]
        xs.append(X)
        ys.append(y)
        segs.extend([x_path.stem.replace("_X", "")] * n)
    X_all = np.vstack(xs)
    y_all = np.concatenate(ys)
    seg_all = np.asarray(segs)
    if max_samples and max_samples > 0:
        X_all, y_all, seg_all = X_all[:max_samples], y_all[:max_samples], seg_all[:max_samples]
    return X_all, y_all, seg_all


def chunk_ranges(n: int, chunk_size: int) -> List[Tuple[int, int]]:
    return [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size) if min(i + chunk_size, n) - i >= max(100, chunk_size // 10)]


def safe_weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def sample_vec(x: np.ndarray, limit: int = 2000) -> np.ndarray:
    if len(x) <= limit:
        return x
    idx = np.random.choice(len(x), limit, replace=False)
    return x[idx]


def compute_msdi(X_ref: np.ndarray, X_new: np.ndarray) -> float:
    """Feature-wise normalized Wasserstein distance, averaged and clipped to [0, 1]."""
    vals = []
    for j in range(X_ref.shape[1]):
        ref = sample_vec(X_ref[:, j])
        new = sample_vec(X_new[:, j])
        denom = np.std(ref) + 1e-6
        vals.append(min(1.0, wasserstein_distance(ref, new) / denom))
    return float(np.mean(vals)) if vals else 0.0


def compute_ks_score(X_ref: np.ndarray, X_new: np.ndarray, max_features: int = 80) -> float:
    """Returns the proportion of checked features with KS p < 0.01."""
    n_features = X_ref.shape[1]
    feat_idx = np.linspace(0, n_features - 1, min(max_features, n_features), dtype=int)
    hits = 0
    for j in feat_idx:
        ref = sample_vec(X_ref[:, j], 1000)
        new = sample_vec(X_new[:, j], 1000)
        try:
            _, p = ks_2samp(ref, new)
            hits += int(p < 0.01)
        except Exception:
            pass
    return float(hits / max(len(feat_idx), 1))


def new_model(seed: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        learning_rate="optimal",
        class_weight=None,
        random_state=seed,
        max_iter=5,
        tol=None,
    )


def proba_or_score(model: SGDClassifier, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X)
            if p.shape[1] == len(classes):
                return p
        except Exception:
            pass
    pred = model.predict(X)
    out = np.zeros((len(pred), len(classes)), dtype=float)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for i, label in enumerate(pred):
        out[i, class_to_idx.get(label, 0)] = 1.0
    return out


def ensemble_predict(models: List[SGDClassifier], weights: np.ndarray, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    probs = np.zeros((len(X), len(classes)), dtype=float)
    for w, m in zip(weights, models):
        probs += float(w) * proba_or_score(m, X, classes)
    return classes[np.argmax(probs, axis=1)]


def evaluate_model(model: SGDClassifier, X: np.ndarray, y: np.ndarray) -> float:
    try:
        return safe_weighted_f1(y, model.predict(X))
    except Exception:
        return 0.0


def update_adawu_weights(
    models: List[SGDClassifier],
    weights: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    msdi_score: float,
    drift_confidence: float,
) -> np.ndarray:
    perf = np.asarray([evaluate_model(m, X, y) for m in models], dtype=float)
    # Drift-aware sharpening: stronger drift gives more weight to locally strong models.
    temperature = max(0.25, 1.0 - 0.6 * min(1.0, msdi_score + drift_confidence))
    adjusted = np.exp(perf / temperature)
    adjusted = adjusted / max(np.sum(adjusted), 1e-12)
    # Smooth update avoids unstable weight oscillation.
    eta = 0.25 + 0.50 * min(1.0, msdi_score + drift_confidence)
    new_w = (1.0 - eta) * weights + eta * adjusted
    new_w = np.maximum(new_w, 1e-3)
    return new_w / np.sum(new_w)


def normalize_weights(weights: np.ndarray, n_models: int) -> np.ndarray:
    if len(weights) != n_models:
        weights = np.ones(n_models, dtype=float) / n_models
    s = np.sum(weights)
    return weights / s if s > 0 else np.ones(n_models, dtype=float) / n_models


def run_variant(args: argparse.Namespace, variant: str) -> Path:
    set_seed(args.seed)
    X_raw, y, _ = load_stream(Path(args.data_dir), args.max_samples)
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("Ablation requires at least two classes.")

    ranges = chunk_ranges(len(y), args.chunk_size)
    min_chunks = args.initial_train_chunks + args.reference_chunks + 1
    if len(ranges) < min_chunks:
        raise ValueError(f"Not enough chunks: got {len(ranges)}, need at least {min_chunks}.")

    train_end = ranges[args.initial_train_chunks - 1][1]
    ref_start = ranges[max(0, args.initial_train_chunks - args.reference_chunks)][0]
    ref_end = train_end

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_raw[:train_end]))
    X_all = scaler.transform(imputer.transform(X_raw))

    X_ref = X_all[ref_start:ref_end]
    y_train = y[:train_end]

    models = [new_model(args.seed + i) for i in range(3)]
    boot_rng = np.random.default_rng(args.seed)
    for i, model in enumerate(models):
        idx = boot_rng.choice(len(X_train), size=len(X_train), replace=True)
        model.partial_fit(X_train[idx], y_train[idx], classes=classes)
    weights = np.ones(len(models), dtype=float) / len(models)

    records: List[ChunkRecord] = []
    for chunk_id, (start, end) in enumerate(ranges[args.initial_train_chunks:], start=args.initial_train_chunks):
        Xc, yc = X_all[start:end], y[start:end]
        y_pred = ensemble_predict(models, weights, Xc, classes)
        acc = float(accuracy_score(yc, y_pred))
        wf1 = safe_weighted_f1(yc, y_pred)

        msdi_score = compute_msdi(X_ref, Xc)
        ks_score = compute_ks_score(X_ref, Xc)

        if variant == "w_o_msdi":
            drift_conf = ks_score
        elif variant == "static_lstm_or_static_sgd":
            drift_conf = 0.0
        else:
            drift_conf = 0.65 * msdi_score + 0.35 * ks_score

        drift_detected = bool(drift_conf >= 0.30)
        response = "none"

        if variant != "static_lstm_or_static_sgd":
            # Weighting ablation.
            if variant == "w_o_dynamic_weighting":
                weights = np.ones(len(models), dtype=float) / len(models)
            else:
                active_msdi = 0.0 if variant == "w_o_msdi" else msdi_score
                weights = update_adawu_weights(models, weights, Xc, yc, active_msdi, drift_conf)

            # Response ablation.
            if variant == "w_o_hierarchical_response":
                for m in models:
                    m.partial_fit(Xc, yc, classes=classes)
                response = "single_mild_update"
            else:
                if drift_detected and drift_conf >= 0.70:
                    candidate = new_model(args.seed + 1000 + chunk_id)
                    candidate.partial_fit(Xc, yc, classes=classes)
                    models.append(candidate)
                    weights = np.append(weights * 0.70, 0.30)
                    weights = normalize_weights(weights, len(models))
                    response = "severe_add_model"
                elif drift_detected and drift_conf >= 0.45:
                    for m in models:
                        m.partial_fit(Xc, yc, classes=classes)
                    response = "moderate_progressive_update"
                elif drift_detected:
                    best = int(np.argmax(weights))
                    models[best].partial_fit(Xc, yc, classes=classes)
                    response = "mild_best_model_update"
                else:
                    # Small maintenance update on the best model.
                    best = int(np.argmax(weights))
                    sample_n = min(512, len(Xc))
                    idx = boot_rng.choice(len(Xc), size=sample_n, replace=False)
                    models[best].partial_fit(Xc[idx], yc[idx], classes=classes)
                    response = "normal_maintenance"
        weights = normalize_weights(weights, len(models))

        # Update reference window only for adaptive variants, preserving chronological behavior.
        if variant != "static_lstm_or_static_sgd" and drift_detected:
            X_ref = Xc.copy()

        records.append(ChunkRecord(
            variant=variant,
            dataset=args.dataset,
            seed=args.seed,
            chunk_id=chunk_id,
            start=start,
            end=end,
            accuracy=acc,
            weighted_f1=wf1,
            drift_detected=drift_detected,
            drift_confidence=float(drift_conf),
            msdi_score=float(msdi_score),
            ks_score=float(ks_score),
            response=response,
            weights=[float(x) for x in weights],
        ))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_trace_{args.dataset}_{variant}_seed{args.seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in records], f, indent=2)
    print(f"[OK] {variant}: {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    variants = VARIANTS if args.variant == "all" else [args.variant]
    for variant in variants:
        run_variant(args, variant)


if __name__ == "__main__":
    main()
