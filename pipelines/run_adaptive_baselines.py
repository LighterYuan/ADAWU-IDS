from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines import DWMEnsemble, OnlineBaggingEnsemble, LeveragingBaggingEnsemble  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_xy(data_dir: Path, stem: str):
    x_path = data_dir / f"{stem}_X.npy"
    y_path = data_dir / f"{stem}_y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing pair for segment: {stem}")
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


def segment_display_name(seg: str) -> str:
    return seg.replace(".pcap_ISCX", "")


def build_base_estimator(cfg: dict, seed: int):
    est_cfg = cfg.get("base_estimator", {})
    name = est_cfg.get("name", "sgd_logistic")
    if name != "sgd_logistic":
        raise ValueError(f"Unsupported base_estimator: {name}")
    return SGDClassifier(
        loss=est_cfg.get("loss", "log_loss"),
        alpha=float(est_cfg.get("alpha", 1e-4)),
        learning_rate=est_cfg.get("learning_rate", "optimal"),
        eta0=float(est_cfg.get("eta0", 0.01)),
        random_state=seed,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def iter_chunks(X: np.ndarray, y: np.ndarray, chunk_size: int):
    for start in range(0, len(y), chunk_size):
        end = min(start + chunk_size, len(y))
        yield start, end, X[start:end], y[start:end]


def build_models(cfg: dict, classes: np.ndarray, seed: int):
    base = build_base_estimator(cfg, seed)
    adaptive_cfg = cfg.get("adaptive_baselines", {})
    models = {}

    if adaptive_cfg.get("dwm", {}).get("enabled", True):
        c = adaptive_cfg["dwm"]
        models["dwm"] = DWMEnsemble(
            base_estimator=base,
            classes=classes,
            beta=float(c.get("beta", 0.5)),
            theta=float(c.get("theta", 0.20)),
            min_weight=float(c.get("min_weight", 0.01)),
            max_experts=int(c.get("max_experts", 16)),
            random_state=seed,
        )

    if adaptive_cfg.get("online_bagging", {}).get("enabled", True):
        c = adaptive_cfg["online_bagging"]
        models["online_bagging"] = OnlineBaggingEnsemble(
            base_estimator=base,
            classes=classes,
            n_estimators=int(c.get("n_estimators", 10)),
            poisson_lambda=float(c.get("poisson_lambda", 1.0)),
            random_state=seed,
        )

    if adaptive_cfg.get("leveraging_bagging", {}).get("enabled", True):
        c = adaptive_cfg["leveraging_bagging"]
        models["leveraging_bagging"] = LeveragingBaggingEnsemble(
            base_estimator=base,
            classes=classes,
            n_estimators=int(c.get("n_estimators", 10)),
            poisson_lambda=float(c.get("poisson_lambda", 6.0)),
            hard_example_boost=float(c.get("hard_example_boost", 2.0)),
            random_state=seed,
        )

    return models


def main():
    parser = argparse.ArgumentParser(description="Run adaptive streaming baselines")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baselines.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    chunk_size = int(cfg.get("chunk_size", 5000))

    paths = cfg.get("paths", {})
    data_dir = PROJECT_ROOT / paths.get("data_dir", "datasets/processed")
    case_dir = PROJECT_ROOT / paths.get("case_dir", "results/cases")
    trace_dir = PROJECT_ROOT / paths.get("trace_dir", "results/traces")
    summary_dir = PROJECT_ROOT / paths.get("summary_dir", "results/summaries")
    case_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    train_segments = cfg["train_segments"]
    test_segments = cfg["test_segments"]

    X_train_list, y_train_list = [], []
    for seg in train_segments:
        X, y = load_xy(data_dir, seg)
        X = ensure_3d(X)
        X_train_list.append(X)
        y_train_list.append(y)

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    classes = np.unique(y_train)

    X_train_2d, shape_info = flatten_3d_to_2d(X_train)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_2d = imputer.fit_transform(X_train_2d)
    X_train_2d = scaler.fit_transform(X_train_2d)

    models = build_models(cfg, classes, seed)
    traces = {name: [] for name in models}
    y_true_all = {name: [] for name in models}
    y_pred_all = {name: [] for name in models}
    seg_all = {name: [] for name in models}
    chunk_all = {name: [] for name in models}

    for name, model in models.items():
        model.partial_fit(X_train_2d, y_train)

    global_chunk = 0
    for seg in test_segments:
        X, y = load_xy(data_dir, seg)
        X = ensure_3d(X)
        X2, _ = flatten_3d_to_2d(X)
        X2 = imputer.transform(X2)
        X2 = scaler.transform(X2)

        for start, end, X_chunk, y_chunk in iter_chunks(X2, y, chunk_size):
            seg_name = segment_display_name(seg)
            for name, model in models.items():
                pred = model.predict(X_chunk)
                metrics = compute_metrics(y_chunk, pred)
                traces[name].append(
                    {
                        "seed": seed,
                        "chunk_id": global_chunk,
                        "segment": seg_name,
                        "start": int(start),
                        "end": int(end),
                        **metrics,
                    }
                )
                y_true_all[name].append(y_chunk)
                y_pred_all[name].append(pred)
                seg_all[name].extend([seg_name] * len(y_chunk))
                chunk_all[name].extend([global_chunk] * len(y_chunk))
                model.partial_fit(X_chunk, y_chunk)
            global_chunk += 1

    summary = {"seed": seed, "chunk_size": chunk_size, "methods": {}}
    for name in models:
        y_true = np.concatenate(y_true_all[name])
        y_pred = np.concatenate(y_pred_all[name])
        metrics = compute_metrics(y_true, y_pred)
        summary["methods"][name] = metrics

        np.savez(
            case_dir / f"{name}_seed{seed}.npz",
            y_true=y_true,
            y_pred=y_pred,
            segments=np.asarray(seg_all[name]),
            chunk_ids=np.asarray(chunk_all[name]),
        )
        with (trace_dir / f"{name}_trace_seed{seed}.json").open("w", encoding="utf-8") as f:
            json.dump(traces[name], f, indent=2)

    with (summary_dir / f"adaptive_baselines_seed{seed}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] adaptive baseline runs completed")
    for name, metrics in summary["methods"].items():
        print(name, metrics)


if __name__ == "__main__":
    main()
