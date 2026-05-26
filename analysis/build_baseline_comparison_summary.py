from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = PROJECT_ROOT / "results" / "cases"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

METHOD_ALIASES = {
    "static_predictions": "static_lstm",
    "adawu_predictions": "adawu_ids",
    "dwm": "dwm",
    "online_bagging": "online_bagging",
    "leveraging_bagging": "leveraging_bagging",
}


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def ci95(values):
    vals = np.asarray(values, dtype=float)
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))


def load_npz_records():
    grouped = defaultdict(list)
    for path in CASE_DIR.glob("*.npz"):
        stem = path.stem
        method = None
        for prefix, alias in METHOD_ALIASES.items():
            if stem == prefix or stem.startswith(prefix + "_") or stem.startswith(prefix + "-"):
                method = alias
                break
        if method is None and "_seed" in stem:
            method = stem.split("_seed")[0]
        if method is None:
            method = stem

        data = np.load(path, allow_pickle=True)
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        grouped[method].append(compute_metrics(y_true, y_pred))
    return grouped


def main():
    grouped = load_npz_records()
    summary = {"methods": {}}
    lines = ["method,accuracy_mean,accuracy_std,weighted_f1_mean,weighted_f1_std,weighted_f1_ci95,macro_f1_mean,macro_f1_std,macro_recall_mean"]

    for method, records in sorted(grouped.items()):
        acc = [r["accuracy"] for r in records]
        wf1 = [r["weighted_f1"] for r in records]
        mf1 = [r["macro_f1"] for r in records]
        mr = [r["macro_recall"] for r in records]

        summary["methods"][method] = {
            "n_runs": len(records),
            "accuracy_mean": float(np.mean(acc)),
            "accuracy_std": float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
            "weighted_f1_mean": float(np.mean(wf1)),
            "weighted_f1_std": float(np.std(wf1, ddof=1)) if len(wf1) > 1 else 0.0,
            "weighted_f1_ci95": ci95(wf1),
            "macro_f1_mean": float(np.mean(mf1)),
            "macro_f1_std": float(np.std(mf1, ddof=1)) if len(mf1) > 1 else 0.0,
            "macro_recall_mean": float(np.mean(mr)),
        }

        lines.append(
            ",".join(
                [
                    method,
                    f"{summary['methods'][method]['accuracy_mean']:.6f}",
                    f"{summary['methods'][method]['accuracy_std']:.6f}",
                    f"{summary['methods'][method]['weighted_f1_mean']:.6f}",
                    f"{summary['methods'][method]['weighted_f1_std']:.6f}",
                    f"{summary['methods'][method]['weighted_f1_ci95']:.6f}",
                    f"{summary['methods'][method]['macro_f1_mean']:.6f}",
                    f"{summary['methods'][method]['macro_f1_std']:.6f}",
                    f"{summary['methods'][method]['macro_recall_mean']:.6f}",
                ]
            )
        )

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    with (SUMMARY_DIR / "baseline_comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (TABLE_DIR / "baseline_comparison_table.csv").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("[OK] baseline comparison summary written")


if __name__ == "__main__":
    main()
