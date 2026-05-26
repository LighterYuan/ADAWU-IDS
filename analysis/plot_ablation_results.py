#!/usr/bin/env python3
"""Plot component ablation figures from ablation_seed_level_metrics.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = [
    "static_lstm_or_static_sgd",
    "w_o_msdi",
    "w_o_dynamic_weighting",
    "w_o_hierarchical_response",
    "full_adawu_ids",
]
LABELS = {
    "static_lstm_or_static_sgd": "Static",
    "w_o_msdi": "w/o MSDI",
    "w_o_dynamic_weighting": "w/o ADAWU\nweighting",
    "w_o_hierarchical_response": "w/o hierarchical\nresponse",
    "full_adawu_ids": "Full\nADAWU-IDS",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-metrics", type=str, default="results/tables/ablation_seed_level_metrics.csv")
    p.add_argument("--out-dir", type=str, default="results/figures/ablations")
    return p.parse_args()


def barplot(df: pd.DataFrame, dataset: str, metric: str, ylabel: str, out_path: Path) -> None:
    sub = df[df["dataset"] == dataset].copy()
    means, stds, labels = [], [], []
    for v in ORDER:
        vals = sub[sub["variant"] == v][metric].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        labels.append(LABELS.get(v, v))
    x = np.arange(len(labels))
    plt.figure(figsize=(9, 4.8))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"{dataset} Component Ablation: {ylabel}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.seed_metrics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(df["dataset"].unique()):
        safe = str(dataset).replace("/", "_").replace(" ", "_")
        barplot(df, dataset, "weighted_f1", "Weighted F1", out_dir / f"{safe}_ablation_weighted_f1.png")
        barplot(df, dataset, "post_drift_min_f1", "Post-drift Minimum F1", out_dir / f"{safe}_ablation_post_drift_min_f1.png")
        barplot(df, dataset, "relative_drop", "Relative Performance Drop", out_dir / f"{safe}_ablation_relative_drop.png")
        barplot(df, dataset, "recovery_steps", "Recovery Steps", out_dir / f"{safe}_ablation_recovery_steps.png")

    print("[OK] figures saved to", out_dir)


if __name__ == "__main__":
    main()
