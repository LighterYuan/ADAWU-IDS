#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate paper-ready ablation tables and figures for CICIDS2017 and UNSW-NB15.

Expected input:
  results/ablation/<DATASET>/seed_<SEED>/ablation_summary.csv
or per-variant files:
  results/ablation/<DATASET>/seed_<SEED>/*_summary.json
  results/ablation/<DATASET>/seed_<SEED>/*_chunk_metrics.csv

Run:
  python analysis/plot_ablation_paper.py \
    --results-dir results/ablation \
    --datasets CICIDS2017 UNSW-NB15 \
    --seeds 42 52 62
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VARIANT_ORDER = [
    "static_lstm_or_static_sgd",
    "w_o_msdi",
    "w_o_dynamic_weighting",
    "w_o_hierarchical_response",
    "full_adawu_ids",
]

VARIANT_LABELS = {
    "static_lstm_or_static_sgd": "Static Model",
    "w_o_msdi": "w/o MSDI",
    "w_o_dynamic_weighting": "w/o ADAWU Weighting",
    "w_o_hierarchical_response": "w/o Hierarchical Response",
    "full_adawu_ids": "Full ADAWU-IDS",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "weighted_f1": "Weighted F1",
    "pre_drift_f1": "Pre-drift F1",
    "post_drift_min_f1": "Post-drift Min F1",
    "final_window_f1": "Final-window F1",
    "relative_drop": "Relative Performance Drop",
    "recovery_steps": "Recovery Steps",
    "n_updates": "Number of Updates",
    "n_drift_chunks": "Detected Drift Chunks",
}

MAIN_METRICS = [
    "accuracy",
    "weighted_f1",
    "post_drift_min_f1",
    "final_window_f1",
    "relative_drop",
]

ROBUSTNESS_METRICS = [
    "pre_drift_f1",
    "post_drift_min_f1",
    "final_window_f1",
    "relative_drop",
]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_seed_summary(results_dir: Path, dataset: str, seed: int) -> pd.DataFrame:
    seed_dir = results_dir / dataset / f"seed_{seed}"
    csv_path = seed_dir / "ablation_summary.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        rows = []
        for p in seed_dir.glob("*_summary.json"):
            with open(p, "r", encoding="utf-8") as f:
                rows.append(json.load(f))
        if not rows:
            raise FileNotFoundError(f"No ablation summary found in {seed_dir}")
        df = pd.DataFrame(rows)

    df["dataset"] = dataset
    df["seed"] = seed
    return df


def read_chunk_metrics(results_dir: Path, dataset: str, seed: int) -> pd.DataFrame:
    seed_dir = results_dir / dataset / f"seed_{seed}"
    rows = []
    for p in seed_dir.glob("*_chunk_metrics.csv"):
        variant = p.name.replace("_chunk_metrics.csv", "")
        tmp = pd.read_csv(p)
        tmp["variant"] = variant
        tmp["dataset"] = dataset
        tmp["seed"] = seed
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_all(results_dir: Path, datasets, seeds):
    summary_rows = []
    chunk_rows = []

    for dataset in datasets:
        for seed in seeds:
            summary_rows.append(read_seed_summary(results_dir, dataset, seed))
            chunk_df = read_chunk_metrics(results_dir, dataset, seed)
            if not chunk_df.empty:
                chunk_rows.append(chunk_df)

    summary = pd.concat(summary_rows, ignore_index=True)
    chunks = pd.concat(chunk_rows, ignore_index=True) if chunk_rows else pd.DataFrame()

    summary["variant_label"] = summary["variant"].map(VARIANT_LABELS).fillna(summary["variant"])
    summary["variant_order"] = summary["variant"].map({v: i for i, v in enumerate(VARIANT_ORDER)}).fillna(99)
    summary = summary.sort_values(["dataset", "variant_order", "seed"])
    return summary, chunks


def mean_std_table(summary: pd.DataFrame, metrics):
    available_metrics = [m for m in metrics if m in summary.columns]
    grouped = summary.groupby(["dataset", "variant"], as_index=False)[available_metrics].agg(["mean", "std"])
    grouped.columns = ["_".join([c for c in col if c]) for col in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index()
    grouped["variant_label"] = grouped["variant"].map(VARIANT_LABELS).fillna(grouped["variant"])
    grouped["variant_order"] = grouped["variant"].map({v: i for i, v in enumerate(VARIANT_ORDER)}).fillna(99)
    return grouped.sort_values(["dataset", "variant_order"])


def format_mean_std(row, metric, decimals=4):
    mean = row.get(f"{metric}_mean", np.nan)
    std = row.get(f"{metric}_std", np.nan)
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def export_tables(summary: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)
    metrics = MAIN_METRICS + ["recovery_steps", "n_updates", "n_drift_chunks"]
    table = mean_std_table(summary, metrics)

    rows = []
    for _, row in table.iterrows():
        item = {"Dataset": row["dataset"], "Variant": VARIANT_LABELS.get(row["variant"], row["variant"])}
        for m in MAIN_METRICS:
            item[METRIC_LABELS[m]] = format_mean_std(row, m, 4)
        item["Recovery Steps"] = format_mean_std(row, "recovery_steps", 2)
        item["Updates"] = format_mean_std(row, "n_updates", 2)
        item["Drift Chunks"] = format_mean_std(row, "n_drift_chunks", 2)
        rows.append(item)

    display = pd.DataFrame(rows)
    display.to_csv(out_dir / "ablation_table_mean_std.csv", index=False)
    display.to_latex(out_dir / "ablation_table_mean_std.tex", index=False, escape=False)
    summary.to_csv(out_dir / "ablation_summary_all_seeds_raw.csv", index=False)
    print(f"[OK] saved tables to {out_dir}")


def savefig(path_base: Path, dpi=300):
    plt.tight_layout()
    plt.savefig(path_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {path_base.with_suffix('.png')}")


def plot_metric_bars(summary: pd.DataFrame, dataset: str, metric: str, out_dir: Path):
    if metric not in summary.columns:
        return
    sub = summary[summary["dataset"] == dataset].copy()
    table = mean_std_table(sub, [metric])
    table = table[table["variant"].isin(VARIANT_ORDER)].sort_values("variant_order")

    labels = [VARIANT_LABELS[v] for v in table["variant"]]
    means = table[f"{metric}_mean"].values
    stds = table[f"{metric}_std"].fillna(0).values
    x = np.arange(len(labels))

    plt.figure(figsize=(10, 5.5))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylabel(METRIC_LABELS.get(metric, metric))
    plt.title(f"{dataset} Ablation Study: {METRIC_LABELS.get(metric, metric)}")

    if metric != "relative_drop":
        ymin = max(0.0, np.nanmin(means - stds) - 0.03)
        ymax = min(1.05, np.nanmax(means + stds) + 0.03)
    else:
        ymin = 0.0
        ymax = max(0.05, np.nanmax(means + stds) + 0.05)
    plt.ylim(ymin, ymax)
    savefig(out_dir / f"{dataset}_ablation_{metric}")


def plot_grouped_accuracy_f1(summary: pd.DataFrame, dataset: str, out_dir: Path):
    if "accuracy" not in summary.columns or "weighted_f1" not in summary.columns:
        return
    sub = summary[summary["dataset"] == dataset].copy()
    table = mean_std_table(sub, ["accuracy", "weighted_f1"])
    table = table[table["variant"].isin(VARIANT_ORDER)].sort_values("variant_order")

    labels = [VARIANT_LABELS[v] for v in table["variant"]]
    x = np.arange(len(labels))
    width = 0.36

    acc_mean = table["accuracy_mean"].values
    acc_std = table["accuracy_std"].fillna(0).values
    f1_mean = table["weighted_f1_mean"].values
    f1_std = table["weighted_f1_std"].fillna(0).values

    plt.figure(figsize=(11, 5.8))
    plt.bar(x - width / 2, acc_mean, width, yerr=acc_std, capsize=4, label="Accuracy")
    plt.bar(x + width / 2, f1_mean, width, yerr=f1_std, capsize=4, label="Weighted F1")
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylabel("Score")
    plt.title(f"{dataset} Ablation Study: Accuracy and Weighted F1")
    plt.legend()

    ymin = max(0.0, min(np.nanmin(acc_mean - acc_std), np.nanmin(f1_mean - f1_std)) - 0.03)
    ymax = min(1.05, max(np.nanmax(acc_mean + acc_std), np.nanmax(f1_mean + f1_std)) + 0.03)
    plt.ylim(ymin, ymax)
    savefig(out_dir / f"{dataset}_ablation_accuracy_weighted_f1")


def plot_robustness_grid(summary: pd.DataFrame, dataset: str, out_dir: Path):
    metrics = [m for m in ROBUSTNESS_METRICS if m in summary.columns]
    if len(metrics) < 4:
        return
    sub = summary[summary["dataset"] == dataset].copy()
    table = mean_std_table(sub, metrics)
    table = table[table["variant"].isin(VARIANT_ORDER)].sort_values("variant_order")

    labels = [VARIANT_LABELS[v] for v in table["variant"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):
        means = table[f"{metric}_mean"].values
        stds = table[f"{metric}_std"].fillna(0).values
        ax.bar(x, means, yerr=stds, capsize=4)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        if metric != "relative_drop":
            ax.set_ylim(max(0.0, np.nanmin(means - stds) - 0.03), min(1.05, np.nanmax(means + stds) + 0.03))
        else:
            ax.set_ylim(0, max(0.05, np.nanmax(means + stds) + 0.05))

    fig.suptitle(f"{dataset} Post-drift Ablation Robustness Summary", y=1.02, fontsize=14)
    savefig(out_dir / f"{dataset}_ablation_post_drift_summary")


def plot_cross_dataset_heatmap(summary: pd.DataFrame, metric: str, out_dir: Path):
    if metric not in summary.columns:
        return
    table = summary.groupby(["dataset", "variant"], as_index=False)[metric].mean()
    pivot = table.pivot(index="dataset", columns="variant", values=metric)
    cols = [v for v in VARIANT_ORDER if v in pivot.columns]
    pivot = pivot[cols].rename(columns=VARIANT_LABELS)
    values = pivot.values.astype(float)

    plt.figure(figsize=(12, 3.6 + 0.35 * len(pivot)))
    im = plt.imshow(values, aspect="auto")
    plt.xticks(np.arange(values.shape[1]), pivot.columns, rotation=20, ha="right")
    plt.yticks(np.arange(values.shape[0]), pivot.index)
    plt.title(f"Cross-dataset Ablation Comparison: {METRIC_LABELS.get(metric, metric)}")
    plt.colorbar(im, label=METRIC_LABELS.get(metric, metric))

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if pd.notna(values[i, j]):
                plt.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center")
    savefig(out_dir / f"cross_dataset_ablation_{metric}_heatmap")


def plot_chunk_trajectories(chunks: pd.DataFrame, dataset: str, out_dir: Path):
    if chunks.empty or "weighted_f1" not in chunks.columns:
        return
    sub = chunks[chunks["dataset"] == dataset].copy()
    if sub.empty:
        return

    grouped = sub.groupby(["variant", "chunk"], as_index=False)["weighted_f1"].agg(["mean", "std"]).reset_index()
    grouped = grouped[grouped["variant"].isin(VARIANT_ORDER)]

    plt.figure(figsize=(11, 5.8))
    for variant in VARIANT_ORDER:
        g = grouped[grouped["variant"] == variant].sort_values("chunk")
        if g.empty:
            continue
        x = g["chunk"].values
        y = g["mean"].values
        s = g["std"].fillna(0).values
        plt.plot(x, y, marker="o", linewidth=1.8, markersize=3, label=VARIANT_LABELS[variant])
        plt.fill_between(x, y - s, y + s, alpha=0.12)

    plt.xlabel("Stream Chunk")
    plt.ylabel("Weighted F1")
    plt.title(f"{dataset} Ablation Weighted F1 Trajectories")
    plt.legend()
    plt.ylim(0, 1.05)
    savefig(out_dir / f"{dataset}_ablation_weighted_f1_trajectory")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/ablation")
    parser.add_argument("--output-dir", default="paper_outputs/ablation")
    parser.add_argument("--datasets", nargs="+", default=["CICIDS2017", "UNSW-NB15"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    ensure_dir(fig_dir)
    ensure_dir(table_dir)

    summary, chunks = load_all(results_dir, args.datasets, args.seeds)
    export_tables(summary, table_dir)

    for dataset in args.datasets:
        plot_grouped_accuracy_f1(summary, dataset, fig_dir)
        plot_metric_bars(summary, dataset, "post_drift_min_f1", fig_dir)
        plot_metric_bars(summary, dataset, "final_window_f1", fig_dir)
        plot_metric_bars(summary, dataset, "relative_drop", fig_dir)
        plot_robustness_grid(summary, dataset, fig_dir)
        plot_chunk_trajectories(chunks, dataset, fig_dir)

    plot_cross_dataset_heatmap(summary, "weighted_f1", fig_dir)
    plot_cross_dataset_heatmap(summary, "post_drift_min_f1", fig_dir)
    plot_cross_dataset_heatmap(summary, "relative_drop", fig_dir)

    print("\n[DONE] Paper-ready ablation tables and figures generated.")
    print(f"Figures: {fig_dir}")
    print(f"Tables : {table_dir}")


if __name__ == "__main__":
    main()
