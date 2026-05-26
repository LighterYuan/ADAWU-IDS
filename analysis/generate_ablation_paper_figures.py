#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate paper-ready ablation figures/tables from mixed output layouts.

Supports:
A) flat JSON traces:
   results/ablation/CICIDS2017/ablation_trace_CICIDS2017_full_adawu_ids_seed42.json

B) seed subfolders:
   results/ablation/UNSW-NB15/seed_42/ablation_summary.csv
   results/ablation/UNSW-NB15/seed_42/full_adawu_ids_chunk_metrics.csv

Run:
  python analysis/generate_ablation_paper_figures.py --input-root results/ablation --datasets CICIDS2017 UNSW-NB15 --seeds 42 52 62
"""

import argparse
import json
import re
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

SUMMARY_METRICS = [
    "accuracy",
    "weighted_f1",
    "pre_drift_f1",
    "post_drift_min_f1",
    "final_window_f1",
    "relative_drop",
    "recovery_steps",
    "n_updates",
    "n_drift_chunks",
]


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_variant(name):
    name = str(name)
    for v in VARIANT_ORDER:
        if v in name:
            return v
    if "msdi" in name and "w_o" in name:
        return "w_o_msdi"
    if "dynamic" in name and "w_o" in name:
        return "w_o_dynamic_weighting"
    if "hierarchical" in name and "w_o" in name:
        return "w_o_hierarchical_response"
    if "static" in name:
        return "static_lstm_or_static_sgd"
    if "full" in name or "adawu" in name:
        return "full_adawu_ids"
    return name


def extract_seed_from_name(name):
    m = re.search(r"seed[_-]?(\d+)", str(name))
    return int(m.group(1)) if m else None


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sequence_to_chunk_df(obj, dataset, variant, seed):
    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        rows = None
        for key in ["trace", "chunk_metrics", "chunks", "history", "metrics", "records"]:
            if key in obj and isinstance(obj[key], list):
                rows = obj[key]
                break
        if rows is None:
            array_keys = [k for k, v in obj.items() if isinstance(v, list)]
            if array_keys:
                n = min(len(obj[k]) for k in array_keys)
                rows = [{k: obj[k][i] for k in array_keys} for i in range(n)]
            else:
                return pd.DataFrame()
    else:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rename = {
        "f1": "weighted_f1",
        "weighted_F1": "weighted_f1",
        "Weighted_F1": "weighted_f1",
        "weighted_f1_score": "weighted_f1",
        "acc": "accuracy",
        "Accuracy": "accuracy",
        "msdi_score": "msdi",
        "drift_score": "msdi",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "chunk" not in df.columns:
        df["chunk"] = np.arange(len(df))

    df["dataset"] = dataset
    df["variant"] = variant
    df["seed"] = seed
    return df


def summarize_from_chunk_df(df):
    if df.empty:
        return {}

    out = {}
    if "accuracy" in df.columns:
        out["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce").mean()

    if "weighted_f1" in df.columns:
        f1 = pd.to_numeric(df["weighted_f1"], errors="coerce")
        out["weighted_f1"] = f1.mean()

        if "drift_level" in df.columns:
            drift_mask = df["drift_level"].astype(str).isin(["low", "high", "drift", "1", "True", "true"])
            pos = np.where(drift_mask.values)[0]
            d0 = int(pos[0]) if len(pos) else max(1, len(df) // 2)
        elif "msdi" in df.columns:
            msdi = pd.to_numeric(df["msdi"], errors="coerce").fillna(0)
            d0 = int(msdi.values.argmax()) if len(msdi) else max(1, len(df) // 2)
            d0 = max(1, min(d0, len(df) - 1))
        else:
            d0 = max(1, len(df) // 2)

        pre = f1.iloc[:max(1, d0)]
        post = f1.iloc[d0:]
        pre_f1 = pre.mean() if len(pre) else f1.iloc[0]
        post_min = post.min() if len(post) else f1.min()
        final_f1 = f1.tail(min(3, len(f1))).mean()

        out["pre_drift_f1"] = pre_f1
        out["post_drift_min_f1"] = post_min
        out["final_window_f1"] = final_f1
        out["relative_drop"] = (pre_f1 - post_min) / max(pre_f1, 1e-8)

        rec = np.nan
        if len(post):
            threshold = 0.95 * pre_f1
            for i, val in enumerate(post.values):
                if val >= threshold:
                    rec = i
                    break
        out["recovery_steps"] = rec

    if "updated" in df.columns:
        out["n_updates"] = pd.to_numeric(df["updated"], errors="coerce").fillna(0).sum()

    if "drift_level" in df.columns:
        out["n_drift_chunks"] = df["drift_level"].astype(str).isin(["low", "high", "drift", "1", "True", "true"]).sum()

    return out


def load_seed_folder(dataset_dir, dataset, seed):
    seed_dir = dataset_dir / f"seed_{seed}"
    summaries = []
    chunks = []

    if not seed_dir.exists():
        return pd.DataFrame(), pd.DataFrame()

    summary_csv = seed_dir / "ablation_summary.csv"
    if summary_csv.exists():
        s = pd.read_csv(summary_csv)
        s["variant"] = s["variant"].map(normalize_variant)
        s["dataset"] = dataset
        s["seed"] = seed
        summaries.append(s)

    for p in seed_dir.glob("*_summary.json"):
        obj = read_json(p)
        if isinstance(obj, dict):
            row = dict(obj)
            row["variant"] = normalize_variant(row.get("variant", p.name.replace("_summary.json", "")))
            row["dataset"] = dataset
            row["seed"] = seed
            summaries.append(pd.DataFrame([row]))

    for p in seed_dir.glob("*_chunk_metrics.csv"):
        variant = normalize_variant(p.name.replace("_chunk_metrics.csv", ""))
        c = pd.read_csv(p)
        c["variant"] = variant
        c["dataset"] = dataset
        c["seed"] = seed
        chunks.append(c)

    s_out = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    c_out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return s_out, c_out


def load_flat_trace_files(dataset_dir, dataset, seeds):
    summaries = []
    chunks = []

    for p in dataset_dir.glob("ablation_trace_*.json"):
        seed = extract_seed_from_name(p.name)
        if seed is None or seed not in seeds:
            continue

        variant = normalize_variant(p.name)
        obj = read_json(p)

        c = sequence_to_chunk_df(obj, dataset, variant, seed)
        if not c.empty:
            chunks.append(c)
            row = summarize_from_chunk_df(c)
        elif isinstance(obj, dict):
            row = dict(obj)
        else:
            row = {}

        row["variant"] = variant
        row["dataset"] = dataset
        row["seed"] = seed
        summaries.append(pd.DataFrame([row]))

    s_out = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    c_out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return s_out, c_out


def load_all(input_root, datasets, seeds):
    input_root = Path(input_root)
    all_summaries = []
    all_chunks = []

    for dataset in datasets:
        dataset_dir = input_root / dataset

        for seed in seeds:
            s, c = load_seed_folder(dataset_dir, dataset, seed)
            if not s.empty:
                all_summaries.append(s)
            if not c.empty:
                all_chunks.append(c)

        s, c = load_flat_trace_files(dataset_dir, dataset, seeds)
        if not s.empty:
            all_summaries.append(s)
        if not c.empty:
            all_chunks.append(c)

    if not all_summaries:
        raise FileNotFoundError(f"No ablation outputs found under {input_root}")

    summary = pd.concat(all_summaries, ignore_index=True)
    summary["variant"] = summary["variant"].map(normalize_variant)
    summary = summary.drop_duplicates(subset=["dataset", "seed", "variant"], keep="first")

    for m in SUMMARY_METRICS:
        if m in summary.columns:
            summary[m] = pd.to_numeric(summary[m], errors="coerce")

    summary["variant_order"] = summary["variant"].map({v: i for i, v in enumerate(VARIANT_ORDER)}).fillna(99)
    summary = summary.sort_values(["dataset", "variant_order", "seed"])

    chunks = pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()
    if not chunks.empty:
        chunks["variant"] = chunks["variant"].map(normalize_variant)
        if "weighted_f1" not in chunks.columns and "f1" in chunks.columns:
            chunks = chunks.rename(columns={"f1": "weighted_f1"})
        if "accuracy" not in chunks.columns and "acc" in chunks.columns:
            chunks = chunks.rename(columns={"acc": "accuracy"})

    return summary, chunks


def grouped_stats(summary, metrics):
    rows = []
    for (dataset, variant), g in summary.groupby(["dataset", "variant"]):
        row = {"dataset": dataset, "variant": variant}
        for m in metrics:
            if m in g.columns:
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_std"] = g[m].std(ddof=1)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["variant_order"] = out["variant"].map({v: i for i, v in enumerate(VARIANT_ORDER)}).fillna(99)
    return out.sort_values(["dataset", "variant_order"])


def fmt(mean, std, decimals=4):
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def export_tables(summary, table_dir):
    mkdir(table_dir)
    stats = grouped_stats(summary, SUMMARY_METRICS)
    stats.to_csv(table_dir / "ablation_summary_stats_raw.csv", index=False)

    rows = []
    for _, r in stats.iterrows():
        row = {"Dataset": r["dataset"], "Variant": VARIANT_LABELS.get(r["variant"], r["variant"])}
        for m in ["accuracy", "weighted_f1", "post_drift_min_f1", "final_window_f1", "relative_drop", "recovery_steps"]:
            row[METRIC_LABELS[m]] = fmt(r.get(f"{m}_mean", np.nan), r.get(f"{m}_std", np.nan), 2 if m == "recovery_steps" else 4)
        rows.append(row)

    paper = pd.DataFrame(rows)
    paper.to_csv(table_dir / "ablation_summary_mean_std.csv", index=False)
    paper.to_latex(table_dir / "ablation_summary_mean_std.tex", index=False, escape=False)
    summary.to_csv(table_dir / "ablation_all_runs_long.csv", index=False)


def savefig(path_base):
    plt.tight_layout()
    plt.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def plot_acc_f1(summary, dataset, fig_dir):
    if "accuracy" not in summary.columns or "weighted_f1" not in summary.columns:
        return
    stats = grouped_stats(summary[summary["dataset"] == dataset], ["accuracy", "weighted_f1"])
    stats = stats[stats["variant"].isin(VARIANT_ORDER)].sort_values("variant_order")
    labels = [VARIANT_LABELS.get(v, v) for v in stats["variant"]]
    x = np.arange(len(labels))
    w = 0.36

    acc = stats["accuracy_mean"].values
    acc_std = stats["accuracy_std"].fillna(0).values
    f1 = stats["weighted_f1_mean"].values
    f1_std = stats["weighted_f1_std"].fillna(0).values

    plt.figure(figsize=(11, 5.8))
    plt.bar(x - w / 2, acc, w, yerr=acc_std, capsize=4, label="Accuracy")
    plt.bar(x + w / 2, f1, w, yerr=f1_std, capsize=4, label="Weighted F1")
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylabel("Score")
    plt.title(f"{dataset} Ablation Study: Accuracy and Weighted F1")
    plt.legend()
    ymin = max(0, min(np.nanmin(acc - acc_std), np.nanmin(f1 - f1_std)) - 0.03)
    ymax = min(1.05, max(np.nanmax(acc + acc_std), np.nanmax(f1 + f1_std)) + 0.03)
    plt.ylim(ymin, ymax)
    savefig(Path(fig_dir) / f"{dataset}_ablation_accuracy_weighted_f1")


def plot_metric_bar(summary, dataset, metric, fig_dir):
    if metric not in summary.columns:
        return
    stats = grouped_stats(summary[summary["dataset"] == dataset], [metric])
    stats = stats[stats["variant"].isin(VARIANT_ORDER)].sort_values("variant_order")
    labels = [VARIANT_LABELS.get(v, v) for v in stats["variant"]]
    x = np.arange(len(labels))
    y = stats[f"{metric}_mean"].values
    e = stats[f"{metric}_std"].fillna(0).values

    plt.figure(figsize=(10.5, 5.5))
    plt.bar(x, y, yerr=e, capsize=4)
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylabel(METRIC_LABELS.get(metric, metric))
    plt.title(f"{dataset} Ablation Study: {METRIC_LABELS.get(metric, metric)}")
    if metric != "relative_drop":
        plt.ylim(max(0, np.nanmin(y - e) - 0.03), min(1.05, np.nanmax(y + e) + 0.03))
    else:
        plt.ylim(0, max(0.05, np.nanmax(y + e) + 0.05))
    savefig(Path(fig_dir) / f"{dataset}_ablation_{metric}")


def plot_robustness_grid(summary, dataset, fig_dir):
    metrics = ["pre_drift_f1", "post_drift_min_f1", "final_window_f1", "relative_drop"]
    if not all(m in summary.columns for m in metrics):
        return
    stats = grouped_stats(summary[summary["dataset"] == dataset], metrics)
    stats = stats[stats["variant"].isin(VARIANT_ORDER)].sort_values("variant_order")
    labels = [VARIANT_LABELS.get(v, v) for v in stats["variant"]]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, metric in zip(axes.ravel(), metrics):
        y = stats[f"{metric}_mean"].values
        e = stats[f"{metric}_std"].fillna(0).values
        ax.bar(x, y, yerr=e, capsize=4)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        if metric != "relative_drop":
            ax.set_ylim(max(0, np.nanmin(y - e) - 0.03), min(1.05, np.nanmax(y + e) + 0.03))
        else:
            ax.set_ylim(0, max(0.05, np.nanmax(y + e) + 0.05))
    fig.suptitle(f"{dataset} Post-drift Ablation Robustness Summary", y=1.02)
    savefig(Path(fig_dir) / f"{dataset}_ablation_post_drift_summary")


def plot_trajectory(chunks, dataset, fig_dir):
    if chunks.empty or "weighted_f1" not in chunks.columns:
        return
    sub = chunks[chunks["dataset"] == dataset].copy()
    if sub.empty:
        return
    sub["weighted_f1"] = pd.to_numeric(sub["weighted_f1"], errors="coerce")
    sub["chunk"] = pd.to_numeric(sub["chunk"], errors="coerce")
    g = sub.groupby(["variant", "chunk"], as_index=False)["weighted_f1"].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(11, 5.8))
    for variant in VARIANT_ORDER:
        one = g[g["variant"] == variant].sort_values("chunk")
        if one.empty:
            continue
        x = one["chunk"].values
        y = one["mean"].values
        e = one["std"].fillna(0).values
        plt.plot(x, y, marker="o", linewidth=1.8, markersize=3, label=VARIANT_LABELS.get(variant, variant))
        plt.fill_between(x, y - e, y + e, alpha=0.12)
    plt.xlabel("Stream Chunk")
    plt.ylabel("Weighted F1")
    plt.title(f"{dataset} Ablation Weighted F1 Trajectories")
    plt.ylim(0, 1.05)
    plt.legend()
    savefig(Path(fig_dir) / f"{dataset}_ablation_weighted_f1_trajectory")


def plot_heatmap(summary, metric, fig_dir):
    if metric not in summary.columns:
        return
    table = summary.groupby(["dataset", "variant"], as_index=False)[metric].mean()
    pivot = table.pivot(index="dataset", columns="variant", values=metric)
    cols = [v for v in VARIANT_ORDER if v in pivot.columns]
    pivot = pivot[cols].rename(columns=VARIANT_LABELS)
    values = pivot.values.astype(float)

    plt.figure(figsize=(12, 3.8 + 0.35 * len(pivot)))
    im = plt.imshow(values, aspect="auto")
    plt.xticks(np.arange(values.shape[1]), pivot.columns, rotation=20, ha="right")
    plt.yticks(np.arange(values.shape[0]), pivot.index)
    plt.title(f"Cross-dataset Ablation Comparison: {METRIC_LABELS.get(metric, metric)}")
    plt.colorbar(im, label=METRIC_LABELS.get(metric, metric))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if pd.notna(values[i, j]):
                plt.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center")
    savefig(Path(fig_dir) / f"cross_dataset_ablation_{metric}_heatmap")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="results/ablation")
    parser.add_argument("--output-root", default="paper_outputs/ablation")
    parser.add_argument("--datasets", nargs="+", default=["CICIDS2017", "UNSW-NB15"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    args = parser.parse_args()

    fig_dir = Path(args.output_root) / "figures"
    table_dir = Path(args.output_root) / "tables"
    mkdir(fig_dir)
    mkdir(table_dir)

    summary, chunks = load_all(args.input_root, args.datasets, args.seeds)
    print("[INFO] loaded summary rows:", len(summary))
    print(summary[["dataset", "seed", "variant"]].to_string(index=False))
    if not chunks.empty:
        print("[INFO] loaded chunk rows:", len(chunks))

    export_tables(summary, table_dir)

    for dataset in args.datasets:
        plot_acc_f1(summary, dataset, fig_dir)
        plot_metric_bar(summary, dataset, "post_drift_min_f1", fig_dir)
        plot_metric_bar(summary, dataset, "final_window_f1", fig_dir)
        plot_metric_bar(summary, dataset, "relative_drop", fig_dir)
        plot_robustness_grid(summary, dataset, fig_dir)
        plot_trajectory(chunks, dataset, fig_dir)

    plot_heatmap(summary, "weighted_f1", fig_dir)
    plot_heatmap(summary, "post_drift_min_f1", fig_dir)
    plot_heatmap(summary, "relative_drop", fig_dir)

    print("\n[DONE]")
    print("Figures:", fig_dir)
    print("Tables :", table_dir)


if __name__ == "__main__":
    main()
