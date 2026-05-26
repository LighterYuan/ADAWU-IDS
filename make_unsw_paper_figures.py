#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METHOD_ORDER = [
    "ADAWU-IDS",
    "Static MLP",
    "DWM",
    "Online Bagging",
    "Leveraging Bagging",
]

def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "method",
        "weighted_f1_mean",
        "weighted_f1_ci95",
        "accuracy_mean",
        "accuracy_ci95",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df = df.copy()
    df["method"] = df["method"].astype(str)
    df["weighted_f1_mean"] = pd.to_numeric(df["weighted_f1_mean"], errors="coerce")
    df["weighted_f1_ci95"] = pd.to_numeric(df["weighted_f1_ci95"], errors="coerce").fillna(0.0)
    df["accuracy_mean"] = pd.to_numeric(df["accuracy_mean"], errors="coerce")
    df["accuracy_ci95"] = pd.to_numeric(df["accuracy_ci95"], errors="coerce").fillna(0.0)

    df["order"] = df["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    df = df[pd.notna(df["order"])].copy()
    df["order"] = df["order"].astype(int)
    df = df.sort_values("order").drop(columns=["order"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid methods found after ordering/filtering.")
    return df

def compute_ylim(values: np.ndarray, ci: np.ndarray, pad_ratio: float = 0.08):
    vmax = float(np.nanmax(values + ci))
    vmin = float(np.nanmin(values - ci))
    lower = max(0.0, min(vmin, values.min()) - pad_ratio * max(1e-6, (vmax - vmin)))
    upper = min(1.02, vmax + pad_ratio * max(1e-6, (vmax - vmin)))
    if vmax > 0.8 and upper - lower > 0.25:
        lower = max(0.70, vmin - 0.05)
        upper = min(1.02, vmax + 0.02)
    return lower, upper

def plot_weighted_f1(df: pd.DataFrame, out_path: Path) -> None:
    x = np.arange(len(df))
    y = df["weighted_f1_mean"].to_numpy(dtype=float)
    ci = df["weighted_f1_ci95"].to_numpy(dtype=float)
    labels = df["method"].tolist()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x, y)
    ax.errorbar(x, y, yerr=ci, fmt="none", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18)
    ax.set_ylabel("Weighted F1")
    ax.set_title("UNSW-NB15 Overall Weighted F1")
    ymin, ymax = compute_ylim(y, ci)
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_accuracy(df: pd.DataFrame, out_path: Path) -> None:
    x = np.arange(len(df))
    y = df["accuracy_mean"].to_numpy(dtype=float)
    ci = df["accuracy_ci95"].to_numpy(dtype=float)
    labels = df["method"].tolist()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x, y)
    ax.errorbar(x, y, yerr=ci, fmt="none", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18)
    ax.set_ylabel("Accuracy")
    ax.set_title("UNSW-NB15 Overall Accuracy")
    ymin, ymax = compute_ylim(y, ci)
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_combined(df: pd.DataFrame, out_path: Path) -> None:
    x = np.arange(len(df))
    width = 0.36
    acc = df["accuracy_mean"].to_numpy(dtype=float)
    acc_ci = df["accuracy_ci95"].to_numpy(dtype=float)
    wf1 = df["weighted_f1_mean"].to_numpy(dtype=float)
    wf1_ci = df["weighted_f1_ci95"].to_numpy(dtype=float)
    labels = df["method"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(x - width / 2, acc, width=width, label="Accuracy")
    ax.bar(x + width / 2, wf1, width=width, label="Weighted F1")
    ax.errorbar(x - width / 2, acc, yerr=acc_ci, fmt="none", capsize=4)
    ax.errorbar(x + width / 2, wf1, yerr=wf1_ci, fmt="none", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18)
    ax.set_ylabel("Score")
    ax.set_title("UNSW-NB15 Accuracy and Weighted F1")
    ax.legend()
    ymin, ymax = compute_ylim(np.concatenate([acc, wf1]), np.concatenate([acc_ci, wf1_ci]))
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready UNSW-NB15 figures from merged summary CSV.")
    parser.add_argument("--csv", type=str, default="merged_summary_aggregated.csv", help="Path to merged_summary_aggregated.csv")
    parser.add_argument("--outdir", type=str, default="figures_unsw_paper", help="Output directory for generated figures")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_summary(csv_path)

    weighted_path = outdir / "unsw_nb15_overall_weighted_f1_paper.png"
    accuracy_path = outdir / "unsw_nb15_overall_accuracy_paper.png"
    combined_path = outdir / "unsw_nb15_accuracy_weighted_f1_paper.png"

    plot_weighted_f1(df, weighted_path)
    plot_accuracy(df, accuracy_path)
    plot_combined(df, combined_path)

    print("[OK] Generated figures:")
    print(" -", weighted_path)
    print(" -", accuracy_path)
    print(" -", combined_path)

if __name__ == "__main__":
    main()
