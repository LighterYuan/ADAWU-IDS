#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate only the aligned drift-centered figure, with the x-range fixed to [-1, 3].

Default behavior:
- Keep only 5 main methods:
  - ADAWU-IDS
  - DWM
  - Online Bagging
  - Leveraging Bagging
  - Static LSTM
- Read an aggregated CSV like aligned_recovery_agg.csv
- Truncate to relative_t in [-1, 3]
- Output only one figure:
  - aligned_post_drift_recovery_fixed.png

Expected CSV columns:
- method
- relative_t
- mean
Optional:
- ci95

Example:
python plot_aligned_recovery_fixed.py \
  --csv results/post_drift_analysis/aligned_recovery_agg.csv \
  --out results/post_drift_analysis/aligned_post_drift_recovery_fixed.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAIN_METHOD_ORDER: List[str] = [
    "ADAWU-IDS",
    "DWM",
    "Online Bagging",
    "Leveraging Bagging",
    "Static LSTM",
]


def normalize_method_name(raw: str) -> str:
    s = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if "leveraging" in s and "bagging" in s:
        return "Leveraging Bagging"
    if "online" in s and "bagging" in s:
        return "Online Bagging"
    if "dwm" in s:
        return "DWM"
    if "adawu" in s or "paper_trace" in s:
        return "ADAWU-IDS"
    if "static" in s or "lstm" in s:
        return "Static LSTM"
    return raw.strip()


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    return (
        pd.Series(values)
        .rolling(window=window, min_periods=1, center=True)
        .mean()
        .to_numpy()
    )


def load_aligned_agg(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"method", "relative_t", "mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    if "ci95" not in df.columns:
        df["ci95"] = 0.0

    df["method"] = df["method"].map(normalize_method_name)
    df = df[df["method"].isin(MAIN_METHOD_ORDER)].copy()

    df["relative_t"] = pd.to_numeric(df["relative_t"], errors="coerce")
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["ci95"] = pd.to_numeric(df["ci95"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["relative_t", "mean"]).copy()
    df["relative_t"] = df["relative_t"].astype(int)

    # Fixed truncation requested by user
    df = df[(df["relative_t"] >= -1) & (df["relative_t"] <= 3)].copy()

    return df.sort_values(["method", "relative_t"]).reset_index(drop=True)


def plot_aligned_recovery_fixed(
    aligned_agg: pd.DataFrame,
    out_path: Path,
    title: str,
    smooth_window: int,
    show_ci: bool,
) -> None:
    plt.figure(figsize=(10, 6))

    if aligned_agg.empty:
        plt.title(title + " (no data in [-1, 3])")
        plt.xlabel("Relative Step Around Drift")
        plt.ylabel("Weighted F1")
        plt.tight_layout()
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close()
        return

    for method in MAIN_METHOD_ORDER:
        sdf = aligned_agg[aligned_agg["method"] == method].sort_values("relative_t")
        if sdf.empty:
            continue

        x = sdf["relative_t"].to_numpy()
        y = smooth_series(sdf["mean"].to_numpy(), smooth_window)
        ci = sdf["ci95"].to_numpy()

        plt.plot(x, y, label=method, linewidth=2.0)
        if show_ci:
            plt.fill_between(
                x,
                np.maximum(0, y - ci),
                np.minimum(1.05, y + ci),
                alpha=0.14,
            )

    plt.axvline(0, linestyle="--", linewidth=1.0, alpha=0.6, color="black")
    plt.title(title)
    plt.xlabel("Relative Step Around Drift")
    plt.ylabel("Weighted F1")
    plt.xlim(-1.0, 3.0)
    plt.ylim(0.90, 1.005)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate only the aligned drift-centered figure with fixed range [-1, 3]."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="aligned_recovery_agg.csv",
        help="Input aggregated aligned CSV file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="aligned_post_drift_recovery_fixed.png",
        help="Output PNG filename.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Rolling smoothing window for displayed mean curves.",
    )
    parser.add_argument(
        "--show-ci",
        action="store_true",
        help="Show 95%% CI shading if ci95 column exists.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Drift-centered Weighted F1 Trajectories",
        help="Figure title.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    aligned_agg = load_aligned_agg(csv_path)
    plot_aligned_recovery_fixed(
        aligned_agg=aligned_agg,
        out_path=out_path,
        title=args.title,
        smooth_window=args.smooth_window,
        show_ci=args.show_ci,
    )

    print(f"[OK] Wrote figure: {out_path}")


if __name__ == "__main__":
    main()
