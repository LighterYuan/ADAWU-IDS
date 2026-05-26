#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_ORDER = [
    "ADAWU-IDS",
    "Static MLP",
    "DWM",
    "Online Bagging",
    "Leveraging Bagging",
]

def ci95(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(1.96 * arr.std(ddof=1) / math.sqrt(arr.size))

def parse_seed(path: Path) -> int:
    name = path.stem.lower()
    for pat in [r"seed[_\-]?(\d+)", r"s[_\-]?(\d+)", r"run[_\-]?(\d+)"]:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return 0

def normalize_method_name(raw: str) -> str:
    s = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if "adawu" in s or "paper_trace" in s:
        return "ADAWU-IDS"
    if "leveraging" in s and "bagging" in s:
        return "Leveraging Bagging"
    if "online" in s and "bagging" in s:
        return "Online Bagging"
    if "dwm" in s:
        return "DWM"
    if "static" in s and "mlp" in s:
        return "Static MLP"
    return raw

def load_adawu_trace(trace_path: Path):
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    if not chunks:
        raise ValueError(f"No 'chunks' found in {trace_path}")

    rows = []
    for rec in chunks:
        rows.append({
            "chunk_id": int(rec.get("chunk_id", len(rows))),
            "accuracy": float(rec.get("ensemble_accuracy", np.nan)),
            "weighted_f1": float(rec.get("ensemble_weighted_f1", np.nan)),
            "method": "ADAWU-IDS",
            "seed": int(data.get("seed", parse_seed(trace_path))),
        })

    trace_df = pd.DataFrame(rows).sort_values("chunk_id").reset_index(drop=True)
    summary = {
        "method": "ADAWU-IDS",
        "seed": int(data.get("seed", parse_seed(trace_path))),
        "accuracy": float(pd.to_numeric(trace_df["accuracy"], errors="coerce").mean()),
        "weighted_f1": float(pd.to_numeric(trace_df["weighted_f1"], errors="coerce").mean()),
        "macro_f1": np.nan,
        "n_chunks": int(len(trace_df)),
        "source_file": str(trace_path),
    }
    return trace_df, summary

def aggregate_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sdf in summary_df.groupby("method"):
        row = {"method": method, "n_seeds": int(len(sdf))}
        for metric in ["accuracy", "weighted_f1", "macro_f1"]:
            vals = pd.to_numeric(sdf[metric], errors="coerce").dropna().to_numpy()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_ci95"] = ci95(vals) if len(vals) else 0.0
        rows.append(row)
    agg = pd.DataFrame(rows)
    if agg.empty:
        return agg
    agg["order"] = agg["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(999)
    agg = agg.sort_values(["order", "method"]).drop(columns=["order"]).reset_index(drop=True)
    return agg

def aggregate_temporal(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sdf in long_df.groupby("method"):
        agg = sdf.groupby("chunk_id")["weighted_f1"].agg(["mean", "std", "count"]).reset_index()
        agg["ci95"] = [
            0.0 if c <= 1 or pd.isna(s) else 1.96 * float(s) / math.sqrt(int(c))
            for s, c in zip(agg["std"], agg["count"])
        ]
        agg["method"] = method
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["order"] = out["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(999)
    out = out.sort_values(["order", "method", "chunk_id"]).drop(columns=["order"]).reset_index(drop=True)
    return out

def plot_overall_weighted_f1(agg_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = agg_df.copy()
    plot_df = plot_df[pd.notna(plot_df["weighted_f1_mean"])].copy()
    if plot_df.empty:
        return
    x = np.arange(len(plot_df))
    plt.figure(figsize=(9, 5))
    plt.bar(x, plot_df["weighted_f1_mean"].to_numpy())
    plt.errorbar(
        x,
        plot_df["weighted_f1_mean"].to_numpy(),
        yerr=plot_df["weighted_f1_ci95"].to_numpy(),
        fmt="none",
        capsize=4,
    )
    plt.xticks(x, plot_df["method"].tolist(), rotation=18)
    plt.ylabel("Weighted F1")
    plt.ylim(0.0, 1.05)
    plt.title("UNSW-NB15 Overall Weighted F1 (Merged)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def plot_temporal_weighted_f1(temporal_agg: pd.DataFrame, out_path: Path) -> None:
    if temporal_agg.empty:
        return
    plt.figure(figsize=(12, 6))
    for method in METHOD_ORDER:
        sdf = temporal_agg[temporal_agg["method"] == method].sort_values("chunk_id")
        if sdf.empty:
            continue
        plt.plot(sdf["chunk_id"].to_numpy(), sdf["mean"].to_numpy(), label=method, linewidth=2.0)
    plt.title("UNSW-NB15 Temporal Weighted F1 Curves (Merged)")
    plt.xlabel("Chunk ID")
    plt.ylabel("Weighted F1")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ADAWU traces with baseline UNSW-NB15 results.")
    parser.add_argument("--adawu-glob", type=str, required=True, help="Glob pattern for ADAWU JSON traces.")
    parser.add_argument("--baseline-summary", type=str, required=True, help="Path to baseline summary_per_seed.csv")
    parser.add_argument("--baseline-temporal", type=str, default="", help="Optional path to baseline temporal_weighted_f1_long.csv")
    parser.add_argument("--outdir", type=str, default="results/unsw_nb15_merged", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trace_paths = [Path(p) for p in sorted(glob.glob(args.adawu_glob))]
    if not trace_paths:
        raise FileNotFoundError(f"No ADAWU trace files matched: {args.adawu_glob}")

    baseline_summary = pd.read_csv(args.baseline_summary)
    baseline_summary["method"] = baseline_summary["method"].map(normalize_method_name)

    adawu_summary_rows = []
    adawu_trace_frames = []
    for trace_path in trace_paths:
        trace_df, summary = load_adawu_trace(trace_path)
        adawu_trace_frames.append(trace_df)
        adawu_summary_rows.append(summary)

    adawu_summary = pd.DataFrame(adawu_summary_rows)
    merged_summary = pd.concat([baseline_summary, adawu_summary], ignore_index=True, sort=False)
    merged_summary["method"] = merged_summary["method"].map(normalize_method_name)
    if "source_file" not in merged_summary.columns:
        merged_summary["source_file"] = ""

    merged_summary = merged_summary.sort_values(["method", "seed"]).reset_index(drop=True)
    merged_summary.to_csv(outdir / "merged_summary_per_seed.csv", index=False)

    merged_agg = aggregate_summary(merged_summary)
    merged_agg.to_csv(outdir / "merged_summary_aggregated.csv", index=False)
    plot_overall_weighted_f1(merged_agg, outdir / "merged_overall_weighted_f1.png")

    if args.baseline_temporal:
        baseline_temporal = pd.read_csv(args.baseline_temporal)
        baseline_temporal["method"] = baseline_temporal["method"].map(normalize_method_name)

        adawu_temporal = pd.concat(adawu_trace_frames, ignore_index=True)
        adawu_temporal = adawu_temporal[["chunk_id", "accuracy", "weighted_f1", "method", "seed"]].copy()

        merged_temporal = pd.concat([baseline_temporal, adawu_temporal], ignore_index=True, sort=False)
        merged_temporal["method"] = merged_temporal["method"].map(normalize_method_name)
        merged_temporal = merged_temporal.sort_values(["method", "seed", "chunk_id"]).reset_index(drop=True)
        merged_temporal.to_csv(outdir / "merged_temporal_long.csv", index=False)

        merged_temporal_agg = aggregate_temporal(merged_temporal)
        merged_temporal_agg.to_csv(outdir / "merged_temporal_agg.csv", index=False)
        plot_temporal_weighted_f1(merged_temporal_agg, outdir / "merged_temporal_weighted_f1.png")

    print("[OK] Merged results saved to:", outdir)

if __name__ == "__main__":
    main()
