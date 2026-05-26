#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build temporal drift analysis figures and CSV outputs from trace JSON and/or NPZ files.

Features
--------
1. Auto-discover multiple methods and multiple seeds.
2. Auto-align drift events from:
   - explicit anchor list (--drift-anchors),
   - segment/phase changes,
   - drift_detected flags in trace files.
3. Generate:
   - temporal_weighted_f1.png
   - aligned_post_drift_recovery.png
   - post_drift_summary.png
   - temporal_f1_long.csv
   - aligned_recovery_long.csv
   - post_drift_summary.csv
   - post_drift_summary_agg.csv
4. Supports both trace JSON and prediction NPZ.
5. Reasonable defaults so you can directly replace your own trace or npz paths.

Typical usage
-------------
A) Auto-scan a result directory:
    python build_post_drift_analysis.py --root ./results

B) Explicitly pass method inputs:
    python build_post_drift_analysis.py \
      --input adawu_ids=results/traces/adawu_ids_seed*.json \
      --input dwm=results/traces/dwm_seed*.json \
      --input online_bagging=results/traces/online_bagging_seed*.json \
      --input static_lstm=results/cases/static_lstm_seed*.npz

C) Give shared drift anchors manually:
    python build_post_drift_analysis.py --root ./results \
      --drift-anchors 96 156 229 343 379

D) Use segment transitions as shared anchors:
    python build_post_drift_analysis.py --root ./results --anchor-source segment_change
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def ci95(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n <= 1:
        return 0.0
    return 1.96 * float(s.std(ddof=1)) / math.sqrt(n)


def normalize_method_name(name: str) -> str:
    name = name.strip()
    mapping = {
        "adawu": "ADAWU-IDS",
        "adawu_ids": "ADAWU-IDS",
        "adawu-ids": "ADAWU-IDS",
        "dwm": "DWM",
        "online_bagging": "Online Bagging",
        "leveraging_bagging": "Leveraging Bagging",
        "static_lstm": "Static LSTM",
    }
    key = name.lower().replace(" ", "_")
    return mapping.get(key, name.replace("_", " ").replace("-", " ").title())


def maybe_read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_seed_from_name(path: Path) -> Optional[int]:
    patterns = [
        r"(?:^|[_\-])seed[_\-]?(\d+)",
        r"(?:^|[_\-])run[_\-]?(\d+)",
        r"(?:^|[_\-])s(\d+)",
    ]
    name = path.stem.lower()
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def parse_list_arg(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("["):
        vals = ast.literal_eval(raw)
        return [int(x) for x in vals]
    return [int(x) for x in raw.replace(",", " ").split() if x.strip()]


def smooth_series(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or len(arr) == 0:
        return arr
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=1, center=True).mean().to_numpy()


TRACE_F1_KEYS = [
    "ensemble_weighted_f1", "weighted_f1", "f1_weighted", "wf1",
    "chunk_weighted_f1", "f1"
]
TRACE_CHUNK_KEYS = ["chunk_id", "chunk", "chunk_index", "step", "t"]
TRACE_SEG_KEYS = ["segment", "phase", "split", "session", "day"]


def _first_key(d: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in d:
            return key
    return None


def _coerce_chunk_records(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ["chunks", "trace", "records", "data", "items"]:
            if key in obj and isinstance(obj[key], list):
                return [x for x in obj[key] if isinstance(x, dict)]
    return []


def load_trace_as_df(path: Path, method: str, seed: int) -> pd.DataFrame:
    obj = maybe_read_json(path)
    records = _coerce_chunk_records(obj)
    if not records:
        raise ValueError(f"No chunk-like records found in trace JSON: {path}")

    rows: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        chunk_key = _first_key(rec, TRACE_CHUNK_KEYS)
        f1_key = _first_key(rec, TRACE_F1_KEYS)
        seg_key = _first_key(rec, TRACE_SEG_KEYS)

        if f1_key is None:
            continue

        rows.append({
            "method": method,
            "seed": seed,
            "chunk_id": int(rec.get(chunk_key, i)),
            "weighted_f1": float(rec[f1_key]),
            "segment": rec.get(seg_key, None) if seg_key else None,
            "drift_detected": bool(rec.get("drift_detected", False)),
            "source_type": "trace",
            "source_file": str(path),
        })

    if not rows:
        raise ValueError(f"Trace JSON found, but no weighted F1 records could be parsed: {path}")

    return pd.DataFrame(rows).sort_values("chunk_id").reset_index(drop=True)


def _pick_npz_key(keys: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    key_set = {k.lower(): k for k in keys}
    for cand in candidates:
        if cand.lower() in key_set:
            return key_set[cand.lower()]
    return None


def _normalize_segments(values: np.ndarray) -> List[str]:
    out = []
    for x in values:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="ignore"))
        else:
            out.append(str(x))
    return out


def build_chunk_df_from_npz(
    path: Path,
    method: str,
    seed: int,
    chunk_size: int,
    group_by_segment: bool = True,
) -> pd.DataFrame:
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    precomputed_key = _pick_npz_key(
        keys,
        ["weighted_f1", "chunk_weighted_f1", "wf1", "f1_weighted", "f1"]
    )
    chunk_key = _pick_npz_key(keys, ["chunk_id", "chunks", "chunk_index", "step"])
    seg_key = _pick_npz_key(keys, ["segments", "segment", "phase", "session"])

    if precomputed_key is not None:
        vals = np.asarray(data[precomputed_key]).reshape(-1)
        chunk_ids = np.arange(len(vals)) if chunk_key is None else np.asarray(data[chunk_key]).reshape(-1)
        if len(chunk_ids) != len(vals):
            chunk_ids = np.arange(len(vals))
        segments = [None] * len(vals)
        if seg_key is not None:
            seg_arr = np.asarray(data[seg_key]).reshape(-1)
            if len(seg_arr) == len(vals):
                segments = _normalize_segments(seg_arr)

        rows = []
        for idx, f1v in enumerate(vals):
            rows.append({
                "method": method,
                "seed": seed,
                "chunk_id": int(chunk_ids[idx]),
                "weighted_f1": float(f1v),
                "segment": segments[idx],
                "drift_detected": False,
                "source_type": "npz",
                "source_file": str(path),
            })
        return pd.DataFrame(rows).sort_values("chunk_id").reset_index(drop=True)

    y_true_key = _pick_npz_key(keys, ["y_true", "true", "labels", "targets"])
    y_pred_key = _pick_npz_key(keys, ["y_pred", "pred", "predictions"])
    if y_true_key is None or y_pred_key is None:
        raise ValueError(
            f"NPZ unsupported. Need chunk-level weighted_f1 or y_true/y_pred keys. keys={keys}"
        )

    y_true = np.asarray(data[y_true_key]).reshape(-1)
    y_pred = np.asarray(data[y_pred_key]).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred length mismatch in {path}")

    segments = None
    if seg_key is not None:
        seg_arr = np.asarray(data[seg_key]).reshape(-1)
        if len(seg_arr) == len(y_true):
            segments = np.asarray(_normalize_segments(seg_arr))

    rows: List[Dict[str, Any]] = []

    if group_by_segment and segments is not None:
        start = 0
        chunk_id = 0
        while start < len(y_true):
            end = start + 1
            while end < len(y_true) and segments[end] == segments[start]:
                end += 1
            yt = y_true[start:end]
            yp = y_pred[start:end]
            f1v = f1_score(yt, yp, average="weighted", zero_division=0)
            rows.append({
                "method": method,
                "seed": seed,
                "chunk_id": chunk_id,
                "weighted_f1": float(f1v),
                "segment": str(segments[start]),
                "drift_detected": False,
                "source_type": "npz",
                "source_file": str(path),
            })
            start = end
            chunk_id += 1
    else:
        chunk_id = 0
        for start in range(0, len(y_true), chunk_size):
            end = min(start + chunk_size, len(y_true))
            yt = y_true[start:end]
            yp = y_pred[start:end]
            seg_val = None
            if segments is not None:
                unique_seg = pd.Series(segments[start:end]).mode()
                seg_val = str(unique_seg.iloc[0]) if not unique_seg.empty else None
            f1v = f1_score(yt, yp, average="weighted", zero_division=0)
            rows.append({
                "method": method,
                "seed": seed,
                "chunk_id": chunk_id,
                "weighted_f1": float(f1v),
                "segment": seg_val,
                "drift_detected": False,
                "source_type": "npz",
                "source_file": str(path),
            })
            chunk_id += 1

    return pd.DataFrame(rows).sort_values("chunk_id").reset_index(drop=True)


def infer_method_from_path(path: Path) -> str:
    name = path.stem.lower()
    parent = path.parent.name.lower()
    joined = " ".join([name, parent, str(path).lower()])
    rules = [
        ("leveraging_bagging", ["leveraging_bagging", "leveraging-bagging", "leveraging bagging"]),
        ("online_bagging", ["online_bagging", "online-bagging", "online bagging"]),
        ("static_lstm", ["static_lstm", "static-lstm", "static lstm"]),
        ("adawu_ids", ["adawu_ids", "adawu-ids", "adawu ids", "adawu"]),
        ("dwm", ["_dwm", "dwm_", "/dwm", "dwm"]),
    ]
    for out, pats in rules:
        for p in pats:
            if p in joined:
                return normalize_method_name(out)
    return normalize_method_name(path.parent.name if path.parent.name not in {"traces", "cases", "results"} else path.stem)


def discover_files(root: Path) -> List[Tuple[str, Path, int]]:
    files: List[Path] = []
    for pattern in ["*.json", "*.npz"]:
        files.extend(root.rglob(pattern))

    out = []
    for path in files:
        if path.name.startswith("."):
            continue
        method = infer_method_from_path(path)
        seed = parse_seed_from_name(path)
        if seed is None:
            seed = 0
        out.append((method, path, seed))
    return out


def parse_inputs(inputs: Sequence[str]) -> List[Tuple[str, Path, int]]:
    import glob
    triples: List[Tuple[str, Path, int]] = []
    for spec in inputs:
        if "=" not in spec:
            raise ValueError(f"--input expects method=glob_pattern, got: {spec}")
        raw_method, raw_glob = spec.split("=", 1)
        method = normalize_method_name(raw_method)
        matched = [Path(x) for x in sorted(glob.glob(raw_glob, recursive=True))]
        if not matched:
            raise FileNotFoundError(f"No files matched input pattern: {spec}")
        for path in matched:
            seed = parse_seed_from_name(path)
            if seed is None:
                seed = 0
            triples.append((method, path, seed))
    return triples


def load_all_runs(
    discovered: Sequence[Tuple[str, Path, int]],
    chunk_size: int,
    group_by_segment: bool = True,
) -> pd.DataFrame:
    frames = []
    for method, path, seed in discovered:
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                df = load_trace_as_df(path, method, seed)
            elif suffix == ".npz":
                df = build_chunk_df_from_npz(
                    path=path,
                    method=method,
                    seed=seed,
                    chunk_size=chunk_size,
                    group_by_segment=group_by_segment,
                )
            else:
                continue
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skip {path}: {e}")

    if not frames:
        raise RuntimeError("No valid trace JSON or NPZ files could be loaded.")
    out = pd.concat(frames, ignore_index=True)
    out["method"] = out["method"].map(normalize_method_name)
    out["seed"] = out["seed"].astype(int)
    out["chunk_id"] = out["chunk_id"].astype(int)
    return out.sort_values(["method", "seed", "chunk_id"]).reset_index(drop=True)


def infer_shared_drift_anchors(
    df: pd.DataFrame,
    anchor_source: str,
    explicit_anchors: Optional[List[int]],
    major_gap: int = 1,
) -> List[int]:
    if explicit_anchors:
        return sorted({int(x) for x in explicit_anchors if int(x) >= 0})

    if anchor_source == "segment_change":
        anchors = []
        for (_, _), sdf in df.groupby(["method", "seed"]):
            sdf = sdf.sort_values("chunk_id").reset_index(drop=True)
            if "segment" not in sdf.columns or sdf["segment"].isna().all():
                continue
            seg = sdf["segment"].astype(str).tolist()
            chunks = sdf["chunk_id"].tolist()
            for i in range(1, len(seg)):
                if seg[i] != seg[i - 1]:
                    anchors.append(int(chunks[i]))
        vc = pd.Series(anchors).value_counts()
        threshold = max(1, int(math.ceil(df[["method", "seed"]].drop_duplicates().shape[0] * 0.5)))
        return sorted([int(k) for k, v in vc.items() if v >= threshold])

    if anchor_source == "drift_flag":
        anchors = []
        for (_, _), sdf in df.groupby(["method", "seed"]):
            sdf = sdf.sort_values("chunk_id").reset_index(drop=True)
            if "drift_detected" not in sdf.columns:
                continue
            active = sdf["drift_detected"].fillna(False).astype(bool).to_numpy()
            chunks = sdf["chunk_id"].to_numpy()
            starts = np.where((active == True) & np.r_[True, active[:-1] == False])[0]
            anchors.extend([int(chunks[i]) for i in starts])
        vc = pd.Series(anchors).value_counts()
        threshold = max(1, int(math.ceil(df[["method", "seed"]].drop_duplicates().shape[0] * 0.3)))
        return sorted([int(k) for k, v in vc.items() if v >= threshold])

    anchors = infer_shared_drift_anchors(df, "segment_change", None, major_gap=major_gap)
    if anchors:
        return anchors
    anchors = infer_shared_drift_anchors(df, "drift_flag", None, major_gap=major_gap)
    if anchors:
        return anchors

    gaps = []
    for (_, _), sdf in df.groupby(["method", "seed"]):
        sdf = sdf.sort_values("chunk_id").reset_index(drop=True)
        chunks = sdf["chunk_id"].to_numpy()
        diff = np.diff(chunks)
        idxs = np.where(diff > major_gap)[0]
        for idx in idxs:
            gaps.append(int(chunks[idx + 1]))
    if gaps:
        vc = pd.Series(gaps).value_counts()
        return sorted([int(k) for k, v in vc.items() if v >= 2])

    return []


def extract_aligned_windows(
    df: pd.DataFrame,
    anchors: Sequence[int],
    pre_window: int,
    post_window: int,
) -> pd.DataFrame:
    rows = []
    if not anchors:
        return pd.DataFrame(columns=[
            "method", "seed", "anchor_chunk", "relative_t", "chunk_id", "weighted_f1"
        ])

    for (method, seed), sdf in df.groupby(["method", "seed"]):
        sdf = sdf.sort_values("chunk_id").reset_index(drop=True)
        f1_map = dict(zip(sdf["chunk_id"].astype(int), sdf["weighted_f1"].astype(float)))
        for anchor in anchors:
            for rel_t in range(-pre_window, post_window + 1):
                chunk_id = anchor + rel_t
                if chunk_id in f1_map:
                    rows.append({
                        "method": method,
                        "seed": int(seed),
                        "anchor_chunk": int(anchor),
                        "relative_t": int(rel_t),
                        "chunk_id": int(chunk_id),
                        "weighted_f1": float(f1_map[chunk_id]),
                    })
    return pd.DataFrame(rows)


def summarize_post_drift(
    df: pd.DataFrame,
    anchors: Sequence[int],
    pre_window: int,
    post_window: int,
    final_window: int,
    recovery_ratio: float,
) -> pd.DataFrame:
    rows = []
    if not anchors:
        return pd.DataFrame(columns=[
            "method", "seed", "anchor_chunk", "pre_drift_f1", "post_drift_min_f1",
            "final_window_f1", "relative_drop", "recovery_time"
        ])

    for (method, seed), sdf in df.groupby(["method", "seed"]):
        sdf = sdf.sort_values("chunk_id").reset_index(drop=True)
        f1_map = dict(zip(sdf["chunk_id"].astype(int), sdf["weighted_f1"].astype(float)))

        for anchor in anchors:
            pre_vals = [f1_map[c] for c in range(anchor - pre_window, anchor) if c in f1_map]
            post_vals = [f1_map[c] for c in range(anchor, anchor + post_window + 1) if c in f1_map]
            final_vals = [f1_map[c] for c in range(anchor + max(0, post_window - final_window + 1), anchor + post_window + 1) if c in f1_map]

            if not pre_vals or not post_vals:
                continue

            pre_f1 = float(np.mean(pre_vals))
            post_min = float(np.min(post_vals))
            final_f1 = float(np.mean(final_vals)) if final_vals else float(post_vals[-1])
            relative_drop = (pre_f1 - post_min) / pre_f1 if pre_f1 > 1e-12 else np.nan

            target = recovery_ratio * pre_f1
            recovery_time = np.nan
            for i, c in enumerate(range(anchor, anchor + post_window + 1)):
                if c in f1_map and f1_map[c] >= target:
                    recovery_time = float(i)
                    break

            rows.append({
                "method": method,
                "seed": int(seed),
                "anchor_chunk": int(anchor),
                "pre_drift_f1": pre_f1,
                "post_drift_min_f1": post_min,
                "final_window_f1": final_f1,
                "relative_drop": relative_drop,
                "recovery_time": recovery_time,
            })
    return pd.DataFrame(rows)


def plot_temporal_curve(
    df: pd.DataFrame,
    out_path: Path,
    anchors: Sequence[int],
    smooth_window: int,
    title: str,
) -> None:
    plt.figure(figsize=(12, 6))
    methods = list(df["method"].drop_duplicates())
    for method in methods:
        sdf = df[df["method"] == method].copy()
        agg = sdf.groupby("chunk_id")["weighted_f1"].mean().reset_index()
        y = smooth_series(agg["weighted_f1"].to_numpy(), smooth_window)
        plt.plot(agg["chunk_id"], y, label=method, linewidth=1.8)

    for anchor in sorted(anchors):
        plt.axvline(anchor, linestyle="--", linewidth=1.0, alpha=0.8, color="tab:blue")

    plt.title(title)
    plt.xlabel("Chunk ID")
    plt.ylabel("Weighted F1")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_aligned_recovery(
    aligned_df: pd.DataFrame,
    out_path: Path,
    title: str,
    smooth_window: int,
) -> None:
    plt.figure(figsize=(10, 6))
    if aligned_df.empty:
        plt.title(title + " (no aligned events available)")
        plt.xlabel("Relative Chunk (t)")
        plt.ylabel("Weighted F1")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return

    methods = list(aligned_df["method"].drop_duplicates())
    for method in methods:
        sdf = aligned_df[aligned_df["method"] == method]
        agg = sdf.groupby("relative_t")["weighted_f1"].agg(["mean", ci95]).reset_index()
        y = smooth_series(agg["mean"].to_numpy(), smooth_window)
        ci = agg["ci95"].to_numpy()
        x = agg["relative_t"].to_numpy()
        plt.plot(x, y, label=method, linewidth=1.8)
        plt.fill_between(x, np.maximum(0, y - ci), np.minimum(1.05, y + ci), alpha=0.15)

    plt.axvline(0, linestyle="--", linewidth=1.0, alpha=0.8, color="black")
    plt.title(title)
    plt.xlabel("Relative Chunk (t)")
    plt.ylabel("Weighted F1")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _bar_panel(ax, stats: pd.DataFrame, metric: str, title: str, ylim=None) -> None:
    if stats.empty:
        ax.set_title(title)
        return
    x = np.arange(len(stats))
    ax.bar(x, stats[f"{metric}_mean"].to_numpy())
    ax.errorbar(
        x,
        stats[f"{metric}_mean"].to_numpy(),
        yerr=stats[f"{metric}_ci95"].to_numpy(),
        fmt="none",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(stats["method"].tolist(), rotation=18)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_summary_bars(stats: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    _bar_panel(axes[0, 0], stats, "pre_drift_f1", "Pre-drift F1", ylim=(0.0, 1.05))
    _bar_panel(axes[0, 1], stats, "post_drift_min_f1", "Post-drift Minimum F1", ylim=(0.0, 1.05))
    _bar_panel(axes[1, 0], stats, "final_window_f1", "Final-window F1", ylim=(0.0, 1.05))
    _bar_panel(axes[1, 1], stats, "recovery_time", "Recovery Time", ylim=None)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def aggregate_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=["method"])

    metrics = [
        "pre_drift_f1",
        "post_drift_min_f1",
        "final_window_f1",
        "relative_drop",
        "recovery_time",
    ]

    rows = []
    for method, sdf in summary_df.groupby("method"):
        row = {"method": method, "n_events": int(len(sdf))}
        for metric in metrics:
            vals = pd.to_numeric(sdf[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_ci95"] = ci95(vals) if len(vals) else 0.0
        rows.append(row)

    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal drift analysis from trace JSON / NPZ.")
    parser.add_argument("--root", type=str, default="results", help="Root directory to auto-scan.")
    parser.add_argument("--input", action="append", default=[], help="Explicit input pattern: method=glob_pattern")
    parser.add_argument("--outdir", type=str, default="results/post_drift_analysis", help="Output directory.")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Chunk size when NPZ only has sample-level predictions.")
    parser.add_argument("--group-by-segment", action="store_true", help="For NPZ, aggregate per segment instead of fixed chunk size when segment labels exist.")
    parser.add_argument("--anchor-source", type=str, default="auto", choices=["auto", "segment_change", "drift_flag"], help="How to infer shared drift anchors.")
    parser.add_argument("--drift-anchors", type=str, default="", help="Explicit anchors, e.g. '96,156,229,343' or '[96,156]'.")
    parser.add_argument("--pre-window", type=int, default=5, help="Chunks before drift used for pre-drift F1.")
    parser.add_argument("--post-window", type=int, default=30, help="Chunks after drift used for aligned recovery and summary.")
    parser.add_argument("--final-window", type=int, default=5, help="Tail window size inside post-window.")
    parser.add_argument("--recovery-ratio", type=float, default=0.95, help="Recovery threshold as ratio of pre-drift F1.")
    parser.add_argument("--smooth-window", type=int, default=1, help="Rolling smoothing for plotted mean curves.")
    parser.add_argument("--temporal-title", type=str, default="Temporal Weighted F1 Curves")
    parser.add_argument("--aligned-title", type=str, default="Aligned Post-drift Performance Trajectories")
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    discovered: List[Tuple[str, Path, int]] = []
    if args.input:
        discovered.extend(parse_inputs(args.input))
    elif root.exists():
        discovered.extend(discover_files(root))
    else:
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    if not discovered:
        raise RuntimeError("No input files were discovered. Use --root or --input.")

    print("[INFO] Discovered files:")
    for method, path, seed in discovered:
        print(f"  - method={method:20s} seed={seed:<3d} file={path}")

    df = load_all_runs(
        discovered=discovered,
        chunk_size=args.chunk_size,
        group_by_segment=args.group_by_segment,
    )

    temporal_csv = outdir / "temporal_f1_long.csv"
    df.to_csv(temporal_csv, index=False)

    anchors = infer_shared_drift_anchors(
        df=df,
        anchor_source=args.anchor_source,
        explicit_anchors=parse_list_arg(args.drift_anchors),
    )
    print(f"[INFO] Shared drift anchors: {anchors if anchors else 'None'}")

    aligned_df = extract_aligned_windows(
        df=df,
        anchors=anchors,
        pre_window=args.pre_window,
        post_window=args.post_window,
    )
    aligned_df.to_csv(outdir / "aligned_recovery_long.csv", index=False)

    summary_df = summarize_post_drift(
        df=df,
        anchors=anchors,
        pre_window=args.pre_window,
        post_window=args.post_window,
        final_window=args.final_window,
        recovery_ratio=args.recovery_ratio,
    )
    summary_df.to_csv(outdir / "post_drift_summary.csv", index=False)

    summary_agg = aggregate_summary(summary_df)
    summary_agg.to_csv(outdir / "post_drift_summary_agg.csv", index=False)

    plot_temporal_curve(
        df=df,
        out_path=outdir / "temporal_weighted_f1.png",
        anchors=anchors,
        smooth_window=args.smooth_window,
        title=args.temporal_title,
    )
    plot_aligned_recovery(
        aligned_df=aligned_df,
        out_path=outdir / "aligned_post_drift_recovery.png",
        title=args.aligned_title,
        smooth_window=args.smooth_window,
    )
    plot_summary_bars(summary_agg, outdir / "post_drift_summary.png")

    print("\n[OK] Outputs saved to:", outdir)
    for name in [
        "temporal_weighted_f1.png",
        "aligned_post_drift_recovery.png",
        "post_drift_summary.png",
        "temporal_f1_long.csv",
        "aligned_recovery_long.csv",
        "post_drift_summary.csv",
        "post_drift_summary_agg.csv",
    ]:
        print("  -", name)


if __name__ == "__main__":
    main()
