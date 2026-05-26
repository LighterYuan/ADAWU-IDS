#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

MAIN_METHOD_ORDER = [
    "ADAWU-IDS",
    "DWM",
    "Online Bagging",
    "Leveraging Bagging",
    "Static LSTM",
]

def normalize_method_name(raw: str) -> str:
    s = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    joined = f" {s} "
    mapping = [
        ("ADAWU-IDS", [" adawu_ids ", " adawu ", " paper_trace ", " adawuids ", " adawu_trace ", " adawu_id "]),
        ("DWM", [" dwm "]),
        ("Online Bagging", [" online_bagging ", " onlinebagging ", " online_bagging_predictions "]),
        ("Leveraging Bagging", [" leveraging_bagging ", " leveragingbagging "]),
        ("Static LSTM", [" static_lstm ", " static ", " static_predictions ", " static_prediction ", " lstm_static ", " static_baseline "]),
    ]
    for canonical, patterns in mapping:
        if any(p in joined for p in patterns):
            return canonical
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

def parse_seed_from_name(path: Path) -> int:
    text = path.stem.lower()
    for pat in [r"seed[_\-]?(\d+)", r"run[_\-]?(\d+)", r"s[_\-]?(\d+)"]:
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
    return 0

def parse_list_arg(raw: str) -> List[int]:
    if not raw:
        return []
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            vals = ast.literal_eval(raw)
            return [int(x) for x in vals]
        except Exception:
            pass
    return [int(x) for x in re.split(r"[,\s]+", raw) if x.strip()]

def ci95(values: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))

def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    return pd.Series(values).rolling(window=window, min_periods=1, center=True).mean().to_numpy()

def _pick_npz_key(keys: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    norm = {str(k).lower(): k for k in keys}
    for cand in candidates:
        if cand.lower() in norm:
            return norm[cand.lower()]
    for k in keys:
        kl = str(k).lower()
        for cand in candidates:
            if cand.lower() in kl:
                return k
    return None

def _normalize_segments(seg_arr: np.ndarray) -> List[str]:
    out = []
    for x in seg_arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="ignore"))
        else:
            out.append(str(x))
    return out

def infer_method_from_path(path: Path) -> str:
    joined = " ".join([path.stem.lower(), path.parent.name.lower(), str(path).lower()])
    return normalize_method_name(joined)

def discover_files(root: Path) -> List[Tuple[str, Path, int]]:
    files = []
    for pattern in ["*.json", "*.npz"]:
        files.extend(root.rglob(pattern))
    triples = []
    for path in sorted(files):
        if path.name.startswith("."):
            continue
        method = infer_method_from_path(path)
        if method not in MAIN_METHOD_ORDER:
            continue
        triples.append((method, path, parse_seed_from_name(path)))
    return triples

def parse_inputs(inputs: Sequence[str]) -> List[Tuple[str, Path, int]]:
    triples = []
    for spec in inputs:
        if "=" not in spec:
            raise ValueError(f"--input expects method=glob_pattern, got: {spec}")
        raw_method, raw_glob = spec.split("=", 1)
        method = normalize_method_name(raw_method)
        if method not in MAIN_METHOD_ORDER:
            continue
        matched = [Path(x) for x in sorted(glob.glob(raw_glob, recursive=True))]
        if not matched:
            raise FileNotFoundError(f"No files matched input pattern: {spec}")
        for path in matched:
            triples.append((method, path, parse_seed_from_name(path)))
    return triples

def load_trace_as_df(path: Path, method: str, seed: int) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ["chunks", "trace", "records", "rows", "data"]:
            if key in data and isinstance(data[key], list):
                records = data[key]
                break
        else:
            if all(k in data for k in ["chunk_id", "weighted_f1"]):
                records = [data]
            else:
                raise ValueError(f"Could not find record list in JSON: {path}")
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unsupported JSON structure: {path}")
    rows = []
    f1_candidates = ["ensemble_weighted_f1", "weighted_f1", "f1_weighted", "wf1", "f1", "weightedF1"]
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        chunk_id = rec.get("chunk_id", rec.get("chunk", rec.get("chunk_index", i)))
        segment = rec.get("segment", rec.get("segment_name", rec.get("phase")))
        drift_detected = rec.get("drift_detected", rec.get("drift_flag", False))
        f1_val = None
        for key in f1_candidates:
            if key in rec and rec[key] is not None:
                f1_val = rec[key]
                break
        if f1_val is None:
            continue
        rows.append({
            "method": method,
            "seed": int(seed),
            "chunk_id": int(chunk_id),
            "weighted_f1": float(f1_val),
            "segment": None if segment is None else str(segment),
            "drift_detected": bool(drift_detected),
            "source_type": "json",
            "source_file": str(path),
        })
    if not rows:
        raise ValueError(f"No usable chunk-level records found in {path}")
    return pd.DataFrame(rows).sort_values("chunk_id").reset_index(drop=True)

def build_chunk_df_from_npz(path: Path, method: str, seed: int, chunk_size: int, group_by_segment: bool) -> pd.DataFrame:
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    f1_key = _pick_npz_key(keys, ["weighted_f1", "f1_weighted", "wf1", "chunk_weighted_f1"])
    seg_key = _pick_npz_key(keys, ["segment", "segments", "segment_id", "segment_ids"])
    if f1_key is not None:
        f1_arr = np.asarray(data[f1_key]).reshape(-1)
        rows = []
        segments = None
        if seg_key is not None:
            seg_arr = np.asarray(data[seg_key]).reshape(-1)
            if len(seg_arr) == len(f1_arr):
                segments = _normalize_segments(seg_arr)
        for i, f1_val in enumerate(f1_arr):
            rows.append({
                "method": method,
                "seed": int(seed),
                "chunk_id": int(i),
                "weighted_f1": float(f1_val),
                "segment": None if segments is None else str(segments[i]),
                "drift_detected": False,
                "source_type": "npz",
                "source_file": str(path),
            })
        return pd.DataFrame(rows).sort_values("chunk_id").reset_index(drop=True)
    y_true_key = _pick_npz_key(keys, ["y_true", "true", "labels", "targets"])
    y_pred_key = _pick_npz_key(keys, ["y_pred", "pred", "predictions"])
    if y_true_key is None or y_pred_key is None:
        raise ValueError(f"NPZ unsupported. Need chunk-level weighted_f1 or y_true/y_pred keys. keys={keys}")
    y_true = np.asarray(data[y_true_key]).reshape(-1)
    y_pred = np.asarray(data[y_pred_key]).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred length mismatch in {path}")
    segments = None
    if seg_key is not None:
        seg_arr = np.asarray(data[seg_key]).reshape(-1)
        if len(seg_arr) == len(y_true):
            segments = np.asarray(_normalize_segments(seg_arr))
    rows = []
    if group_by_segment and segments is not None:
        start = 0
        chunk_id = 0
        while start < len(y_true):
            end = start + 1
            while end < len(y_true) and segments[end] == segments[start]:
                end += 1
            rows.append({
                "method": method,
                "seed": int(seed),
                "chunk_id": int(chunk_id),
                "weighted_f1": float(f1_score(y_true[start:end], y_pred[start:end], average="weighted", zero_division=0)),
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
            seg_val = None
            if segments is not None:
                mode_seg = pd.Series(segments[start:end]).mode()
                seg_val = str(mode_seg.iloc[0]) if not mode_seg.empty else None
            rows.append({
                "method": method,
                "seed": int(seed),
                "chunk_id": int(chunk_id),
                "weighted_f1": float(f1_score(y_true[start:end], y_pred[start:end], average="weighted", zero_division=0)),
                "segment": seg_val,
                "drift_detected": False,
                "source_type": "npz",
                "source_file": str(path),
            })
            chunk_id += 1
    return pd.DataFrame(rows).sort_values("chunk_id").reset_index(drop=True)

def load_all_runs(discovered: Sequence[Tuple[str, Path, int]], chunk_size: int, group_by_segment: bool) -> pd.DataFrame:
    frames = []
    for method, path, seed in discovered:
        try:
            if path.suffix.lower() == ".json":
                frames.append(load_trace_as_df(path, method, seed))
            elif path.suffix.lower() == ".npz":
                frames.append(build_chunk_df_from_npz(path, method, seed, chunk_size, group_by_segment))
        except Exception as exc:
            print(f"[WARN] Skipped {path}: {exc}")
    if not frames:
        raise RuntimeError("No valid trace JSON or NPZ files could be loaded.")
    df = pd.concat(frames, ignore_index=True)
    df["method"] = df["method"].map(normalize_method_name)
    df = df[df["method"].isin(MAIN_METHOD_ORDER)].copy()
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").fillna(0).astype(int)
    df["chunk_id"] = pd.to_numeric(df["chunk_id"], errors="coerce").astype(int)
    df["weighted_f1"] = pd.to_numeric(df["weighted_f1"], errors="coerce")
    return df.dropna(subset=["weighted_f1"]).sort_values(["method", "seed", "chunk_id"]).reset_index(drop=True)

def infer_shared_drift_anchors(df: pd.DataFrame, anchor_source: str, explicit_anchors: Optional[List[int]]) -> List[int]:
    if explicit_anchors:
        return sorted({int(x) for x in explicit_anchors if int(x) >= 0})
    if anchor_source in {"segment_change", "auto"}:
        anchors = []
        runs = df[["method", "seed"]].drop_duplicates().shape[0]
        threshold = max(1, int(math.ceil(runs * 0.5)))
        for (_, _), sdf in df.groupby(["method", "seed"]):
            sdf = sdf.sort_values("chunk_id").reset_index(drop=True)
            if "segment" not in sdf.columns or sdf["segment"].isna().all():
                continue
            seg = sdf["segment"].astype(str).tolist()
            chunks = sdf["chunk_id"].tolist()
            for i in range(1, len(seg)):
                if seg[i] != seg[i - 1]:
                    anchors.append(int(chunks[i]))
        if anchors:
            vc = pd.Series(anchors).value_counts()
            chosen = sorted([int(k) for k, v in vc.items() if v >= threshold])
            if chosen or anchor_source == "segment_change":
                return chosen
    if anchor_source in {"drift_flag", "auto"}:
        anchors = []
        runs = df[["method", "seed"]].drop_duplicates().shape[0]
        threshold = max(1, int(math.ceil(runs * 0.3)))
        for (_, _), sdf in df.groupby(["method", "seed"]):
            sdf = sdf.sort_values("chunk_id").reset_index(drop=True)
            active = sdf.get("drift_detected", pd.Series(False, index=sdf.index)).fillna(False).astype(bool).to_numpy()
            chunks = sdf["chunk_id"].to_numpy()
            starts = np.where((active == True) & np.r_[True, active[:-1] == False])[0]
            anchors.extend([int(chunks[i]) for i in starts])
        if anchors:
            vc = pd.Series(anchors).value_counts()
            return sorted([int(k) for k, v in vc.items() if v >= threshold])
    return []

def deduplicate_close_anchors(anchors: Sequence[int], min_gap: int = 8) -> List[int]:
    if not anchors:
        return []
    anchors = sorted(set(int(a) for a in anchors))
    cleaned = [anchors[0]]
    for a in anchors[1:]:
        if a - cleaned[-1] >= min_gap:
            cleaned.append(a)
    return cleaned

def extract_aligned_windows(df: pd.DataFrame, anchors: Sequence[int], pre_window: int, post_window: int) -> pd.DataFrame:
    rows = []
    for (method, seed), sdf in df.groupby(["method", "seed"]):
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

def summarize_post_drift(df: pd.DataFrame, anchors: Sequence[int], pre_window: int, post_window: int, final_window: int, recovery_ratio: float, recovery_consecutive_k: int) -> pd.DataFrame:
    rows = []
    for (method, seed), sdf in df.groupby(["method", "seed"]):
        f1_map = dict(zip(sdf["chunk_id"].astype(int), sdf["weighted_f1"].astype(float)))
        for anchor in anchors:
            pre_vals = [f1_map[c] for c in range(anchor - pre_window, anchor) if c in f1_map]
            post_range = [c for c in range(anchor, anchor + post_window + 1) if c in f1_map]
            post_vals = [f1_map[c] for c in post_range]
            if not pre_vals or not post_vals:
                continue
            final_start = max(anchor, anchor + post_window - final_window + 1)
            final_vals = [f1_map[c] for c in range(final_start, anchor + post_window + 1) if c in f1_map]
            pre_f1 = float(np.mean(pre_vals))
            post_min = float(np.min(post_vals))
            final_f1 = float(np.mean(final_vals)) if final_vals else float(post_vals[-1])
            relative_drop = float((pre_f1 - post_min) / pre_f1) if pre_f1 > 1e-12 else np.nan
            target = recovery_ratio * pre_f1
            recovery_time = np.nan
            for idx, c in enumerate(post_range):
                ok = True
                for j in range(recovery_consecutive_k):
                    cj = c + j
                    if cj not in f1_map or f1_map[cj] < target:
                        ok = False
                        break
                if ok:
                    recovery_time = float(idx)
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

def aggregate_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["pre_drift_f1", "post_drift_min_f1", "final_window_f1", "relative_drop", "recovery_time"]
    rows = []
    for method in MAIN_METHOD_ORDER:
        sdf = summary_df[summary_df["method"] == method].copy()
        if sdf.empty:
            continue
        row = {"method": method, "n_events": int(len(sdf))}
        for metric in metrics:
            vals = pd.to_numeric(sdf[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_ci95"] = ci95(vals) if len(vals) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

def build_temporal_agg(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in MAIN_METHOD_ORDER:
        sdf = df[df["method"] == method].copy()
        if sdf.empty:
            continue
        agg = sdf.groupby("chunk_id")["weighted_f1"].agg(["mean", "std", "count"]).reset_index()
        agg["ci95"] = [0.0 if c <= 1 or pd.isna(s) else 1.96 * float(s) / math.sqrt(int(c)) for s, c in zip(agg["std"], agg["count"])]
        agg["method"] = method
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def build_aligned_agg(aligned_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in MAIN_METHOD_ORDER:
        sdf = aligned_df[aligned_df["method"] == method].copy()
        if sdf.empty:
            continue
        agg = sdf.groupby("relative_t")["weighted_f1"].agg(["mean", "std", "count"]).reset_index()
        agg["ci95"] = [0.0 if c <= 1 or pd.isna(s) else 1.96 * float(s) / math.sqrt(int(c)) for s, c in zip(agg["std"], agg["count"])]
        agg["method"] = method
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def truncate_aligned_to_common_horizon(aligned_agg: pd.DataFrame) -> pd.DataFrame:
    if aligned_agg.empty:
        return aligned_agg
    max_t_by_method = aligned_agg.groupby("method")["relative_t"].max().to_dict()
    min_t_by_method = aligned_agg.groupby("method")["relative_t"].min().to_dict()
    valid_methods = [m for m in MAIN_METHOD_ORDER if m in max_t_by_method and m in min_t_by_method]
    if not valid_methods:
        return aligned_agg
    common_max_t = min(max_t_by_method[m] for m in valid_methods)
    common_min_t = max(min_t_by_method[m] for m in valid_methods)
    out = aligned_agg[(aligned_agg["relative_t"] >= common_min_t) & (aligned_agg["relative_t"] <= common_max_t)].copy()
    print(f"[INFO] Truncated aligned horizon to common range: [{common_min_t}, {common_max_t}]")
    return out

def check_temporal_coverage(temporal_agg: pd.DataFrame) -> None:
    if temporal_agg.empty:
        return
    coverage = temporal_agg.groupby("method")["chunk_id"].agg(["min", "max", "count"]).reset_index()
    print("\n[INFO] Temporal coverage by method:")
    print(coverage.to_string(index=False))

def plot_temporal_curve(temporal_agg: pd.DataFrame, anchors: Sequence[int], out_path: Path, title: str, smooth_window: int, show_ci: bool) -> None:
    plt.figure(figsize=(12, 6))
    for method in MAIN_METHOD_ORDER:
        sdf = temporal_agg[temporal_agg["method"] == method].sort_values("chunk_id")
        if sdf.empty:
            continue
        x = sdf["chunk_id"].to_numpy()
        y = smooth_series(sdf["mean"].to_numpy(), smooth_window)
        ci = sdf["ci95"].to_numpy()
        plt.plot(x, y, label=method, linewidth=2.0)
        if show_ci:
            plt.fill_between(x, np.maximum(0, y - ci), np.minimum(1.05, y + ci), alpha=0.14)
    for anchor in sorted(set(int(a) for a in anchors)):
        plt.axvline(anchor, linestyle="--", linewidth=1.0, alpha=0.55)
    plt.title(title)
    plt.xlabel("Chunk ID")
    plt.ylabel("Weighted F1")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def plot_aligned_recovery(aligned_agg: pd.DataFrame, out_path: Path, title: str, smooth_window: int, show_ci: bool) -> None:
    plt.figure(figsize=(10, 6))
    if aligned_agg.empty:
        plt.title(title + " (no aligned events available)")
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
            plt.fill_between(x, np.maximum(0, y - ci), np.minimum(1.05, y + ci), alpha=0.14)
    plt.axvline(0, linestyle="--", linewidth=1.0, alpha=0.6, color="black")
    plt.title(title)
    plt.xlabel("Relative Step Around Drift")
    plt.ylabel("Weighted F1")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def _bar_panel(ax, stats: pd.DataFrame, metric: str, title: str, ylim=None) -> None:
    if stats.empty:
        ax.set_title(title)
        return
    x = np.arange(len(stats))
    means = stats[f"{metric}_mean"].to_numpy()
    ci = stats[f"{metric}_ci95"].to_numpy()
    ax.bar(x, means)
    ax.errorbar(x, means, yerr=ci, fmt="none", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(stats["method"].tolist(), rotation=18)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)

def plot_summary_bars(stats: pd.DataFrame, out_path: Path) -> None:
    ordered = stats.copy()
    ordered["order"] = ordered["method"].map({m: i for i, m in enumerate(MAIN_METHOD_ORDER)})
    ordered = ordered.sort_values("order").drop(columns=["order"])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    _bar_panel(axes[0, 0], ordered, "pre_drift_f1", "Pre-drift F1", ylim=(0.0, 1.05))
    _bar_panel(axes[0, 1], ordered, "post_drift_min_f1", "Post-drift Minimum F1", ylim=(0.0, 1.05))
    _bar_panel(axes[1, 0], ordered, "final_window_f1", "Final-window F1", ylim=(0.0, 1.05))
    _bar_panel(axes[1, 1], ordered, "relative_drop", "Relative Performance Drop", ylim=(0.0, 1.05))
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready post-drift figures from JSON/NPZ traces.")
    parser.add_argument("--root", type=str, default="results", help="Root directory to auto-scan.")
    parser.add_argument("--input", action="append", default=[], help="Explicit input pattern: method=glob_pattern")
    parser.add_argument("--outdir", type=str, default="results/post_drift_analysis", help="Output directory.")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Chunk size when NPZ only has sample-level predictions.")
    parser.add_argument("--group-by-segment", action="store_true", help="For NPZ, aggregate by segment when segment labels exist.")
    parser.add_argument("--anchor-source", type=str, default="auto", choices=["auto", "segment_change", "drift_flag"], help="How to infer shared drift anchors.")
    parser.add_argument("--drift-anchors", type=str, default="", help="Explicit anchors like '96,156,229,343'.")
    parser.add_argument("--anchor-min-gap", type=int, default=8, help="Minimum gap used to deduplicate nearby drift anchors.")
    parser.add_argument("--pre-window", type=int, default=5, help="Chunks before drift for pre-drift F1.")
    parser.add_argument("--post-window", type=int, default=30, help="Chunks after drift for alignment and summaries.")
    parser.add_argument("--final-window", type=int, default=5, help="Tail window size inside post-window.")
    parser.add_argument("--recovery-ratio", type=float, default=0.95, help="Recovery threshold as ratio of pre-drift F1.")
    parser.add_argument("--recovery-consecutive-k", type=int, default=3, help="Consecutive chunks required to count as recovered.")
    parser.add_argument("--smooth-window", type=int, default=1, help="Rolling smoothing window for displayed mean curves.")
    parser.add_argument("--temporal-show-ci", action="store_true", help="Show 95%% CI shading on the temporal figure.")
    parser.add_argument("--aligned-show-ci", action="store_true", help="Show 95%% CI shading on the aligned figure.")
    parser.add_argument("--temporal-title", type=str, default="Temporal Weighted F1 Curves")
    parser.add_argument("--aligned-title", type=str, default="Drift-centered Weighted F1 Trajectories")
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    discovered = parse_inputs(args.input) if args.input else discover_files(root)
    if not discovered:
        raise RuntimeError("No input files were discovered for the 5 main methods.")

    print("[INFO] Discovered files:")
    for method, path, seed in discovered:
        print(f"  - method={method:20s} seed={seed:<3d} file={path}")

    df = load_all_runs(discovered, args.chunk_size, args.group_by_segment)
    df.to_csv(outdir / "temporal_f1_long.csv", index=False)

    anchors = infer_shared_drift_anchors(df, args.anchor_source, parse_list_arg(args.drift_anchors))
    anchors = deduplicate_close_anchors(anchors, min_gap=args.anchor_min_gap)
    print(f"[INFO] Cleaned drift anchors: {anchors if anchors else 'None'}")

    aligned_df = extract_aligned_windows(df, anchors, args.pre_window, args.post_window)
    aligned_df.to_csv(outdir / "aligned_recovery_long.csv", index=False)

    summary_df = summarize_post_drift(
        df, anchors, args.pre_window, args.post_window,
        args.final_window, args.recovery_ratio, args.recovery_consecutive_k
    )
    summary_df.to_csv(outdir / "post_drift_summary.csv", index=False)

    summary_agg = aggregate_summary(summary_df)
    summary_agg.to_csv(outdir / "post_drift_summary_agg.csv", index=False)

    temporal_agg = build_temporal_agg(df)
    check_temporal_coverage(temporal_agg)
    temporal_agg.to_csv(outdir / "temporal_f1_agg.csv", index=False)

    aligned_agg = build_aligned_agg(aligned_df)
    aligned_agg = truncate_aligned_to_common_horizon(aligned_agg)
    aligned_agg.to_csv(outdir / "aligned_recovery_agg.csv", index=False)

    plot_temporal_curve(
        temporal_agg, anchors, outdir / "temporal_weighted_f1.png",
        args.temporal_title, args.smooth_window, args.temporal_show_ci
    )
    plot_aligned_recovery(
        aligned_agg, outdir / "aligned_post_drift_recovery.png",
        args.aligned_title, args.smooth_window, args.aligned_show_ci
    )
    plot_summary_bars(summary_agg, outdir / "post_drift_summary.png")

    print("\n[OK] Paper-ready outputs saved to:", outdir)

if __name__ == "__main__":
    main()
