# analysis/plot_post_drift_recovery.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
TRACE_DIR = RESULTS_DIR / "traces"
CASE_DIR = RESULTS_DIR / "cases"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Config
# =========================

METHOD_PATTERNS: Dict[str, List[str]] = {
    "ADAWU-IDS": [
        "paper_trace_seed",
        "paper_trace",
        "adawu_trace_seed",
        "adawu_trace",
        "adawu_predictions_seed",
        "adawu_predictions",
    ],
    "Static LSTM": [
        "static_trace_seed",
        "static_trace",
        "static_predictions_seed",
        "static_predictions",
    ],
    "DWM": [
        "dwm_trace_seed",
        "dwm_trace",
        "dwm_seed",
        "dwm_predictions_seed",
        "dwm_predictions",
    ],
    "Online Bagging": [
        "online_bagging_trace_seed",
        "online_bagging_trace",
        "online_bagging_seed",
        "online_bagging_predictions_seed",
        "online_bagging_predictions",
    ],
    "Leveraging Bagging": [
        "leveraging_bagging_trace_seed",
        "leveraging_bagging_trace",
        "leveraging_bagging_seed",
        "leveraging_bagging_predictions_seed",
        "leveraging_bagging_predictions",
    ],
}

# 统一的“段切换漂移锚点”
DEFAULT_DRIFT_ANCHOR_SEGMENTS = [
    "Thursday-WorkingHours-Morning-WebAttacks",
    "Thursday-WorkingHours-Afternoon-Infilteration",
    "Friday-WorkingHours-Morning",
    "Friday-WorkingHours-Afternoon-PortScan",
]

# 每次 drift 之后最多向后看多少个 chunk
RECOVERY_HORIZON = 6

# summary 统计参数
PRE_WINDOW = 1
POST_MIN_WINDOW = 3
FINAL_WINDOW = 1
RECOVERY_RATIO = 0.95

# 如果 recovery_time 全部缺失，是否仍然保留该面板
SHOW_RECOVERY_PANEL_WHEN_EMPTY = True


# =========================
# Basic helpers
# =========================

def normalize_segment_name(s: Any) -> str:
    s = str(s)
    s = s.replace(".pcap_ISCX", "")
    s = s.replace("_X.npy", "")
    s = s.replace("_y.npy", "")
    s = s.replace("\\", "/")
    s = s.split("/")[-1]
    return s.strip()


def extract_seed_from_name(path: Path) -> Optional[int]:
    stem = path.stem
    if "seed" not in stem:
        return None

    idx = stem.rfind("seed")
    suffix = stem[idx + 4 :]
    digits: List[str] = []
    for ch in suffix:
        if ch.isdigit():
            digits.append(ch)
        else:
            break

    if not digits:
        return None
    return int("".join(digits))


def ci95(std: float, n: int) -> float:
    if n <= 1 or pd.isna(std):
        return float("nan")
    return 1.96 * float(std) / math.sqrt(n)


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted"))


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        val = float(x)
        if math.isnan(val):
            return None
        return val
    except Exception:
        return None


def first_existing_value(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        if key in d:
            val = safe_float(d.get(key))
            if val is not None:
                return val
    return None


def find_weighted_f1_in_item(item: Dict[str, Any]) -> Optional[float]:
    # 顶层常见字段
    top_level_keys = [
        "ensemble_weighted_f1",
        "weighted_f1",
        "f1_weighted",
        "f1",
    ]
    val = first_existing_value(item, top_level_keys)
    if val is not None:
        return val

    # 嵌套字段
    for nested_key in ["metrics", "results", "scores"]:
        nested = item.get(nested_key)
        if isinstance(nested, dict):
            val = first_existing_value(
                nested,
                [
                    "ensemble_weighted_f1",
                    "weighted_f1",
                    "f1_weighted",
                    "f1",
                ],
            )
            if val is not None:
                return val

    return None


def ordered_unique(seq: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =========================
# Loaders
# =========================

def load_trace_json(path: Path, method_name: str) -> Optional[pd.DataFrame]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"[WARN] failed to read trace json: {path} ({e})")
        return None

    seed = extract_seed_from_name(path)
    chunks: Optional[List[Any]] = None

    # 兼容：
    # 1) {"seed": 1, "chunks": [...]}
    # 2) {"chunks": [...]}
    # 3) [ {...}, {...} ]
    if isinstance(obj, dict):
        chunks_obj = obj.get("chunks")
        if isinstance(chunks_obj, list):
            chunks = chunks_obj
        else:
            # 有些 trace 可能直接把 chunk 们放在 "trace" / "records" 之类
            for alt in ["trace", "records", "items"]:
                alt_obj = obj.get(alt)
                if isinstance(alt_obj, list):
                    chunks = alt_obj
                    break
        file_seed = obj.get("seed")
        if file_seed is not None:
            try:
                seed = int(file_seed)
            except Exception:
                pass

    elif isinstance(obj, list):
        chunks = obj

    else:
        print(f"[WARN] unsupported root type in {path}: {type(obj).__name__}")
        return None

    if not isinstance(chunks, list) or len(chunks) == 0:
        print(f"[WARN] no valid chunks found in {path}")
        return None

    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(chunks):
        if not isinstance(item, dict):
            continue

        raw_chunk_id = item.get("chunk_id", i)
        try:
            chunk_id = int(raw_chunk_id)
        except Exception:
            chunk_id = i

        segment = normalize_segment_name(item.get("segment", f"chunk_{chunk_id}"))
        f1_val = find_weighted_f1_in_item(item)

        if f1_val is None:
            continue

        rows.append(
            {
                "method": method_name,
                "seed": -1 if seed is None else int(seed),
                "chunk_id_raw": chunk_id,
                "segment": segment,
                "weighted_f1": float(f1_val),
                "source": "trace_json",
                "file_name": path.name,
            }
        )

    if not rows:
        print(f"[WARN] no usable rows parsed from {path}")
        return None

    return pd.DataFrame(rows)


def load_prediction_npz(path: Path, method_name: str) -> Optional[pd.DataFrame]:
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"[WARN] failed to read npz: {path} ({e})")
        return None

    files = set(data.files)

    if not {"y_true", "y_pred", "segments"}.issubset(files):
        print(f"[WARN] npz missing required arrays in {path}: {sorted(files)}")
        return None

    y_true = np.asarray(data["y_true"])
    y_pred = np.asarray(data["y_pred"])
    segments = np.asarray(data["segments"])

    if not (len(y_true) == len(y_pred) == len(segments)):
        print(f"[WARN] inconsistent lengths in {path}")
        return None

    seed = extract_seed_from_name(path)
    if "seed" in files:
        try:
            seed_arr = np.asarray(data["seed"]).reshape(-1)
            if len(seed_arr) > 0:
                seed = int(seed_arr[0])
        except Exception:
            pass

    seg_list = [normalize_segment_name(s) for s in segments]
    seg_series = pd.Series(seg_list)
    unique_segments = list(pd.unique(seg_series))

    rows: List[Dict[str, Any]] = []
    for i, seg in enumerate(unique_segments):
        mask = (seg_series == seg).to_numpy()
        if mask.sum() == 0:
            continue
        f1_val = weighted_f1(y_true[mask], y_pred[mask])

        rows.append(
            {
                "method": method_name,
                "seed": -1 if seed is None else int(seed),
                "chunk_id_raw": i,
                "segment": seg,
                "weighted_f1": float(f1_val),
                "source": "prediction_npz",
                "file_name": path.name,
            }
        )

    if not rows:
        return None

    return pd.DataFrame(rows)


def discover_files() -> Dict[str, List[Path]]:
    found: Dict[str, List[Path]] = {k: [] for k in METHOD_PATTERNS}

    for method, patterns in METHOD_PATTERNS.items():
        bucket: List[Path] = []
        for pat in patterns:
            bucket.extend(sorted(TRACE_DIR.glob(f"{pat}*.json")))
            bucket.extend(sorted(CASE_DIR.glob(f"{pat}*.npz")))
        # 去重
        unique_bucket = ordered_unique(bucket)
        found[method] = unique_bucket

    return found


# =========================
# Curve normalization
# =========================

def assign_global_chunk_ids(sub: pd.DataFrame) -> pd.DataFrame:
    """
    解决常见问题：
    - chunk_id 在不同 segment 里重置
    - chunk_id 不连续
    - 同一 segment 内可能只有单点
    这里统一为 method-seed 内的全局顺序索引。
    """
    sub = sub.copy()

    # 先按原始 chunk_id 排，再按 segment，最后按文件名保证稳定
    sub = sub.sort_values(
        by=["chunk_id_raw", "segment", "file_name", "weighted_f1"],
        kind="mergesort"
    ).reset_index(drop=True)

    # 如果同一 seed 下 segment 出现顺序能提供更稳定时间顺序，则按首次出现重排
    segment_order = (
        sub.groupby("segment", as_index=False)["chunk_id_raw"]
        .min()
        .sort_values("chunk_id_raw", kind="mergesort")["segment"]
        .tolist()
    )
    segment_rank = {seg: i for i, seg in enumerate(segment_order)}
    sub["segment_rank"] = sub["segment"].map(segment_rank)

    sub = sub.sort_values(
        by=["segment_rank", "chunk_id_raw", "file_name"],
        kind="mergesort"
    ).reset_index(drop=True)

    sub["chunk_id"] = np.arange(len(sub), dtype=int)
    sub.drop(columns=["segment_rank"], inplace=True)
    return sub


def load_all_method_curves() -> pd.DataFrame:
    found = discover_files()
    frames: List[pd.DataFrame] = []

    for method, files in found.items():
        if not files:
            print(f"[WARN] no files found for method={method}")
            continue

        print(f"[INFO] discovered {len(files)} files for method={method}")
        for path in files:
            print(f"[INFO] loading: {path}")

            df: Optional[pd.DataFrame] = None
            suffix = path.suffix.lower()

            if suffix == ".json":
                df = load_trace_json(path, method)
            elif suffix == ".npz":
                df = load_prediction_npz(path, method)

            if df is None or df.empty:
                print(f"[WARN] skipped: {path}")
                continue

            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No usable trace json or prediction npz files found in results/traces or results/cases."
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["method", "seed", "segment", "weighted_f1"]).copy()

    # method-seed 内统一全局 chunk_id
    fixed_parts: List[pd.DataFrame] = []
    for (method, seed), sub in df.groupby(["method", "seed"], sort=True):
        fixed_sub = assign_global_chunk_ids(sub)
        fixed_parts.append(fixed_sub)

    df = pd.concat(fixed_parts, ignore_index=True)
    df["seed"] = df["seed"].astype(int)
    df["chunk_id"] = df["chunk_id"].astype(int)

    df = df.sort_values(["method", "seed", "chunk_id"]).reset_index(drop=True)

    print("[INFO] loaded curve records by method:")
    print(df.groupby("method")["seed"].nunique())
    print("[INFO] total rows:", len(df))

    return df


# =========================
# Drift anchors
# =========================

def infer_drift_anchors(df: pd.DataFrame) -> List[int]:
    """
    基于 segment 切换点推断统一 drift anchors。
    这里不直接用原始 chunk_id_raw，而用归一化后的全局 chunk_id。
    """
    seg_chunk = (
        df[["chunk_id", "segment"]]
        .drop_duplicates()
        .sort_values("chunk_id")
        .reset_index(drop=True)
    )

    anchors: List[int] = []

    # 先按显式 segment 名匹配
    for seg in DEFAULT_DRIFT_ANCHOR_SEGMENTS:
        matched = seg_chunk[seg_chunk["segment"] == seg]
        if not matched.empty:
            anchors.append(int(matched.iloc[0]["chunk_id"]))

    anchors = sorted(set(anchors))

    # 如果一个都匹配不到，则退化为“segment 首次出现位置（除第一个段外）”
    if not anchors:
        ordered_segments = seg_chunk["segment"].tolist()
        seen = set()
        first_occurrence: List[Tuple[str, int]] = []
        for _, row in seg_chunk.iterrows():
            seg = row["segment"]
            cid = int(row["chunk_id"])
            if seg not in seen:
                seen.add(seg)
                first_occurrence.append((seg, cid))

        anchors = [cid for _, cid in first_occurrence[1:]]

    return sorted(set(anchors))


# =========================
# Plot 1: temporal curve
# =========================

def build_curve_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for method, sub in df.groupby("method", sort=True):
        pivot = sub.pivot_table(
            index="chunk_id",
            columns="seed",
            values="weighted_f1",
            aggfunc="mean",
        ).sort_index()

        for cid, vals in pivot.iterrows():
            arr = vals.dropna().to_numpy(dtype=float)
            if len(arr) == 0:
                continue
            rows.append(
                {
                    "method": method,
                    "chunk_id": int(cid),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "n": int(len(arr)),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Curve summary is empty.")
    return out.sort_values(["method", "chunk_id"]).reset_index(drop=True)


def plot_temporal_curve(df: pd.DataFrame, drift_anchors: List[int]) -> None:
    summary = build_curve_summary(df)

    plt.figure(figsize=(12, 6))

    for method in summary["method"].unique():
        part = summary[summary["method"] == method].sort_values("chunk_id")
        x = part["chunk_id"].to_numpy(dtype=float)
        y = part["mean"].to_numpy(dtype=float)
        s = part["std"].fillna(0.0).to_numpy(dtype=float)

        plt.plot(x, y, label=method, linewidth=2)
        plt.fill_between(x, y - s, y + s, alpha=0.15)

    for anchor in drift_anchors:
        plt.axvline(anchor, linestyle="--", linewidth=1)

    plt.xlabel("Chunk ID")
    plt.ylabel("Weighted F1")
    plt.title("Temporal Weighted F1 Curves")
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "temporal_weighted_f1.png", dpi=300)
    plt.close()


# =========================
# Plot 2: aligned post-drift trajectories
# =========================

def build_aligned_recovery_records(
    df: pd.DataFrame,
    drift_anchors: List[int],
    horizon: int = RECOVERY_HORIZON,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for (method, seed), sub in df.groupby(["method", "seed"], sort=True):
        sub = sub.sort_values("chunk_id")
        curve = dict(zip(sub["chunk_id"], sub["weighted_f1"]))

        for drift_id, anchor in enumerate(drift_anchors):
            for rel_t in range(horizon + 1):
                cid = anchor + rel_t
                if cid in curve:
                    rows.append(
                        {
                            "method": method,
                            "seed": int(seed),
                            "drift_id": int(drift_id),
                            "anchor_chunk": int(anchor),
                            "relative_t": int(rel_t),
                            "weighted_f1": float(curve[cid]),
                        }
                    )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["method", "seed", "drift_id", "relative_t"]).reset_index(drop=True)


def plot_aligned_recovery(aligned_df: pd.DataFrame) -> None:
    if aligned_df.empty:
        raise RuntimeError("Aligned recovery dataframe is empty.")

    rows: List[Dict[str, Any]] = []
    for (method, rel_t), sub in aligned_df.groupby(["method", "relative_t"], sort=True):
        vals = sub["weighted_f1"].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        rows.append(
            {
                "method": method,
                "relative_t": int(rel_t),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": int(len(vals)),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("Aligned recovery summary is empty.")

    plt.figure(figsize=(10, 5))

    for method in summary["method"].unique():
        part = summary[summary["method"] == method].sort_values("relative_t")
        x = part["relative_t"].to_numpy(dtype=float)
        y = part["mean"].to_numpy(dtype=float)
        s = part["std"].fillna(0.0).to_numpy(dtype=float)

        plt.plot(x, y, label=method, linewidth=2)
        plt.fill_between(x, y - s, y + s, alpha=0.15)

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Relative Chunk Index After Drift")
    plt.ylabel("Weighted F1")
    plt.title("Aligned Post-drift Performance Trajectories")
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "aligned_post_drift_recovery.png", dpi=300)
    plt.close()


# =========================
# Post-drift summary
# =========================

def compute_post_drift_summary(
    df: pd.DataFrame,
    drift_anchors: List[int],
    pre_window: int = PRE_WINDOW,
    post_min_window: int = POST_MIN_WINDOW,
    final_window: int = FINAL_WINDOW,
    recovery_ratio: float = RECOVERY_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回：
    1) per_event_df: 每个 method-seed-drift 的事件级结果
    2) summary_df: 按 method 汇总后的均值/std/n/ci95
    """
    event_rows: List[Dict[str, Any]] = []

    for (method, seed), sub in df.groupby(["method", "seed"], sort=True):
        sub = sub.sort_values("chunk_id")
        curve = dict(zip(sub["chunk_id"], sub["weighted_f1"]))
        max_chunk = int(sub["chunk_id"].max())

        for drift_id, anchor in enumerate(drift_anchors):
            pre_ids = [c for c in range(anchor - pre_window, anchor) if c in curve]
            if not pre_ids:
                continue

            pre_val = float(np.mean([curve[c] for c in pre_ids]))

            post_ids = [
                c
                for c in range(anchor, min(anchor + post_min_window, max_chunk + 1))
                if c in curve
            ]
            if not post_ids:
                continue

            post_min = float(np.min([curve[c] for c in post_ids]))

            next_anchors = [d for d in drift_anchors if d > anchor]
            seg_end = min(next_anchors) - 1 if next_anchors else max_chunk

            final_ids = [
                c
                for c in range(max(anchor, seg_end - final_window + 1), seg_end + 1)
                if c in curve
            ]
            final_val = (
                float(np.mean([curve[c] for c in final_ids]))
                if final_ids
                else float("nan")
            )

            threshold = pre_val * recovery_ratio
            recovery_time = np.nan

            for rel_t, cid in enumerate(range(anchor, seg_end + 1)):
                if cid in curve and curve[cid] >= threshold:
                    recovery_time = float(rel_t)
                    break

            event_rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "drift_id": int(drift_id),
                    "anchor_chunk": int(anchor),
                    "pre_drift_f1": pre_val,
                    "post_drift_min_f1": post_min,
                    "final_window_f1": final_val,
                    "recovery_time": recovery_time,
                    "recovery_threshold": threshold,
                }
            )

    per_event_df = pd.DataFrame(event_rows)
    if per_event_df.empty:
        return per_event_df, per_event_df

    metric_cols = [
        "pre_drift_f1",
        "post_drift_min_f1",
        "final_window_f1",
        "recovery_time",
    ]

    summary_rows: List[Dict[str, Any]] = []
    for method, sub in per_event_df.groupby("method", sort=True):
        row: Dict[str, Any] = {"method": method, "n_events": int(len(sub))}
        for col in metric_cols:
            vals = sub[col].dropna().to_numpy(dtype=float)
            row[f"{col}_mean"] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
            row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else (
                0.0 if len(vals) == 1 else float("nan")
            )
            row[f"{col}_n"] = int(len(vals))
            row[f"{col}_ci95"] = ci95(row[f"{col}_std"], int(len(vals))) if len(vals) > 0 else float("nan")
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("method").reset_index(drop=True)
    return per_event_df, summary_df


def plot_post_drift_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        raise RuntimeError("Summary dataframe is empty.")

    metrics = [
        ("pre_drift_f1_mean", "pre_drift_f1_ci95", "Pre-drift F1"),
        ("post_drift_min_f1_mean", "post_drift_min_f1_ci95", "Post-drift Minimum F1"),
        ("final_window_f1_mean", "final_window_f1_ci95", "Final-window F1"),
        ("recovery_time_mean", "recovery_time_ci95", "Recovery Time"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    methods = summary_df["method"].tolist()

    for ax, (mean_col, err_col, title) in zip(axes, metrics):
        values = summary_df[mean_col].to_numpy(dtype=float)
        errs = summary_df[err_col].to_numpy(dtype=float)

        valid_mask = np.isfinite(values)

        if valid_mask.any():
            x_labels = [m for m, keep in zip(methods, valid_mask) if keep]
            plot_values = values[valid_mask]
            plot_errs = errs[valid_mask]
            plot_errs = np.where(np.isfinite(plot_errs), plot_errs, 0.0)

            ax.bar(x_labels, plot_values, yerr=plot_errs, capsize=4)
            ax.tick_params(axis="x", rotation=20)
            ax.set_title(title)
        else:
            if SHOW_RECOVERY_PANEL_WHEN_EMPTY and title == "Recovery Time":
                ax.set_title(title)
                ax.text(
                    0.5,
                    0.5,
                    "No valid recovery time\nwithin current threshold/window",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "post_drift_summary.png", dpi=300)
    plt.close(fig)


# =========================
# Save tables
# =========================

def save_tables(per_event_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    per_event_df.to_csv(TABLE_DIR / "post_drift_per_event.csv", index=False)
    summary_df.to_csv(TABLE_DIR / "post_drift_summary.csv", index=False)


# =========================
# Main
# =========================

def main() -> None:
    df = load_all_method_curves()

    print("\n[INFO] curve rows by method / seed:")
    print(df.groupby(["method", "seed"]).size())

    drift_anchors = infer_drift_anchors(df)
    if not drift_anchors:
        raise RuntimeError(
            "No drift anchors inferred. Please check segment labels or edit "
            "DEFAULT_DRIFT_ANCHOR_SEGMENTS."
        )

    print(f"\n[INFO] inferred drift anchors: {drift_anchors}")

    # 图1：全局时间曲线
    plot_temporal_curve(df, drift_anchors)

    # 图2：对齐后的 post-drift trajectory
    aligned_df = build_aligned_recovery_records(
        df=df,
        drift_anchors=drift_anchors,
        horizon=RECOVERY_HORIZON,
    )
    if aligned_df.empty:
        raise RuntimeError("Aligned recovery dataframe is empty.")
    plot_aligned_recovery(aligned_df)

    # 图3 + 表
    per_event_df, summary_df = compute_post_drift_summary(
        df=df,
        drift_anchors=drift_anchors,
        pre_window=PRE_WINDOW,
        post_min_window=POST_MIN_WINDOW,
        final_window=FINAL_WINDOW,
        recovery_ratio=RECOVERY_RATIO,
    )
    if summary_df.empty:
        raise RuntimeError("Post-drift summary dataframe is empty.")

    save_tables(per_event_df, summary_df)
    plot_post_drift_summary(summary_df)

    print("\n[OK] Saved figures:")
    print(FIG_DIR / "temporal_weighted_f1.png")
    print(FIG_DIR / "aligned_post_drift_recovery.png")
    print(FIG_DIR / "post_drift_summary.png")

    print("\n[OK] Saved tables:")
    print(TABLE_DIR / "post_drift_per_event.csv")
    print(TABLE_DIR / "post_drift_summary.csv")


if __name__ == "__main__":
    main()
