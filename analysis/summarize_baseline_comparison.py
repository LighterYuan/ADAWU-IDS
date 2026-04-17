from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = PROJECT_ROOT / "results" / "cases"
TRACES_DIR = PROJECT_ROOT / "results" / "traces"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FILES = {
    "Static LSTM": CASES_DIR / "static_chunkwise_predictions.npz",
    "DWM-LSTM": CASES_DIR / "dwm_predictions.npz",
    "Online Bagging-LSTM": CASES_DIR / "online_bagging_predictions.npz",
    "Drift-Triggered Ensemble": CASES_DIR / "drift_triggered_predictions.npz",
    "ADAWU-IDS": CASES_DIR / "adawu_predictions.npz",
}


TRACE_FILES = {
    "Static LSTM": TRACES_DIR / "static_chunkwise_trace.json",
    "DWM-LSTM": TRACES_DIR / "dwm_trace.json",
    "Online Bagging-LSTM": TRACES_DIR / "online_bagging_trace.json",
    "Drift-Triggered Ensemble": TRACES_DIR / "drift_triggered_trace.json",
    "ADAWU-IDS": TRACES_DIR / "adawu_trace.json",
}


POST_DRIFT_SEGMENTS = {
    "Thursday-WorkingHours-Afternoon-Infilteration",
    "Friday-WorkingHours-Morning",
    "Friday-WorkingHours-Afternoon-PortScan",
}


def load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    return data["y_true"], data["y_pred"], data["segments"]


def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision(weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall(weighted)": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1(weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1(macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def load_trace(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_drift_zones_from_static(
    df_static: pd.DataFrame,
    low_f1_threshold: float = 0.80,
    min_run_length: int = 2,
    expand_left: int = 1,
    expand_right: int = 2,
) -> List[Tuple[int, int]]:
    """
    Detect severe degradation zones from Static LSTM chunk trace.
    A zone is a consecutive run of chunks where weighted_f1 < low_f1_threshold.
    Then expand the zone slightly on both sides for robustness.
    """
    df_static = df_static.sort_values("GlobalChunkID").reset_index(drop=True)
    bad = (df_static["F1(weighted)"] < low_f1_threshold).astype(int).to_numpy()
    chunk_ids = df_static["GlobalChunkID"].to_numpy()

    zones = []
    start = None

    for i, flag in enumerate(bad):
        if flag == 1 and start is None:
            start = i
        elif flag == 0 and start is not None:
            end = i - 1
            if end - start + 1 >= min_run_length:
                zone_start = max(0, start - expand_left)
                zone_end = min(len(chunk_ids) - 1, end + expand_right)
                zones.append((int(chunk_ids[zone_start]), int(chunk_ids[zone_end])))
            start = None

    if start is not None:
        end = len(bad) - 1
        if end - start + 1 >= min_run_length:
            zone_start = max(0, start - expand_left)
            zone_end = min(len(chunk_ids) - 1, end + expand_right)
            zones.append((int(chunk_ids[zone_start]), int(chunk_ids[zone_end])))

    # merge overlapping zones
    if not zones:
        return []

    merged = [zones[0]]
    for s, e in zones[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + 1:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def zone_label(idx: int, start: int, end: int) -> str:
    return f"Zone{idx + 1}[{start}-{end}]"


def compute_zone_stats(df_chunk: pd.DataFrame, zones: List[Tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for method, sub in df_chunk.groupby("Method"):
        sub = sub.sort_values("GlobalChunkID").reset_index(drop=True)

        for i, (z_start, z_end) in enumerate(zones):
            z = sub[(sub["GlobalChunkID"] >= z_start) & (sub["GlobalChunkID"] <= z_end)].copy()
            if z.empty:
                continue
            rows.append(
                {
                    "Method": method,
                    "Zone": zone_label(i, z_start, z_end),
                    "StartChunk": int(z_start),
                    "EndChunk": int(z_end),
                    "NumChunks": int(len(z)),
                    "Mean F1(weighted)": float(z["F1(weighted)"].mean()),
                    "Min F1(weighted)": float(z["F1(weighted)"].min()),
                    "Mean F1(macro)": float(z["F1(macro)"].mean()),
                    "Below0.8": int((z["F1(weighted)"] < 0.8).sum()),
                    "Below0.5": int((z["F1(weighted)"] < 0.5).sum()),
                }
            )
    return pd.DataFrame(rows)


def compute_worst_case_stats(df_chunk: pd.DataFrame, ks=(5, 10, 20)) -> pd.DataFrame:
    rows = []
    for method, sub in df_chunk.groupby("Method"):
        sub = sub.sort_values("F1(weighted)", ascending=True).reset_index(drop=True)

        base = {
            "Method": method,
            "NumChunksBelow0.8": int((sub["F1(weighted)"] < 0.8).sum()),
            "NumChunksBelow0.5": int((sub["F1(weighted)"] < 0.5).sum()),
            "MinChunkF1": float(sub["F1(weighted)"].min()),
            "MedianChunkF1": float(sub["F1(weighted)"].median()),
        }

        for k in ks:
            worst_k = sub.head(min(k, len(sub)))
            base[f"Worst{k}MeanF1"] = float(worst_k["F1(weighted)"].mean())
            base[f"Worst{k}MinF1"] = float(worst_k["F1(weighted)"].min())

        rows.append(base)
    return pd.DataFrame(rows)


def compute_old_post_drift_rows(df_trace: pd.DataFrame, method: str) -> List[Dict]:
    """
    Keep old segment-based post-drift stats for debugging/comparison,
    but do not rely on them as the main drift analysis.
    """
    rows = []
    post_df = df_trace[df_trace["segment"].isin(POST_DRIFT_SEGMENTS)].copy()
    if post_df.empty:
        return rows

    post_df = post_df.sort_values("global_chunk_id").reset_index(drop=True)

    immediate = post_df.iloc[: min(3, len(post_df))]
    recovery = post_df.iloc[3: min(10, len(post_df))]
    overall_post = post_df

    rows.append(
        {
            "Method": method,
            "Window": "Immediate Post-drift",
            "Mean F1(weighted)": float(immediate["weighted_f1"].mean()),
            "Mean F1(macro)": float(immediate["macro_f1"].mean()),
            "NumChunks": int(len(immediate)),
        }
    )

    if len(recovery) > 0:
        rows.append(
            {
                "Method": method,
                "Window": "Recovery",
                "Mean F1(weighted)": float(recovery["weighted_f1"].mean()),
                "Mean F1(macro)": float(recovery["macro_f1"].mean()),
                "NumChunks": int(len(recovery)),
            }
        )

    rows.append(
        {
            "Method": method,
            "Window": "All Post-drift Chunks",
            "Mean F1(weighted)": float(overall_post["weighted_f1"].mean()),
            "Mean F1(macro)": float(overall_post["macro_f1"].mean()),
            "NumChunks": int(len(overall_post)),
        }
    )

    return rows


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    seg_rows = []
    chunk_rows = []
    old_drift_rows = []

    # ---------- case-level and segment-level ----------
    for method, path in FILES.items():
        if not path.exists():
            print(f"[WARN] missing: {path}")
            continue

        y_true, y_pred, segments = load_npz(path)

        row = {"Method": method}
        row.update(compute_metrics(y_true, y_pred))

        mask = np.isin(segments, list(POST_DRIFT_SEGMENTS))
        if mask.any():
            row["Post-drift F1(weighted)"] = f1_score(
                y_true[mask], y_pred[mask], average="weighted", zero_division=0
            )
            row["Post-drift F1(macro)"] = f1_score(
                y_true[mask], y_pred[mask], average="macro", zero_division=0
            )
        else:
            row["Post-drift F1(weighted)"] = np.nan
            row["Post-drift F1(macro)"] = np.nan

        rows.append(row)

        for seg in np.unique(segments):
            seg_mask = segments == seg
            seg_rows.append(
                {
                    "Method": method,
                    "Segment": seg,
                    "Accuracy": accuracy_score(y_true[seg_mask], y_pred[seg_mask]),
                    "F1(weighted)": f1_score(
                        y_true[seg_mask], y_pred[seg_mask], average="weighted", zero_division=0
                    ),
                    "F1(macro)": f1_score(
                        y_true[seg_mask], y_pred[seg_mask], average="macro", zero_division=0
                    ),
                    "Samples": int(np.sum(seg_mask)),
                }
            )

    # ---------- trace/chunk-level ----------
    for method, path in TRACE_FILES.items():
        if not path.exists():
            print(f"[WARN] missing trace: {path}")
            continue

        trace = load_trace(path)
        df_trace = pd.DataFrame(trace)

        for _, r in df_trace.iterrows():
            chunk_rows.append(
                {
                    "Method": method,
                    "Segment": r["segment"],
                    "ChunkName": r["chunk_name"],
                    "LocalChunkID": r["local_chunk_id"],
                    "GlobalChunkID": r["global_chunk_id"],
                    "ChunkSize": r["size"],
                    "Accuracy": r["accuracy"],
                    "F1(weighted)": r["weighted_f1"],
                    "F1(macro)": r["macro_f1"],
                }
            )

        old_drift_rows.extend(compute_old_post_drift_rows(df_trace, method))

    df_main = pd.DataFrame(rows).sort_values("F1(weighted)", ascending=False)
    df_seg = pd.DataFrame(seg_rows)
    df_chunk = pd.DataFrame(chunk_rows).sort_values(["Method", "GlobalChunkID"]).reset_index(drop=True)
    df_old_drift = pd.DataFrame(old_drift_rows)

    # ---------- unified drift zones from Static LSTM ----------
    if not df_chunk.empty and "Static LSTM" in set(df_chunk["Method"]):
        df_static = df_chunk[df_chunk["Method"] == "Static LSTM"].copy()
        drift_zones = detect_drift_zones_from_static(
            df_static,
            low_f1_threshold=0.80,
            min_run_length=2,
            expand_left=1,
            expand_right=2,
        )
    else:
        drift_zones = []

    df_zones = pd.DataFrame(
        [
            {
                "Zone": zone_label(i, s, e),
                "StartChunk": int(s),
                "EndChunk": int(e),
                "Length": int(e - s + 1),
            }
            for i, (s, e) in enumerate(drift_zones)
        ]
    )

    df_zone_stats = compute_zone_stats(df_chunk, drift_zones) if (not df_chunk.empty and drift_zones) else pd.DataFrame()
    df_worst = compute_worst_case_stats(df_chunk, ks=(5, 10, 20)) if not df_chunk.empty else pd.DataFrame()

    # ---------- save ----------
    df_main.to_csv(TABLES_DIR / "baseline_comparison.csv", index=False, encoding="utf-8-sig")
    df_seg.to_csv(TABLES_DIR / "baseline_segment_comparison.csv", index=False, encoding="utf-8-sig")

    if not df_chunk.empty:
        df_chunk.to_csv(TABLES_DIR / "baseline_chunk_comparison.csv", index=False, encoding="utf-8-sig")

    if not df_old_drift.empty:
        df_old_drift.to_csv(TABLES_DIR / "baseline_drift_windows_legacy.csv", index=False, encoding="utf-8-sig")

    if not df_zones.empty:
        df_zones.to_csv(TABLES_DIR / "baseline_detected_drift_zones.csv", index=False, encoding="utf-8-sig")

    if not df_zone_stats.empty:
        df_zone_stats.to_csv(TABLES_DIR / "baseline_unified_drift_zone_stats.csv", index=False, encoding="utf-8-sig")

    if not df_worst.empty:
        df_worst.to_csv(TABLES_DIR / "baseline_worst_chunk_stats.csv", index=False, encoding="utf-8-sig")

    print("[OK] saved:", TABLES_DIR / "baseline_comparison.csv")
    print("[OK] saved:", TABLES_DIR / "baseline_segment_comparison.csv")

    if not df_chunk.empty:
        print("[OK] saved:", TABLES_DIR / "baseline_chunk_comparison.csv")
    if not df_old_drift.empty:
        print("[OK] saved:", TABLES_DIR / "baseline_drift_windows_legacy.csv")
    if not df_zones.empty:
        print("[OK] saved:", TABLES_DIR / "baseline_detected_drift_zones.csv")
    if not df_zone_stats.empty:
        print("[OK] saved:", TABLES_DIR / "baseline_unified_drift_zone_stats.csv")
    if not df_worst.empty:
        print("[OK] saved:", TABLES_DIR / "baseline_worst_chunk_stats.csv")


if __name__ == "__main__":
    main()
