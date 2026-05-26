#!/usr/bin/env python3
"""Build paper-ready component ablation tables from ablation_trace_*.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

VARIANT_ORDER = [
    "static_lstm_or_static_sgd",
    "w_o_msdi",
    "w_o_dynamic_weighting",
    "w_o_hierarchical_response",
    "full_adawu_ids",
]

DISPLAY = {
    "static_lstm_or_static_sgd": "Static model",
    "w_o_msdi": "w/o MSDI",
    "w_o_dynamic_weighting": "w/o ADAWU weighting",
    "w_o_hierarchical_response": "w/o hierarchical response",
    "full_adawu_ids": "Full ADAWU-IDS",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", type=str, default="results/traces/ablations")
    p.add_argument("--out-dir", type=str, default="results/tables")
    p.add_argument("--recovery-ratio", type=float, default=0.95)
    return p.parse_args()


def load_records(trace_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(trace_dir.glob("ablation_trace_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            rows.extend(json.load(f))
    if not rows:
        raise FileNotFoundError(f"No ablation_trace_*.json files found in {trace_dir}")
    return pd.DataFrame(rows)


def first_recovery_steps(sub: pd.DataFrame, ratio: float) -> float:
    sub = sub.sort_values("chunk_id")
    if sub.empty:
        return np.nan
    pre = sub.iloc[: max(1, min(2, len(sub)))] ["weighted_f1"].mean()
    threshold = ratio * pre
    drift_rows = sub[sub["drift_detected"] == True]
    if drift_rows.empty:
        anchor = int(sub["chunk_id"].iloc[0])
    else:
        anchor = int(drift_rows["chunk_id"].iloc[0])
    after = sub[sub["chunk_id"] >= anchor]
    recovered = after[after["weighted_f1"] >= threshold]
    if recovered.empty:
        return np.nan
    return float(int(recovered["chunk_id"].iloc[0]) - anchor)


def summarize_seed(sub: pd.DataFrame, ratio: float) -> Dict[str, float]:
    sub = sub.sort_values("chunk_id")
    pre = sub.iloc[: max(1, min(2, len(sub)))] ["weighted_f1"].mean()
    post_min = sub["weighted_f1"].min()
    final_f1 = sub.iloc[-1]["weighted_f1"]
    return {
        "accuracy": float(sub["accuracy"].mean()),
        "weighted_f1": float(sub["weighted_f1"].mean()),
        "pre_drift_f1": float(pre),
        "post_drift_min_f1": float(post_min),
        "final_window_f1": float(final_f1),
        "relative_drop": float((pre - post_min) / max(pre, 1e-12)),
        "recovery_steps": first_recovery_steps(sub, ratio),
        "detected_drifts": float(sub["drift_detected"].sum()),
        "mean_msdi": float(sub["msdi_score"].mean()),
        "mean_drift_confidence": float(sub["drift_confidence"].mean()),
    }


def mean_std(x: pd.Series) -> str:
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"


def main() -> None:
    args = parse_args()
    df = load_records(Path(args.trace_dir))
    seed_rows = []
    for (dataset, variant, seed), sub in df.groupby(["dataset", "variant", "seed"]):
        row = {"dataset": dataset, "variant": variant, "seed": seed}
        row.update(summarize_seed(sub, args.recovery_ratio))
        seed_rows.append(row)
    seed_df = pd.DataFrame(seed_rows)

    metrics = [
        "accuracy", "weighted_f1", "pre_drift_f1", "post_drift_min_f1",
        "final_window_f1", "relative_drop", "recovery_steps", "detected_drifts",
    ]
    table_rows = []
    for (dataset, variant), sub in seed_df.groupby(["dataset", "variant"]):
        row = {
            "Dataset": dataset,
            "Variant": DISPLAY.get(variant, variant),
            "MSDI": "✓" if variant not in ["static_lstm_or_static_sgd", "w_o_msdi"] else "×",
            "ADAWU weighting": "✓" if variant not in ["static_lstm_or_static_sgd", "w_o_dynamic_weighting"] else "×",
            "Hierarchical response": "✓" if variant not in ["static_lstm_or_static_sgd", "w_o_hierarchical_response"] else "×",
        }
        for m in metrics:
            row[m] = mean_std(sub[m])
        row["_order"] = VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else 99
        table_rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_path = out_dir / "ablation_seed_level_metrics.csv"
    table_path = out_dir / "ablation_component_table.csv"
    latex_path = out_dir / "ablation_component_table.tex"

    seed_df.to_csv(seed_path, index=False)
    table = pd.DataFrame(table_rows).sort_values(["Dataset", "_order"]).drop(columns=["_order"])
    table.to_csv(table_path, index=False)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(table.to_latex(index=False, escape=False))

    print("[OK] seed metrics:", seed_path)
    print("[OK] paper table:", table_path)
    print("[OK] latex table:", latex_path)


if __name__ == "__main__":
    main()
