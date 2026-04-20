from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = PROJECT_ROOT / "results" / "traces" / "paper_trace_6_4.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "summaries" / "weight_summary_6_4.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_records(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and "chunks" in data:
        return data["chunks"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported trace format")


def dominance_intervals(dominants: List[int]) -> Dict[int, List[int]]:
    out = {}
    if not dominants:
        return out

    cur = dominants[0]
    length = 1
    for d in dominants[1:]:
        if d == cur:
            length += 1
        else:
            out.setdefault(cur, []).append(length)
            cur = d
            length = 1
    out.setdefault(cur, []).append(length)
    return out


def main():
    parser = argparse.ArgumentParser(description="Build weight dynamics summary")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = load_json(args.trace)
    records = get_records(data)

    weights_after = np.array([r["weights_after"] for r in records], dtype=float)
    deltas_l1 = np.array([r["weight_delta_l1"] for r in records], dtype=float)
    deltas_l2 = np.array([r["weight_delta_l2"] for r in records], dtype=float)
    severity = [r["drift_severity"] for r in records]
    dominants = [int(r["dominant_model_after"]) for r in records]

    mean_weights = weights_after.mean(axis=0).tolist()
    dominance_counts = {str(i): int(np.sum(np.array(dominants) == i)) for i in range(weights_after.shape[1])}
    dominance_switch_count = int(np.sum(np.diff(np.array(dominants)) != 0))

    intervals = dominance_intervals(dominants)
    interval_stats = {}
    for k, vals in intervals.items():
        interval_stats[str(k)] = {
            "intervals": vals,
            "avg_interval": round(float(np.mean(vals)), 3),
            "max_interval": int(np.max(vals)),
        }

    severity_groups = {"mild": [], "moderate": [], "severe": [], "none": []}
    severity_groups_l2 = {"mild": [], "moderate": [], "severe": [], "none": []}
    for s, d1, d2 in zip(severity, deltas_l1, deltas_l2):
        severity_groups[s].append(float(d1))
        severity_groups_l2[s].append(float(d2))

    def mean_or_zero(xs):
        return round(float(np.mean(xs)), 4) if len(xs) > 0 else 0.0

    out = {
        "run_meta": {
            "n_chunks": len(records),
            "trace_path": str(args.trace),
        },
        "weight_statistics": {
            "mean_weights": [round(float(x), 4) for x in mean_weights],
            "mean_delta_l1": round(float(np.mean(deltas_l1)), 4),
            "mean_delta_l2": round(float(np.mean(deltas_l2)), 4),
        },
        "dominance_statistics": {
            "dominance_counts": dominance_counts,
            "dominance_switch_count": dominance_switch_count,
            "interval_stats": interval_stats,
        },
        "severity_update_statistics": {
            "mean_delta_l1_by_severity": {k: mean_or_zero(v) for k, v in severity_groups.items()},
            "mean_delta_l2_by_severity": {k: mean_or_zero(v) for k, v in severity_groups_l2.items()},
        },
    }

    save_json(out, args.output)
    print(f"[OK] saved: {args.output}")


if __name__ == "__main__":
    main()
