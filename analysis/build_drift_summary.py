from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = PROJECT_ROOT / "results" / "traces" / "paper_trace_6_3.json"
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "summaries" / "drift_summary.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_records(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and "chunks" in data:
        return data["chunks"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported trace format")


def compute_detection_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((pred == 1) & (true == 1)))
    fp = int(np.sum((pred == 1) & (true == 0)))
    fn = int(np.sum((pred == 0) & (true == 1)))
    tn = int(np.sum((pred == 0) & (true == 0)))

    ddr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "DDR": round(ddr, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "FAR": round(far, 3),
    }


def count_severity(records: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {"mild": 0, "moderate": 0, "severe": 0}
    for r in records:
        if not r.get("drift_detected", False):
            continue
        s = str(r.get("drift_severity", "none")).lower()
        if s in out:
            out[s] += 1
    return out


def detector_vote_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    detector_names = sorted({
        name
        for r in records
        for name in r.get("detector_votes", {}).keys()
    })

    pos_counts = {name: 0 for name in detector_names}
    agreement = {
        "single_detector": 0,
        "two_detectors": 0,
        "three_or_more_detectors": 0,
    }

    for r in records:
        if not r.get("drift_detected", False):
            continue

        votes = r.get("detector_votes", {})
        n_pos = 0
        for name in detector_names:
            if bool(votes.get(name, False)):
                pos_counts[name] += 1
                n_pos += 1

        if n_pos == 1:
            agreement["single_detector"] += 1
        elif n_pos == 2:
            agreement["two_detectors"] += 1
        elif n_pos >= 3:
            agreement["three_or_more_detectors"] += 1

    return {
        "detector_positive_counts": pos_counts,
        "agreement_counts": agreement,
    }


def main():
    parser = argparse.ArgumentParser(description="Build drift summary for Section 6.3")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    if not args.trace.exists():
        raise FileNotFoundError(f"Trace not found: {args.trace}")

    data = load_json(args.trace)
    records = get_records(data)

    pred = np.array([1 if r.get("drift_detected", False) else 0 for r in records], dtype=int)
    true = np.array([1 if r.get("true_drift", False) else 0 for r in records], dtype=int)

    metrics = compute_detection_metrics(pred, true)
    sev = count_severity(records)
    vote_stats = detector_vote_stats(records)

    out = {
        "run_meta": {
            "n_chunks": len(records),
            "trace_path": str(args.trace),
            "true_drift_source": "trace_true_drift",
        },
        "drift_metrics": metrics,
        "drift_counts": {
            "detected_drifts": int(pred.sum()),
            "true_drifts": int(true.sum()),
        },
        "severity_statistics": sev,
        "detector_vote_statistics": vote_stats,
        "series_preview": {
            "chunk_ids": [r["chunk_id"] for r in records],
            "msdi": [r["msdi_score"] for r in records],
            "f1": [r["ensemble_weighted_f1"] for r in records],
            "detected": pred.tolist(),
            "true": true.tolist(),
        },
    }

    save_json(out, args.output)
    print(f"[OK] saved: {args.output}")


if __name__ == "__main__":
    main()
