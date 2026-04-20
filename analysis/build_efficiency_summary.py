from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "traces" / "paper_trace_6_6.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "summaries" / "efficiency_summary_6_6.json"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input trace not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * q
    lower = int(idx)
    upper = min(lower + 1, len(values) - 1)
    frac = idx - lower
    return float(values[lower] * (1 - frac) + values[upper] * frac)


def avg_for(records: List[Dict[str, Any]], key: str) -> float:
    return float(mean(float(r[key]) for r in records)) if records else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 6.6 efficiency summary")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    trace = load_json(args.input)
    windows = trace.get("efficiency_windows", [])
    if not windows:
        raise ValueError("Efficiency trace contains no efficiency_windows.")

    end_to_end_values = [float(w["end_to_end_latency_ms"]) for w in windows]
    retrain_count = sum(int(w["retrain_triggered"]) for w in windows)

    by_severity: Dict[str, List[Dict[str, Any]]] = {}
    for w in windows:
        sev = str(w["drift_severity"])
        by_severity.setdefault(sev, []).append(w)

    severity_summary: Dict[str, Dict[str, Any]] = {}
    for sev, records in by_severity.items():
        severity_summary[sev] = {
            "count": len(records),
            "mean_inference_latency_ms": round(avg_for(records, "inference_latency_ms"), 4),
            "mean_adaptation_latency_ms": round(avg_for(records, "adaptation_latency_ms"), 4),
            "mean_end_to_end_latency_ms": round(avg_for(records, "end_to_end_latency_ms"), 4),
            "mean_peak_memory_mb": round(avg_for(records, "peak_memory_mb"), 2),
            "mean_throughput_samples_per_sec": round(avg_for(records, "throughput_samples_per_sec"), 2),
        }

    summary = {
        "run_meta": trace.get("run_meta", {}),
        "overall": {
            "num_windows": len(windows),
            "mean_inference_latency_ms": round(avg_for(windows, "inference_latency_ms"), 4),
            "mean_adaptation_latency_ms": round(avg_for(windows, "adaptation_latency_ms"), 4),
            "mean_end_to_end_latency_ms": round(avg_for(windows, "end_to_end_latency_ms"), 4),
            "p95_end_to_end_latency_ms": round(percentile(end_to_end_values, 0.95), 4),
            "mean_throughput_samples_per_sec": round(avg_for(windows, "throughput_samples_per_sec"), 2),
            "peak_memory_mb": round(max(float(w["peak_memory_mb"]) for w in windows), 2),
            "retrain_events": int(retrain_count),
            "retrain_event_rate": round(retrain_count / len(windows), 4),
        },
        "by_severity": severity_summary,
    }

    ensure_parent(args.output)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {args.output}")


if __name__ == "__main__":
    main()
