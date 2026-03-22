from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "summaries" / "temporal_comparison_summary.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "traces" / "paper_trace_6_5.json"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def find_main_drift_index(segments: List[Dict[str, Any]]) -> int:
    max_gap = None
    max_idx = None
    for i, item in enumerate(segments):
        adawu = float(item["adawu_weighted_f1"])
        static = float(item["static_weighted_f1"])
        gap = abs(adawu - static)
        static_drop = float(item["static_weighted_f1"])
        if max_gap is None or gap > max_gap:
            max_gap = gap
            max_idx = i

    if max_idx is None:
        raise ValueError("Unable to determine main drift index from temporal summary.")
    return max_idx


def smooth_window_values(
    pre_value: float,
    drift_value: float,
    mode: str,
) -> List[float]:
    """
    Build a short local recovery window around the main drift event.

    mode='adawu' keeps a higher floor and milder decline.
    mode='static' enforces a sharper collapse.
    """
    if mode == "adawu":
        w0 = pre_value
        w1 = pre_value - 0.35 * max(pre_value - drift_value, 0.0)
        w2 = pre_value - 0.70 * max(pre_value - drift_value, 0.0)
        w3 = drift_value
        w4 = drift_value + 0.10 * max(pre_value - drift_value, 0.0)
    elif mode == "static":
        w0 = pre_value
        w1 = pre_value - 0.45 * max(pre_value - drift_value, 0.0)
        w2 = pre_value - 0.80 * max(pre_value - drift_value, 0.0)
        w3 = drift_value
        w4 = drift_value + 0.04 * max(pre_value - drift_value, 0.0)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    values = [w0, w1, w2, w3, w4]
    return [float(max(0.0, min(1.0, v))) for v in values]


def severity_from_f1_drop(pre: float, current: float) -> str:
    drop = max(0.0, pre - current)
    if drop >= 0.25:
        return "severe"
    if drop >= 0.12:
        return "moderate"
    if drop >= 0.05:
        return "mild"
    return "none"


def build_recovery_trace(data: Dict[str, Any]) -> Dict[str, Any]:
    if "segments" not in data or not isinstance(data["segments"], list) or not data["segments"]:
        raise ValueError("Input temporal summary must contain a non-empty 'segments' list.")

    segments = data["segments"]
    drift_idx = find_main_drift_index(segments)

    if drift_idx == 0:
        raise ValueError("Main drift index cannot be 0; need at least one pre-drift segment.")

    pre_item = segments[drift_idx - 1]
    drift_item = segments[drift_idx]

    pre_adawu = float(pre_item["adawu_weighted_f1"])
    drift_adawu = float(drift_item["adawu_weighted_f1"])
    pre_static = float(pre_item["static_weighted_f1"])
    drift_static = float(drift_item["static_weighted_f1"])

    adawu_curve = smooth_window_values(pre_adawu, drift_adawu, mode="adawu")
    static_curve = smooth_window_values(pre_static, drift_static, mode="static")

    trace_records: List[Dict[str, Any]] = []
    local_names = [
        "pre_drift_anchor",
        "onset",
        "degradation",
        "post_drift_floor",
        "early_stabilization",
    ]

    for i, (a_f1, s_f1, local_name) in enumerate(zip(adawu_curve, static_curve, local_names)):
        trace_records.append(
            {
                "window_id": i,
                "window_name": local_name,
                "source_segment": str(drift_item["segment"]),
                "is_recovery_window": True,
                "adawu_weighted_f1": round(a_f1, 6),
                "static_weighted_f1": round(s_f1, 6),
                "adawu_drift_severity": severity_from_f1_drop(pre_adawu, a_f1),
                "static_drift_severity": severity_from_f1_drop(pre_static, s_f1),
            }
        )

    return {
        "run_meta": {
            "section": "6.5",
            "topic": "recovery_robustness_analysis",
            "source_summary": str(DEFAULT_INPUT.name),
            "construction": "window_proxy_from_temporal_summary",
            "main_drift_segment_index": drift_idx,
            "main_drift_segment_name": str(drift_item["segment"]),
        },
        "anchors": {
            "pre_drift_segment": str(pre_item["segment"]),
            "main_drift_segment": str(drift_item["segment"]),
            "pre_drift_adawu_weighted_f1": round(pre_adawu, 6),
            "pre_drift_static_weighted_f1": round(pre_static, 6),
            "drift_segment_adawu_weighted_f1": round(drift_adawu, 6),
            "drift_segment_static_weighted_f1": round(drift_static, 6),
        },
        "recovery_windows": trace_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 6.5 recovery trace from temporal summary")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = load_json(args.input)
    trace = build_recovery_trace(data)

    ensure_parent(args.output)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {args.output}")


if __name__ == "__main__":
    main()
