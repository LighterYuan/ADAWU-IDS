from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "traces" / "paper_trace_6_5.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "summaries" / "recovery_summary_6_5.json"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input trace not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def first_recovery_step(
    curve: List[float],
    pre_value: float,
    ratio: float = 0.95,
) -> Optional[int]:
    threshold = pre_value * ratio
    for i, v in enumerate(curve):
        if v >= threshold:
            return i
    return None


def summarize_model(curve: List[float], pre_value: float) -> Dict[str, Any]:
    post_curve = curve[1:] if len(curve) > 1 else curve[:]
    min_post = min(post_curve) if post_curve else curve[0]
    drop_abs = pre_value - min_post
    drop_ratio = (drop_abs / pre_value) if pre_value > 0 else 0.0
    final_value = curve[-1]
    rec95 = first_recovery_step(curve[1:], pre_value, ratio=0.95)
    rec90 = first_recovery_step(curve[1:], pre_value, ratio=0.90)

    return {
        "pre_drift_weighted_f1": round(pre_value, 6),
        "post_drift_min_weighted_f1": round(min_post, 6),
        "absolute_drop": round(drop_abs, 6),
        "relative_drop_ratio": round(drop_ratio, 6),
        "final_window_weighted_f1": round(final_value, 6),
        "recovery_window_to_90pct": None if rec90 is None else int(rec90 + 1),
        "recovery_window_to_95pct": None if rec95 is None else int(rec95 + 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 6.5 recovery summary")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    trace = load_json(args.input)
    windows = trace.get("recovery_windows", [])
    anchors = trace.get("anchors", {})

    if not windows:
        raise ValueError("Trace contains no 'recovery_windows'.")

    adawu_curve = [float(w["adawu_weighted_f1"]) for w in windows]
    static_curve = [float(w["static_weighted_f1"]) for w in windows]

    pre_adawu = float(anchors["pre_drift_adawu_weighted_f1"])
    pre_static = float(anchors["pre_drift_static_weighted_f1"])

    summary = {
        "run_meta": trace.get("run_meta", {}),
        "anchors": anchors,
        "adawu_ids": summarize_model(adawu_curve, pre_adawu),
        "static_lstm": summarize_model(static_curve, pre_static),
    }

    ensure_parent(args.output)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {args.output}")


if __name__ == "__main__":
    main()
