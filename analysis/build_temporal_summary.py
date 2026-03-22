from pathlib import Path
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_STATIC = PROJECT_ROOT / "results" / "cases" / "static_predictions.npz"
INPUT_ADAWU = PROJECT_ROOT / "results" / "cases" / "adawu_predictions.npz"

OUTPUT_SUMMARY = PROJECT_ROOT / "results" / "summaries" / "temporal_comparison_summary.json"

SEGMENTS = [
    "Tuesday-WorkingHours",
    "Wednesday-workingHours",
    "Thursday-WorkingHours-Morning-WebAttacks",
    "Thursday-WorkingHours-Afternoon-Infilteration",
    "Friday-WorkingHours-Morning",
    "Friday-WorkingHours-Afternoon-PortScan",
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def validate_npz(data, name: str) -> None:
    required = {"y_true", "y_pred", "segments"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{name} missing keys: {sorted(missing)}")


def main():
    static = np.load(INPUT_STATIC, allow_pickle=True)
    adawu = np.load(INPUT_ADAWU, allow_pickle=True)

    validate_npz(static, "static_predictions.npz")
    validate_npz(adawu, "adawu_predictions.npz")

    s_seg = static["segments"].astype(str)
    a_seg = adawu["segments"].astype(str)

    results = []
    for seg in SEGMENTS:
        s_mask = s_seg == seg
        a_mask = a_seg == seg

        if s_mask.sum() == 0:
            raise ValueError(f"No static samples for segment: {seg}")
        if a_mask.sum() == 0:
            raise ValueError(f"No ADAWU samples for segment: {seg}")

        s_metrics = compute_metrics(static["y_true"][s_mask], static["y_pred"][s_mask])
        a_metrics = compute_metrics(adawu["y_true"][a_mask], adawu["y_pred"][a_mask])

        results.append({
            "segment": seg,
            "static_accuracy": s_metrics["accuracy"],
            "adawu_accuracy": a_metrics["accuracy"],
            "static_weighted_f1": s_metrics["weighted_f1"],
            "adawu_weighted_f1": a_metrics["weighted_f1"],
        })

    output = {
        "run_meta": {
            "dataset": "CICIDS2017",
            "protocol": "temporal_generalization_and_adaptation",
        },
        "segments": results,
    }

    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("[OK] saved:", OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
